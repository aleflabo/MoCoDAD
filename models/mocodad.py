import argparse
import os
from math import prod
from typing import Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.stsae.stsae import STSAE, STSE
from models.stsae.stsae_unet import STSAE_Unet, STSE_Unet
from sklearn.metrics import roc_auc_score
from torch.optim import Adam
from tqdm import tqdm
from utils.diffusion_utils import Diffusion
from utils.eval_utils import (compute_var_matrix, filter_vectors_by_cond,
                              get_avenue_mask, get_hr_ubnormal_mask, pad_scores, score_process)
from utils.model_utils import processing_data


class MoCoDAD(pl.LightningModule):
    
    losses = {'l1':nn.L1Loss, 'smooth_l1':nn.SmoothL1Loss, 'l2':nn.MSELoss}
    conditioning_strategies = {'cat':'concat', 'concat':'concat', 
                               'add2layers':'inject', 'inject':'inject', 
                               'inbetween_imp':'interleave', 'interleave':'interleave',  
                               'random_indices':'random_imp', 'random_imp':'random_imp', 
                               'no_condition':'no_condition', 'none':'no_condition'}

    def __init__(self, args:argparse.Namespace) -> None:
        """
        This class implements MoCoDAD model.
        
        Args:
            args (argparse.Namespace): arguments containing the hyperparameters of the model
        """
        
        super(MoCoDAD, self).__init__()

        # Log the hyperparameters of the model
        self.save_hyperparameters(args)
        
        # Set the internal variables of the model
        # Data parameters
        self.n_frames = args.seg_len
        self.num_coords = args.num_coords
        self.n_joints = self._infer_number_of_joint(args)
        
        # Model parameters
        # Main network
        self.device_ = args.device
        self.embedding_dim = args.embedding_dim 
        self.dropout = args.dropout
        self.conditioning_strategy = self.conditioning_strategies[args.conditioning_strategy]
        # Conditioning network
        self.conditioning_indices = args.conditioning_indices
        self.n_frames_condition, self.n_frames_corrupt = self._set_conditioning_strategy()        
        self.conditioning_architecture = args.conditioning_architecture
        self.cond_h_dim = args.h_dim
        self.cond_latent_dim = args.latent_dim
        self.cond_channels = args.channels
        self.cond_dropout = args.dropout
        
        # Training and inference parameters
        self.learning_rate = args.opt_lr
        self.loss_fn = self.losses[args.loss_fn](reduction='none')
        self.rec_weight = args.rec_weight # weight of the reconstruction loss
        self.noise_steps = args.noise_steps
        self.aggregation_strategy = args.aggregation_strategy
        self.n_generated_samples = args.n_generated_samples
        self.model_return_value = args.model_return_value
        self.gt_path = args.gt_path
        self.split = args.split
        self.use_hr = args.use_hr
        self.ckpt_dir = args.ckpt_dir
        self.save_tensors = args.save_tensors
        self.num_transforms = args.num_transform
        self.pad_size = args.pad_size
        self.dataset_name = args.dataset_choice
        
        # Set the noise scheduler for the diffusion process
        self._set_diffusion_variables()
        
        # Build the model
        self.build_model()
        
    
    def build_model(self) -> None:
        """
        Build the model according to the specified hyperparameters.
        If the conditioning strategy is 'inject', the conditioning network is built and the available architectures are:
        AutoEncoder (AE), Encoder (E), Encoder-UNet (E_unet). For the other conditioning strategies, the conditioning network is set to `None`.

        Raises:
            NotImplementedError: if the conditioning architecture is not implemented
        """
        
        if self.conditioning_strategy == 'inject':
            if self.conditioning_architecture == 'AE':
                condition_encoder = STSAE(c_in=self.num_coords, h_dim=self.cond_h_dim, 
                                          latent_dim=self.cond_latent_dim, n_frames=self.n_frames_condition, 
                                          dropout=self.cond_dropout, n_joints=self.n_joints, 
                                          layer_channels=self.cond_channels, device=self.device_)
            elif self.conditioning_architecture == 'E':
                condition_encoder = STSE(c_in=self.num_coords, h_dim=self.cond_h_dim, 
                                         latent_dim=self.cond_latent_dim, n_frames=self.n_frames_condition, 
                                         dropout=self.cond_dropout, n_joints=self.n_joints, 
                                         layer_channels=self.cond_channels, device=self.device_)
            elif self.conditioning_architecture == 'E_unet':
                condition_encoder = STSE_Unet(c_in=self.num_coords, embedding_dim=self.embedding_dim,
                                              latent_dim=self.cond_latent_dim, n_frames=self.n_frames_condition,
                                              n_joints=self.n_joints, dropout=self.cond_dropout,
                                              unet_down_channels=self.cond_channels, device=self.device_, set_out_layer=True)
            else:
                raise NotImplementedError(f'Conditioning architecture {self.conditioning_architecture} not implemented.')
        else:
            condition_encoder = None
            
        model = STSAE_Unet(c_in=self.num_coords, embedding_dim=self.embedding_dim, 
                           n_frames=self.n_frames_corrupt, dropout=self.dropout, 
                           n_joints=self.n_joints, device=self.device_,
                           concat_condition=(self.conditioning_strategy == 'concat'), 
                           inject_condition=(self.conditioning_strategy == 'inject'))
        
        self.condition_encoder, self.model = condition_encoder, model
    
        
    def forward(self, input_data:List[torch.Tensor], aggr_strategy:str=None, return_:str=None) -> List[torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            input_data (List[torch.Tensor]): list containing the following tensors:
                                             - tensor_data: tensor of shape (B, C, T, V) containing the input sequences
                                             - transformation_idx
                                             - metadata
                                             - actual_frames
            aggr_strategy (str, optional): aggregation strategy to use. If not specified as a function parameter, the aggregation strategy 
                                           specified in the model hyperparameters is used. Defaults to None. 
            return_ (str, optional): return value of the model: 
                                     - only the selected poses according to the aggregation strategy ('pose')
                                     - only the loss of the selected poses ('loss')
                                     - both ('all'). 
                                     If not specified as a function parameter, the return value specified in the model hyperparameters is used. Defaults to None.

        Returns:
            List[torch.Tensor]: [predicted poses and the loss, tensor_data, transformation_idx, metadata, actual_frames]
        """
        
        # Unpack data: tensor_data is the input data, meta_out is a list of metadata
        tensor_data, meta_out = self._unpack_data(input_data)
        B = tensor_data.shape[0]
        # Select frames to condition on and to corrupt according to the conditioning strategy
        condition_data, corrupt_data, idxs = self._select_frames(tensor_data)
        # Encode the condition data
        condition_embedding, _ = self._encode_condition(condition_data)

        generated_xs = []
        for _ in range(self.n_generated_samples):
            # Generate gaussian noise of the same shape as the corrupt_data
            x = torch.randn_like(corrupt_data, device=self.device_)
            for i in reversed(range(1, self.noise_steps)):
                
                # Set the time step
                t = torch.full(size=(B,), fill_value=i, dtype=torch.long, device=self.device_)
                # Predict the noise
                predicted_noise = self._unet_forward(x, t=t, condition_data=condition_embedding, corrupt_idxs=idxs[1])
                # Get the alpha and beta values and expand them to the shape of the predicted noise
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                # Generate gaussian noise of the same shape as the predicted noise
                noise = torch.randn_like(x, device=self.device_) if i > 1 else torch.zeros_like(x, device=self.device_)
                # Recover the predicted sequence
                x = (1 / torch.sqrt(alpha) ) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
            # Append the generated sequence to the list of generated sequences
            generated_xs.append(x)
            
        selected_x, loss_of_selected_x = self._aggregation_strategy(generated_xs, corrupt_data, aggr_strategy)
        
        return self._pack_out_data(selected_x, loss_of_selected_x, [tensor_data] + meta_out, return_=return_)
        
        
    def training_step(self, batch:List[torch.Tensor], batch_idx:int) -> torch.float32:
        """
        Training step of the model.

        Args:
            batch (List[torch.Tensor]): list containing the following tensors:
                                        - tensor_data: tensor of shape (B, C, T, V) containing the input sequences
                                        - transformation_idx
                                        - metadata
                                        - actual_frames
            batch_idx (int): index of the batch

        Returns:
            torch.float32: loss of the model
        """
        
        # Unpack data: tensor_data is the input data, meta_out is a list of metadata
        tensor_data, _ = self._unpack_data(batch)
        # Select frames to condition on and to corrupt according to the conditioning strategy
        condition_data, corrupt_data, idxs = self._select_frames(tensor_data)
        # Encode the condition data
        condition_embedding, rec_cond_data = self._encode_condition(condition_data)
        # Sample the time steps and currupt the data
        t = self.noise_scheduler.sample_timesteps(corrupt_data.shape[0]).to(self.device_)
        x_t, noise = self.noise_scheduler.noise_graph(corrupt_data, t) 
        # Predict the noise
        predicted_noise = self._unet_forward(x_t, t=t, condition_data=condition_embedding, corrupt_idxs=idxs[1])
        # Compute the loss
        loss_noise = torch.mean(self.loss_fn(predicted_noise, noise))
        self.log('loss_noise', loss_noise)      

        if self.conditioning_architecture == 'AE':
            loss_rec_cond = F.mse_loss(rec_cond_data, condition_data)
            loss = loss_noise + loss_rec_cond * self.rec_weight
            self.log("loss_recons", loss_rec_cond)
        else:
            loss = loss_noise
            
        return loss
    
    
    def test_step(self, batch:List[torch.Tensor], batch_idx:int) -> None:
        """
        Test step of the model. It saves the output of the model and the input data as 
        List[torch.Tensor]: [predicted poses and the loss, tensor_data, transformation_idx, metadata, actual_frames]

        Args:
            batch (List[torch.Tensor]): list containing the following tensors:
                                        - tensor_data: tensor of shape (B, C, T, V) containing the input sequences
                                        - transformation_idx
                                        - metadata
                                        - actual_frames
            batch_idx (int): index of the batch
        """
        
        self._test_output_list.append(self.forward(batch))
        return
    
    
    def on_test_epoch_start(self) -> None:
        """
        Called when the test epoch begins.
        """
        
        super().on_test_epoch_start()
        self._test_output_list = []
        return
    
    
    def on_test_epoch_end(self) -> float:
        """
        Test epoch end of the model.

        Returns:
            float: test auc score
        """
        
        out, gt_data, trans, meta, frames = processing_data(self._test_output_list)
        del self._test_output_list
        if self.save_tensors:
            tensors = {'prediction':out, 'gt_data':gt_data, 
                       'trans':trans, 'metadata':meta, 'frames':frames}
            self._save_tensors(tensors, split_name=self.split, aggr_strategy=self.aggregation_strategy, n_gen=self.n_generated_samples)
        return self.post_processing(out, gt_data, trans, meta, frames)
    
    
    def validation_step(self, batch:List[torch.Tensor], batch_idx:int) -> None:
        """
        Validation step of the model. It saves the output of the model and the input data as 
        List[torch.Tensor]: [predicted poses and the loss, tensor_data, transformation_idx, metadata, actual_frames]

        Args:
            batch (List[torch.Tensor]): list containing the following tensors:
                                        - tensor_data: tensor of shape (B, C, T, V) containing the input sequences
                                        - transformation_idx
                                        - metadata
                                        - actual_frames
            batch_idx (int): index of the batch
        """
        
        self._validation_output_list.append(self.forward(batch))
        return
    
    
    def on_validation_epoch_start(self) -> None:
        """
        Called when the test epoch begins.
        """
        
        super().on_validation_epoch_start()
        self._validation_output_list = []
        return


    def on_validation_epoch_end(self) -> float:
        """
        Validation epoch end of the model.

        Returns:
            float: validation auc score
        """
        
        out, gt_data, trans, meta, frames = processing_data(self._validation_output_list)
        del self._validation_output_list
        if self.save_tensors:
            tensors = {'prediction':out, 'gt_data':gt_data, 
                       'trans':trans, 'metadata':meta, 'frames':frames}
            self._save_tensors(tensors, split_name=self.split, aggr_strategy=self.aggregation_strategy, n_gen=self.n_generated_samples)
        return self.post_processing(out, gt_data, trans, meta, frames)

    
    def configure_optimizers(self) -> Dict:
        """
        Configure the optimizers and the learning rate schedulers.

        Returns:
            Dict: dictionary containing the optimizers, the learning rate schedulers and the metric to monitor
        """
        
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99, last_epoch=-1, verbose=False)
        return {'optimizer':optimizer,'lr_scheduler':scheduler,'monitor':'validation_auc'}
            

    def post_processing(self, out:np.ndarray, gt_data:np.ndarray, trans:np.ndarray, meta:np.ndarray, frames:np.ndarray) -> float:
        """
        Post processing of the model.

        Args:
            out (np.ndarray): output of the model
            gt_data (np.ndarray): ground truth data
            trans (np.ndarray): transformation index
            meta (np.ndarray): metadata
            frames (np.ndarray): frame indexes of the data

        Returns:
            float: auc score
        """
        
        all_gts = [file_name for file_name in os.listdir(self.gt_path) if file_name.endswith('.npy')]
        all_gts = sorted(all_gts)
        scene_clips = [(int(fn.split('_')[0]), int(fn.split('_')[1].split('.')[0])) for fn in all_gts]

        num_transform = self.num_transforms
        model_scores_transf = {}
        dataset_gt_transf = {}

        for transformation in tqdm(range(num_transform)):
            # iterating over each transformation T
            
            dataset_gt = []
            model_scores = []
            cond_transform = (trans == transformation)

            out_transform, gt_data_transform, meta_transform, frames_transform = filter_vectors_by_cond([out, gt_data, meta, frames], cond_transform)

            for idx in range(len(all_gts)):
                # iterating over each clip C with transformation T
                
                scene_idx, clip_idx = scene_clips[idx]
                gt = np.load(os.path.join(self.gt_path, all_gts[idx]))
                n_frames = gt.shape[0]
                
                cond_scene_clip = (meta_transform[:, 0] == scene_idx) & (meta_transform[:, 1] == clip_idx)
                out_scene_clip, gt_scene_clip, meta_scene_clip, frames_scene_clip = filter_vectors_by_cond([out_transform, gt_data_transform, 
                                                                                                           meta_transform, frames_transform], 
                                                                                                           cond_scene_clip) 
                
                figs_ids = sorted(list(set(meta_scene_clip[:, 2])))
                error_per_person = []
                
                for fig in figs_ids:
                    # iterating over each actor A in each clip C with transformation T

                    cond_fig = (meta_scene_clip[:, 2] == fig)
                    out_fig, _, frames_fig = filter_vectors_by_cond([out_scene_clip, gt_scene_clip, frames_scene_clip], cond_fig) 

                    loss_matrix = compute_var_matrix(out_fig, frames_fig, n_frames)
                    loss_matrix = np.where(loss_matrix == 0.0, np.nan, loss_matrix)
                    fig_reconstruction_loss = np.nanmedian(loss_matrix, 0)
                    fig_reconstruction_loss = np.where(np.isnan(fig_reconstruction_loss), 0, fig_reconstruction_loss) 
                    if self.pad_size != -1:
                        fig_reconstruction_loss = pad_scores(fig_reconstruction_loss, gt, self.pad_size)                    
                    
                    error_per_person.append(fig_reconstruction_loss)

                clip_score = np.mean(np.stack(error_per_person, axis=0), axis=0)
                
                if self.use_hr:
                    hr_ubnormal_masked_clips = get_hr_ubnormal_mask(self.split)
                    if (scene_idx, clip_idx) in hr_ubnormal_masked_clips:
                        clip_score = clip_score[hr_ubnormal_masked_clips[(scene_idx, clip_idx)]]
                        gt = gt[hr_ubnormal_masked_clips[(scene_idx, clip_idx)]]
                
                # removing the non-HR frames for Avenue dataset
                masked_clips = get_avenue_mask()
                if clip_idx in masked_clips:
                    clip_score = clip_score[np.array(masked_clips[clip_idx])==1]
                    gt = gt[np.array(masked_clips[clip_idx])==1]

                clip_score = score_process(clip_score)
                model_scores.append(clip_score)
                dataset_gt.append(gt)
                    
            model_scores = np.concatenate(model_scores, axis=0)
            dataset_gt = np.concatenate(dataset_gt, axis=0)

            model_scores_transf[transformation] = model_scores
            dataset_gt_transf[transformation] = dataset_gt

        # aggregating the anomaly scores for all transformations
        pds = np.mean(np.stack(list(model_scores_transf.values()),0),0)
        gt = dataset_gt_transf[0]
        
        # computing the AUC
        auc = roc_auc_score(gt,pds)
        self.log('AUC', auc)
        
        return auc
    
    
    def test_on_saved_tensors(self, split_name:str) -> float:
        """
        Skip the prediction step and test the model on the saved tensors.

        Args:
            split_name (str): split name (val, test)

        Returns:
            float: auc score
        """
        
        tensors = self._load_tensors(split_name, self.aggregation_strategy, self.n_generated_samples)
        return self.post_processing(tensors['prediction'], tensors['gt_data'], tensors['trans'],
                                    tensors['metadata'], tensors['frames'])
        
    
    
    ## Helper functions
    
    def _aggregation_strategy(self, generated_xs:List[torch.Tensor], input_sequence:torch.Tensor, aggr_strategy:str) -> Tuple[torch.Tensor]:
        """
        Aggregates the generated samples and returns the selected one and its reconstruction error.
        Strategies:
            - all: returns all the generated samples
            - random: returns a random sample
            - best: returns the sample with the lowest reconstruction loss
            - worst: returns the sample with the highest reconstruction loss
            - mean: returns the mean of the losses of the generated samples
            - median: returns the median of the losses of the generated samples
            - mean_pose: returns the mean of the generated samples
            - median_pose: returns the median of the generated samples

        Args:
            generated_xs (List[torch.Tensor]): list of generated samples
            input_sequence (torch.Tensor): ground truth sequence
            aggr_strategy (str): aggregation strategy

        Raises:
            ValueError: if the aggregation strategy is not valid

        Returns:
            Tuple[torch.Tensor]: selected sample and its reconstruction error
        """

        aggr_strategy = self.aggregation_strategy if aggr_strategy is None else aggr_strategy 
        if aggr_strategy == 'random':
            return generated_xs[np.random.randint(len(generated_xs))]
        
        B, repr_shape = input_sequence.shape[0], input_sequence.shape[1:]
        compute_loss = lambda x: torch.mean(self.loss_fn(x, input_sequence).reshape(-1, prod(repr_shape)), dim=-1)
        losses = [compute_loss(x) for x in generated_xs]

        if aggr_strategy == 'all':
            dims_idxs = list(range(2, len(repr_shape)+2))
            dims_idxs = [1,0] + dims_idxs
            selected_x = torch.stack(generated_xs).permute(*dims_idxs)
            loss_of_selected_x = torch.stack(losses).permute(1,0)
        elif aggr_strategy == 'mean':
            selected_x = None
            loss_of_selected_x = torch.mean(torch.stack(losses), dim=0)
        elif aggr_strategy == 'mean_pose':
            selected_x = torch.mean(torch.stack(generated_xs), dim=0)
            loss_of_selected_x = compute_loss(selected_x)
        elif aggr_strategy == 'median':
            loss_of_selected_x, _ = torch.median(torch.stack(losses), dim=0)
            selected_x = None
        elif aggr_strategy == 'median_pose':
            selected_x, _ = torch.median(torch.stack(generated_xs), dim=0)
            loss_of_selected_x = compute_loss(selected_x)
        elif aggr_strategy == 'best' or aggr_strategy == 'worst':
            strategy = (lambda x,y: x < y) if aggr_strategy == 'best' else (lambda x,y: x > y)
            loss_of_selected_x = torch.full((B,), fill_value=(1e10 if aggr_strategy == 'best' else -1.), device=self.device_)
            selected_x = torch.zeros((B, *repr_shape)).to(self.device)

            for i in range(len(generated_xs)):
                mask = strategy(losses[i], loss_of_selected_x)
                loss_of_selected_x[mask] = losses[i][mask]
                selected_x[mask] = generated_xs[i][mask]
        else:
            raise ValueError(f'Unknown aggregation strategy {aggr_strategy}')
        
        return selected_x, loss_of_selected_x
    
    
    def _cut_array_from_indices(self, x:torch.Tensor, indices:torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Cuts the input array according to the indices. This function is called when the conditioning strategy is 'random imputation'.

        Args:
            x (torch.Tensor): input sequence
            indices (torch.Tensor): indices of the conditioning frames

        Returns:
            Tuple[torch.Tensor]: conditioning frames, non-conditioning frames, conditioning indices, non-conditioning indices
        """
        
        mask = indices < self.conditioning_indices
        indices_ = indices[mask].reshape(-1, self.conditioning_indices)
        not_indices = indices[~mask].reshape(-1, self.n_frames-self.conditioning_indices)
        mask = mask[:,None,None,:].expand(-1, self.num_coords, self.n_joints, -1)
        container = x[mask].reshape(-1, self.num_coords, self.n_joints, self.conditioning_indices)
        not_container = x[~mask].reshape(-1, self.num_coords, self.n_joints, self.n_frames-self.conditioning_indices)
        
        return container, not_container, indices_, not_indices
    
    
    def _encode_condition(self, condition_data:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes the conditioning data if the conditioning strategy is 'inject', returns None otherwise.

        Args:
            condition_data (torch.Tensor): conditioning data

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: encoded conditioning data, reconstructed conditioning data (if AE is used as condition encoder)
        """
        
        if self.condition_encoder is None:
            return None, None
        
        return self.condition_encoder(condition_data, t=None)
        
    
    def _infer_number_of_joint(self, args:argparse.Namespace) -> int:
        """
        Infer the number of joints based on the dataset parameters.

        Args:
            args (argparse.Namespace): arguments containing the hyperparameters of the model

        Returns:
            int: number of joints
        """
        
        if args.headless:
            joints_to_consider = 14
        elif args.kp18_format:
            joints_to_consider = 18
        else:
            joints_to_consider = 17
        return joints_to_consider
    
    
    def _load_tensors(self, split_name:str, aggr_strategy:str, n_gen:int) -> Dict[str, torch.Tensor]:
        name = 'saved_tensors_{}_{}_{}'.format(split_name, aggr_strategy, n_gen)
        path = os.path.join(self.ckpt_dir, name)
        tensor_files = os.listdir(path)
        tensors = {}
        for t_file in tensor_files:
            t_name = t_file.split('.')[0]
            tensors[t_name] = torch.load(os.path.join(path, t_file))
        return tensors
    
    
    def _pack_out_data(self, selected_x:torch.Tensor, loss_of_selected_x:torch.Tensor, additional_out:List[torch.Tensor], return_:str) -> List[torch.Tensor]:
        """
        Packs the output data according to the return_ argument.

        Args:
            selected_x (torch.Tensor): generated samples selected among the others according to the aggregation strategy
            loss_of_selected_x (torch.Tensor): loss of the selected samples
            additional_out (List[torch.Tensor]): additional output data (ground truth, meta-data, etc.)
            return_ (str): return strategy. Can be 'pose', 'loss', 'all'

        Raises:
            ValueError: if return_ is None and self.model_return_value is None

        Returns:
            List[torch.Tensor]: output data
        """
        
        if return_ is None:
            if self.model_return_value is None:
                raise ValueError('Either return_ or self.model_return_value must be set')
            else:
                return_ = self.model_return_value

        if return_ == 'pose':
            out = [selected_x]
        elif return_ == 'loss':
            out = [loss_of_selected_x]
        elif return_ == 'all':
            out = [loss_of_selected_x, selected_x]
            
        return out + additional_out
    
    
    def _pos_encoding(self, t:torch.Tensor, channels:int) -> torch.Tensor:
        """
        Encodes the time information of the input sequence.

        Args:
            t (torch.Tensor): time steps
            channels (int): dimension of the encoding

        Returns:
            torch.Tensor: encoded time steps
        """
        
        return self.model.pos_encoding(t, channels)
    
    
    def _save_tensors(self, tensors:Dict[str, torch.Tensor], split_name:str, aggr_strategy:str, n_gen:int) -> None:
        """
        Saves the tensors in the experiment directory.

        Args:
            tensors (Dict[str, torch.Tensor]): tensors to save
            split_name (str): name of the split (val, test)
            aggr_strategy (str): aggregation strategy
            n_gen (int): number of generated samples
        """
        
        name = 'saved_tensors_{}_{}_{}'.format(split_name, aggr_strategy, n_gen)
        path = os.path.join(self.ckpt_dir, name)
        if not os.path.exists(path):
            os.mkdir(path)
        for t_name, tensor in tensors.items():
            torch.save(tensor, os.path.join(path, t_name+'.pt'))
    
    
    def _select_frames(self, data:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Selects the conditioning frames according to the conditioning strategy.

        Args:
            data (torch.Tensor): input sequence

        Returns:
            Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]: conditioning frames, non-conditioning frames, indexes
        """
        
        if self.conditioning_strategy == 'random_imp':
            # Randomly select the indices of the conditioning frames and cut the input sequence accordingly
            indices = torch.tensor([torch.randperm(self.n_frames).tolist() for _ in range(data.shape[0])], device=self.device_)
            cond_data, corrupt_data, cond_idxs, corrupt_idxs = self._cut_array_from_indices(data.permute((0,1,3,2)), indices)
            cond_data = cond_data.permute((0,1,3,2))
            corrupt_data = corrupt_data.permute((0,1,3,2))
        
        elif self.conditioning_strategy == 'no_condition':
            # The input to the model is the whole sequence
            cond_data, cond_idxs = None, None
            corrupt_data, corrupt_idxs = data, torch.arange(self.n_frames, device=self.device_)
            
        elif len(self.conditioning_indices) == 0:
            if self.conditioning_strategy == 'interleave': 
                # Take conditioning frames with step equal to `conditioning indices` starting from the first frame
                cond_idxs = torch.arange(start=0, end=self.n_frames, step=self.conditioning_indices, device=self.device_)
                corrupt_idxs = torch.arange(start=1, end=self.n_frames, step=self.conditioning_indices, device=self.device_)
            else:
                # Use the integer in `conditioning indices` to split the input sequence in two parts
                cond_idxs = torch.arange(start=0, end=self.n_frames//self.conditioning_indices, device=self.device_)
                corrupt_idxs = torch.arange(start=self.n_frames//self.conditioning_indices, end=self.n_frames, device=self.device_)
            cond_data = torch.index_select(data, 2, cond_idxs)
            corrupt_data = torch.index_select(data, 2, corrupt_idxs)
            
        else:
            # Take the indices explicitly specified in `conditioning indices`
            cond_idxs = torch.tensor(self.conditioning_indices, device=self.device_)
            corrupt_idxs = torch.tensor([i for i in range(self.n_frames) if i not in self.conditioning_indices], device=self.device_)
            cond_data = torch.index_select(data, 2, cond_idxs)
            corrupt_data = torch.index_select(data, 2, corrupt_idxs) 

        return cond_data, corrupt_data, [cond_idxs, corrupt_idxs]
    
    
    def _set_conditioning_strategy(self) -> Tuple[int]:
        """
        Sets the conditioning strategy.

        Raises:
            NotImplementedError: if the conditioning strategy is not implemented

        Returns:
            Tuple[int]: number of conditioning frames, number of non-conditioning frames (input to the model)
        """
        
        if self.conditioning_strategy == 'no_condition':
            n_frames_cond = 0
            
        elif self.conditioning_strategy == 'concat' or self.conditioning_strategy == 'inject':
            if isinstance(self.conditioning_indices, int):
                n_frames_cond = self.n_frames // self.conditioning_indices
            else:
                n_frames_cond = len(self.conditioning_indices)
                assert self.conditioning_indices == list(range(min(self.conditioning_indices), max(self.conditioning_indices)+1)), \
                    'Conditioning indices must be a list of consecutive integers'
                assert (min(self.conditioning_indices) == 0) or (max(self.conditioning_indices) == (self.n_frames-1)), \
                    'Conditioning indices must start from 0 or end at the last frame'
                    
        elif self.conditioning_strategy == 'interleave' or self.conditioning_strategy == 'random_imp':
            if isinstance(self.conditioning_indices, int):
                n_frames_cond = self.conditioning_indices
            else:
                n_frames_cond = len(self.conditioning_indices)
                assert self.conditioning_strategy != 'random_imp', \
                    'Random imputation requires an integer number of frames to condition on, not a list of indices'
                    
        else:
            raise NotImplementedError(f'Conditioning strategy {self.conditioning_strategy} not implemented')
        
        n_frames_to_corrupt = self.n_frames - n_frames_cond
        return n_frames_cond, n_frames_to_corrupt
    
    
    def _set_diffusion_variables(self) -> None:
        """
        Sets the diffusion variables.
        """
        
        self.noise_scheduler = Diffusion(noise_steps=self.noise_steps, n_joints=self.n_joints,
                                         device=self.device_, time=self.n_frames)
        self.beta = self.noise_scheduler.schedule_noise().to(self.device_)
        self.alpha = (1. - self.beta).to(self.device_)
        self.alpha_hat = torch.cumprod(self.alpha, dim=0).to(self.device_)
        
        
    def _unet_forward(self, corrupt_data:torch.Tensor, t:torch.Tensor=None, condition_data:torch.Tensor=None, 
                      *, corrupt_idxs:torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass of the UNet model.

        Args:
            corrupt_data (torch.Tensor): input tensor of shape (batch_size, n_coords, n_joints, n_frames)
            corrupt_idxs (torch.Tensor): indices of the frames for which the model should predict the noise
            t (torch.Tensor, optional): encoded time tensor. Defaults to None.
            condition_data (torch.Tensor, optional): conditioning embedding of shape (batch_size, latent_dim). Defaults to None.

        Returns:
            torch.Tensor: predicted noise of shape (batch_size, n_coords, n_joints, n_frames)
        """
        
        prediction, _ = self.model(corrupt_data, t, condition_data=condition_data)
        
        if self.conditioning_strategy != 'inject':
            prediction = prediction[:, corrupt_idxs]
        
        return prediction 
    
    
    def _unpack_data(self, x:torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Unpacks the data.

        Args:
            x (torch.Tensor): list containing the input data, the transformation index, the metadata and the actual frames.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: input data, list containing the transformation index, the metadata and the actual frames.
        """
        tensor_data = x[0].to(self.device_)
        transformation_idx = x[1]
        metadata = x[2]
        actual_frames = x[3]
        meta_out = [transformation_idx, metadata, actual_frames]
        return tensor_data, meta_out

