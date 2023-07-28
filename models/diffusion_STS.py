import os
from typing import Dict, List, Tuple, Union
import argparse
from math import prod
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.stsae.stsae_unet import STSAE_Unet, STSE_Unet
from models.stsae.stsae import STSAE, STSE
from sklearn.metrics import roc_auc_score
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.diffusion_utils import Diffusion
from utils.eval_utils import (compute_var_matrix, filter_vectors_by_cond,
                              pad_scores, score_process)
from utils.model_utils import processing_data

V_01 = [1] * 75 + [0] * 46 + [1] * 269 + [0] * 47 + [1] * 427 + [0] * 47 + [1] * 20 + [0] * 70 + [1] * 438  # 1439 Frames
V_02 = [1] * 272 + [0] * 48 + [1] * 403 + [0] * 41 + [1] * 447  # 1211 Frames
V_03 = [1] * 293 + [0] * 48 + [1] * 582  # 923 Frames
V_04 = [1] * 947  # 947 Frames
V_05 = [1] * 1007  # 1007 Frames
V_06 = [1] * 561 + [0] * 64 + [1] * 189 + [0] * 193 + [1] * 276  # 1283 Frames
V_07_to_15 = [1] * 6457
V_16 = [1] * 728 + [0] * 12  # 740 Frames
V_17_to_21 = [1] * 1317
AVENUE_MASK = np.array(V_01 + V_02 + V_03 + V_04 + V_05 + V_06 + V_07_to_15 + V_16 + V_17_to_21) == 1

masked_clips = {
    1: V_01,
    2: V_02,
    3: V_03,
    6: V_06,
    16: V_16
}



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
        self.batch_size = args.dataset_batch_size
        self.n_frames = args.dataset_seg_len
        self.num_coords = args.num_coords
        self.n_joints = self._infer_number_of_joint(args)
        
        # Model parameters
        # Main network
        self.device_ = args.device
        self.embedding_dim = args.embedding_dim 
        self.dropout = args.dropout
        self.conditioning_strategy = self.conditioning_strategies[args.conditioning_strategy]
        self.conditioning_indices = args.conditioning_indices
        self.n_frames_condition, self.n_frames_corrupt = self._set_conditioning_strategy()        
        # Conditioning network
        self.conditioning_architecture = args.conditioning_architecture
        self.cond_h_dim = args.h_dim
        self.cond_latent_dim = args.latent_dim
        self.cond_channels = args.channels
        self.cond_dropout = args.dropout
        
        # Training and inference parameters
        self.learning_rate = args.opt_lr
        self.loss_fn = self.losses[args.loss_fn](reduction='none')
        self.lambda_ = args.lambda_ # weight of the reconstruction loss
        self.noise_steps = args.noise_steps
        self.aggregation_strategy = args.aggregation_strategy
        self.n_generated_samples = args.n_generated_samples
        self.model_return_value = args.model_return_value
        self.val_loss = 0 
        
        # Set the noise scheduler for the diffusion process
        self._set_diffusion_variables()
        
        # Build the model
        self.build_model()
        
    
    def build_model(self) -> None:
        
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
        
    
    def _set_diffusion_variables(self) -> None:
        self.noise_scheduler = Diffusion(noise_steps=self.noise_steps, n_joints=self.n_joints,
                                         device=self.device_, time=self.n_frames)
        self.beta = self.noise_scheduler.schedule_noise().to(self.device_)
        self.alpha = (1. - self.beta).to(self.device_)
        self.alpha_hat = torch.cumprod(self.alpha, dim=0).to(self.device_)
        
    
    def _infer_number_of_joint(self, args:argparse.Namespace) -> int:
        """
        Infer the number of joints based on the dataset parameters.

        Args:
            args (argparse.Namespace): arguments containing the hyperparameters of the model

        Returns:
            int: number of joints
        """
        
        if args.dataset_headless:
            joints_to_consider = 14
        elif args.dataset_kp18_format:
            joints_to_consider = 18
        else:
            joints_to_consider = 17
        return joints_to_consider
    
    
    def _set_conditioning_strategy(self) -> Tuple[int]:
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
    
    
    def _pos_encoding(self, t, channels):
        return self.model.pos_encoding(t, channels)
    
    
    def _encode_condition(self, condition_data:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.condition_encoder is None:
            return None, None
        
        return self.condition_encoder(condition_data, t=None)
    
    
    def _unpack_data(self, x:torch.Tensor) -> List[torch.Tensor]:
        tensor_data = x[0].to(self.device_)
        transformation_idx = x[1]
        metadata = x[2]
        actual_frames = x[3]
        meta_out = [transformation_idx, metadata, actual_frames]
        return tensor_data, meta_out
    
    
    def _pack_out_data(self, selected_x:torch.Tensor, loss_of_selected_x:torch.Tensor, meta_out:List[torch.Tensor], return_:str) -> torch.Tensor:
        
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
            out = [selected_x, loss_of_selected_x]
            
        return out + meta_out
    
    
    def _cut_array_from_indices(self, x:torch.Tensor, indices:torch.Tensor) -> Tuple[torch.Tensor]:
    
        mask = indices < self.conditioning_indices
        indices_ = indices[mask].reshape(-1, self.conditioning_indices)
        not_indices = indices[~mask].reshape(-1, self.n_frames-self.conditioning_indices)
        mask = mask[:,None,None,:].expand(-1, self.num_coords, self.n_joints, -1)
        container = x[mask].reshape(-1, self.num_coords, self.n_joints, self.conditioning_indices)
        not_container = x[~mask].reshape(-1, self.num_coords, self.n_joints, self.n_frames-self.conditioning_indices)
        
        return container, not_container, indices_, not_indices
    
    
    def _select_frames(self, data:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        
        if self.conditioning_strategy == 'random_imp':

            indices = torch.tensor([torch.randperm(self.n_frames).tolist() for _ in range(data.shape[0])], device=self.device_)
            cond_data, corrupt_data, cond_idxs, corrupt_idxs = self._cut_array_from_indices(data.permute((0,1,3,2)), indices)
            cond_data = cond_data.permute((0,1,3,2))
            corrupt_data = corrupt_data.permute((0,1,3,2))
        
        elif self.conditioning_strategy == 'no_condition':
            cond_data, cond_idxs = None, None
            corrupt_data, corrupt_idxs = data, torch.arange(self.n_frames, device=self.device_)
            
        elif len(self.conditioning_indices) == 0:
            if self.conditioning_strategy == 'interleave': 
                cond_idxs = torch.arange(start=0, end=self.n_frames, step=self.conditioning_indices, device=self.device_)
                corrupt_idxs = torch.arange(start=1, end=self.n_frames, step=self.conditioning_indices, device=self.device_)
            else:
                cond_idxs = torch.arange(start=0, end=self.n_frames//self.conditioning_indices, device=self.device_)
                corrupt_idxs = torch.arange(start=self.n_frames//self.conditioning_indices, end=self.n_frames, device=self.device_)
            cond_data = torch.index_select(data, 2, cond_idxs)
            corrupt_data = torch.index_select(data, 2, corrupt_idxs)
            
        else:
            cond_idxs = torch.tensor(self.conditioning_indices, device=self.device_)
            corrupt_idxs = torch.tensor([i for i in range(self.n_frames) if i not in self.conditioning_indices], device=self.device_)
            cond_data = torch.index_select(data, 2, cond_idxs)
            corrupt_data = torch.index_select(data, 2, corrupt_idxs) 

        return cond_data, corrupt_data, [cond_idxs, corrupt_idxs]
    
    
    def _unet_forward(self, corrupt_data:torch.Tensor, t:torch.Tensor=None, condition_data:torch.Tensor=None, 
                      *, corrupt_idxs:torch.Tensor) -> torch.Tensor:
        
        prediction, _ = self.model(corrupt_data, t, condition_data=condition_data)
        
        if self.conditioning_strategy != 'inject':
            prediction = prediction[:, corrupt_idxs]
        
        return prediction 
    
    
    def _aggregation_strategy(self, generated_xs:List[torch.Tensor], input_sequence:torch.Tensor, aggr_strategy:str):
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
    
        
    def forward(self, input_data:List[torch.Tensor], condition_data:torch.Tensor=None, aggr_strategy:str=None, return_:str=None) -> List[torch.Tensor]:
        
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
        
        return self._pack_out_data(selected_x, loss_of_selected_x, meta_out, return_=return_)
        
        
    def training_step(self, batch:List[torch.Tensor], batch_idx:int) -> torch.float32:
        # Unpack data: tensor_data is the input data, meta_out is a list of metadata
        tensor_data, _ = self._unpack_data(batch)
        # Select frames to condition on and to corrupt according to the conditioning strategy
        condition_data, corrupt_data, idxs = self._select_frames(tensor_data)
        # Encode the condition data
        condition_embedding, rec_cond_data = self._encode_condition(condition_data)
        # Sample the time steps and currupt the data
        t = self.noise_scheduler.sample_timesteps(corrupt_data.shape[0]).to(self.device_)
        x_t, noise = self.noise_scheduler.noise_graph(corrupt_data, t.to(corrupt_data.get_device())) 
        # Predict the noise
        predicted_noise = self._unet_forward(x_t, t=t, condition_data=condition_embedding, corrupt_idxs=idxs[1])
        # Compute the loss
        loss_noise = torch.mean(self.loss_fn(predicted_noise, noise))
        self.log('loss_noise', loss_noise)      

        if self.conditioning_architecture == 'AE':
            loss_rec_cond = F.mse_loss(rec_cond_data, condition_data)
            loss = loss_noise + loss_rec_cond * self.lambda_
            self.log("loss_recons", loss_rec_cond)
        else:
            loss = loss_noise
            
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        return self.forward(batch)


    def validation_epoch_end(self, validation_step_outputs:Union[Tuple[torch.tensor],List[torch.tensor]]):
        out, gt_data, trans, meta, frames = processing_data(validation_step_outputs)
        return self.post_processing(out, gt_data, trans, meta, frames)

    
    def configure_optimizers(self) -> Dict:
        optimizer = Adam(self.parameters(), lr=self.learning_rate)

        """scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                    mode='max',
                                                                    factor=0.2,
                                                                    patience=5,
                                                                    min_lr=1e-6,
                                                                    verbose=True)"""
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99, last_epoch=- 1, verbose=False)
        return {'optimizer':optimizer,'lr_scheduler':scheduler,'monitor':'validation_auc'}
        
        
                    

    def post_processing(self, out:np.ndarray, gt_data:np.ndarray, trans:int, meta:np.ndarray, frames:np.ndarray) -> float:
        all_gts = [file_name for file_name in os.listdir(self.args.gt_path) if file_name.endswith('.npy')]
        all_gts = sorted(all_gts)
        scene_clips = [(int(fn.split('_')[0]), int(fn.split('_')[1].split('.')[0])) for fn in all_gts]

        num_transform = self.args.dataset_num_transform
        smoothing = self.args.smoothing
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
                gt = np.load(os.path.join(self.args.gt_path, all_gts[idx]))
                n_frames = gt.shape[0]
                
                cond_scene_clip = (meta_transform[:, 0] == scene_idx) & (meta_transform[:, 1] == clip_idx)
                out_scene_clip, gt_scene_clip, meta_scene_clip, frames_scene_clip = filter_vectors_by_cond([out_transform, gt_data_transform, meta_transform, frames_transform], cond_scene_clip) 
                
                figs_ids = sorted(list(set(meta_scene_clip[:, 2])))
                error_per_person = []
                
                for fig in figs_ids:
                    # iterating over each actor A in each clip C with transformation T
                    
                    cond_fig = (meta_scene_clip[:, 2] == fig)
                    out_fig, gt_fig, frames_fig = filter_vectors_by_cond([out_scene_clip, gt_scene_clip, frames_scene_clip], cond_fig) 

                    
                    loss_matrix = compute_var_matrix(out_fig, frames_fig, n_frames)
                    loss_matrix = np.where(loss_matrix == 0.0, np.nan, loss_matrix)
                    fig_reconstruction_loss = np.nanmedian(loss_matrix, 0)
                    fig_reconstruction_loss = np.where(np.isnan(fig_reconstruction_loss), 0, fig_reconstruction_loss) 
                    if self.args.pad_size!=-1:
                        fig_reconstruction_loss = pad_scores(fig_reconstruction_loss, gt, self.args.pad_size)                    
                    
                    error_per_person.append(fig_reconstruction_loss)

                clip_score = np.amax(np.stack(error_per_person, axis=0), axis=0)
                
                # removing the non-HR frames for Avenue dataset
                if clip_idx in masked_clips:
                    clip_score = clip_score[np.array(masked_clips[clip_idx])==1]
                    gt = gt[np.array(masked_clips[clip_idx])==1]

                clip_score = score_process(clip_score, win_size=smoothing, dataname=self.args.dataset_choice, use_scaler=False)
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
        self.log('validation_auc', auc)
        
        return auc





# using LightningDataModule
class LitDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, train_dataset):
        super().__init__()
        self.save_hyperparameters()
        # or
        self.batch_size = batch_size
        self.train_dataset = train_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size | self.hparams.batch_size, num_workers=8, pin_memory=True)

