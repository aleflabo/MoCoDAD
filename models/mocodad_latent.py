import argparse
from typing import List
import numpy as np

import torch
import torch.nn.functional as F
from models.mocodad import MoCoDAD
from models.common.components import Denoiser
from models.stsae.stsae_unet import STSAE_Unet, STSE_Unet
from utils.model_utils import processing_data



class MoCoDADlatent(MoCoDAD):
    
    def __init__(self, args:argparse.Namespace) -> None:
        """
        This class implements the MoCoDAD model that runs the diffusion process into the latent space.
        
        Args:
            args (argparse.Namespace): arguments containing the hyperparameters of the model
        """
        
        # MoCoDAD parameters for the latent space
        self.stage = args.stage
        self.latent_embedding_dim = args.latent_embedding_dim
        self.hidden_sizes = args.hidden_sizes
        self.pretrained_model_ckpt_path = args.pretrained_model_ckpt_path
        
        super().__init__(args)
        
        assert self.conditioning_strategy == 'inject', 'Conditioning strategy must be inject. Other strategies are not supported for the latent space'
        self.model_return_value = 'pose' if self.stage == 'pretrain' else self.model_return_value
        
        # Load the pretrained model
        if self.stage == 'diffusion':
            assert self.pretrained_model_ckpt_path is not None or self.pretrained_model_ckpt_path != '', 'Pretrained model checkpoint path not specified'
            self._freeze_main_net_and_load_ckpt()
        
        
        
    def build_model(self) -> None:
        """
        Build the model. Overrides the parent method to build the model for the latent space.
        """
        
        super().build_model()
        
        # Overwrite the model
        if self.stage == 'diffusion':
            self.model = STSE_Unet(c_in=self.num_coords, embedding_dim=self.embedding_dim,
                                   latent_dim=self.latent_embedding_dim, n_frames=self.n_frames_corrupt,
                                   n_joints=self.n_joints, dropout=self.dropout,
                                   device=self.device, set_out_layer=True,
                                   unet_down_channels=[16, 32, 32, 64, 64, 128, 64])
            self.denoiser = Denoiser(input_size=self.latent_embedding_dim, hidden_sizes=self.hidden_sizes, 
                                     cond_size=self.embedding_dim, bias=True, device=self.device)
        elif self.stage == 'pretrain':
            self.model = STSAE_Unet(c_in=self.num_coords, embedding_dim=self.embedding_dim, 
                                    n_frames=self.n_frames_corrupt, dropout=self.dropout, 
                                    n_joints=self.n_joints, device=self.device,
                                    concat_condition=(self.conditioning_strategy == 'concat'), 
                                    inject_condition=(self.conditioning_strategy == 'inject'),
                                    use_bottleneck=True, latent_dim=self.latent_embedding_dim)
        else:
            raise ValueError(f'Unknown stage {self.stage}')
    
    
    def forward(self, input_data:torch.Tensor, condition_data:torch.Tensor=None, aggr_strategy:str='best', 
                *, return_:str=None) -> List[torch.Tensor]:
        """
        Forward pass of the model. Overrides the parent method to run the diffusion process into the latent space.

        Args:
            input_data (List[torch.Tensor]): list containing the following tensors:
                                             - tensor_data: tensor of shape (B, C, T, V) containing the input sequences
                                             - transformation_idx
                                             - metadata
                                             - actual_frames
            aggr_strategy (str, optional): aggregation strategy to use. If not specified as a function parameter, the aggregation strategy 
                                           specified in the model hyperparameters is used. Defaults to None. 
            return_ (str, optional): return value of the model: 
                                     - only the selected latents according to the aggregation strategy ('pose')
                                     - only the loss of the selected latents ('loss')
                                     - both ('all'). 
                                     If not specified as a function parameter, the return value specified in the model hyperparameters is used. Defaults to None.

        Returns:
            List[torch.Tensor]: [predicted latents and the loss, tensor_data, transformation_idx, metadata, actual_frames]
        """
        
        # Unpack data: tensor_data is the input data, meta_out is a list of metadata
        tensor_data, meta_out = self._unpack_data(input_data)
        B = tensor_data.shape[0]
        constant_t = torch.full(size=(B,), fill_value=-1, dtype=torch.long, device=self.device) # pass a constant value to the unet
        
        # Select frames to condition on and to corrupt according to the conditioning strategy
        condition_data, corrupt_data, idxs = self._select_frames(tensor_data)
        # Encode the condition data
        condition_embedding, _ = self._encode_condition(condition_data)
        
        if self.stage == 'diffusion':
            # Encode the input
            latent_code = self._unet_forward(corrupt_data, t=constant_t, condition_data=condition_embedding, corrupt_idxs=idxs[1])
            generated_latent = []
            
            for _ in range(self.n_generated_samples):
                # Generate gaussian noise of the same shape as the latent code        
                x = torch.randn((corrupt_data.shape[0], self.latent_embedding_dim), device=self.device)
                for i in reversed(range(1, self.noise_steps)):
                    
                    # Set the time step
                    t = torch.full(size=(B,), fill_value=i, dtype=torch.long, device=self.device)
                    # Predict the noise
                    predicted_noise = self.denoiser(x, t, condition_embedding)                     
                    # Get the alpha and beta values and expand them to the shape of the predicted noise
                    alpha = self._alpha[t][:, None]
                    alpha_hat = self._alpha_hat[t][:, None]
                    beta = self._beta[t][:, None]
                    # Generate gaussian noise of the same shape as the predicted noise
                    noise = torch.randn_like(latent_code, device=self.device) if i > 1 else torch.zeros_like(latent_code, device=self.device)   

                    x = (1 / torch.sqrt(alpha) ) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

                generated_latent.append(x)

            selected_latent, loss_of_selected_latent = self._aggregation_strategy(generated_latent, latent_code, aggr_strategy)

            return self._pack_out_data(selected_latent, loss_of_selected_latent, [tensor_data] + meta_out, return_=return_)
        else:
            pose = self._unet_forward(corrupt_data, t=constant_t, condition_data=condition_embedding, corrupt_idxs=idxs[1])
            return self._pack_out_data(pose, None, [corrupt_data] + meta_out, return_=return_)
    
    
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
        constant_t = torch.full(size=(tensor_data.shape[0],), fill_value=-1, dtype=torch.long, device=self.device) # pass a constant value to the unet
        
        if self.stage == 'diffusion':
            # Encode the input
            latent_code = self._unet_forward(corrupt_data, t=constant_t, condition_data=condition_embedding, corrupt_idxs=idxs[1])
            # Sample the time steps and corrupt the data
            t = self.noise_scheduler.sample_timesteps(corrupt_data.shape[0]).to(self.device)
            x_t, latent_noise = self.noise_scheduler.noise_latent(latent_code, t) 
            # Predict the noise
            predicted_noise = self.denoiser(x_t, t, condition_embedding)
            # Compute the loss
            loss_noise = loss = torch.mean(self.loss_fn(predicted_noise, latent_noise))
            self.log('loss_noise', loss_noise)
        
        elif self.stage == 'pretrain':
            prediction = self._unet_forward(corrupt_data, t=constant_t, condition_data=condition_embedding, corrupt_idxs=idxs[1])
            loss_unet = loss = torch.mean(self.loss_fn(prediction, corrupt_data))
            self.log('loss_unet', loss_unet)

            if self.conditioning_architecture == 'AE':
                loss_rec_cond = F.mse_loss(rec_cond_data, condition_data)
                loss = loss + loss_rec_cond * self.rec_weight
                self.log("loss_recons", loss_rec_cond)
            
        return loss
    
    
    def on_validation_epoch_end(self) -> float:
        """
        Validation epoch end of the model.

        Returns:
            float: validation auc score
        """
        
        if self.stage == 'pretrain':
            out, gt_data, trans, meta, frames = processing_data(self._validation_output_list)
            del self._validation_output_list
            rec_loss = self.post_processing(out, gt_data, trans, meta, frames)
            self.log('pretrain_rec_loss', rec_loss, sync_dist=True)
            return rec_loss
            
        return super().on_validation_epoch_end()
    
    
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
            float: auc score or reconstruction loss (if stage is pretrain)
        """
        
        if self.stage == 'pretrain':
            return torch.mean(self.loss_fn(torch.tensor(out), torch.tensor(gt_data))).item()
        
        return super().post_processing(out, gt_data, trans, meta, frames)
    
    
    def _freeze_main_net_and_load_ckpt(self):
        self.load_state_dict(torch.load(self.pretrained_model_ckpt_path)['state_dict'], strict=False)
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.condition_encoder.parameters():
            param.requires_grad = False