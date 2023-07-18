
import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.eval_utils import (pad_scores, filter_vectors_by_cond, score_process,
                              compute_var_matrix)
from utils.model_utils import processing_data
from typing import Dict, List, Tuple, Union

from models.stsae.stsae_diffusion_unet import STSAE, STSENC
from models.stsae.stsae_hidden_hypersphere import STSAE as STSAE_enc
from utils.diffusion_utils import Diffusion

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



class LitAutoEncoder(pl.LightningModule):
    def __init__(self, 
                 args,
                 ):
        super().__init__()
        self.save_hyperparameters()
        
        self.args = args 
        self.learning_rate = args.opt_lr
        self.batch_size = args.dataset_batch_size
        channels = args.channels
        self.lambda_ = args.lambda_ # weight of the reconstruction loss
        self.device_ = args.device
        self.noise_steps = args.noise_steps
        self.emb_dim = args.emb_dim
        
        
        self.seq_len = args.dataset_seg_len
        self.true_every = args.true_every
        self.interleave = args.interleave
        self.num_random_indices = args.num_random_indices
        self.l1 = torch.nn.SmoothL1Loss(reduction='none')
        self.val_loss = 0 
        self.num_coords = args.num_coords
        
        
        if args.dataset_headless:
            joints_to_consider = 14
        elif args.dataset_kp18_format:
            joints_to_consider = 18
        else:
            joints_to_consider = 17
        self.num_joints = joints_to_consider
        self.diffusion = Diffusion(noise_steps=self.noise_steps, n_joints=joints_to_consider, channels=channels,
                                   device=self.device_, time=args.dataset_seg_len)
        
        # condition information
        self.concat_condition = args.concat_condition
        self.inject_condition = args.inject_condition
        self.no_condition = args.no_condition

        if self.concat_condition:
            n_frames_cond = args.dataset_seg_len
            n_frames_to_be_noised = n_frames_cond
        elif self.no_condition:
            n_frames_cond = 0
            n_frames_to_be_noised = self.seq_len
        elif self.interleave:
            if self.num_random_indices>0:
                n_frames_cond = self.num_random_indices
            else:
                n_frames_cond = len(self.args.indices)
            n_frames_to_be_noised = self.seq_len - n_frames_cond
            print('n_frames_cond', n_frames_cond)
        else:
            n_frames_cond = args.dataset_seg_len//2
            
        if self.inject_condition:
            if self.args.ae:
                self.condition_encoder = STSAE_enc(c_in=args.num_coords, h_dim=args.h_dim, latent_dim=args.latent_dim, 
                                   n_frames=int(n_frames_cond), 
                                   dropout=args.dropout, n_joints=joints_to_consider, channels=channels, emb_dim=self.emb_dim,  device=self.device_)
            else:
                self.condition_encoder = STSENC(c_in=args.num_coords, h_dim=args.h_dim, latent_dim=args.latent_dim, 
                            n_frames=n_frames_cond, dropout=args.dropout, 
                            n_joints=joints_to_consider, device=self.device_, channels=channels, emb_dim=self.emb_dim)
        else:
            self.condition_encoder = None
            
            
        self.model = STSAE(c_in=args.num_coords, h_dim=args.h_dim, latent_dim=args.latent_dim, 
                           n_frames=n_frames_to_be_noised, dropout=args.dropout, 
                           n_joints=joints_to_consider, device=self.device_, channels=channels,
                           concat_condition=self.concat_condition, inject_condition=self.inject_condition, emb_dim=self.emb_dim)
        
            
        
        
    def forward(self, x:List[torch.tensor]) -> Tuple[torch.tensor]:
        # implement diffusion process used at evaluation time
        
        pst = None
        condition_data = None
        tensor_data = x[0]
        
        if self.interleave:
            condition_data, to_be_noised_data, indices = self.interleave_fn(tensor_data)
            if self.args.ae:
                if self.num_random_indices>0:
                    recons , condition_data = self.condition_encoder(condition_data,indices=indices[0].to(self.device))
                else:
                    recons, condition_data = self.condition_encoder(condition_data)                
            else:
                condition_data, _ = self.condition_encoder(condition_data, None, None)
        
        elif self.no_condition:
            to_be_noised_data = tensor_data
            
        else:
            if self.concat_condition:
                pst = tensor_data[:,:,:self.seq_len//2,:]
            if self.inject_condition:
                if self.args.ae:
                    recons, condition_data = self.condition_encoder(tensor_data[:,:,:self.args.dataset_seg_len//2,:])
                else:
                    condition_data, _ = self.condition_encoder(tensor_data[:,:,:self.seq_len//2,:], None, None)  
        
            to_be_noised_data = tensor_data[:,:,self.seq_len//2:,:]

        transformation_idx = x[1]
        metadata = x[2]
        actual_frames = x[3]
        
        
        self.beta = self.diffusion.my_schedule_noise().to(self.device_)
        self.alpha = (1. - self.beta).to(self.device_)
        self.alpha_hat = torch.cumprod(self.alpha, dim=0).to(self.device_)
        accumulate = []
        losses = []
        
        mode = 'best'
        if mode == 'best':
            actual_loss = (torch.ones(to_be_noised_data.shape[0])*1e10).to(self.device_)
        else:
            actual_loss = (torch.zeros(to_be_noised_data.shape[0])).to(self.device_)
        actual_x = torch.zeros((to_be_noised_data.shape[0], to_be_noised_data.shape[1], to_be_noised_data.shape[2], to_be_noised_data.shape[3])).to(self.device_)
        
        n_gen = 50
        for _ in range(n_gen):

            # generate inital random noise based on the condition type
            if self.interleave:
                x = torch.randn((to_be_noised_data.shape[0], to_be_noised_data.shape[1], to_be_noised_data.shape[2], to_be_noised_data.shape[3])).to(self.device_)
            elif self.concat_condition:
                x = torch.randn((tensor_data.shape[0], tensor_data.shape[1], self.seq_len//2, tensor_data.shape[3])).to(self.device_)
            else:    
                x = torch.randn((tensor_data.shape[0], tensor_data.shape[1], self.seq_len, tensor_data.shape[3])).to(self.device_)

            for i in reversed(range(1, self.noise_steps)):
                # reverse diffusion process
                
                t = (torch.ones(to_be_noised_data.shape[0]) * i).long().to(self.device_)
                
                predicted_noise, hidden_out = self.model(x, t, pst=pst, condition_data=condition_data)
                
                if self.concat_condition:
                    predicted_noise = predicted_noise[:, :, self.seq_len//2:, :]
                    
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                x = (1 / torch.sqrt(alpha) ) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

            # calculate loss between ground truth and generated data
            loss_fn=nn.SmoothL1Loss(reduction="none")
            w, dim, timesteps, joints = x.shape
            xs = x.cpu().numpy().transpose(0, 2, 3, 1).reshape(-1, timesteps*joints*dim)
            to_be_noised_datas = to_be_noised_data.cpu().numpy().transpose(0, 2, 3, 1).reshape(-1, timesteps*joints*dim)
            loss = loss_fn(torch.from_numpy(xs).to(self.device_),torch.from_numpy(to_be_noised_datas).to(self.device_) )
            loss = torch.mean(loss,dim=-1)
            
            # based on the selected mode, save the best or worst generated data
            if mode == 'best':
                mask = actual_loss > loss
            else:
                mask = actual_loss < loss
            actual_loss[mask] = loss[mask]
            actual_x[mask] = x[mask]

            accumulate.append(x)
            losses.append(loss.unsqueeze(-1))

            
        if self.concat_condition:
            tensor_data = pst
        
        return actual_loss, tensor_data, transformation_idx, metadata, actual_frames
    
    
    def interleave_fn(self, data):
        #data = torch.tensor(data).permute((0,2,3,1))
        if len(self.args.indices)==0:
            indices = torch.tensor([i for i in range(self.seq_len) if i%self.true_every==0]).to(self.device)
            not_indices = torch.tensor([i for i in range(self.seq_len) if i%self.true_every!=0]).to(self.device)
            cond = torch.index_select(data,2,indices)
            to_be_noised = torch.index_select(data,2,not_indices)

        elif self.num_random_indices>0:
            indices = torch.tensor([torch.randperm(self.seq_len).tolist() for j in range(data.shape[0])]).to(self.device)
            #indices = torch.tensor([torch.randperm(self.seq_len)[:self.num_random_indices].tolist() for j in range(data.shape[0])]).to(self.device)
            #not_indices = torch.tensor([[i for i in range(self.seq_len) if i not in indices[j]] for j in range(data.shape[0])]).to(self.device)
            data = data.permute((0,1,3,2))

            cond, to_be_noised, indices, not_indices = self.cut_array_from_indices(data,indices)
            cond = cond.permute((0,1,3,2))
            to_be_noised = to_be_noised.permute((0,1,3,2))
            data = data.permute((0,1,3,2))
            #cond = torch.index_select(data,2,indices)
            #to_be_noised = torch.index_select(data,2,not_indices)

        else:
            indices =torch.tensor(self.args.indices).to(self.device)
            not_indices = torch.tensor([i for i in range(self.seq_len) if i not in self.args.indices]).to(self.device)
            cond = torch.index_select(data,2,indices)
            to_be_noised = torch.index_select(data,2,not_indices) 

        return cond, to_be_noised, [indices,not_indices]

    def cut_array_from_indices(self,x,indices): #before using permute so that dimention that is being cut is in last position 
        
        """container = []
        not_container = []
        for batch_ind in range(len(x)):
                container.append(torch.index_select(x[batch_ind],-1,indices[batch_ind]).unsqueeze(0))
                not_container.append(torch.index_select(x[batch_ind],-1,not_indices[batch_ind]).unsqueeze(0))
        container = torch.cat((container),0)
        not_container = torch.cat((not_container),0)"""
       
        #indices = torch.tensor([torch.randperm(self.seq_len).tolist() for j in range(data.shape[0])])
        mask = indices < self.num_random_indices
        indices_ = indices[mask].reshape(-1,self.num_random_indices)
        not_indices = indices[~mask].reshape(-1,self.seq_len-self.num_random_indices)
        mask = mask[:,None,None,:].expand(-1,self.num_coords,self.num_joints,-1)
        container = x[mask].reshape(-1,self.num_coords,self.num_joints,self.num_random_indices)
        not_container = x[~mask].reshape(-1,self.num_coords,self.num_joints,self.seq_len-self.num_random_indices)
        """container = []
        not_container = []



        container = x[indices[:,None,None,:]]
        not_cointainer = x[not_indices] 

        for batch_ind in range(len(x)):
            
            container.append(torch.index_select(x[batch_ind],-1,indices[batch_ind]).unsqueeze(0))
            not_container.append(torch.index_select(x[batch_ind],-1,not_indices[batch_ind]).unsqueeze(0))
        
        cointainer = torch.cat(container,0)
        not_cointainer = torch.cat(not_container,0)"""

        
        return container, not_container, indices_, not_indices


    def training_step(self, batch:List[torch.tensor], batch_idx:int) -> torch.float32:
        data = batch[0] 

        if self.interleave:
            condition_data, to_be_noised_data, indices = self.interleave_fn(data)
        elif self.no_condition:
            condition_data = None
            to_be_noised_data = data
        else:
            condition_data = data[:,:,:self.seq_len//2,:]# [2048,2,12,17]
            to_be_noised_data = data[:,:,self.seq_len//2:,:]
        
        ### Diffusion ###
        t = self.diffusion.sample_timesteps(to_be_noised_data.shape[0]).to(self.device_)
        x_t, noise = self.diffusion.noise_graph(to_be_noised_data, t.to(to_be_noised_data.get_device())) 
        
        if not self.concat_condition and not self.inject_condition:
            predicted_noise, _ = self.model(x_t, t, pst=condition_data)

        elif self.inject_condition:
            if self.args.ae:
                if self.num_random_indices>0:
                    gt_data = condition_data
                    recons_data , condition_data = self.condition_encoder(condition_data,indices=indices[0].to(self.device))
                else:
                    gt_data = condition_data
                    recons_data , condition_data = self.condition_encoder(condition_data)
            else:
                condition_data, _ = self.condition_encoder(condition_data, None, None) # [2048, 2, 24, 17] -> [2048, 64, 12, 10]
            pst = condition_data
            
            predicted_noise, _ = self.model(x_t, t, pst=pst, condition_data=condition_data)

        if self.concat_condition:
            predicted_noise, _ = self.model(x_t, t, pst=condition_data)
            predicted_noise = predicted_noise[:, :, self.seq_len//2:, :] 
        loss_noise = F.smooth_l1_loss(predicted_noise, noise)
        self.log("loss",loss_noise)      

        if self.args.ae:
                loss_recons = F.mse_loss(recons_data, gt_data)
                """if self.current_epoch < self.args.pretrain:
                    loss = loss_recons
                    for param in self.model.parameters():
                        param.requires_grad = False
                else:      
                    for param in self.model.parameters():
                        param.requires_grad = True"""
                
                loss = loss_noise + loss_recons*self.args.rec_weight
                self.log("loss_recons",loss_recons)
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

