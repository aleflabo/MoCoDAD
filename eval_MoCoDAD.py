import argparse
import os
import warnings
from glob import glob

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from sklearn.metrics import roc_auc_score
from utils.model_utils import processing_data

from utils.argparser import init_sub_args
from utils.eval_utils import ROC, score_process, compute_var_matrix, filter_vectors_by_cond, pad_scores
from utils.dataset import get_dataset_and_loader
from models.diffusion_STS import LitAutoEncoder as Litmodel

warnings.filterwarnings("ignore")

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


def hr_ubnormal(path_to_boolean_masks):
    """
    """
    
    hr_boolean_masks = glob(path_to_boolean_masks)
    hr_ubnormal_masked_clips = {}
    
    for boolean_mask_path in hr_boolean_masks:
        scene_clip_id = os.path.basename(boolean_mask_path).split('.')[0]
        scene_id, clip_id = list(map(int, scene_clip_id.split('_')))
        hr_ubnormal_masked_clips[(scene_id, clip_id)] = np.load(boolean_mask_path)
    
    return hr_ubnormal_masked_clips
    







if __name__== '__main__':
    

    parser = argparse.ArgumentParser(description='Pose_AD_Experiment')
    parser.add_argument('-c', '--config', type=str, required=True,
                       default='./config/old_ckpt.yaml')
    args = parser.parse_args()
    args = yaml.load(open(args.config), Loader=yaml.FullLoader)
    args = argparse.Namespace(**args)
    
    args, ae_args, dcec_args, res_args, opt_args = init_sub_args(args)
    

    
    ### For HR UBnormal
    if args.use_hr:
        if 'test' in args.split:
            split = 'testing'
        else:
            split = 'validating'
            
        ubnormal_path_to_boolean_masks = f'/media/hdd/data/anomaly_detection/UBnormal/hr_bool_masks/{split}/test_frame_mask/*'
        hr_ubnormal_masked_clips = hr_ubnormal(ubnormal_path_to_boolean_masks)
    else:
        hr_ubnormal_masked_clips = {}
    
    exp_dir = os.path.join(args.exp_dir, args.dataset_choice, args.dir_name)
    
    # Pass arguments as dataset arguments for PoseDatasetRobust
    ae_args.exp_dir = exp_dir

    print('Done\n')
    print(args.dataset_choice)

    
    gen = 50
    aggregator = 'best'
    split = 'test_solved'

    name = 'saved_tensors_{}_{}_{}'.format(split, aggregator, gen)
    path = os.path.join(args.exp_dir,args.dataset_choice,args.dir_name, args.load_ckpt)
    if not os.path.exists(os.path.join(exp_dir,name)):
        os.mkdir(os.path.join(exp_dir,name))
    if (not args.load_tensors):
        print('Loading data and creating loaders.....')
        dataset, loader = get_dataset_and_loader(ae_args,split=args.split)
        
        # init model
        model = Litmodel(args)
        
        
        print('Loading model from {}'.format(path))
        
        trainer = pl.Trainer(strategy="ddp",accelerator=args.accelerator,devices= args.devices)
        out = trainer.predict(model, dataloaders=loader,ckpt_path=path,return_predictions=True)
        out, gt_data, trans, meta, frames = processing_data(out)
        torch.save(out, os.path.join(exp_dir,name,'out.pt'))
        torch.save(gt_data, os.path.join(exp_dir,name,'gt_data.pt'))
        torch.save(trans, os.path.join(exp_dir,name,'trans.pt'))
        torch.save(meta, os.path.join(exp_dir,name,'meta.pt'))
        torch.save(frames, os.path.join(exp_dir,name,'frames.pt'))

    else:
        out = torch.load(os.path.join(exp_dir,name,'out.pt'))
        gt_data = torch.load(os.path.join(exp_dir,name,'gt_data.pt'))
        trans = torch.load(os.path.join(exp_dir,name,'trans.pt'))
        meta = torch.load(os.path.join(exp_dir,name,'meta.pt'))
        frames = torch.load(os.path.join(exp_dir,name,'frames.pt'))

    
    print('Checkpoint loaded')
    print('Processing data.....')

    
    print('Dataset: {}, Test path: {}'.format(args.dataset_choice,args.gt_path))

    all_gts = [file_name for file_name in os.listdir(args.gt_path) if file_name.endswith('.npy')]
    all_gts = sorted(all_gts)
    scene_clips = [(int(fn.split('_')[0]), int(fn.split('_')[1].split('.')[0])) for fn in all_gts]

    model_scores_transf = {}
    dataset_gt_transf = {}
    loss_fn = torch.nn.SmoothL1Loss(reduction='none')
    num_transform = ae_args.num_transform
    idx_transf = 0
    smoothing = args.smoothing


    print('Starting evaluation.....')
    for transformation in range(num_transform):
        # iterating over each transformation T
        
        dataset_gt = []
        model_scores = []
        errors = []
        scenes_division = {}
        cond_transform = (trans == transformation)

        out_transform, gt_data_transform, meta_transform, frames_transform = filter_vectors_by_cond([out, gt_data, meta, frames], cond_transform)

        for idx in range(len(all_gts)):
            # iterating over each clip C with transformation T
            
            scene_idx, clip_idx = scene_clips[idx]
            if not scene_idx in scenes_division.keys():
                scenes_division[scene_idx] = []
            

            gt = np.load(os.path.join(args.gt_path, all_gts[idx]))
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
                
                if args.pad_size!=-1:
                    fig_reconstruction_loss = pad_scores(fig_reconstruction_loss, gt, self.args.pad_size)                    
                
                error_per_person.append(fig_reconstruction_loss)
                    
                    
            clip_score = np.mean(np.stack(error_per_person, axis=0), axis=0)

            if (scene_idx, clip_idx) in hr_ubnormal_masked_clips:
                clip_score = clip_score[hr_ubnormal_masked_clips[(scene_idx, clip_idx)]]
                gt = gt[hr_ubnormal_masked_clips[(scene_idx, clip_idx)]]
            
            # removing the non-HR frames for Avenue dataset
            if clip_idx in masked_clips:
                clip_score = clip_score[np.array(masked_clips[clip_idx])==1]
                gt = gt[np.array(masked_clips[clip_idx])==1]

            clip_score = score_process(clip_score, win_size=smoothing, dataname=args.dataset_choice,use_scaler=False)
            scenes_division[scene_idx].append(clip_score)
            model_scores.append(clip_score)
            dataset_gt.append(gt)
            try:
                auc=roc_auc_score(gt, clip_score)
                errors.append(auc)
            except Exception as e: 
                pass
            
                
        model_scores = np.concatenate(model_scores, axis=0)
        dataset_gt = np.concatenate(dataset_gt, axis=0)

        print('\nTest set score for transformation {}\n'.format(transformation+1))
        if args.dataset_choice == 'HR-Avenue':
            best_threshold, auc = ROC(dataset_gt, model_scores)
        else:
            best_threshold, auc = ROC(dataset_gt, model_scores, path=path+f'_t{transformation}_roc_hyp.png')

        print('auc = {}'.format(auc))

        model_scores_transf[transformation] = model_scores
        dataset_gt_transf[transformation] = dataset_gt
        
    # aggregating the anomaly scores for all transformations
    pds = np.mean(np.stack(list(model_scores_transf.values()),0),0)
    gt = dataset_gt_transf[0]
    
    # computing the AUC
    auc=roc_auc_score(gt,pds)
    print('final AUC score: {}'.format(auc))
