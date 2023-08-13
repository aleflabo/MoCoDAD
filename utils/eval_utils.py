import os
from glob import glob
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import MinMaxScaler


def compute_fig_matrix(pos, frames_pos, n_frames):
    assert len(pos.shape) == 4
    w, dim, timesteps, joints = pos.shape

    pos = pos.transpose(0, 2, 3, 1).reshape(-1, timesteps, joints*dim)

    pose = np.zeros(shape=(w, n_frames, joints*dim))

    for n in range(pose.shape[0]):
        pose[n, frames_pos[n] - 1, :] = pos[n, :, :]
    
    return pose


def compute_var_matrix(pos, frames_pos, n_frames):

    pose = np.zeros(shape=(pos.shape[0], n_frames))

    for n in range(pose.shape[0]):
        pose[n, frames_pos[n] - 1] = pos[n]

    return pose


def extract_single_pose(pose_matrix, method="median", std=False, std_method="mean", std_lift=1e-7):
    single_pose = np.zeros((pose_matrix.shape[1], pose_matrix.shape[2]))
    if std:
        std_vec = np.zeros(shape=(pose_matrix.shape[1]))
    non_zero_frames = list(set(np.nonzero(pose_matrix)[1]))

    for index in non_zero_frames:
        col = pose_matrix[:,index,:]
        used_col = np.stack([x for x in col if np.sum(x) != 0], 0)
        
        if method == "unique":
            single_pose[index, :] = used_col[0]
            if std & (std_method == "mean"):
                std_vec[index] = np.mean(np.std(used_col, 0), -1)
            elif std & (std_method == "median"):
                std_vec[index] = np.median(np.std(used_col, 0), -1)
        if method == "mean":
            single_pose[index,:] = np.mean(used_col, 0)
            if std & (std_method == "mean"):
                std_vec[index] = np.mean(np.std(used_col, 0), -1)
            elif std & (std_method == "median"):
                std_vec[index] = np.median(np.std(used_col, 0), -1)
        elif method == "median":
            single_pose[index,:] = np.median(used_col, 0)
            if std & (std_method == "mean"):
                std_vec[index] = np.mean(np.std(used_col, 0), -1)
            elif std & (std_method == "median"):
                std_vec[index] = np.median(np.std(used_col, 0), -1)
                
    if std:
        scaler = MinMaxScaler()
        std_score = std_vec + std_lift
        std_score = scaler.fit_transform(std_score.reshape(-1, 1)).reshape(-1)

        return single_pose, std_score
    return single_pose, []


def filter_vectors_by_cond(vecs, cond):
    return [filter_by_cond(vec, cond) for vec in vecs]


def filter_by_cond(vec, cond):
    return vec[cond]


def calculate_loss(loss_func, input, target, dataname, to_pow=False):
    if 'array' in str(type(input)):    
        input = torch.from_numpy(input).cuda()
    if 'array' in str(type(target)):
        target = torch.from_numpy(target).cuda()

    
    reco_loss = loss_func(target, input)
    
    if to_pow:
        reco_loss = torch.mean(reco_loss, dim=-1)**4
    else:
        reco_loss = torch.mean(reco_loss, dim=-1)
    
    return reco_loss.detach().cpu().numpy()


def score_process(score, shift, kernel_size):
    
    scores_shifted = np.zeros_like(score)
    scores_shifted[shift:] = score[:-shift]
    score = gaussian_filter1d(scores_shifted, kernel_size)
    
    return score


def ranges(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))


def ROC(y_test,y_pred, path=''):
    fpr,tpr,tr=roc_curve(y_test,y_pred)
    auc=roc_auc_score(y_test,y_pred)
    idx=np.argwhere(np.diff(np.sign(tpr-(1-fpr)))).flatten()

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.plot(fpr,tpr,label="AUC="+str(auc))
    plt.plot(fpr,1-fpr,'r:')
    plt.plot(fpr[idx],tpr[idx], 'ro')
    plt.legend(loc=4)
    plt.grid()
    plt.savefig(path)
    
    return tr[idx], auc


def pad_scores(fig_reconstruction_loss, gt, pad_size):
    
        zero_interval = set(list(range(len(gt)-1))) - set(np.nonzero(fig_reconstruction_loss)[0])
        non_presence_intervals = ranges(zero_interval)
        nope = []
        for _,interval in enumerate(non_presence_intervals):
                start,end = interval
                if start == 0 and end == len(gt)-2: continue
                elif start==0 and end!= len(gt)-2:
                    nope.append((start,min(end+pad_size,len(gt))))
                elif start!=0 and end==len(gt)-2:
                    nope.append((max(start-pad_size,0),end))
                elif start!=0 and end!=len(gt)-2:
                    nope.append((max(start-pad_size,0),min(end+pad_size,len(gt))))
        for interval in nope: 
            fig_reconstruction_loss[range(interval[0],interval[1])] = 0
        return fig_reconstruction_loss
    
    
def get_avenue_mask() -> Dict[int, List[int]]:
    V_01 = [1] * 75 + [0] * 46 + [1] * 269 + [0] * 47 + [1] * 427 + [0] * 47 + [1] * 20 + [0] * 70 + [1] * 438  # 1439 Frames
    V_02 = [1] * 272 + [0] * 48 + [1] * 403 + [0] * 41 + [1] * 447  # 1211 Frames
    V_03 = [1] * 293 + [0] * 48 + [1] * 582  # 923 Frames
    V_06 = [1] * 561 + [0] * 64 + [1] * 189 + [0] * 193 + [1] * 276  # 1283 Frames
    V_16 = [1] * 728 + [0] * 12  # 740 Frames

    masked_clips = {
        1: V_01,
        2: V_02,
        3: V_03,
        6: V_06,
        16: V_16
    }
    return masked_clips


def get_hr_ubnormal_mask(split:str):
    
    if 'test' in split:
        split = 'testing'
    else:
        split = 'validating'
    
    path_to_boolean_masks = f'./data/UBnormal/hr_bool_masks/{split}/test_frame_mask/*'
    hr_boolean_masks = glob(path_to_boolean_masks)
    hr_ubnormal_masked_clips = {}
    
    for boolean_mask_path in hr_boolean_masks:
        scene_clip_id = os.path.basename(boolean_mask_path).split('.')[0]
        scene_id, clip_id = list(map(int, scene_clip_id.split('_')))
        hr_ubnormal_masked_clips[(scene_id, clip_id)] = np.load(boolean_mask_path)
    
    return hr_ubnormal_masked_clips