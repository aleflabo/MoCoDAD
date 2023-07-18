import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from sklearn.metrics import roc_curve,roc_auc_score
import matplotlib.pyplot as plt
import torch.nn as nn
import warnings
from scipy.ndimage import gaussian_filter1d
import geoopt.manifolds.stereographic.math as gmath
import pandas as pd
warnings.filterwarnings("ignore")


def compute_fig_matrix(pos, frames_pos, n_frames):
    assert len(pos.shape) == 4
    w, dim, timesteps, joints = pos.shape

    pos = pos.transpose(0, 2, 3, 1).reshape(-1, timesteps, joints*dim)

    pose = np.zeros(shape=(w, n_frames, joints*dim))

    for n in range(pose.shape[0]):
        pose[n, frames_pos[n] - 1, :] = pos[n, :, :] # added -1
    
    return pose

def compute_var_matrix(pos, frames_pos, n_frames):

    pose = np.zeros(shape=(pos.shape[0], n_frames))

    for n in range(pose.shape[0]):
        pose[n, frames_pos[n] - 1] = pos[n] # added -1

    return pose


def mahalanobis(u:torch.tensor, v:torch.tensor, VI:torch.tensor, reduce='mean'):
    if len(u.size()) < 3:
        u = torch.reshape(u, (*u.size(), 1))
    if len(v.size()) < 3:
        v = torch.reshape(v, (*v.size(), 1))
    distance = torch.sqrt(torch.matmul(torch.matmul(torch.transpose(u - v, 1, 2), VI), u - v))
    
    if reduce == 'mean':
        return distance.mean()
    else:
        return distance
    
    
def windows_based_loss_mahalanobis(hidden_c, hidden_out_fig, VI, frames_fig, n_frames): # VI is the inverse covariance matrix
    
    w = hidden_out_fig.shape[0]
    
    hidden_out_fig = torch.from_numpy(hidden_out_fig).cuda()
    loss_hypersphere = mahalanobis(hidden_out_fig, hidden_c, VI, reduce='none')
    loss_hypersphere = torch.mean(loss_hypersphere, dim=-1)

    pose = np.zeros(shape=(w, n_frames))

    for n in range(pose.shape[0]):
        pose[n, frames_fig[n] - 1] = loss_hypersphere[n].cpu() # added -1

    return pose




def windows_based_loss_rec_and_hy(gt_fig, out_fig, hidden_c, hidden_out_fig, frames_fig, n_frames, loss_fn=nn.MSELoss(reduction='none'), rec_loss_weight=0.2, loss_type='rec'):
    assert len(gt_fig.shape) == 4
    w, dim, timesteps, joints = gt_fig.shape

    gt_fig = gt_fig.transpose(0, 2, 3, 1).reshape(-1, timesteps*joints*dim)
    out_fig = out_fig.transpose(0, 2, 3, 1).reshape(-1, timesteps*joints*dim)

    gt_fig = torch.from_numpy(gt_fig).cuda()
    out_fig = torch.from_numpy(out_fig).cuda()
    loss = loss_fn(gt_fig,out_fig)
    loss = torch.mean(loss, dim=-1)
    
    hidden_out_fig = torch.from_numpy(hidden_out_fig).cuda()
    loss_hypersphere = loss_fn(hidden_c,hidden_out_fig)
    loss_hypersphere = torch.mean(loss_hypersphere, dim=-1)
    if loss_type == 'rec+hyp':
        loss /= rec_loss_weight     
    # print(torch.mean(loss),torch.mean(loss_hypersphere))
    pose = np.zeros(shape=(w, n_frames))

    for n in range(pose.shape[0]):
        # pose[n, frames_fig[n] - 1] = rec_loss_weight*loss[n].cpu() + loss_hypersphere[n].cpu() # added -1
        if loss_type == 'rec':
            pose[n, frames_fig[n] - 1] = loss[n].cpu()
        if loss_type == 'hyp':
            pose[n, frames_fig[n] - 1] = loss_hypersphere[n].cpu()
        if loss_type == 'rec+hyp':
            pose[n, frames_fig[n] - 1] = loss[n].cpu() + loss_hypersphere[n].cpu()
            
    return pose


def windows_based_loss(gt_fig, out_fig, frames_fig, n_frames, loss_fn=nn.MSELoss(reduction="none")):
    assert len(gt_fig.shape) == 4
    w, dim, timesteps, joints = gt_fig.shape

    gt_fig = gt_fig.transpose(0, 2, 3, 1).reshape(-1, timesteps*joints*dim)
    out_fig = out_fig.transpose(0, 2, 3, 1).reshape(-1, timesteps*joints*dim)

    # loss_fn = nn.MSELoss(reduction="none")
    gt_fig = torch.from_numpy(gt_fig).cuda()
    out_fig = torch.from_numpy(out_fig).cuda()
    loss = loss_fn(gt_fig,out_fig)
    loss = torch.mean(loss, dim=-1)

    pose = np.zeros(shape=(w, n_frames))

    for n in range(pose.shape[0]):
        pose[n, frames_fig[n] - 1] = loss[n].cpu() # added -1

    return pose
 
def windows_based_loss_diffusion(gt_fig, out_fig, frames_fig, n_frames, loss_fn=nn.MSELoss(reduction="none")):
    assert len(gt_fig.shape) == 4
    w, dim, timesteps, joints = gt_fig.shape

    gt_fig = gt_fig.transpose(0, 2, 3, 1).reshape(-1, timesteps*joints*dim)
    out_fig = out_fig.transpose(0, 2, 3, 1).reshape(-1, timesteps*joints*dim)

    # loss_fn = nn.MSELoss(reduction="none")
    gt_fig = torch.from_numpy(gt_fig).cuda()
    out_fig = torch.from_numpy(out_fig).cuda()
    loss = loss_fn(gt_fig,out_fig)
    loss = torch.mean(loss, dim=-1)

    pose = np.zeros(shape=(w, n_frames))

    for n in range(pose.shape[0]):
        pose[n, frames_fig[n] - 1] = loss[n].cpu() # added -1

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


# def score_process(score, win_size=50, dataname='STC', use_scaler = False):
    
#     score = pd.Series(score).rolling(win_size, center=True, min_periods=win_size//2).mean().values
#     if use_scaler:
#         scaler = MinMaxScaler()
#         score = scaler.fit_transform(score.reshape(-1, 1)).reshape(-1)
#     return score


def score_process(score, win_size=50, dataname='STC', use_scaler = False):
    
    # score = pd.Series(score).rolling(win_size, center=True, min_periods=win_size//2).mean().values
    # # if use_scaler:
    # scaler = MinMaxScaler()
    # score = scaler.fit_transform(score.reshape(-1, 1)).reshape(-1)
        
    scores_shifted = np.zeros_like(score)
    shift = 8 + (8 // 2) - 1
    scores_shifted[shift:] = score[:-shift]
    score = gaussian_filter1d(scores_shifted, 30)
    
    #### STC
    # scores_shifted = np.zeros_like(score)
    # shift = 6 + (6 // 2) - 1
    # scores_shifted[shift:] = score[:-shift]
    # score = gaussian_filter1d(scores_shifted, 10)   
    
    
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