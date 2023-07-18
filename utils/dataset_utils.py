import numpy as np
import os
import torch
import math
from sklearn.preprocessing import RobustScaler

def keypoints17_to_coco18(kps):
    """
    Convert a 17 keypoints coco format skeleton to an 18 keypoint one.
    New keypoint (neck) is the average of the shoulders, and points
    are also reordered.
    """
    kp_np = np.array(kps)
    neck_kp_vec = 0.5 * (kp_np[..., 5, :] + kp_np[..., 6, :])
    kp_np = np.concatenate([kp_np, neck_kp_vec[..., None, :]], axis=-2)
    opp_order = [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
    opp_order = np.array(opp_order, dtype=np.int)
    kp_coco18 = kp_np[..., opp_order, :]
    return kp_coco18

def normalize_pose(pose_data, **kwargs):
    """
    Normalize keypoint values to the range of [-1, 1]
    :param pose_data: Formatted as [N, T, V, F], e.g. (Batch=64, Frames=12, 18, 3)
    :param vid_res:
    :param symm_range:
    :return:
    """
    vid_res = kwargs.get('vid_res', [856, 480])

    symm_range = kwargs.get('symm_range', True)
    sub_mean = kwargs.get('sub_mean', True)
    return_mean = kwargs.get('return_mean', True)
    vid_res_wconf = vid_res + [1]
    norm_factor = np.array(vid_res_wconf)
    pose_data_normalized = pose_data / norm_factor
    pose_data_centered = pose_data_normalized
    hip_center = kwargs.get('hip_center', False)
    
    if symm_range:  
        # Means shift data to [-1, 1] range
        pose_data_centered[..., :2] = 2 * pose_data_centered[..., :2] - 1
    
    pose_data_scaled = pose_data_centered

    if sub_mean:  # Inner frame scaling requires mean subtraction
        pose_data_zero_mean = pose_data_centered
        mean_kp_val = np.mean(pose_data_zero_mean[..., :2], (1, 2))
        pose_data_zero_mean[..., :2] -= mean_kp_val[:, None, None, :]
        pose_data_scaled = pose_data_zero_mean

    if hip_center:
        hips_xy = pose_data_scaled.copy()[:, :, [8, 11], :2]
        hips_mean = np.mean(hips_xy, axis=2)
        pose_data_scaled[..., :2] -= hips_mean[:, :, None, :]
        pose_data_scaled = pose_data_centered
    
    if return_mean:
        return pose_data_scaled, mean_kp_val
    return pose_data_scaled, None

def normalize_pose_robust(pose_data, scaler = None, **kwargs):
    """
    Normalize keypoint values to the range of [-1, 1]
    :param pose_data: Formatted as [N, T, V, F], e.g. (Batch=64, Frames=12, 18, 3)
    :param vid_res:
    :param symm_range:
    :return:
    """
    # STC
    # vid_res = kwargs.get('vid_res', [856, 480])
    # AVENUE
    vid_res = kwargs.get('vid_res', [640, 360])

    symm_range = kwargs.get('symm_range', True)
    # symm_range = False
    vid_res_wconf = vid_res + [1]
    norm_factor = np.array(vid_res_wconf)
    pose_data_normalized = pose_data / norm_factor
    pose_data_centered = pose_data_normalized
    if symm_range:  # Means shift data to [-1, 1] range
        pose_data_centered[..., :2] = 2 * pose_data_centered[..., :2] - 1
    pose_data_scaled = pose_data_centered

    original_shape = pose_data_scaled[..., :2].shape
    input_dim = original_shape[-1]*original_shape[-2]
    X = pose_data_scaled[..., :2].reshape(-1, input_dim)
    X_scaled = np.where(X == 0.0, np.nan, X)

    if scaler is None:
        scaler = RobustScaler(quantile_range=(10.0, 90.0))
        scaler.fit(X_scaled)

    X_scaled = scaler.transform(X_scaled)
    X_scaled = np.where(np.isnan(X_scaled), 0.0, X_scaled).reshape(original_shape)
    new_shape = list(original_shape)
    new_shape[-1]=1
    X_scaled = np.concatenate([X_scaled,np.ones(tuple(new_shape))],-1)

    return X_scaled, scaler


def normalize_pose_stan(pose_data, **kwargs):
    """
    pose_data: Formatted as [N, T, V, F], e.g. (Batch=64, Frames=12, 18, 3)
    """
    vid_res = kwargs.get('vid_res', [640, 360])

    symm_range = kwargs.get('symm_range', True)
    # symm_range = False
    vid_res_wconf = vid_res + [1]
    norm_factor = np.array(vid_res_wconf)
    pose_data_normalized = pose_data / norm_factor
    pose_data_centered = pose_data_normalized
    if symm_range:  # Means shift data to [-1, 1] range
        pose_data_centered[..., :2] = 2 * pose_data_centered[..., :2] - 1
    pose_data_scaled = pose_data_centered
    
    temporal_mean = np.mean(pose_data_scaled, axis=1, keepdims=True)
    pose_data_scaled -= temporal_mean
    spatial_mean = np.mean(pose_data_scaled[...,:2], axis=(2,3), keepdims=True)
    std_dev = np.sqrt(np.mean(np.square(pose_data_scaled[...,:2] - spatial_mean), axis=(2,3), keepdims=True) + 1e-5)
    pose_data_scaled[..., :2] /= std_dev
    return pose_data_scaled, None


def normalize_pose_bbox(pose_data, **kwargs):
    """
    pose_data: Formatted as [N, T, V, F], e.g. (Batch=64, Frames=12, 18, 3)
    """
    
    vid_res = kwargs.get('vid_res', [640, 360])

    symm_range = kwargs.get('symm_range', True)
    # symm_range = False
    vid_res_wconf = vid_res + [1]
    norm_factor = np.array(vid_res_wconf)
    pose_data_normalized = pose_data / norm_factor
    pose_data_centered = pose_data_normalized
    if symm_range:  # Means shift data to [-1, 1] range
        pose_data_centered[..., :2] = 2 * pose_data_centered[..., :2] - 1
    pose_data_scaled = pose_data_centered
    
    pose_data_scaled_w = pose_data_scaled[..., 0].max(axis=-2, keepdims=True) - pose_data_scaled[..., 0].min(axis=-2, keepdims=True)
    pose_data_scaled_h = pose_data_scaled[..., 1].max(axis=-2, keepdims=True) - pose_data_scaled[..., 1].min(axis=-2, keepdims=True)
    pose_data_scaled[..., 0] = pose_data_scaled[..., 0] / pose_data_scaled_w
    pose_data_scaled[..., 1] = pose_data_scaled[..., 1] / pose_data_scaled_h
    
    return pose_data_scaled, None
    
    
    


def gen_clip_seg_data_np(clip_dict, start_ofst=0, seg_stride=4, seg_len=12, scene_id='', clip_id='', ret_keys=False, kp_threshold=0, debug = False):
    """
    Generate an array of segmented sequences, each object is a segment and a corresponding metadata array
    """
    pose_segs_data = []
    pose_segs_meta = []
    pose_segs_ids = []
    person_keys = {}
    for idx in sorted(clip_dict.keys(), key=lambda x: int(x)):
        sing_pose_np, sing_pose_meta, sing_pose_keys = single_pose_dict2np(clip_dict, idx, kp_threshold)
        key = ('{:02d}_{:04d}_{:02d}'.format(int(scene_id), int(clip_id), int(idx)))
        person_keys[key] = sing_pose_keys

        curr_pose_segs_np, curr_pose_segs_meta, curr_pose_ids_meta = split_pose_to_segments(sing_pose_np, sing_pose_meta, sing_pose_keys,
                                                                        start_ofst, seg_stride, seg_len,
                                                                        scene_id=scene_id, clip_id=clip_id)
        pose_segs_data.append(curr_pose_segs_np)
        pose_segs_meta += curr_pose_segs_meta
        pose_segs_ids += curr_pose_ids_meta
    pose_segs_data_np = np.concatenate(pose_segs_data, axis=0)

    del pose_segs_data
    if ret_keys:
        return pose_segs_data_np, pose_segs_meta, person_keys, pose_segs_ids
    else:
        return pose_segs_data_np, pose_segs_meta, pose_segs_ids
    
def single_pose_dict2np(person_dict, idx, kp_threshold):
    single_person = person_dict[str(idx)]
    sing_pose_np = []
    if isinstance(single_person, list):
        single_person_dict = {}
        for sub_dict in single_person:
            single_person_dict.update(**sub_dict)
        single_person = single_person_dict
    single_person_dict_keys = sorted(single_person.keys())
    sing_pose_meta = [int(idx), int(single_person_dict_keys[0])]  # Meta is [person_idx, first_frame]
    for key in single_person_dict_keys:
        curr_pose_np = np.array(single_person[key]['keypoints']).reshape(-1, 3)
        # added a threshold to the keypoints. All keypoints < kp_threshold are set to 0
        if kp_threshold > 0:
            lower_than_th = np.where(curr_pose_np[:,2] < kp_threshold)
            curr_pose_np[lower_than_th,:2] = 0
        sing_pose_np.append(curr_pose_np)
    sing_pose_np = np.stack(sing_pose_np, axis=0)
    return sing_pose_np, sing_pose_meta, single_person_dict_keys

def is_single_person_dict_continuous(sing_person_dict):
    """
    Checks if an input clip is continuous or if there are frames missing
    :return:
    """
    start_key = min(sing_person_dict.keys())
    person_dict_items = len(sing_person_dict.keys())
    sorted_seg_keys = sorted(sing_person_dict.keys(), key=lambda x: int(x))
    return is_seg_continuous(sorted_seg_keys, start_key, person_dict_items)


def is_seg_continuous(sorted_seg_keys, start_key, seg_len, missing_th=2):
    """
    Checks if an input clip is continuous or if there are frames missing
    :param sorted_seg_keys:
    :param start_key:
    :param seg_len:
    :param missing_th: The number of frames that are allowed to be missing on a sequence,
    i.e. if missing_th = 1 then a seg for which a single frame is missing is considered continuous
    :return:
    """
    start_idx = sorted_seg_keys.index(start_key)
    expected_idxs = list(range(start_key, start_key + seg_len))
    act_idxs = sorted_seg_keys[start_idx: start_idx + seg_len]
    min_overlap = seg_len - missing_th
    key_overlap = len(set(act_idxs).intersection(expected_idxs))
    if key_overlap >= min_overlap:
        return True
    else:
        return False


def split_pose_to_segments(single_pose_np, single_pose_meta, single_pose_keys, start_ofst=0, seg_dist=6, seg_len=12,
                           scene_id='', clip_id=''):
    clip_t, kp_count, kp_dim = single_pose_np.shape
    pose_segs_np = np.empty([0, seg_len, kp_count, kp_dim])
    pose_segs_meta = []
    pose_ids_meta = []
    num_segs = np.ceil((clip_t - seg_len) / seg_dist).astype(np.int)
    single_pose_keys_sorted = sorted([int(i) for i in single_pose_keys])  # , key=lambda x: int(x))
    for seg_ind in range(num_segs):
        start_ind = start_ofst + seg_ind * seg_dist
        start_key = single_pose_keys_sorted[start_ind]
        if is_seg_continuous(single_pose_keys_sorted, start_key, seg_len):
            start_idx = single_pose_keys_sorted.index(start_key)
            curr_segment = single_pose_np[start_ind:start_ind + seg_len].reshape(1, seg_len, kp_count, kp_dim)
            pose_segs_np = np.append(pose_segs_np, curr_segment, axis=0)
            act_idxs = single_pose_keys_sorted[start_idx: start_idx + seg_len]
            pose_segs_meta.append([int(scene_id), int(clip_id), int(single_pose_meta[0]), int(start_key)] )
            pose_ids_meta.append(act_idxs)

    return pose_segs_np, pose_segs_meta, pose_ids_meta

def get_aff_trans_mat(sx=1, sy=1, tx=0, ty=0, rot=0, flip=False):
    """
    Generate affine transfomation matrix (torch.tensor type) for transforming pose sequences
    :rot is given in degrees
    """
    cos_r = math.cos(math.radians(rot))
    sin_r = math.sin(math.radians(rot))
    flip_mat = torch.eye(3, dtype=torch.float32)
    if flip:
        flip_mat[0, 0] = -1.0
    trans_scale_mat = torch.tensor([[sx, 0, tx], [0, sy, ty], [0, 0, 1]], dtype=torch.float32)
    rot_mat = torch.tensor([[cos_r, -sin_r, 0], [sin_r, cos_r, 0], [0, 0, 1]], dtype=torch.float32)
    aff_mat = torch.matmul(rot_mat, trans_scale_mat)
    aff_mat = torch.matmul(flip_mat, aff_mat)
    return aff_mat


def apply_pose_transform(pose, trans_mat):
    """ Given a set of pose sequences of shape (Channels, Time_steps, Vertices, M[=num of figures])
    return its transformed form of the same sequence. 3 Channels are assumed (x, y, conf) """

    # We isolate the confidence vector, replace with ones, than plug back after transformation is done
    conf = np.expand_dims(pose[2], axis=0)
    ones_vec = np.ones_like(conf)
    pose_w_ones = np.concatenate([pose[:2], ones_vec], axis=0)
    if len(pose.shape) == 3:
        einsum_str = 'ktv,ck->ctv'
    else:
        einsum_str = 'ktvm,ck->ctvm'
    pose_transformed_wo_conf = np.einsum(einsum_str, pose_w_ones, trans_mat)
    pose_transformed = np.concatenate([pose_transformed_wo_conf[:2], conf], axis=0)
    return pose_transformed


class PoseTransform(object):
    """ A general class for applying transformations to pose sequences, empty init returns identity """

    def __init__(self, sx=1, sy=1, tx=0, ty=0, rot=0, flip=False, trans_mat=None):
        """ An explicit matrix overrides all parameters"""
        if trans_mat is not None:
            self.trans_mat = trans_mat
        else:
            self.trans_mat = get_aff_trans_mat(sx, sy, tx, ty, rot, flip)

    def __call__(self, x):
        x = apply_pose_transform(x, self.trans_mat)
        return x


ae_trans_list = [
    PoseTransform(sx=1, sy=1, tx=0.0, ty=0, rot=0, flip=False),  # 0
    PoseTransform(sx=1, sy=1, tx=0.0, ty=0, rot=0, flip=True),  # 3
    PoseTransform(sx=1, sy=1, tx=0.0, ty=0, rot=90, flip=False),  # 6
    PoseTransform(sx=1, sy=1, tx=0.0, ty=0, rot=90, flip=True),  # 9
    PoseTransform(sx=1, sy=1, tx=0.0, ty=0, rot=45, flip=False),  # 12
]

def collate_fn(data):
    """
    Collate function for the dataloader. Pad the keypoints tensor to have the same size.
    """
    keypoints, matrix, meta = zip(*data)
    lenghts = [int(kp.shape[0]) for kp in keypoints]
    max_len = max(lenghts)
    features = torch.zeros([len(keypoints), max_len, keypoints[0].shape[1], keypoints[0].shape[2], keypoints[0].shape[3]])
    adjacency = torch.zeros([len(keypoints), matrix[0].shape[0], max_len, max_len])

    for i in range(len(keypoints)):
        features[i, :lenghts[i]] = keypoints[i]
        adjacency[i, :, :lenghts[i], :lenghts[i]] = matrix[i]
        
    
    return features.float, adjacency, meta

class StdScaler:
    def __init__(self, stds=3):
        self.stds = stds
        self.mu = None
        self.sigma = None

    def fit(self, X):
        self.mu = np.nanmean(X, axis=0, keepdims=True)
        self.sigma = np.nanstd(X, axis=0, keepdims=True)

    def transform(self, X):
        reps = [X.shape[0], 1]
        mu = np.tile(self.mu, reps=reps)
        sigma = np.tile(self.sigma, reps=reps)
        X = (X - (mu - self.stds * sigma)) / (2 * self.stds * sigma)

        return X

    def inverse_transform(self, X):
        reps = [X.shape[0], 1]
        mu = np.tile(self.mu, reps=reps)
        sigma = np.tile(self.sigma, reps=reps)
        X = X * (2 * self.stds * sigma) + (mu - self.stds * sigma)

        return X
