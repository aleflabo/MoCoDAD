import os
import json
import numpy as np
import pickle

from torch.utils.data import Dataset, DataLoader
import utils.tools as tools
from utils.get_robust_data import data_of_combined_model
from utils.dataset_utils import keypoints17_to_coco18, normalize_pose, normalize_pose_bbox, normalize_pose_robust, gen_clip_seg_data_np, ae_trans_list, normalize_pose_stan
import random


class PoseDataset(Dataset):
    def __init__(self, path_to_json_dir,
                 transform_list=None,
                 return_indices=False, return_metadata=False,
                 debug=False, dataset_clips=None,
                 **dataset_args):
        super(PoseDataset).__init__()
        self.path_to_json = path_to_json_dir
        self.headless = dataset_args.get('headless', False)
        self.normalize_pose_seg = dataset_args.get('normalize_pose', True)
        self.kp18_format = dataset_args.get('kp18_format', True)
        self.vid_res = dataset_args.get('vid_res', [856, 480])
        self.num_coords = dataset_args.get('num_coords', 2)
        self.return_mean = dataset_args.get('return_mean', True)
        self.kp_threshold = dataset_args.get('kp_threshold', 0)
        self.debug = debug
        num_clips = 5 if debug else None
        if dataset_clips:
            num_clips = dataset_clips
        self.num_clips = num_clips
        self.return_indices = return_indices
        self.return_metadata = return_metadata
        self.scaler=dataset_args.get('scaler', None)
        self.normalization_strategy = dataset_args.get('normalization_strategy', 'markovitz')
        self.ckpt = dataset_args.get('ckpt','')
        if (transform_list is None) or (transform_list == []):
            self.apply_transforms = False
            self.num_transform = 1
        else:
            self.apply_transforms = True
            self.num_transform = len(transform_list)
        
        self.double_item = dataset_args.get('double_item', False)
        # self.split = dataset_args.get('split', 'train')
        self.transform_list = transform_list
        self.start_ofst = dataset_args.get('start_offset', 0)
        self.seg_len = dataset_args.get('seg_len', 12)
        self.seg_stride = dataset_args.get('seg_stride', 1)
        if (self.return_mean) & (self.normalize_pose_seg):
            self.segs_data_np, self.segs_meta, self.person_keys, self.segs_ids, self.segs_mean = self.gen_dataset(ret_keys=True,
                                                                                                                  **dataset_args)
        else:
            self.segs_data_np, self.segs_meta, self.person_keys, self.segs_ids = self.gen_dataset(ret_keys=True,
                                                                                                  **dataset_args)
        self.person_keys = {k: [int(i) for i in v] for k, v in self.person_keys.items()}
        self.segs_meta = np.array(self.segs_meta)
        self.segs_ids = np.array(self.segs_ids)
        self.metadata = self.segs_meta
        self.num_samples, self.C, self.T, self.V = self.segs_data_np.shape
        self.shear_amplitude = 0.5
        self.temperal_padding_ratio = 6
        
    def __repr__(self):
        return '(DATA [B])'
    def __getitem__(self, index: int):
        # Select sample and augmentation. I.e. given 5 samples and 2 transformations,
        # sample 7 is data sample 7%5=2 and transform is 7//5=1
        if self.apply_transforms:
            sample_index = index % self.num_samples
            trans_index = index // self.num_samples
            data_numpy = np.array(self.segs_data_np[sample_index])
            data_transformed = self.transform_list[trans_index](data_numpy)
            data_transformed = data_transformed[:self.num_coords,:,:]
        else:
            sample_index = index
            # data_transformed = data_numpy = np.array(self.segs_data_np[index])
            # data_transformed = data_transformed[:self.num_coords,:,:]
            data_numpy = np.array(self.segs_data_np[index])
            data_transformed = self._old_aug(data_numpy[:self.num_coords,:,:])
            trans_index = 0  # No transformations
        seg_metadata = self.segs_meta[sample_index]
        self.ids = self.segs_ids[sample_index]
        
        if (self.return_mean) & (self.normalize_pose_seg):
            seg_mean = self.segs_mean[sample_index]
        
        ret_arr = [data_transformed, trans_index]
        if self.return_metadata:
            ret_arr += [seg_metadata]
            ret_arr += [self.ids]
        if (self.return_mean) & (self.normalize_pose_seg):
            ret_arr += [seg_mean]

        if self.return_indices:
            ret_arr += [index]
            
        if self.double_item and (self.dataset_split=='train'):
            data_transformed_2 = self._old_strong_aug(data_numpy[:self.num_coords,:,:])

            # trans_index_2 = random.choice(list(set(range(self.num_transform))-set([trans_index])))
            
            # data_transformed_2 = self.transform_list[trans_index_2](data_numpy)
            # data_transformed_2 = data_transformed_2[:self.num_coords,:,:]
            ret_arr_2 = [data_transformed_2, 0]
            
            return ret_arr, ret_arr_2 
            
        return ret_arr
    
    def _old_strong_aug(self, data_numpy):
        if self.temperal_padding_ratio > 0:
            data_numpy = tools.temporal_crop(data_numpy, self.temperal_padding_ratio)
        # if self.shear_amplitude > 0:
        #     data_numpy = tools.shear(data_numpy, self.shear_amplitude)
        # data_numpy = tools.random_spatial_flip(data_numpy)
        data_numpy = tools.random_rotate(data_numpy)
        data_numpy = tools.gaus_noise(data_numpy)
        data_numpy = tools.gaus_filter(data_numpy)
        data_numpy = tools.axis_mask(data_numpy)
        data_numpy = tools.random_time_flip(data_numpy)
        return data_numpy

    def _old_aug(self, data_numpy):
        if self.temperal_padding_ratio > 0:
            data_numpy = tools.temporal_crop(data_numpy, self.temperal_padding_ratio)
        # if self.shear_amplitude > 0:
        #     data_numpy = tools.shear(data_numpy, self.shear_amplitude)
        return data_numpy
    
    def gen_dataset(self, ret_keys=False, **dataset_args):

        segs_data_np = []
        segs_meta = []
        segs_ids = []
        
        person_keys = dict()
        
        dir_list = os.listdir(self.path_to_json)
        json_list = sorted([fn for fn in dir_list if fn.endswith('.json')])
        
        if self.num_clips is not None:
            json_list = json_list[:self.num_clips]
            json_list += json_list[-self.num_clips:]
        
        for person_dict_fn in json_list:
            scene_id, clip_id = person_dict_fn.split('_')[:2]
            clip_json_path = os.path.join(self.path_to_json, person_dict_fn)
            with open(clip_json_path, 'r') as f:
                clip_dict = json.load(f)
            clip_segs_data_np, clip_segs_meta, clip_keys, clip_segs_ids = gen_clip_seg_data_np(clip_dict=clip_dict, 
                                                                                               start_ofst=self.start_ofst, 
                                                                                               seg_stride=self.seg_stride,
                                                                                               seg_len=self.seg_len, 
                                                                                               scene_id=scene_id,
                                                                                               clip_id=clip_id, 
                                                                                               ret_keys=ret_keys,
                                                                                               kp_threshold=self.kp_threshold,
                                                                                               debug = self.debug)
            segs_data_np.append(clip_segs_data_np)
            segs_meta += clip_segs_meta
            segs_ids += clip_segs_ids
            person_keys = {**person_keys, **clip_keys}
        
        segs_data_np = np.concatenate(segs_data_np, axis=0)
        
        if self.kp18_format and segs_data_np.shape[-2] == 17:
            segs_data_np = keypoints17_to_coco18(segs_data_np)
        
        if self.headless:
            segs_data_np = segs_data_np[:,:,:14]
        
        if self.normalize_pose_seg:
            if self.normalization_strategy == 'markovitz':
                segs_data_np, segs_means = normalize_pose(segs_data_np,
                                                            **dataset_args)
            elif self.normalization_strategy == 'robust':
                segs_data_np, scaler_out = normalize_pose_robust(segs_data_np, **dataset_args)
                if self.scaler is None:
                    with open(self.ckpt+'/robust.pkl', 'wb') as handle:
                        pickle.dump(scaler_out, handle, protocol=pickle.HIGHEST_PROTOCOL)
            elif self.normalization_strategy == 'stan':
                segs_data_np, segs_means = normalize_pose_stan(segs_data_np, **dataset_args)
            elif self.normalization_strategy == 'bbox':
                segs_data_np, segs_means = normalize_pose_bbox(segs_data_np, **dataset_args)

        segs_data_np = np.transpose(segs_data_np, (0, 3, 1, 2)).astype(np.float32)

        if ret_keys:
            if (self.return_mean) & (self.normalize_pose_seg):
                return segs_data_np, segs_meta, person_keys, segs_ids, segs_means
            else:
                return segs_data_np, segs_meta, person_keys, segs_ids
        else:
            if (self.return_mean) & (self.normalize_pose_seg):
                return segs_data_np, segs_meta, segs_ids, segs_means
            else:
                return segs_data_np, segs_meta, segs_ids

    def __len__(self):
        return self.num_transform * self.num_samples
    
    
    

class PoseDatasetRobust(PoseDataset):
    """
    """
    
    def __init__(self, path_to_data, include_global=False, split='train', exp_dir='',
                 transform_list=None,
                 return_indices=False, return_metadata=False,
                 debug=False, dataset_clips=None,
                 **dataset_args):
        
        dataset_args['return_mean'] = False
        self.include_global = include_global

        self.dataset_split = split
        self.exp_dir = exp_dir

        super(PoseDatasetRobust, self).__init__(path_to_data,
                                                transform_list,
                                                return_indices, return_metadata,
                                                debug, dataset_clips,
                                                **dataset_args)
        
        
        
    def gen_dataset(self, ret_keys=False, **dataset_args):
        
        global_, local_ = data_of_combined_model(trajectories_path=self.path_to_json, 
                                                 split=self.dataset_split, seg_len=self.seg_len, 
                                                 seg_stride=self.seg_stride,
                                                 vid_res=self.vid_res,
                                                 normalization_strategy=self.normalization_strategy,
                                                 exp_dir=self.exp_dir, reconstruct_original_data=False,
                                                 normalize_pose=self.normalize_pose_seg,
                                                 include_global=self.include_global,
                                                 debug=self.debug)
        
        X_global, _ = global_
        X_local, X_local_meta = local_ # X_local has shape (number of segments, window lenght, 34)

        segs_meta, segs_ids = X_local_meta
        person_keys = dict()
        
        for seg_meta_item in segs_meta:
            
            person_id = '_'.join(map(lambda x: str(x), seg_meta_item[:-1]))
            
            person_keys[person_id] = seg_meta_item[:2]
    
        X_local = X_local.reshape((*X_local.shape[:2], 17, 2))
    
        if not self.include_global:
              
            segs_data_np = np.empty(shape=(*X_local.shape[:-1], 3))
            segs_data_np[..., :2] = X_local
            segs_data_np[..., 2] = 1.0
            
        else:
            segs_data_np = np.empty(shape=(*X_local.shape[:-1], 7)) # check this when there are transformations
            segs_data_np[..., :2] = X_local
            segs_data_np[..., 2:6] = X_global
            segs_data_np[..., 6] = 1.0
    
        if self.kp18_format and segs_data_np.shape[-2] == 17:
            segs_data_np = keypoints17_to_coco18(segs_data_np)
        
        if self.headless:
            segs_data_np = segs_data_np[:,:,:14]
    
        segs_data_np = np.transpose(segs_data_np, (0, 3, 1, 2)).astype(np.float32)
            

        if ret_keys:
            return segs_data_np, segs_meta, person_keys, segs_ids
        else:
            return segs_data_np, segs_meta, segs_ids




def get_dataset_and_loader(args, split='train', validation=False):
    
    if args.num_transform > 0:
        trans_list = ae_trans_list[:args.num_transform]
    else: trans_list = None
    
    if args.use_fitted_scaler:
        with open('{}/robust.pkl'.format(args.ckpt_dir), 'rb') as handle:
            scaler = pickle.load(handle)
        print('Scaler loaded from {}'.format('{}/robust.pkl'.format(args.ckpt_dir)))
    else: scaler = None 
    
    dataset_args = {'transform_list': trans_list, 'debug': args.debug, 'headless': args.headless,
                    'seg_len': args.seg_len, 'normalize_pose': (args.normalization_strategy != 'none'), 'kp18_format': args.kp18_format,
                    'vid_res': args.vid_res, 'num_coords': args.num_coords, 'sub_mean': False,
                    'return_indices': False, 'return_metadata': True, 'return_mean': False,
                    'symm_range': args.symm_range, 'hip_center': args.hip_center, 
                    'normalization_strategy': args.normalization_strategy, 'ckpt': args.ckpt_dir, 'scaler': scaler, 
                    'kp_threshold': 0, 'double_item': False}

    loader_args = {'batch_size': args.batch_size, 'num_workers': args.num_workers, 'pin_memory': True}
    
    dataset_args['seg_stride'] = args.seg_stride if split == 'train' else 1  # No strides for test set
    if args.normalization_strategy=='robust': 
        dataset = PoseDatasetRobust(path_to_data=args.data_dir, 
                                    exp_dir=args.ckpt_dir,
                                    include_global=(args.num_coords==6), split=split, **dataset_args)
    else:
        dataset = PoseDataset(args.pose_path[split], **dataset_args) 
        
    loader = DataLoader(dataset, **loader_args, shuffle=(split == 'train'))
    if validation:
        dataset_args['seg_stride'] = 1
        if args.normalization_strategy=='robust': 
            val_dataset = PoseDatasetRobust(path_to_data=args.data_dir, 
                                    exp_dir=args.ckpt_dir,
                                    include_global=(args.num_coords==6), split='validation', **dataset_args)
        else:
            val_dataset = PoseDataset(args.pose_path['validation'], **dataset_args)
            
        val_loader = DataLoader(val_dataset, **loader_args, shuffle=False)
    else:
        val_dataset, val_loader = None, None
    
    return dataset, loader, val_dataset, val_loader
    
