import os
import numpy as np
import pickle

from copy import deepcopy

from utils.data import load_trajectories, extract_global_features
from utils.data import change_coordinate_system, scale_trajectories, aggregate_autoencoder_data
from utils.data import input_trajectories_missing_steps
from utils.preprocessing import remove_short_trajectories, aggregate_rnn_autoencoder_data


def save_scaler(scaler, path):
    with open(path, 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
    
        
def load_scaler(path):
    with open(path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return scaler


def data_of_combined_model(**args):
    # General
    exp_dir = args.get('exp_dir', '')
    split = args.get('split', 'train')
    normalize_pose = args.get('normalize_pose', True)
    trajectories_path = args.get('trajectories_path', '')
    include_global = args.get('include_global', True)
    debug = args.get('debug', False)
    if 'train' in split:
        subfolder = 'training'
    elif 'test' in split:
        subfolder = 'testing'
    else:
        subfolder = 'validating'
    trajectories_path = os.path.join(trajectories_path, f'{subfolder}/trajectories')
    video_resolution = args.get('vid_res', (1080,720))
    video_resolution = np.array(video_resolution, dtype=np.float32)
    # Architecture
    reconstruct_original_data = args.get('reconstruct_original_data', False) 
    input_length = args.get('seg_len', 12)
    seg_stride = args.get('seg_stride', 1) - 1 
    pred_length = 0 
    # Training
    input_missing_steps = False # args.input_missing_steps
    
    if normalize_pose == True:
        global_normalisation_strategy = args.get('normalization_strategy', 'robust')
        local_normalisation_strategy = args.get('normalization_strategy', 'robust')
        out_normalisation_strategy = args.get('normalization_strategy', 'robust')


    trajectories = load_trajectories(trajectories_path, debug=debug, split=split)
    print('\nLoaded %d trajectories.' % len(trajectories))

    trajectories = remove_short_trajectories(trajectories, input_length=input_length,
                                             input_gap=seg_stride, pred_length=pred_length)
    print('\nRemoved short trajectories. Number of trajectories left: %d.' % len(trajectories))

    # trajectories, trajectories_val = split_into_train_and_test(trajectories, train_ratio=0.8, seed=42)

    if input_missing_steps:
        trajectories = input_trajectories_missing_steps(trajectories)
        print('\nInputted missing steps of trajectories.')

    # Global
    if include_global:
        global_trajectories = extract_global_features(deepcopy(trajectories), video_resolution=video_resolution)

        global_trajectories = change_coordinate_system(global_trajectories, video_resolution=video_resolution,
                                                        coordinate_system='global', invert=False)

        print('\nChanged global trajectories\'s coordinate system to global.')
        
        X_global, y_global, X_global_meta, y_global_meta = aggregate_rnn_autoencoder_data(global_trajectories, 
                                                                                        input_length=input_length,
                                                                                        input_gap=seg_stride, pred_length=pred_length, 
                                                                                        return_ids=True)
        
        if normalize_pose == True:
            scaler_path = os.path.join(exp_dir, f'global_{global_normalisation_strategy}.pickle')
            
            if split == 'train':
                _, global_scaler = scale_trajectories(aggregate_autoencoder_data(global_trajectories),
                                                    strategy=global_normalisation_strategy)
                save_scaler(global_scaler, scaler_path)
            else:
                global_scaler = load_scaler(scaler_path)

            X_global, _ = scale_trajectories(X_global, scaler=global_scaler, strategy=global_normalisation_strategy)
            
            if y_global is not None:
                y_global, _ = scale_trajectories(y_global, scaler=global_scaler,
                                                strategy=global_normalisation_strategy)
                
            print('\nNormalised global trajectories using the %s normalisation strategy.' % global_normalisation_strategy)
    
    else:
        X_global, X_global_meta = None, None
    
    # Local
    local_trajectories = deepcopy(trajectories) if reconstruct_original_data else trajectories

    local_trajectories = change_coordinate_system(local_trajectories, video_resolution=video_resolution,
                                                  coordinate_system='bounding_box_centre', invert=False)

    print('\nChanged local trajectories\'s coordinate system to bounding_box_centre.')
    
    X_local, y_local, X_local_meta, y_local_meta = aggregate_rnn_autoencoder_data(local_trajectories, input_length=input_length, 
                                                                                  input_gap=seg_stride, pred_length=pred_length,
                                                                                  return_ids=True)
    
    if normalize_pose == True:
        scaler_path = os.path.join(exp_dir, f'local_{local_normalisation_strategy}.pickle')
        
        if split == 'train':
            _, local_scaler = scale_trajectories(aggregate_autoencoder_data(local_trajectories),
                                                strategy=local_normalisation_strategy)
            save_scaler(local_scaler, scaler_path)
        elif split == 'validation' and 'UBnormal' not in trajectories_path:
            _, local_scaler = scale_trajectories(aggregate_autoencoder_data(local_trajectories),
                                                strategy=local_normalisation_strategy)
            save_scaler(local_scaler, scaler_path.removesuffix('.pickle') + '_val.pickle') 
        else:
            local_scaler = load_scaler(scaler_path)

        X_local, _ = scale_trajectories(X_local, scaler=local_scaler, strategy=local_normalisation_strategy)

        if y_local is not None:
            y_local, _ = scale_trajectories(y_local, scaler=local_scaler, strategy=local_normalisation_strategy)
        
        print('\nNormalised local trajectories using the %s normalisation strategy.' % local_normalisation_strategy)

    # (Optional) Reconstruct the original data
    if reconstruct_original_data:
        print('\nReconstruction/Prediction target is the original data.')
        out_trajectories = trajectories
        
        out_trajectories = change_coordinate_system(out_trajectories, video_resolution=video_resolution,
                                                    coordinate_system='global', invert=False)
    
        print('\nChanged target trajectories\'s coordinate system to global.')
        
        scaler_path = os.path.join(exp_dir, f'out_{out_normalisation_strategy}.pickle')
    
        if split == 'train':
            _, out_scaler = scale_trajectories(aggregate_autoencoder_data(out_trajectories),
                                               strategy=out_normalisation_strategy)
            save_scaler(out_scaler, scaler_path)
        else:
            out_scaler = load_scaler(scaler_path)
        
        ######## X_out_{}, y_out_{} numpy arrays

        X_out, y_out, X_out_meta, y_out_meta = aggregate_rnn_autoencoder_data(out_trajectories, input_length=input_length, 
                                                                              input_gap=seg_stride, pred_length=pred_length,
                                                                              return_ids=True)

        X_out, _ = scale_trajectories(X_out, scaler=out_scaler, strategy=out_normalisation_strategy)
        
        if y_out is not None:
            y_out, _ = scale_trajectories(y_out, scaler=out_scaler, strategy=out_normalisation_strategy)
            
        print('\nNormalised target trajectories using the %s normalisation strategy.' % out_normalisation_strategy)
        
            
    if pred_length > 0:
        
        if reconstruct_original_data:
            return (X_global, X_global_meta), \
                   (X_local, X_local_meta), \
                   (X_out, X_out_meta), \
                   (y_global, y_global_meta), \
                   (y_local, y_local_meta), \
                   (y_out, y_out_meta) 
        else:
            return (X_global, X_global_meta), \
                   (X_local, X_local_meta), \
                   (y_global, y_global_meta), \
                   (y_local, y_local_meta)
    else:
        if reconstruct_original_data:
            return (X_global, X_global_meta), \
                   (X_local, X_local_meta), \
                   (X_out, X_out_meta)
        else:
            return (X_global, X_global_meta), \
                   (X_local, X_local_meta)