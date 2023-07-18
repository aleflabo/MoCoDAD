import os
import time
import argparse

def init_args():
    parser = init_parser()
    args = parser.parse_args()
    return init_sub_args(args)

def init_sub_args(args):
    if args.debug:
        args.ae_epochs = 10
    
    args.gt_path = args.test_path # os.path.join(args.data_dir, 'testing', 'test_frame_mask')
    
    if args.dataset_choice in ['STC','HR-STC','HR-Avenue', 'UBnormal']:
        args.pose_path = {
            'train' : os.path.join(args.data_dir, 'pose', 'training/tracked_person/'),
            'test' : os.path.join(args.data_dir, 'pose', 'testing/tracked_person/'),
            'validation': os.path.join(args.data_dir, 'pose', 'validating/tracked_person/')}
        if args.validation:
            if args.dataset_choice == 'UBnormal':
                args.pose_path['validation'] = os.path.join(args.data_dir, 'pose', 'validating/tracked_person/')
                args.gt_path = os.path.join(args.data_dir, 'validating', 'test_frame_mask')
            else:
                args.pose_path['validation'] = os.path.join(args.data_dir, 'pose', 'testing/tracked_person/')
                args.gt_path = os.path.join(args.data_dir, 'testing', 'test_frame_mask')
            
    elif args.dataset_choice == 'Avenue':
        args.pose_path = {
            'train' : os.path.join(args.data_dir, 'pose', 'training/tracked_person/'),
            'test' : os.path.join(args.data_dir, 'pose', 'testing/tracked_person/')}
        print('Not usable yet.')
        exit(-1)

    # if args.is_train:
    args.ckpt_dir = create_experiment_dirs(args)

    if not args.dataset_sub_mean:
        args.dataset_return_mean = False

    opt_args = args_rm_prefix(args, 'opt_')
    data_args = args_rm_prefix(args, 'dataset_')
    ae_args = args_rm_prefix(args, 'ae_')
    res_args = args_rm_prefix(args, 'res_')
    return args, data_args, ae_args, res_args, opt_args

def init_parser(default_data_dir='../data/STC',
                default_exp_dir='./checkpoints', is_train=True):
    parser = argparse.ArgumentParser("Pose_AD_Experiment")
    
    # General arguments:
    parser.add_argument('--split', type=str, default='train',
                        help="Flag to know wheter we are training or testing. Default:True")
    parser.add_argument('--debug', type=bool, default=False,
                        help='Debug experiment script with minimal epochs. (default: False)')
    parser.add_argument('--device', type=str, default='cuda:1', metavar='DEV',
                        help='Device for feature calculation (default: \'cuda:0\')')
    parser.add_argument('--seed', type=int, metavar='S', default=999,
                        help='Random seed, use 999 for random (default: 999)')
    parser.add_argument('--data_dir', type=str, default=default_data_dir, metavar='DATA_DIR',
                        help="Path to directory holding .npy and .pkl files (default: {})".format(default_data_dir))
    parser.add_argument('--exp_dir', type=str, default=default_exp_dir, metavar='EXP_DIR',
                    help="Path to the directory where models will be saved (default: {})".format(default_exp_dir))
    parser.add_argument('--dir_name', type=str, default='',
                        help="Path to the directory where models will be saved (format: args.exp_dir/args.dataset_choice/args.dir_name)")
    parser.add_argument('--num_coords', '-n_c', type=int, default=2,
                        help='Number of AlphaPose coordinates to consider (default=2)')
    parser.add_argument('--alpha', '-a', type=float, default=1e-3,  metavar='G',
                    help='Alpha value for weighting L2 regularization (default: 1e-3)')
    parser.add_argument('--create_experiment_dir', type=bool, default=False,
                        help='Whether if create the experiment directory')
    parser.add_argument('--pretrained', action='store_true',help='Load pretrained model')
    parser.add_argument('--test_path', type=str, default='/home/zeus/aleflabo/gepc/data/testing/test_frame_mask', metavar='test_path',
                    help="Path to the test mask containing the ground truth labels")
    parser.add_argument('--dropout', type=float, default=0., metavar='DROPOUT',
                        help='Dropout training Parameter (default: 0.3)')
    parser.add_argument('--conv_oper', type=str, default='sagc', metavar='CONV_OPER',
                        help="Convolutional Operator to be used [sagc, gcn] (default: 'sagc')")
    parser.add_argument('--act', type=str, default='relu', metavar='ACT_TYPE', 
                        help="Activation used in ST-GCN [relu, mish] (default: 'relu')")
    parser.add_argument('--pad_size', type=int, default=20, metavar='pad_size',
                        help='number of frames to cut out when a person enters in the middle of the scene or exits before it ends')
    parser.add_argument('--load_ckpt', type=str, default='', 
                        help="Load a ckpt from a specific path. (default:'')")

    # Dataset arguments
    parser.add_argument('--dataset_headless', type=bool, default=False,
                        help='Remove head keypoints (14-17) and use 14 kps only. (default: False)')
    parser.add_argument('--dataset_choice', type=str, default='STC', metavar='dataset_choice',
                    help="Select which dataset to use from [STC, HR-STC, HR-Avenue]")
    parser.add_argument('--dataset_seg_len', type=int, default=12, metavar='SGLEN',
                        help='Number of frames for training segment sliding window, a multiply of 6 (default: 12)')
    parser.add_argument('--dataset_seg_stride', type=int, default=8, metavar='SGST',
                        help='Stride for training segment sliding window (default: 1)')
    parser.add_argument('--dataset_start_offset', type=int, default=0,
                        help='sliding window offset from the start of the video (default=0)')
    parser.add_argument('--dataset_num_transform', type=int, default=5, metavar='T',
                        help='number of transformations to use for augmentation (default: 5)')
    parser.add_argument('--dataset_symm_range', action='store_false',
                        help='Whether if embed the input in [-1, 1] (symmetric) interval (default=True)')
    parser.add_argument('--dataset_return_indices',type=bool, default=False,
                        help='Return the indices to reconstruct the original pose (default=True)')
    parser.add_argument('--dataset_sub_mean', '-smean', action='store_false',
                        help='wether if apply the translation wrt the mean in each window (default=True)')
    parser.add_argument('--dataset_vid_res', type=int, nargs=2,
                        help='Resolution of the video in the dataset (default=[856, 480])',)
    parser.add_argument('--dataset_normalize_pose', '-np', type=bool, default=True,
                        help='Whether if apply normalization (default=True)')
    parser.add_argument('--dataset_kp18_format', '-kp18', type=bool, default=True,
                        help='18 Keypoints in COCO format (default=True)')
    parser.add_argument('--dataset_batch_size', '-ae_b', type=int, default=512,  metavar='B',
                        help='Batch sizes for autoencoder. (default: 512)')
    parser.add_argument('--dataset_hip_center', '-hc', type=bool, default=False,
                        help='Center the skeleton on the hip center, thougt to be alternative to --dataset_sub_mean (default=True)')
    parser.add_argument('--dataset_num_workers', type=int, default=8, metavar='W',
                        help='number of dataloader workers (0=current thread) (default: 32)')
    parser.add_argument('--dataset_normalization_strategy', type=str, default='markovitz',
                        help='Decide which normalization has to be used. options: [markovitz,robust]')
    parser.add_argument('--dataset_use_fitted_scaler', action='store_true',
                        help='Load a fitted scaler. The location of the scaler is in args.ckpt_dir/robust.pkl by default')
    parser.add_argument('--dataset_kp_th', type=float, default=0,
                        help='Use a threshold to determine wheter the keypoint is usable or not. When a keypoint_score < kp_th then the coords are set to 0. (deafult=0)')

    # AE arguments:
    parser.add_argument('--ae_act', type=str, default='relu', metavar='ACT_TYPE', 
                        help="Activation used in ST-GCN [relu, mish] (default: 'relu')")
    parser.add_argument('--ae_fn', type=str, metavar='AE_FN',
                        help="Path to a trained AE models to start with")
    parser.add_argument('--ae_test_every', type=int, default=20, metavar='T',
                        help='How many epochs between test evaluations (default: 20)')
    parser.add_argument('--ae_epochs', '-ae_e', type=int, default=10,  metavar='E',
                        help='Number of epochs per cycle. (default: 10)')
    
    # Optim & Scheduler Arguments:
    parser.add_argument('--opt_optimizer', '-opt_o', type=str, default='adam', metavar='opt_OPT',
                        help="Optimizer (default: 'adam')")
    parser.add_argument('--opt_scheduler', '-opt_s', type=str, default='tri', metavar='opt_SCH',
                        help="Optimization LR scheduler (default: 'step')")
    parser.add_argument('--opt_lr', type=float, default=1e-4, metavar='LR',
                        help='Optimizer Learning Rate Parameter (default: 1e-4)')
    parser.add_argument('--opt_lr_decay', '-optim_ld', type=float, default=0.99, metavar='LD',
                        help='Optimizer Learning Rate Decay Parameter (default: 0.99)')
    parser.add_argument('--opt_weight_decay', type=float, default=1e-5, metavar='WD',
                    help='Optimizer Weight Decay Parameter (default: 1e-5)')

    # Postprocessing arguments:
    parser.add_argument('--smoothing', '-smooth', type=int, default=50,
                        help='size of the smoothing_kernel')
    # Viz arguments:
    parser.add_argument('--viz_gif_folder', '-vf', type=str, default=os.path.join(default_data_dir, 'viz'),
                        help='Folder to store visual gif')
    return parser

def args_rm_prefix(args, prefix):
    wp_args = argparse.Namespace(**vars(args))
    args_dict = vars(args)
    wp_args_dict = vars(wp_args)
    
    # single_dict = {}
    for key, value in args_dict.items():
        if key.startswith(prefix):
            ae_key = key[len(prefix):]
            wp_args_dict[ae_key] = value
            # single_dict[ae_key] = value

    return wp_args


def create_experiment_dirs(args):
    # default: experiments are saved in the folder ./checkpoints
    # time_str = time.strftime("%b%d_%H%M")

    # dirmap = 'av'
    dataset = args.dataset_choice
    
    checkpoints_dir = os.path.join(args.exp_dir, dataset, args.dir_name)
    # checkpoints_dir = os.path.join(experiment_dir, 'checkpoints/')
    dirs = [checkpoints_dir]
    try:
        for dir_ in dirs:
            os.makedirs(dir_, exist_ok=True)
        print("Experiment directories created in {}".format(checkpoints_dir))
        return checkpoints_dir
    except Exception as err:
        print("Experiment directories creation Failed, error {}".format(err))
        exit(-1)