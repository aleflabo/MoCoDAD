import os


def init_args(args):
    if args.debug:
        args.ae_epochs = 10
    
    args.gt_path = args.test_path
    
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

    return args


def create_experiment_dirs(args):
    dataset = args.dataset_choice
    checkpoints_dir = os.path.join(args.exp_dir, dataset, args.dir_name)
    dirs = [checkpoints_dir]
    try:
        for dir_ in dirs:
            os.makedirs(dir_, exist_ok=True)
        print("Experiment directories created in {}".format(checkpoints_dir))
        return checkpoints_dir
    except Exception as err:
        print("Experiment directories creation Failed, error {}".format(err))
        exit(-1)