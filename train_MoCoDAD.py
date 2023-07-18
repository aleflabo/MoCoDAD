import argparse
import os

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from utils.argparser import init_sub_args
from utils.dataset import get_dataset_and_loader
from utils.ema import EMACallback
import torch 
import numpy as np
import random
from models.diffusion_STS import LitAutoEncoder as Litmodel



if __name__== '__main__':

    
    parser = argparse.ArgumentParser(description='Pose_AD_Experiment')
    parser.add_argument('-c', '--config', type=str, required=True,
                        default='/your_default_config_file_path')
    
    args = parser.parse_args()
    config_path = args.config
    args = yaml.load(open(args.config), Loader=yaml.FullLoader)
    args = argparse.Namespace(**args)
    
    args, dataset_args, ae_args, res_args, opt_args = init_sub_args(args)
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed) 
    pl.seed_everything(args.seed)
    
    exp_dir = os.path.join(args.exp_dir, args.dataset_choice, args.dir_name)
    
    os.system(f'cp {config_path} {os.path.join(exp_dir, "config.yaml")}')     

    project_name = "Diffusion_" + args.project_name

   
    # Pass arguments as dataset arguments for PoseDatasetRobust
    dataset_args.exp_dir = exp_dir
    

    model = Litmodel(args)


    checkpoint_callback = ModelCheckpoint(dirpath=args.ckpt_dir, save_top_k=2,
                                          monitor="validation_auc" if (dataset_args.choice == 'UBnormal' or args.validation) else 'loss',
                                          mode="max" if (dataset_args.choice == 'UBnormal' or args.validation) else 'min'
                                         )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    print('ema = ',args.use_ema)
    if args.use_wandb:
        wandb_logger = WandbLogger(project=args.project_name, group=args.group_name, entity=args.wandb_entity, name=args.dir_name, config=args.__dict__,log_model='all')
        trainer = pl.Trainer.from_argparse_args(args, default_root_dir=args.ckpt_dir, 
                        logger = wandb_logger, log_every_n_steps=20, max_epochs=args.ae_epochs,
                        callbacks=[checkpoint_callback,lr_monitor, EMACallback()] if args.use_ema else [checkpoint_callback,lr_monitor],
                        val_check_interval=0.5, num_sanity_val_steps=0, 
                        strategy = DDPStrategy(find_unused_parameters=True), deterministic=True)
    else:
        wandb_logger = None
        trainer = pl.Trainer.from_argparse_args(args, default_root_dir=args.ckpt_dir, 
                            logger = wandb_logger, log_every_n_steps=20, max_epochs=args.ae_epochs,
                            callbacks=[checkpoint_callback,EMACallback()] if args.use_ema else [checkpoint_callback] ,
                            val_check_interval=0.5, num_sanity_val_steps=0, 
                            strategy = DDPStrategy(find_unused_parameters=True))

    if args.validation:
        train_dataset, train_loader, val_dataset, val_loader = get_dataset_and_loader(dataset_args, split=args.split, validation=args.validation)
        
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    else:
        train_dataset, train_loader = get_dataset_and_loader(dataset_args, split=args.split)
        trainer.fit(model=model, train_dataloaders=train_loader)
    
