import argparse
import os

import torch
import pytorch_lightning as pl
import yaml
from models.mocodad import MoCoDAD
from utils.argparser import init_args
from utils.dataset import get_dataset_and_loader
from utils.model_utils import processing_data


# Parse command line arguments and load config file
parser = argparse.ArgumentParser(description='MoCoDAD')
parser.add_argument('-c', '--config', type=str, required=True)
args = parser.parse_args()
args = yaml.load(open(args.config), Loader=yaml.FullLoader)
args = argparse.Namespace(**args)
args = init_args(args)

# Initialize the model
model = MoCoDAD(args)

print('Loading data and creating loaders.....')
ckpt_path = os.path.join(args.ckpt_dir, args.load_ckpt)
dataset, loader, _, _ = get_dataset_and_loader(args, split=args.split)

# Initialize trainer and test
trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices,
                     default_root_dir=args.ckpt_dir, max_epochs=1, logger=False)
out = trainer.predict(model, dataloaders=loader, ckpt_path=ckpt_path, return_predictions=True)
unpacked_result = processing_data(out)
file_names = ['prediction', 'gt_data', 'trans', 'metadata', 'frames']
save_dir = os.path.join(args.ckpt_dir, f'saved_tensors_{args.split}_{args.aggregation_strategy}_{args.n_generated_samples}')
for i in range(len(unpacked_result)):
    torch.save(unpacked_result[i], os.path.join(save_dir, file_names[i]+'.pt'))