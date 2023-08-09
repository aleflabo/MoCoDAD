# Multimodal Motion Conditioned Diffusion Model for Skeleton-based Video Anomaly Detection
_Alessandro Flaborea, Luca Collorone, Guido D'Amely, Stefano D'Arrigo, Bardh Prenkaj, Fabio Galasso_

<p align="center">
    <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/pytorch-lightning-blue.svg?logo=PyTorch%20Lightning"></a>
    <a href="https://wandb.ai/site"><img alt="Logging: wandb" src="https://img.shields.io/badge/logging-wandb-yellow"></a>

</p>


The official PyTorch implementation of the IEEE/CVF International Conference on Computer Vision (ICCV) '23 paper [**Multimodal Motion Conditioned Diffusion Model for Skeleton-based Video Anomaly Detection**](https://arxiv.org/abs/2307.07205).

<!-- Visit our [**webpage**](https://www.pinlab.org/coskad) for more details. -->

![teaser](assets/mocodad-1.png) 

## Content
```
.
├── assets
│   └── model.png
├── config
│   ├── Avenue
│   │   └── diffusion_ave.yaml
│   ├── STC
│   │   └── diffusion_stc.yaml
│   └── UBnormal
│       └── diffusion.yaml
├── environment.yaml
├── eval_MoCoDAD.py
├── models
│   ├── common
│   │   ├── components.py
│   │   └── diffusion_components.py
│   ├── diffusion_STS.py
│   ├── gcae
│   │   ├── stsgcn_diffusion.py
│   │   ├── stsgcn_diffusion_unet.py
│   │   └── stsgcn.py
│   ├── stsae
│   │   ├── stsae_diffusion.py
│   │   ├── stsae_diffusion_unet.py
│   │   └── stsae_hidden_hypersphere.py
│   ├── stse
│   │   └── stse_hidden_hypersphere.py
│   └── stsve
│       └── stsve_hidden_hypersphere.py
├── README.md
├── train_MoCoDAD.py
└── utils
    ├── argparser.py
    ├── data.py
    ├── dataset.py
    ├── dataset_utils.py
    ├── diffusion_utils.py
    ├── ema.py
    ├── eval_utils.py
    ├── get_robust_data.py
    ├── __init__.py
    ├── model_utils.py
    ├── preprocessing.py
    └── tools.py
```

## Setup
### Environment
```
conda env create -f environment.yaml
conda activate mocodad
```

### Datasets
Send a mail to flaborea@di.uniroma1.it to have the datasets. (We're planning to uploading them to a publicly available repository)


### **Training** 

To train MoCoDAD, you can select the different type of conditioning of the model. The default parameters achieve the best results reported in the paper 

In each config file you can choose the conditioning strategy and change the diffusion process parameters:

1. Conditioning
    -  inject_condition: best performing conditioning techniques. Inject condition information into the model. The indexes to be used as conditioning can be set using the 'indices' parameter. Enabled by default. 
    - concat_condition: concat conditioning and noise data to be passed to the model
    - no_condition: if enabled, no condition is passed to the model
    - interleave: if 'num_random_indices'=0 the poses in 'indices' are used as conditioning. If 'num_random_indices'>0 the conditioning poses are chosen at random 

2. Diffusion Process
    -  noise_steps: how many diffusion steps have to be performed

Update the args 'data_dir', 'test_path', 'dataset_path_to_robust' with the path where you stored the datasets.  To better track your experiments, change 'dir_name' and the wandb parameters.

To train MoCoDAD:
```
python train_MoCoDAD.py --config config/[Avenue/UBnormal/STC]/{config_name}.yaml
```


### Once trained, you can run the **Evaluation**

The training config is saved the associated experiment directory (/args.exp_dir/args.dataset_choice/args.dir_name). 
In order to evaluate the model on the test set, you need to change the following parameters in the config:

- split: 'Test'
- validation: 'False'
- load_ckpt: 'path_to_the_saved_ckpt'

Test MoCoDAD
```
python eval_MoCoDAD.py --config /args.exp_dir/args.dataset_choice/args.dir_name/config.yaml
```
additional flag you can use:
- use_hr: False -> just for test. Use the entire version of the dataset or the Human-Related one.


