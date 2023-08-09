# Multimodal Motion Conditioned Diffusion Model for Skeleton-based Video Anomaly Detection
_Alessandro Flaborea, Luca Collorone, Guido D'Amely, Stefano D'Arrigo, Bardh Prenkaj, Fabio Galasso_

<p align="center">
    <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/pytorch-lightning-blue.svg?logo=PyTorch%20Lightning"></a>
    <a href="https://wandb.ai/site"><img alt="Logging: wandb" src="https://img.shields.io/badge/logging-wandb-yellow"></a>

</p>


The official PyTorch implementation of the IEEE/CVF International Conference on Computer Vision (ICCV) '23 paper [**Multimodal Motion Conditioned Diffusion Model for Skeleton-based Video Anomaly Detection**](https://arxiv.org/abs/2307.07205).

<!-- Visit our [**webpage**](https://www.pinlab.org/coskad) for more details. -->

![teaser](assets/mocodad.jpg) 

## Content
```
.
├── assets
│   ├── mocodad.jpg
├── config
│   ├── Avenue
│   │   ├── mocodad_test.yaml
│   │   └── mocodad_train.yaml
│   ├── STC
│   │   ├── mocodad_test.yaml
│   │   └── mocodad_train.yaml
│   └── UBnormal
│       ├── mocodad-latent_train.yaml
│       ├── mocodad_test.yaml
│       └── mocodad_train.yaml
├── environment.yaml
├── eval_MoCoDAD.py
├── models
│   ├── common
│   │   └── components.py
│   ├── gcae
│   │   └── stsgcn.py
│   ├── mocodad_latent.py
│   ├── mocodad.py
│   └── stsae
│       ├── stsae.py
│       └── stsae_unet.py
├── README.md
├── scripts
│   ├── count_anomalies.py
│   ├── to_morais_format.py
│   └── visualization.py
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

1. conditioning_strategy
    -  'inject': Inject condition information into the model. The indices to be used as conditioning can be set using the 'conditioning_indices' parameter. Enabled by default. 
    - 'concat': concat conditioning and noised data to be passed to the model. The indices to be used as conditioning can be set using the 'conditioning_indices' parameter.
    - 'inbetween_imp': Uses the list of indices of the 'conditioning_indices' parameter to select the indices to be used as conditioning.
    - 'random_imp': 'conditioning_indices' must be int and it is used as the number of random indices that will be selected 
    - 'no_condition': if enabled, no motion condition is passed to the model

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


