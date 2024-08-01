# Code for generating the annotations with Alphapose+Poseflow
---

> [!NOTE]
> This is the official code that has been used for annotating UBnormal. It also provides some examples on other datasets.

> [!WARNING]
> This code has been updated up to May 2022. It may rely on outdated software or may not work due to deprecated code.


## Annotating with Alphapose+Poseflow

> [!WARNING]
> Please use it with caution and refer to the official implementations of Alphapose and Poseflow for updated code.

### Build the container for Alphapose+Poseflow

In order to build the container for Alphapose+Poseflow, open a terminal within the same folder of `Dockerfile` and issue the following command:

```sh
docker build -t pose_and_track .
```

### Run and annotate

Create a folder `shared_memory` in your host system and place the dataset you want to annotate in a subfolder `data`:

```sh
mkdir -p /your_path/shared_memory/data
# place your dataset within shared_memory/data
```

To start the container, run:

```sh
docker run -dp 127.0.0.1:8888:8888 -v /your_path/shared_memory:/code/shared_memory pose_and_track
```

> [!TIP]
> Refer to the Docker's official documention.


Follow the steps of the notebook and modify them according to your needs.

### Notes on the configurations

>#### Alphapose
>
> - Config: pretrained ResNet-50, lr1e-3; [config](https://github.com/MVIG-SJTU/AlphaPose/blob/master/configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml) 
> - Detector: YOLOv3
> - Checkpoint: FastRes-50
> - Pose batch (pose estimation maximum batch size PER GPU): 20 due to GPU memory limits
> - Detection batch (detection batch size PER GPU): 1 due to GPU memory limits
> - Min box area (min box area to filter out): 0 (default)
> - All other parameters left as default

>#### Poseflow
>
> - All parameters left as default
> - NOTE: the DROP option for pruning tracked keypoints with score \<DROP (default 2) is not applied in the script 
