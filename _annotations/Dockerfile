# Dockerfile for Alphapose 
# syntax=docker/dockerfile:1

# Configure the container to use the GPU
FROM nvidia/cuda:11.6.0-runtime-ubuntu20.04 as base

ENV NV_CUDA_LIB_VERSION "11.6.0-1"

FROM base as base-amd64

ENV NV_CUDA_CUDART_DEV_VERSION 11.6.55-1
ENV NV_NVML_DEV_VERSION 11.6.55-1
ENV NV_LIBCUSPARSE_DEV_VERSION 11.7.1.55-1
ENV NV_LIBNPP_DEV_VERSION 11.6.0.55-1
ENV NV_LIBNPP_DEV_PACKAGE libnpp-dev-11-6=${NV_LIBNPP_DEV_VERSION}

ENV NV_LIBCUBLAS_DEV_VERSION 11.8.1.74-1
ENV NV_LIBCUBLAS_DEV_PACKAGE_NAME libcublas-dev-11-6
ENV NV_LIBCUBLAS_DEV_PACKAGE ${NV_LIBCUBLAS_DEV_PACKAGE_NAME}=${NV_LIBCUBLAS_DEV_VERSION}

ENV NV_LIBNCCL_DEV_PACKAGE_NAME libnccl-dev
ENV NV_LIBNCCL_DEV_PACKAGE_VERSION 2.11.4-1
ENV NCCL_VERSION 2.11.4-1
ENV NV_LIBNCCL_DEV_PACKAGE ${NV_LIBNCCL_DEV_PACKAGE_NAME}=${NV_LIBNCCL_DEV_PACKAGE_VERSION}+cuda11.6

LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libtinfo5 libncursesw5 \
    cuda-cudart-dev-11-6=${NV_CUDA_CUDART_DEV_VERSION} \
    cuda-command-line-tools-11-6=${NV_CUDA_LIB_VERSION} \
    cuda-minimal-build-11-6=${NV_CUDA_LIB_VERSION} \
    cuda-libraries-dev-11-6=${NV_CUDA_LIB_VERSION} \
    cuda-nvml-dev-11-6=${NV_NVML_DEV_VERSION} \
    ${NV_LIBNPP_DEV_PACKAGE} \
    libcusparse-dev-11-6=${NV_LIBCUSPARSE_DEV_VERSION} \
    ${NV_LIBCUBLAS_DEV_PACKAGE} \
    ${NV_LIBNCCL_DEV_PACKAGE} \
    && rm -rf /var/lib/apt/lists/*

# Keep apt from auto upgrading the cublas and nccl packages. See https://gitlab.com/nvidia/container-images/cuda/-/issues/88
RUN apt-mark hold ${NV_LIBCUBLAS_DEV_PACKAGE_NAME} ${NV_LIBNCCL_DEV_PACKAGE_NAME}

ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Configure AlphaPose
ENV TZ=Europe/Rome
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN echo Installing Conda
RUN apt-get update -y && apt-get install software-properties-common wget -y

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda
# Put conda in path so we can use conda activate
ENV PATH $CONDA_DIR/bin:$PATH

# Activate the virtual environment
RUN echo "conda activate" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

RUN echo Installing Alphapose
WORKDIR /code
RUN apt-get install git -y
RUN git clone https://github.com/MVIG-SJTU/AlphaPose.git
WORKDIR /code/AlphaPose
RUN export PATH=/usr/local/cuda/bin/:$PATH
RUN export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH
RUN apt-get install libyaml-dev libgl1 -y

RUN conda install pyyaml==5.2 scipy==1.1.0 pytorch==1.2.0 torchvision==0.4.0 \
    pillow==6.2.1 matplotlib==3.2.2 cython notebook
RUN python setup.py build develop

# Set the working directory and the mount point for the shared memory
WORKDIR /code
RUN mkdir shared_memory

# Run a notebook instance
EXPOSE 8888                                           
ENTRYPOINT ["jupyter", "notebook", "--no-browser","--ip=0.0.0.0","--NotebookApp.token=''","--NotebookApp.password=''", "--allow-root"]
