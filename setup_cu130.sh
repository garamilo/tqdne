#!/bin/bash

conda create -n tqdne python=3.11 -y
conda activate tqdne

conda install main::setuptools==80.10.2 -n tqdne -y
conda install main::numpy==1.25.2 -n tqdne -y
conda install main::einops==0.8.1 -n tqdne -y
conda install main::h5py==3.13.0 -n tqdne -y
conda install main::librosa==0.11.0 -n tqdne -y
conda install main::jupyterlab==4.2.5 -n tqdne -y
conda install main::scikit-image==0.25.2 -n tqdne -y
conda install main::scikit-learn==1.6.1 -n tqdne -y
conda install main::seaborn==0.13.2 -n tqdne -y
conda install main::tqdm==4.68.2 -n tqdne -y
pip install pytorch-lightning==2.5.1
pip install obspy==1.4.1
pip install seisbench==0.8.2
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
pip install torchmetrics==1.7.0
pip install wandb==0.25.0
pip install -r ./experiments/envs/oq-engine_requirements-py311-linux64.txt
pip install -e .