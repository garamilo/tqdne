#!/bin/bash
set -e

conda create -n tqdne_stead python=3.11 -y

conda install -n tqdne_stead main::setuptools==80.10.2 -y
conda install -n tqdne_stead main::numpy==1.25.2 -y
conda install -n tqdne_stead main::einops==0.8.1 -y
conda install -n tqdne_stead main::h5py==3.13.0 -y
conda install -n tqdne_stead main::librosa==0.11.0 -y
conda install -n tqdne_stead main::jupyterlab==4.2.5 -y
conda install -n tqdne_stead main::scikit-image==0.25.2 -y
conda install -n tqdne_stead main::scikit-learn==1.6.1 -y
conda install -n tqdne_stead main::seaborn==0.13.2 -y
conda install -n tqdne_stead main::tqdm==4.68.2 -y

echo "=== [pip 1/8] installing pytorch-lightning==2.5.1 ==="
conda run -n tqdne_stead pip install pytorch-lightning==2.5.1

echo "=== [pip 2/8] installing obspy==1.4.1 ==="
conda run -n tqdne_stead pip install obspy==1.4.1

echo "=== [pip 3/8] installing seisbench==0.8.2 ==="
conda run -n tqdne_stead pip install seisbench==0.8.2

echo "=== [pip 4/8] installing torch, torchvision, torchaudio (cu130) ==="
conda run -n tqdne_stead pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130

echo "=== [pip 5/8] installing torchmetrics==1.7.0 ==="
conda run -n tqdne_stead pip install torchmetrics==1.7.0

echo "=== [pip 6/8] installing wandb==0.25.0 ==="
conda run -n tqdne_stead pip install wandb==0.25.0

echo "=== [pip 7/8] installing oq-engine requirements ==="
conda run -n tqdne_stead pip install -r $PWD/experiments_stead/envs/oq-engine_requirements-py311-linux64.txt

echo "=== [pip 8/8] installing tqdne local package (editable install) ==="
conda run -n tqdne_stead pip install -e $PWD

echo ""
echo ">>> All installations complete. Environment 'tqdne_stead' is ready. <<<"
