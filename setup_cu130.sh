#!/bin/bash
set -e

conda create -n tqdne python=3.11 -y

conda install -n tqdne main::setuptools==80.10.2 -y
conda install -n tqdne main::numpy==1.25.2 -y
conda install -n tqdne main::einops==0.8.1 -y
conda install -n tqdne main::h5py==3.13.0 -y
conda install -n tqdne main::librosa==0.11.0 -y
conda install -n tqdne main::jupyterlab==4.2.5 -y
conda install -n tqdne main::scikit-image==0.25.2 -y
conda install -n tqdne main::scikit-learn==1.6.1 -y
conda install -n tqdne main::seaborn==0.13.2 -y
conda install -n tqdne main::tqdm==4.68.2 -y

echo "=== [pip 1-5/10] installing pytorch lightning, obspy, seisbench, torchmetrics, and wandb ==="
conda run -n tqdne pip install pytorch-lightning==2.5.1 obspy==1.4.1 seisbench==0.8.2 torchmetrics==1.7.0 wandb==0.25.0

echo "=== [pip 6-8/10] installing torch, torchvision, torchaudio (cu130) ==="
conda run -n tqdne pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130

echo "=== [pip 9/10] installing oq-engine requirements ==="
conda run -n tqdne pip install -r $PWD/experiments_stead/envs/oq-engine_requirements-py311-linux64.txt

echo "=== [pip 10/8] installing tqdne local package (editable install) ==="
conda run -n tqdne pip install -e $PWD

echo ""
echo ">>> All installations complete. Environment 'tqdne' is ready. <<<"
