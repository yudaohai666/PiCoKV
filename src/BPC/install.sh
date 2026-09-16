#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# source /root/miniconda3/etc/profile.d/conda.sh
# conda activate base

conda create -n bpc python=3.11 -y
conda activate bpc

# PyTorch 2.6.0 + CUDA 11.8
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118

cd "$SCRIPT_DIR"

pip install -r requirements.txt
pip install -e .

# cd 3rd/lm-evaluation-harness/
# pip install -e .

cd "$SCRIPT_DIR"

# flash-attn prebuilt wheel
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.1.post4/flash_attn-2.7.1.post4+cu11torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl


echo "安装完成!"