#!/usr/bin/env bash
# Install KIVI into the project venv (keep torch/transformers as-is), then verify
# example.py and mem_spd_test.py (KIVI vs. standard attention).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$(cd "$SCRIPT_DIR/../.." && pwd)/.venv"

# Base model used by both verification scripts. Override with:
#   MODEL_NAME_OR_PATH=/path/to/model bash setup_env_verify.sh
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/data/Llama-3.1-8B-Instruct}"

source "$VENV/bin/activate"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"

# runtime deps (do NOT install KIVI root pkg: it would downgrade torch/transformers)
uv pip install  setuptools wheel ninja
uv pip install datasets sentencepiece protobuf accelerate triton

# CUDA kernel kivi_gemv (--no-build-isolation: build against venv torch)
cd "$SCRIPT_DIR/quant"
uv pip install -e . --no-build-isolation

python -c "import torch, transformers, datasets, kivi_gemv; print('deps OK')"

# ---- verification ----
cd "$SCRIPT_DIR"
echo "=========================================="
echo "model: $MODEL_NAME_OR_PATH"
echo "=========================================="

echo "[1/2] verifying example.py ..."
python example.py --model_name_or_path "$MODEL_NAME_OR_PATH"

echo "[2/2] verifying mem_spd_test.py (KIVI vs. standard attention) ..."
python mem_spd_test.py --model_name_or_path "$MODEL_NAME_OR_PATH"

echo "verification done."
