#!/bin/bash

# Compute InfiniteBench scores.
# Usage:
#   bash scripts/infinitebench/compute_scores.sh                    # all methods
#   bash scripts/infinitebench/compute_scores.sh origin             # single method
#   bash scripts/infinitebench/compute_scores.sh "origin snapkv"    # multiple methods

source /root/miniconda3/etc/profile.d/conda.sh
conda activate bpc
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

# model_name=${2:-"Llama-3.1-8B-Instruct"}
model_name=${2:-"Qwen2.5-7B-Instruct-1M"}
# Methods that do not take max_capacity_prompt.
METHODS_NO_CAPACITY="bpc origin magicpig bpc-offline"
# Methods that require max_capacity_prompt.
METHODS_WITH_CAPACITY="pyramidkv quest cakekv compresskv"

if [ -n "$1" ]; then
    METHODS="$1"
else
    METHODS="$METHODS_NO_CAPACITY $METHODS_WITH_CAPACITY"
fi

echo "=========================================="
echo "Computing scores for model: $model_name"
echo "Methods: $METHODS"
echo "=========================================="

for method in $METHODS; do
    echo ""
    echo ">>> Processing method: $method"
    
    if [[ " $METHODS_NO_CAPACITY " =~ " $method " ]]; then
        echo "Running without max_capacity_prompt..."
        python -m benchmark.infinitebench.compute_scores \
            --task all \
            --output_dir outputs/infinitebench \
            --model_name $model_name \
            --method $method
    else
        # Default max_capacity_prompt = 2048.
        echo "Running with max_capacity_prompt..."
        python -m benchmark.infinitebench.compute_scores \
            --task all \
            --output_dir outputs/infinitebench \
            --model_name $model_name \
            --method $method \
            --max_capacity_prompt 2048
    fi
done

echo ""
echo "=========================================="
echo "All scores computed!"
echo "=========================================="
