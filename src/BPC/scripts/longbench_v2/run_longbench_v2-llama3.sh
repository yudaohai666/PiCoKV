#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
source /root/miniconda3/etc/profile.d/conda.sh
conda activate bpc
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

models=(
    llama3-1-8b-ins
    # mistral-7b-ins
)

max_capacity_prompts=(
    # 1024
    2048
    # 4096
)

methods=(
    # "bpc"
    "bpc-offline"
    # "origin"
    # "cakekv"
    # "compresskv"
    # "magicpig"
    # "pyramidkv"
    # "quest"
)

save_dir="outputs/longbench_v2"

# Methods that do not take a budget argument.
NO_BUDGET_METHODS="origin bpc bpc-offline magicpig spotlight"

for model in "${models[@]}"
do
    for method in "${methods[@]}"
    do
        config_file="config/${model}/${model}-${method}.json"

        if [[ " $NO_BUDGET_METHODS " =~ " $method " ]]; then
            echo "========================================"
            echo "Running model=$model method=$method"
            echo "Config: $config_file"
            echo "========================================"
            python -m benchmark.longbench_v2.pred_v2 \
                --save_dir $save_dir \
                --env_conf $config_file
        else
            for budget in "${max_capacity_prompts[@]}"
            do
                echo "========================================"
                echo "Running model=$model method=$method budget=$budget"
                echo "Config: $config_file"
                echo "========================================"
                python -m benchmark.longbench_v2.pred_v2 \
                    --save_dir $save_dir \
                    --env_conf $config_file \
                    --max_capacity_prompt $budget
            done
        fi
    done
done
