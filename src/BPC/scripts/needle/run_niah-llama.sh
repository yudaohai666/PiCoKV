#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
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
    "bpc"
    "bpc-offline"
    "origin"
    "cakekv"
    "compresskv"
    "magicpig"
    "pyramidkv"
    "quest"
    # "rocketkv"
    "snapkv"
    
    # "streamingllm"
)

# Methods that do not take a budget argument.
NO_BUDGET_METHODS="origin bpc bpc-offline magicpig sparse sparsetopk sparsetopk-offline"

# context_lengths: 8192 to 131072 step 8192
context_lengths=""
for i in $(seq 8192 8192 131072); do
    context_lengths="${context_lengths}${i} "
done
context_lengths=${context_lengths% }
echo "Context lengths: $context_lengths"

for model in "${models[@]}"
do
    for method in "${methods[@]}"
    do
        config_file="config/${model}/${model}-${method}.json"
        save_dir="outputs/needle/${model}-${method}"
        
        mkdir -p "$save_dir"
        
        if [[ " $NO_BUDGET_METHODS " =~ " $method " ]]; then
            echo "========================================"
            echo "Running model=$model method=$method (Multi-GPU)"
            echo "Config: $config_file"
            echo "Save dir: $save_dir"
            echo "========================================"
            
            python -m benchmark.needle.run_needle_in_haystack_multi_gpu \
                --context_length $context_lengths \
                --save_dir "$save_dir" \
                --env_conf "$config_file"
        else
            for budget in "${max_capacity_prompts[@]}"
            do
                echo "========================================"
                echo "Running model=$model method=$method budget=$budget (Multi-GPU)"
                echo "Config: $config_file"
                echo "Save dir: $save_dir"
                echo "========================================"
                
                python -m benchmark.needle.run_needle_in_haystack_multi_gpu \
                    --context_length $context_lengths \
                    --save_dir "$save_dir" \
                    --env_conf "$config_file" \
                    --max_capacity_prompt $budget
            done
        fi
    done
done
