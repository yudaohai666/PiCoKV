#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HUB_OFFLINE=1
export HF_ENDPOINT=https://hf-mirror.com
# source /root/miniconda3/etc/profile.d/conda.sh
# conda activate bpc
# (auto cd to project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

# Per-task sample cap; empty or 0 means run everything.
NUM_SAMPLES=${NUM_SAMPLES:-100}

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
    # "origin"
    "bpc"
    # "bpc-offline"
    "origin"
    # "cakekv"
    # "compresskv"
    # "magicpig"
    # "pyramidkv"
    # "quest"
)

tasks=(
    passkey
    # number_string
    math_find
    # code_debug
    longbook_choice_eng
    longbook_qa_chn
    longbook_qa_eng
    longbook_sum_eng
    longdialogue_qa_eng
)

# Methods that do not take a budget argument.
NO_BUDGET_METHODS="origin bpc bpc-offline magicpig sparse sparsetopk sparsetopk-offline"

for model in "${models[@]}"
do
    for method in "${methods[@]}"
    do
        config_file="config/${model}/${model}-${method}.json"

        sample_args=""
        if [[ -n "$NUM_SAMPLES" && "$NUM_SAMPLES" -gt 0 ]]; then
            sample_args="--start_idx 0 --stop_idx $NUM_SAMPLES"
            echo "Limiting to $NUM_SAMPLES samples per task"
        fi

        if [[ " $NO_BUDGET_METHODS " =~ " $method " ]]; then
            for task in "${tasks[@]}"
            do
                echo "========================================"
                echo "Running model=$model method=$method task=$task"
                echo "Config: $config_file"
                echo "========================================"
                python -m benchmark.infinitebench.llama_parallel \
                    --task $task \
                    --output_dir outputs/infinitebench \
                    --data_dir benchmark/infinitebench/data \
                    --env_conf $config_file \
                    $sample_args
            done
        else
            for budget in "${max_capacity_prompts[@]}"
            do
                for task in "${tasks[@]}"
                do
                    echo "========================================"
                    echo "Running model=$model method=$method budget=$budget task=$task"
                    echo "Config: $config_file"
                    echo "========================================"
                    python -m benchmark.infinitebench.llama_parallel \
                        --task $task \
                        --output_dir outputs/infinitebench \
                        --data_dir benchmark/infinitebench/data \
                        --env_conf $config_file \
                        --max_capacity_prompt $budget \
                        $sample_args
                done
            done
        fi
    done
done
