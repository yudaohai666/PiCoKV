#!/bin/bash

# General Tasks (GLUE, SuperGLUE, etc.) Evaluation Script
# Uses mysparse.lmeval.MyModel for evaluation

export CUDA_VISIBLE_DEVICES=5
source /root/miniconda3/etc/profile.d/conda.sh
conda activate bpc
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."


models=(
    # llama3-8b
    llama3-1-8b-ins
    # mistral-7b-ins
)

max_capacity_prompts=(
    # 1024
    # 2048
    128
)

methods=(
    # "origin"
    "bpc"
    # "compresskv"
    # "cakekv"
    # "snapkv"
    # "pyramidkv"
    # "streamingllm"
)

# Evaluation parameters
MODEL_MAX_LENGTH=4096
BATCH_SIZE=1
MAX_GEN_TOKS=128
LIMIT=1000
# FEWSHOT removed - use each task's default num_fewshot:
# gsm8k_cot: 8-shot, mmlu_flan_cot_fewshot: 4-shot, coqa: from train split

for model in "${models[@]}"
do
    for method in "${methods[@]}"
    do
        config_file="config/${model}/${model}-${method}.json"
        
        # Log directory with model and method
        LOG_DIR="logs/general_tasks/${model}/${method}"
        mkdir -p $LOG_DIR
        
        # Check if config file exists
        if [ ! -f "$config_file" ]; then
            echo "Warning: Config file $config_file not found, skipping..."
            continue
        fi
        
        # Methods that don't need budget parameter
        if [[ "$method" == "origin" || "$method" == "bpc" || "$method" == "spotlight" ]]; then
            LOG_FILE="${LOG_DIR}/eval.log"
            echo "========================================"
            echo "Running model=$model method=$method"
            echo "Config: $config_file"
            echo "Log: $LOG_FILE"
            echo "========================================"
            python benchmark/general_tasks/lmeval.py \
                --env_conf $config_file \
                --model_max_length $MODEL_MAX_LENGTH \
                --batch_size $BATCH_SIZE \
                --max_gen_toks $MAX_GEN_TOKS \
                --limit $LIMIT \
                2>&1 | tee "${LOG_FILE}"
        else
            # Methods that need budget parameter (KV cache compression methods)
            for budget in "${max_capacity_prompts[@]}"
            do
                LOG_FILE="${LOG_DIR}/${budget}.log"
                echo "========================================"
                echo "Running model=$model method=$method budget=$budget"
                echo "Config: $config_file"
                echo "Log: $LOG_FILE"
                echo "========================================"
                python benchmark/general_tasks/lmeval.py \
                    --env_conf $config_file \
                    --model_max_length $MODEL_MAX_LENGTH \
                    --batch_size $BATCH_SIZE \
                    --max_gen_toks $MAX_GEN_TOKS \
                    --limit $LIMIT \
                    --max_capacity_prompt $budget \
                    2>&1 | tee "${LOG_FILE}"
            done
        fi
    done
done
