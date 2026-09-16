#!/bin/bash

# RULER Tasks Evaluation Script
# Uses mysparse.lmeval.MyModel for evaluation

export CUDA_VISIBLE_DEVICES=6
source /root/miniconda3/etc/profile.d/conda.sh
conda activate bpc
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

models=(
    llama3-1-8b-ins
    # qwen2.5-7b-ins
)

max_capacity_prompts=(
    # 1024
    2048
    # 4096
)

methods=(
    # "origin"
    # "compresskv"
    # "cakekv"
    # "pyramidkv"
    # "quest"
    # "magicpig"
    "bpc"
    # "bpc-offline"
)

# RULER specific parameters
# MAX_SEQ_LENGTHS="4096,8192,16384,32768,65536,131072"  # Comma-separated: "4096,8192,16384,32768"
MAX_SEQ_LENGTHS="131072"
MAX_LENGTH=131072
BATCH_SIZE=1
MAX_GEN_TOKS=128
LIMIT=100
# All RULER tasks - use "ruler" to run all tasks at once
TASKS="ruler"
# Or specify individual tasks:
# TASKS="niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multiquery,niah_multivalue,ruler_vt,ruler_cwe,ruler_fwe,ruler_qa_squad,ruler_qa_hotpot"

for model in "${models[@]}"
do
    for method in "${methods[@]}"
    do
        config_file="config/${model}/${model}-${method}.json"
        
        # Check if config file exists
        if [ ! -f "$config_file" ]; then
            echo "Warning: Config file $config_file not found, skipping..."
            continue
        fi
        
        # Methods that don't need budget parameter
        if [[ "$method" == "origin" || "$method" == "bpc" || "$method" == "bpc-offline" || "$method" == "magicpig" ]]; then
            echo "========================================"
            echo "Running model=$model method=$method"
            echo "Config: $config_file"
            echo "========================================"
            python benchmark/ruler_task/run_ruler_evaluation.py \
                --env_conf $config_file \
                --tasks $TASKS \
                --max_seq_lengths $MAX_SEQ_LENGTHS \
                --max_length $MAX_LENGTH \
                --batch_size $BATCH_SIZE \
                --max_gen_toks $MAX_GEN_TOKS \
                --limit $LIMIT \
                --apply_chat_template
        else
            # Methods that need budget parameter (KV cache compression methods)
            for budget in "${max_capacity_prompts[@]}"
            do
                echo "========================================"
                echo "Running model=$model method=$method budget=$budget"
                echo "Config: $config_file"
                echo "========================================"
                python benchmark/ruler_task/run_ruler_evaluation.py \
                    --env_conf $config_file \
                    --tasks $TASKS \
                    --max_seq_lengths $MAX_SEQ_LENGTHS \
                    --max_length $MAX_LENGTH \
                    --batch_size $BATCH_SIZE \
                    --max_gen_toks $MAX_GEN_TOKS \
                    --limit $LIMIT \
                    --max_capacity_prompt $budget \
                    --apply_chat_template
            done
        fi
    done
done
