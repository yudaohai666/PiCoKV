#!/bin/bash
export CUDA_VISIBLE_DEVICES=0


# source /root/miniconda3/etc/profile.d/conda.sh
# conda activate bpc
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

eval_paths=(

    # "outputs/longbench/llama-3-8b/spotlight/"
    # "outputs/longbench/llama-3-8b/bpc/"
    # "outputs/longbench/llama-3-8b/origin/"
    # "outputs/longbench/llama-3-8b/magicpig/"
    # "outputs/longbench/llama-3-8b/quest/1024/"
    # "outputs/longbench/llama-3-8b/bpc_offline/"

    # "outputs/longbench/Llama-3.1-8B-Instruct/topk/"




    "outputs/longbench/Llama-3.1-8B-Instruct/bpc/"
    # "outputs/longbench/Llama-3.1-8B-Instruct/bpc_offline/"
    "outputs/longbench/Llama-3.1-8B-Instruct/origin/"
    # "outputs/longbench/Llama-3.1-8B-Instruct/magicpig/"
    # "outputs/longbench/Llama-3.1-8B-Instruct/cakekv/1024/"
    # "outputs/longbench/Llama-3.1-8B-Instruct/compresskv/1024/"  
    # "outputs/longbench/Llama-3.1-8B-Instruct/pyramidkv/1024/"
    # "outputs/longbench/Llama-3.1-8B-Instruct/quest/1024/"       
    # "outputs/longbench/Llama-3.1-8B-Instruct/snapkv/1024/" 
    
    # "outputs/longbench/Mistral-7B-Instruct-v0.3/bpc/"  
    # "outputs/longbench/Mistral-7B-Instruct-v0.3/bpc_offline/"  

    # "outputs/longbench/Mistral-7B-Instruct-v0.3/origin/" 
    # "outputs/longbench/Mistral-7B-Instruct-v0.3/magicpig/"
    # "outputs/longbench/Mistral-7B-Instruct-v0.3/quest/1024/"
    # "outputs/longbench/Mistral-7B-Instruct-v0.3/cakekv/1024/"
    # "outputs/longbench/Mistral-7B-Instruct-v0.3/compresskv/1024/"   
    # "outputs/longbench/Mistral-7B-Instruct-v0.3/snapkv/1024/"
    # "outputs/longbench/Mistral-7B-Instruct-v0.3/pyramidkv/1024/"
    
    # "outputs/longbench/Qwen2.5-7B-Instruct-1M/bpc/"
    # "outputs/longbench/Qwen2.5-7B-Instruct-1M/origin/"
    # "outputs/longbench/Qwen2.5-7B-Instruct-1M/magicpig/"
    # "outputs/longbench/Qwen2.5-7B-Instruct-1M/pyramidkv/1024/"
    # "outputs/longbench/Qwen2.5-7B-Instruct-1M/quest/1024/"
    # "outputs/longbench/Qwen2.5-7B-Instruct-1M/cakekv/1024/"
    # "outputs/longbench/Qwen2.5-7B-Instruct-1M/snapkv/1024/"
    # "outputs/longbench/Qwen2.5-7B-Instruct-1M/bpc_offline/"  
)


for path in "${eval_paths[@]}"; do
    echo "Processing ${path} ..."
    python -m benchmark.longbench.eval \
        --path "${path}" \
        --eval_avg
done
