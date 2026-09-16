#!/bin/bash
# BPC Multi-GPU Efficiency Benchmark
# Usage:
#   bash benchmark/test_efficiency/run_efficiency_e2e.sh [origin|bpc-offline|all]
#   bash benchmark/test_efficiency/run_efficiency_e2e.sh origin --dynamic
#
# Multiple BATCH_SIZE and PREFIX_LEN values run all combinations.

set -e
cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MODEL="meta-llama/Llama-3.1-8B-Instruct"
PREFIX_LENS=(1048576)
GEN_LEN=64
BATCH_SIZES=(1)
WARM_UP=32
CHUNK_SIZE=16384

MASK_OUT=0.98
MIN_REMAIN=128

# Multi-GPU strategy: auto, balanced, balanced_low_0, sequential
DEVICE_MAP_STRATEGY="balanced"

# Per-GPU memory cap (e.g. "80GiB"); empty = unlimited
MAX_MEMORY_PER_GPU="80GiB"

OUTPUT_DIR="outputs/efficiency_e2e_B1"
mkdir -p $OUTPUT_DIR

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

USE_DYNAMIC_CACHE=""
CACHE_SUFFIX=""

for arg in "$@"; do
    if [ "$arg" == "--dynamic" ]; then
        USE_DYNAMIC_CACHE="--use_dynamic_cache"
        CACHE_SUFFIX="_dynamic"
    fi
done

if [ -n "$1" ] && [ "$1" != "--dynamic" ]; then
    if [ "$1" == "all" ]; then
        METHODS=("origin" "bpc-offline")
    else
        METHODS=("$1")
    fi
else
    METHODS=("origin" "bpc-offline")
fi

echo "=============================================="
echo "Model: $MODEL | GPUs: $NUM_GPUS"
echo "BATCH_SIZES: ${BATCH_SIZES[@]}"
echo "PREFIX_LENS: ${PREFIX_LENS[@]}"
echo "G=$GEN_LEN, chunk=$CHUNK_SIZE"
echo "Device Map Strategy: $DEVICE_MAP_STRATEGY"
echo "Methods: ${METHODS[@]}"
echo "=============================================="

SUMMARY_FILE="${OUTPUT_DIR}/summary_${NUM_GPUS}gpu${CACHE_SUFFIX}_${TIMESTAMP}.json"
echo "[" > "$SUMMARY_FILE"
FIRST_ENTRY=1

for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    for PREFIX_LEN in "${PREFIX_LENS[@]}"; do
        for METHOD in "${METHODS[@]}"; do
            echo ""
            echo ">>> Testing: $METHOD | B=$BATCH_SIZE | P=$PREFIX_LEN"
            
            OUTPUT_FILE="${OUTPUT_DIR}/${METHOD}_B${BATCH_SIZE}_P${PREFIX_LEN}_${NUM_GPUS}gpu${CACHE_SUFFIX}_${TIMESTAMP}.json"
            
            EXTRA_ARGS=""
            if [ "$METHOD" == "bpc" ] || [ "$METHOD" == "bpc-offline" ]; then
                EXTRA_ARGS="--mask_out $MASK_OUT --min_remain $MIN_REMAIN --fix_layers 0,1"
            fi
            
            MAX_MEM_ARG=""
            if [ -n "$MAX_MEMORY_PER_GPU" ]; then
                MAX_MEM_ARG="--max_memory_per_gpu $MAX_MEMORY_PER_GPU"
            fi
            
            python benchmark/test_efficiency/efficiency_e2e.py \
                --model $MODEL \
                --method $METHOD \
                --B $BATCH_SIZE \
                --P $PREFIX_LEN \
                --G $GEN_LEN \
                --chunk_size $CHUNK_SIZE \
                --warm_up $WARM_UP \
                --device_map_strategy $DEVICE_MAP_STRATEGY \
                $MAX_MEM_ARG \
                $EXTRA_ARGS \
                $USE_DYNAMIC_CACHE \
                --output_file $OUTPUT_FILE
            
            echo "Saved: $OUTPUT_FILE"

            # Append key fields to summary file
            if [ "$FIRST_ENTRY" == "1" ]; then
                FIRST_ENTRY=0
            else
                echo "," >> "$SUMMARY_FILE"
            fi
            python -c "
import json
with open('$OUTPUT_FILE') as f:
    d = json.load(f)
print(json.dumps({
    'method': d['method'],
    'batch_size': d['batch_size'],
    'prefix_len': d['prefix_len'],
    'decode_latency_ms': d['decode_latency_ms'],
    'decode_throughput': d['decode_throughput']
}, indent=2))
" >> "$SUMMARY_FILE"
        done
    done
done

echo "" >> "$SUMMARY_FILE"
echo "]" >> "$SUMMARY_FILE"
echo "Summary saved: $SUMMARY_FILE"

echo ""
echo "All benchmarks completed!"
