#!/bin/bash
# BPC Profiler Benchmark
# Usage:
#   bash benchmark/test_efficiency/run_profile.sh [origin|bpc-offline|all]
#   bash benchmark/test_efficiency/run_profile.sh origin --dynamic
#   bash benchmark/test_efficiency/run_profile.sh all --lengths "131072 262144"

set -e
cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MODEL="meta-llama/Llama-3.1-8B-Instruct"
# Multiple lengths, space-separated
PREFIX_LENS=(8192 32768 131072 524288)
GEN_LEN=64
WARM_UP=32
CHUNK_SIZE=16384
BATCH_SIZE=1

MASK_OUT=0.98
MIN_REMAIN=128

# Multi-GPU strategy: auto, balanced, balanced_low_0, sequential
DEVICE_MAP_STRATEGY="balanced"

OUTPUT_DIR="outputs/traces"
mkdir -p $OUTPUT_DIR

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

USE_DYNAMIC_CACHE=""
CACHE_SUFFIX=""
METHODS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --dynamic)
            USE_DYNAMIC_CACHE="--use_dynamic_cache"
            CACHE_SUFFIX="_dynamic"
            shift
            ;;
        --lengths)
            # Parse length list, e.g. --lengths "131072 262144"
            IFS=' ' read -r -a PREFIX_LENS <<< "$2"
            shift 2
            ;;
        origin|baseline|bpc|bpc-offline|all)
            if [ "$1" == "all" ]; then
                METHODS=("origin" "bpc-offline")
            else
                METHODS+=("$1")
            fi
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Default methods
if [ ${#METHODS[@]} -eq 0 ]; then
    METHODS=("origin" "bpc-offline")
fi

echo "=============================================="
echo "Model: $MODEL | GPUs: $NUM_GPUS"
echo "PREFIX_LENS: ${PREFIX_LENS[@]}"
echo "G=$GEN_LEN, warm_up=$WARM_UP"
echo "Device Map Strategy: $DEVICE_MAP_STRATEGY"
echo "Methods: ${METHODS[@]}"
echo "=============================================="

for PREFIX_LEN in "${PREFIX_LENS[@]}"; do
    echo ""
    echo ">>>>>>>>>> Context Length: P=$PREFIX_LEN <<<<<<<<<<"
    
    for METHOD in "${METHODS[@]}"; do
        echo ""
        echo ">>> Profiling: $METHOD (P=$PREFIX_LEN)"
        
        # Use "baseline" filename for both origin and baseline
        if [ "$METHOD" == "origin" ] || [ "$METHOD" == "baseline" ]; then
            TRACE_FILE="${OUTPUT_DIR}/baseline_P${PREFIX_LEN}_G${GEN_LEN}${CACHE_SUFFIX}_${TIMESTAMP}.json"
            METHOD_ARG="origin"
        else
            TRACE_FILE="${OUTPUT_DIR}/${METHOD}_P${PREFIX_LEN}_G${GEN_LEN}${CACHE_SUFFIX}_${TIMESTAMP}.json"
            METHOD_ARG="$METHOD"
        fi
        
        EXTRA_ARGS=""
        if [ "$METHOD_ARG" == "bpc" ] || [ "$METHOD_ARG" == "bpc-offline" ]; then
            EXTRA_ARGS="--mask_out $MASK_OUT --min_remain $MIN_REMAIN --fix_layers 0,1"
        fi
        
        python benchmark/test_efficiency/run_profile.py \
            --model $MODEL \
            --method $METHOD_ARG \
            --B $BATCH_SIZE \
            --P $PREFIX_LEN \
            --G $GEN_LEN \
            --chunk_size $CHUNK_SIZE \
            --warm_up $WARM_UP \
            --record_shapes \
            --with_stack \
            --device_map_strategy $DEVICE_MAP_STRATEGY \
            $EXTRA_ARGS \
            $USE_DYNAMIC_CACHE \
            --output_trace $TRACE_FILE
        
        echo "Saved: $TRACE_FILE"
    done
done

echo ""
echo "All profiling completed! Traces in: $OUTPUT_DIR"
