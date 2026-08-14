#!/bin/bash
# Compare prefill (Enc) / decode (Dec) speed across KV-cache methods.
# Runs benchmark_throughput.py for each method under the same config and
# saves per-run logs, then prints the "Avg Enc/Dec Time" lines side by side.

set -euo pipefail
cd "$(dirname "$0")"

# ---- config (override via env, e.g. GPU=1 BATCH_SIZE=8 bash run_benchmark.sh) ----
GPU=${GPU:-3}
MODEL_PATH=${MODEL_PATH:-/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct}
BATCH_SIZE=${BATCH_SIZE:-6}
INPUT_LENGTH=${INPUT_LENGTH:-32000}
OUTPUT_LENGTH=${OUTPUT_LENGTH:-16}
METHODS=${METHODS:-"polar exp"}
# polar bit-width: 4/4 -> "44", 3/3 -> "33" (only affects method=polar)
RBITS=${RBITS:-3}
TBITS=${TBITS:-3}

LOG_DIR=${LOG_DIR:-logs}
mkdir -p "$LOG_DIR"
TAG="bs${BATCH_SIZE}_in${INPUT_LENGTH}_out${OUTPUT_LENGTH}"

echo "=============================================="
echo "model      : $MODEL_PATH"
echo "gpu        : $GPU"
echo "batch_size : $BATCH_SIZE"
echo "input_len  : $INPUT_LENGTH"
echo "output_len : $OUTPUT_LENGTH"
echo "methods    : $METHODS"
echo "polar bits : rbits=${RBITS}, tbits=${TBITS}"
echo "=============================================="

for method in $METHODS; do
    # include polar bit-width in the log name so 33 vs 44 runs don't overwrite each other
    if [ "$method" = "polar" ]; then
        log="$LOG_DIR/${method}${RBITS}${TBITS}_${TAG}.log"
    else
        log="$LOG_DIR/${method}_${TAG}.log"
    fi
    echo ""
    echo ">>> running method=$method  (log: $log)"
    CUDA_VISIBLE_DEVICES=$GPU python benchmark_throughput.py \
        --method "$method" \
        --model_path "$MODEL_PATH" \
        --batch_size "$BATCH_SIZE" \
        --input_length "$INPUT_LENGTH" \
        --output_length "$OUTPUT_LENGTH" \
        --rbits "$RBITS" \
        --tbits "$TBITS" \
        2>&1 | tee "$log"
done

echo ""
echo "==================== SUMMARY ===================="
printf "%-10s | %-16s | %-16s\n" "method" "Avg Enc Time" "Avg Dec Time"
echo "---------------------------------------------------"
for method in $METHODS; do
    if [ "$method" = "polar" ]; then
        log="$LOG_DIR/${method}${RBITS}${TBITS}_${TAG}.log"
        name="${method}${RBITS}${TBITS}"
    else
        log="$LOG_DIR/${method}_${TAG}.log"
        name="$method"
    fi
    enc=$(grep "Avg Enc Time" "$log" | tail -n 1 | awk -F': ' '{print $2}')
    dec=$(grep "Avg Dec Time" "$log" | tail -n 1 | awk -F': ' '{print $2}')
    printf "%-10s | %-16s | %-16s\n" "$name" "${enc:-N/A}" "${dec:-N/A}"
done
echo "==================================================="
