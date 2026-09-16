#!/bin/bash

# Run calibrate_projection.py to compute the projection matrix.
# Multi-source mixed sampling. ~210 NUM_SAMPLES finishes in under 2 hours.
export CUDA_VISIBLE_DEVICES=7

source /root/miniconda3/etc/profile.d/conda.sh
conda activate bpc
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ============= Config =============
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"

OUTPUT_PATH="./projections/llama3.1-8b-ins-mixlen-mixdata-projection.pt"

MIX_DATA_DIR="./mix_data"

# Comma-separated data sources; each contributes NUM_SAMPLES_PER_SOURCE samples.
DATA_SOURCES="${MIX_DATA_DIR}/codeparrot.json,${MIX_DATA_DIR}/pg19.json,${MIX_DATA_DIR}/proof-pile.json"

# Fallback used when a source lacks enough samples.
FALLBACK_DATA="${MIX_DATA_DIR}/arxiv.json"

NUM_SAMPLES_PER_SOURCE=60

# 3 sources x 60 = 180.
NUM_SAMPLES=180

MAX_LENGTH=131072

# Per-source length allocation (auto-derived from MAX_LENGTH):
# - MAX_LENGTH: NUM_SAMPLES_PER_SOURCE/2 samples
# - MAX_LENGTH/{2,4,8,16,32}: NUM_SAMPLES_PER_SOURCE/2/5 samples each
HALF_SAMPLES=$((NUM_SAMPLES_PER_SOURCE / 2))
EACH_SHORT_SAMPLES=$((HALF_SAMPLES / 5))

L1=${MAX_LENGTH}
L2=$((MAX_LENGTH / 2))
L3=$((MAX_LENGTH / 4))
L4=$((MAX_LENGTH / 8))
L5=$((MAX_LENGTH / 16))
L6=$((MAX_LENGTH / 32))

# Per-source "length:count" config.
LENGTH_CONFIG="${L1}:${HALF_SAMPLES},${L2}:${EACH_SHORT_SAMPLES},${L3}:${EACH_SHORT_SAMPLES},${L4}:${EACH_SHORT_SAMPLES},${L5}:${EACH_SHORT_SAMPLES},${L6}:${EACH_SHORT_SAMPLES}"

NUM_BITS=64

NUM_ITERS=8

# fp16 | bf16 | fp32
DTYPE="bf16"

# Forward chunk size to avoid OOM.
CHUNK_SIZE=16384

# ============= Run =============
echo "=========================================="
echo "Calibrating Projection Matrices (Multi-Source)"
echo "=========================================="
echo "Model: ${MODEL_NAME}"
echo "Output: ${OUTPUT_PATH}"
echo "Data Sources: ${DATA_SOURCES}"
echo "Fallback Data: ${FALLBACK_DATA}"
echo "Samples per Source: ${NUM_SAMPLES_PER_SOURCE}"
echo "Total Samples: ${NUM_SAMPLES}"
echo "Max Length: ${MAX_LENGTH}"
echo "Length Config (per source): ${LENGTH_CONFIG}"
echo "  - ${L1}: ${HALF_SAMPLES} samples"
echo "  - ${L2}: ${EACH_SHORT_SAMPLES} samples"
echo "  - ${L3}: ${EACH_SHORT_SAMPLES} samples"
echo "  - ${L4}: ${EACH_SHORT_SAMPLES} samples"
echo "  - ${L5}: ${EACH_SHORT_SAMPLES} samples"
echo "  - ${L6}: ${EACH_SHORT_SAMPLES} samples"
echo "Num Bits: ${NUM_BITS}"
echo "Num Iters: ${NUM_ITERS}"
echo "Dtype: ${DTYPE}"
echo "Chunk Size: ${CHUNK_SIZE}"
echo "=========================================="

CMD="python calibrate_projection.py \
    --model_name ${MODEL_NAME} \
    --output_path ${OUTPUT_PATH} \
    --data_sources ${DATA_SOURCES} \
    --fallback_data ${FALLBACK_DATA} \
    --num_samples_per_source ${NUM_SAMPLES_PER_SOURCE} \
    --max_length ${MAX_LENGTH} \
    --length_config ${LENGTH_CONFIG} \
    --num_bits ${NUM_BITS} \
    --num_iters ${NUM_ITERS} \
    --dtype ${DTYPE} \
    --chunk_size ${CHUNK_SIZE}"

echo "Running: ${CMD}"
echo ""
${CMD}

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
