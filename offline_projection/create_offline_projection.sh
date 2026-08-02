#!/bin/bash
# Create fixed-width offline QuanPC block-32 projections.
# Runs offline_projection/create_offline_projection.py (stage=both by default).
# Output: k{K}_v{V}_zrope{Z}_block32.pt under OUTPUT_DIR.
# All tunables live here; override any via env var, e.g. K_BITS=256 bash create_offline_projection.sh
set -e
cd "$(dirname "$0")"

export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH}"

# --- paths (relative to this script) ---
MODEL_NAME=${MODEL_NAME:-/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct}
DATASET_DIR=${DATASET_DIR:-data}
OUTPUT_DIR=${OUTPUT_DIR:-projections}
TMP_STATE_DIR=${TMP_STATE_DIR:-offline_states.tmp}

# --- projection config ---
# DATASETS=${DATASETS:-codeparrot,pg19,proof-pile}
DATASETS=${DATASETS:-codeparrot,pg19,proof-pile}
# LENGTH_CONFIG=${LENGTH_CONFIG:-131072:20,65536:20,32768:20,8192:20}
LENGTH_CONFIG=${LENGTH_CONFIG:-131072:4}
SINK_SIZE=${SINK_SIZE:-64}
LOCAL_SIZE=${LOCAL_SIZE:-1024}
K_BITS=${K_BITS:-256}
V_BITS=${V_BITS:-256}
ZEROROPE=${ZEROROPE:-20}
BLOCK32_LEARN_TOKENS=${BLOCK32_LEARN_TOKENS:-1048576}

STAGE=${STAGE:-both}
MODEL_TAG=$(basename "${MODEL_NAME}")
OUTPUT_PATH=${OUTPUT_PATH:-${OUTPUT_DIR}/${MODEL_TAG}_k${K_BITS}_v${V_BITS}_block32.pt}

CUDA_VISIBLE_DEVICES=7 python create_offline_projection.py \
    --stage "${STAGE}" \
    --model_name "${MODEL_NAME}" \
    --dataset_dir "${DATASET_DIR}" \
    --datasets "${DATASETS}" \
    --length_config "${LENGTH_CONFIG}" \
    --tmp_state_dir "${TMP_STATE_DIR}" \
    --output_path "${OUTPUT_PATH}" \
    --sink_size "${SINK_SIZE}" \
    --local_size "${LOCAL_SIZE}" \
    --k_bits "${K_BITS}" \
    --v_bits "${V_BITS}" \
    --zerorope "${ZEROROPE}" \
    --block32-learn-tokens "${BLOCK32_LEARN_TOKENS}" \
    "$@"
