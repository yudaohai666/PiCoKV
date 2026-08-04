#!/bin/bash
# Build all three offline matrices in one pass: whitening, encoder, decoder.
# Output: {MODEL_TAG}_k{K}_v{V}_block32_folded.pt under OUTPUT_DIR, in format
# quanpc_standalone_block32_folded_v1 -- encoders and decoders with the whitening
# already folded in (E W and D W^-1), so prefill never rotates K, decode never
# rotates Q, and key_projection_t is W^-1 D^T.
#
# Consume it with:
#   run_standalone_niah.py --offline-proj-path <file> --offline-decoder
# and *without* --whiten (the loader guard rejects that combination).
#
# Override any tunable via env, e.g. WHITEN_RIDGE=1e-2 bash create_offline_projections.sh
set -e
cd "$(dirname "$0")"

export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

MODEL_NAME=${MODEL_NAME:-/data/zxh/models/Llama-3.1-8B-Instruct}
DATASET_DIR=${DATASET_DIR:-data}
OUTPUT_DIR=${OUTPUT_DIR:-projections}
TMP_STATE_DIR=${TMP_STATE_DIR:-offline_states.tmp}

DATASETS=${DATASETS:-codeparrot,pg19,proof-pile}
LENGTH_CONFIG=${LENGTH_CONFIG:-131072:10,98304:10,65536:10,32768:10}
K_BITS=${K_BITS:-320}
V_BITS=${V_BITS:-256}
ZEROROPE=${ZEROROPE:-20}
SINK_SIZE=${SINK_SIZE:-64}
LOCAL_SIZE=${LOCAL_SIZE:-1024}
DECODER_ITERS=${DECODER_ITERS:-5}
# Ridge on the normalized query Gram before its square root.  The online
# find_orth_proj gets this by accident (float32 Gram, unresolvable small
# eigenvalues); at float64 it has to be asked for.  First knob to sweep if a
# whitened build loses accuracy -- re-fitting needs only STAGE=fit when the
# shards were kept.
WHITEN_RIDGE=${WHITEN_RIDGE:-1e-3}
WHITEN_LAYERS=${WHITEN_LAYERS:-all}
# Block-32 learning subsamples to BLOCK32_LEARN_TOKENS anyway, so striding at
# collect time costs no fit quality and cuts shard bytes proportionally.
COLLECT_STRIDE=${COLLECT_STRIDE:-8}
BLOCK32_LEARN_TOKENS=${BLOCK32_LEARN_TOKENS:-1048576}
# All 32 layers in one forward: two half-passes cost 1.5 forwards per sample.
COLLECT_LAYERS_PER_FORWARD=${COLLECT_LAYERS_PER_FORWARD:-32}

STAGE=${STAGE:-all}
GPUS=${GPUS:-0,1,2,3}
MODEL_TAG=$(basename "${MODEL_NAME}")
OUTPUT_PATH=${OUTPUT_PATH:-${OUTPUT_DIR}/${MODEL_TAG}_k${K_BITS}_v${V_BITS}_block32_folded.pt}

CUDA_VISIBLE_DEVICES=${GPUS} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python create_offline_projections.py \
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
    --decoder_iters "${DECODER_ITERS}" \
    --whiten_ridge "${WHITEN_RIDGE}" \
    --whiten_layers "${WHITEN_LAYERS}" \
    --collect_stride "${COLLECT_STRIDE}" \
    --collect_layers_per_forward "${COLLECT_LAYERS_PER_FORWARD}" \
    --block32-learn-tokens "${BLOCK32_LEARN_TOKENS}" \
    --verbose_fit --keep_tmp \
    "$@"
