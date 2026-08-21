#!/bin/bash
# ============================== STAGE ==============================
# STAGE=collect : only run calibration and save per-layer KV/query-gram shards.
# STAGE=fit     : reuse existing shards and only fit/save projection matrices.
# STAGE=all     : run both stages above; this is the default and corresponds to
#                 "both" collect and fit stages.
#
# The usual workflow is STAGE=all for a fresh model, then STAGE=fit when the
# same collected shards are reused to build another WHITEN/K_BITS/V_BITS pair.
# =====================================================================
#
# ============================ EXAMPLE: QWEN3 ============================
# Qwen3-8B, build a non-whitened projection.
#   cd /home/zxh/zxh/QuantPC &&
#   . .venv/bin/activate &&
#   MODEL_NAME=/home/zxh/zxh/models/Qwen3-8B \
#   GPUS=0,1 \
#   STAGE=all \
#   WHITEN=false \
#   COLLECT_LAYERS_PER_FORWARD=36 \
#   bash offline_projection/create_offline_projections.sh
# ========================================================================
#
# ============================ EXAMPLE: LLAMA ============================
# Llama-3.1-8B-Instruct, build a non-whitened projection.
#   cd /home/zxh/zxh/QuantPC &&
#   . .venv/bin/activate &&
#   MODEL_NAME=/data/zxh/models/Llama-3.1-8B-Instruct \
#   GPUS=0,1 \
#   STAGE=all \
#   WHITEN=false \
#   bash offline_projection/create_offline_projections.sh
# ========================================================================
#
# The example above runs both collect and fit for the non-whitened
# variant. To build the corresponding whitened variant from the same shards,
# keep TMP_STATE_DIR unchanged and run the script again with STAGE=fit and
# WHITEN=true.
#
# WHITEN=true  computes whitening bases from the calibration data, learns the
#              encoder/decoder in that basis, and folds them into the saved
#              projection pair. The output ends with `_whiten.pt`.
# WHITEN=false (or omitting WHITEN) uses the original model basis directly and
#              saves a `_nowhiten.pt` projection. The same collect shards can
#              be reused to build both variants; only the fit stage differs.
# Both outputs are folded offline projections. At evaluation time, pass the
# projection with --offline-proj-path and do not also pass --whiten.
#
# Build all three offline matrices in one pass: whitening, encoder, decoder.
# Output: {MODEL_TAG}_k{K}_v{V}_{whiten|nowhiten}.pt under OUTPUT_DIR, in format
# quanpc_standalone_block32_folded_v1 -- encoders and decoders with whitening,
# when requested, already folded in (E W and D W^-1).
#
# Consume it with:
#   benchmarks/Needle/run_bin.py --offline-proj-path <file>
# and *without* --whiten (the loader guard rejects that combination).
#
# Whitening ridge is fixed internally at 1e-3 for comparable artifacts.
set -e
cd "$(dirname "$0")"

export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

MODEL_NAME=${MODEL_NAME:-/data/Llama-3.1-8B-Instruct}
DATASET_DIR=${DATASET_DIR:-data}
OUTPUT_DIR=${OUTPUT_DIR:-projections}
TMP_STATE_DIR=${TMP_STATE_DIR:-offline_states.tmp}

DATASETS=${DATASETS:-codeparrot,pg19,proof-pile}
LENGTH_CONFIG=${LENGTH_CONFIG:-131072:10,98304:10,65536:10,32768:10}
K_BITS=${K_BITS:-320}
V_BITS=${V_BITS:-320}
ZEROROPE=${ZEROROPE:-20}
SINK_SIZE=${SINK_SIZE:-64}
LOCAL_SIZE=${LOCAL_SIZE:-256}
DECODER_ITERS=${DECODER_ITERS:-5}
DECODER_RIDGE=${DECODER_RIDGE:-1e-3}
WHITEN=${WHITEN:-false}
# Block-32 learning subsamples to BLOCK32_LEARN_TOKENS anyway, so striding at
# collect time costs no fit quality and cuts shard bytes proportionally.
COLLECT_STRIDE=${COLLECT_STRIDE:-8}
BLOCK32_LEARN_TOKENS=${BLOCK32_LEARN_TOKENS:-1048576}
# All 32 layers in one forward: two half-passes cost 1.5 forwards per sample.
COLLECT_LAYERS_PER_FORWARD=${COLLECT_LAYERS_PER_FORWARD:-32}

STAGE=${STAGE:-all}
GPUS=${GPUS:-0,1,2,3}
MODEL_TAG=$(basename "${MODEL_NAME}")
if [ "${WHITEN}" = "true" ] || [ "${WHITEN}" = "1" ]; then
  WHITEN_TAG=whiten
else
  WHITEN_TAG=nowhiten
fi
OUTPUT_PATH=${OUTPUT_PATH:-${OUTPUT_DIR}/${MODEL_TAG}_k${K_BITS}_v${V_BITS}_${WHITEN_TAG}.pt}

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
    --decoder_ridge "${DECODER_RIDGE}" \
    --whiten "${WHITEN}" \
    --collect_stride "${COLLECT_STRIDE}" \
    --collect_layers_per_forward "${COLLECT_LAYERS_PER_FORWARD}" \
    --block32-learn-tokens "${BLOCK32_LEARN_TOKENS}" \
    --verbose_fit --keep_tmp \
    "$@"
