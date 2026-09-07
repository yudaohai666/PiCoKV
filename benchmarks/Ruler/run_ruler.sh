#!/bin/bash
# RULER launcher for quantpc / kivi / oscar / polar / dense baselines.
# Results are auto-written by run_ruler_evaluation.py to results/ruler/{MODEL}/{TAG}/
# Select methods via the METHODS list below (METHOD=oscar forces a single one).
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
# Fragmentation, not live memory: the block-32 online solve churns a 126 MiB
# fp32 copy of its +-1 code matrix per group across two CUDA streams, and the
# default allocator opens fresh segments for that size class instead of reusing
# them. Measured on Qwen3-8B at 131k: nvidia-smi peak 35 -> 32 GB with
# max_memory_allocated unchanged. Must precede torch's allocator init, so it
# cannot move into python. Keep identical across all benchmark launchers.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${REPO_ROOT}"

GPU="${GPU:-3}"
MODEL="${MODEL:-/data/Llama-3.1-8B-Instruct}"
MODEL_TAG="$(basename "${MODEL%/}")"

# ── benchmark runtime params ──
MAX_SEQ_LENGTHS="${MAX_SEQ_LENGTHS:-8192,16384,32768,65536,131072}"
MAX_LENGTH="${MAX_LENGTH:-131072}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-128}"
LIMIT="${LIMIT:-100}"
TASKS="${TASKS:-niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multiquery,niah_multivalue,ruler_vt,ruler_cwe,ruler_fwe,ruler_qa_squad,ruler_qa_hotpot}"
APPLY_CHAT_TEMPLATE="${APPLY_CHAT_TEMPLATE:-1}"
# Override the python runner's auto results dir (results/ruler/{MODEL}/{TAG}/).
# Needed when the same method is split across processes (e.g. one GPU per
# sequence length): without it every process writes the same results.json.
OUTPUT_PATH="${OUTPUT_PATH:-}"

# ── methods to run (comment out to skip; METHOD=oscar forces a single one) ──
METHODS=(
  quantpc
  # kivi
  # oscar
  # polar
  # flash_attention_2
  # sdpa
  # eager
)
[ -n "${METHOD:-}" ] && METHODS=("${METHOD}")

# ── per-method defaults (identical across all benchmark launchers) ──
# quantpc; default = online
K_BITS="${K_BITS:-320}"; V_BITS="${V_BITS:-320}"
SINK="${SINK:-64}"; LOCAL="${LOCAL:-256}"
EXP_BITS="${EXP_BITS:-4}"; EAS="${EAS:-0.01}"; ZEROROPE="${ZEROROPE:-20}"
USE_OFFLINE="${USE_OFFLINE:-0}"
OFFLINE_PROJ_PATH="${OFFLINE_PROJ_PATH:-}"
# kivi (2/4-bit KV)
KIVI_K="${KIVI_K:-2}"; KIVI_V="${KIVI_V:-2}"; KIVI_GROUP="${KIVI_GROUP:-32}"; KIVI_RESIDUAL="${KIVI_RESIDUAL:-128}"
# oscar INT2 recipe (group MUST be 32; int2 V-param packer hardcodes 32-wide groups)
OSCAR_NUM_BITS="${OSCAR_NUM_BITS:-2}"; OSCAR_QUANT_MODE="${OSCAR_QUANT_MODE:-k-channel}"; OSCAR_GROUP="${OSCAR_GROUP:-32}"
OSCAR_KV_ROTATION="${OSCAR_KV_ROTATION:-hadamard}"; OSCAR_KV_NORM="${OSCAR_KV_NORM:-1}"; OSCAR_EVICT="${OSCAR_EVICT:-256}"; OSCAR_OFFLINE_V_HAD="${OSCAR_OFFLINE_V_HAD:-1}"
# polar (src/PolarQuant)
POLAR_MODE="${POLAR_MODE:-mixed}"; POLAR_RBITS="${POLAR_RBITS:-3}"; POLAR_TBITS="${POLAR_TBITS:-3}"; POLAR_VBITS="${POLAR_VBITS:-2}"

# Build METHOD_ARGS[] and TAG for a method. Needs REPO_ROOT, MODEL_TAG + the
# per-method defaults above. quantpc TAG matches run_ruler_evaluation.py's python
# run_tag(): quantpc_{mode}_k{K}_v{V} (informational only here -- the python
# runner auto-writes the results dir).
build_method_args() {
  METHOD_ARGS=(); TAG="$1"
  case "$1" in
    quantpc)
      METHOD_ARGS=(--k-bits "${K_BITS}" --v-bits "${V_BITS}" --sink-size "${SINK}" --local-size "${LOCAL}"
        --exp-bits "${EXP_BITS}" --eas "${EAS}" --zerorope "${ZEROROPE}")
      local mode proj
      if [ "${USE_OFFLINE}" = "1" ] || [ -n "${OFFLINE_PROJ_PATH}" ]; then
        mode="offline"; METHOD_ARGS+=(--offline); proj="${OFFLINE_PROJ_PATH}"
        [ -n "${proj}" ] && METHOD_ARGS+=(--offline-proj-path "${proj}")
      else
        mode="online"
      fi
      TAG="quantpc_${mode}_k${K_BITS}_v${V_BITS}"
      ;;
    kivi)
      METHOD_ARGS=(--k-bits "${KIVI_K}" --v-bits "${KIVI_V}" --group-size "${KIVI_GROUP}" --residual-length "${KIVI_RESIDUAL}")
      TAG="kivi_k${KIVI_K}_v${KIVI_V}"
      ;;
    oscar)
      METHOD_ARGS=(--num-bits "${OSCAR_NUM_BITS}" --quant-mode "${OSCAR_QUANT_MODE}" --group-size "${OSCAR_GROUP}")
      [ "${OSCAR_KV_ROTATION}" != "none" ] && METHOD_ARGS+=(--kv-rotation "${OSCAR_KV_ROTATION}")
      [ "${OSCAR_KV_NORM}" != "0" ]       && METHOD_ARGS+=(--kv-norm "${OSCAR_KV_NORM}")
      [ -n "${OSCAR_EVICT}" ]             && METHOD_ARGS+=(--residual-evict-size "${OSCAR_EVICT}")
      [ "${OSCAR_OFFLINE_V_HAD}" = "1" ]  && METHOD_ARGS+=(--offline-v-hadamard)
      TAG="oscar_int${OSCAR_NUM_BITS}"
      ;;
    polar)
      METHOD_ARGS=(--polar-mode "${POLAR_MODE}" --polar-rbits "${POLAR_RBITS}" --polar-tbits "${POLAR_TBITS}" --polar-vbits "${POLAR_VBITS}")
      TAG="polar_${POLAR_MODE}_${POLAR_RBITS}${POLAR_TBITS}${POLAR_VBITS}"
      ;;
  esac
}

CHAT_ARGS=()
if [ "${APPLY_CHAT_TEMPLATE}" = "1" ]; then CHAT_ARGS+=(--apply_chat_template); else CHAT_ARGS+=(--no-apply_chat_template); fi

OUT_ARGS=()
[ -n "${OUTPUT_PATH}" ] && OUT_ARGS+=(--output_path "${OUTPUT_PATH}")

for method in "${METHODS[@]}"; do
  build_method_args "${method}"
  echo "=== RULER: method=${method} (${TAG}) ==="
  CUDA_VISIBLE_DEVICES="${GPU}" python "${SCRIPT_DIR}/run_ruler_evaluation.py" \
    --model "${MODEL}" --method "${method}" \
    --tasks "${TASKS}" --max_seq_lengths "${MAX_SEQ_LENGTHS}" \
    --max_length "${MAX_LENGTH}" --batch_size "${BATCH_SIZE}" \
    --max_gen_toks "${MAX_GEN_TOKS}" --limit "${LIMIT}" \
    "${CHAT_ARGS[@]}" "${OUT_ARGS[@]}" "${METHOD_ARGS[@]}"
done

echo "All done."
