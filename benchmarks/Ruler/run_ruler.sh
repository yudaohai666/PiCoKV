#!/bin/bash
# RULER evaluation launcher for QuanPC, PolarQuant, KIVI, OScaR and dense baselines.
# Results are auto-written by run_ruler_evaluation.py to:
#   results/ruler/{MODEL}/{TAG}/
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
cd "${REPO_ROOT}"

GPU="${GPU:-3}"
MODEL="${MODEL:-/data/Llama-3.1-8B-Instruct}"

# RULER runtime parameters. Override any of these through the environment.
MAX_SEQ_LENGTHS="${MAX_SEQ_LENGTHS:-32768,65536}"
MAX_LENGTH="${MAX_LENGTH:-131072}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-128}"
LIMIT="${LIMIT:-100}"
TASKS="${TASKS:-niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multiquery,niah_multivalue,ruler_vt,ruler_cwe,ruler_fwe,ruler_qa_squad,ruler_qa_hotpot}"
APPLY_CHAT_TEMPLATE="${APPLY_CHAT_TEMPLATE:-1}"

# Methods: quantpc | polar | kivi | oscar | flash_attention_2 | sdpa | eager.
METHODS=(quantpc)
if [ -n "${METHOD:-}" ]; then METHODS=("${METHOD}"); fi

# quantpc offline projection settings.
USE_OFFLINE="${USE_OFFLINE:-0}"
OFFLINE_PROJ_PATH="${OFFLINE_PROJ_PATH:-}"
WHITEN="${WHITEN:-0}"
ONLINE_DECODER_REFIT="${ONLINE_DECODER_REFIT:-none}"
DECODER_RIDGE="${DECODER_RIDGE:-1e-2}"
OFFLINE_ARGS=()
if [ "${USE_OFFLINE}" = "1" ] || [ -n "${OFFLINE_PROJ_PATH}" ]; then
  OFFLINE_ARGS=(--offline)
  [ -n "${OFFLINE_PROJ_PATH}" ] && OFFLINE_ARGS+=(--offline-proj-path "${OFFLINE_PROJ_PATH}")
fi

# quantpc K/V and decode settings. Multiple entries are useful for ablations;
# run_tag() includes the non-default exp/zr values to avoid overwriting runs.
CONFIGS=(
  "320 320 4 20"
  # "320 320 2 20"
  # "320 320 6 20"
  # "320 320 8 20"
  # "320 320 4 0"
  # "320 320 4 5"
  # "320 320 4 10"
  # "320 320 4 15"
  # "320 320 4 25"
  # "320 320 4 30"
)
EAS="${EAS:-0.01}"

# KIVI parameters.
KIVI_K="${KIVI_K:-2}"
KIVI_V="${KIVI_V:-2}"
KIVI_GROUP="${KIVI_GROUP:-32}"
KIVI_RESIDUAL="${KIVI_RESIDUAL:-128}"

# OScaR INT2 parameters.
OSCAR_NUM_BITS="${OSCAR_NUM_BITS:-2}"
OSCAR_QUANT_MODE="${OSCAR_QUANT_MODE:-k-channel}"
OSCAR_GROUP="${OSCAR_GROUP:-32}"
OSCAR_KV_ROTATION="${OSCAR_KV_ROTATION:-hadamard}"
OSCAR_KV_NORM="${OSCAR_KV_NORM:-1}"
OSCAR_EVICT="${OSCAR_EVICT:-256}"
OSCAR_OFFLINE_V_HAD="${OSCAR_OFFLINE_V_HAD:-1}"

# PolarQuant parameters, matching quantpc_v2.
POLAR_MODE="${POLAR_MODE:-mixed}"
POLAR_RBITS="${POLAR_RBITS:-3}"
POLAR_TBITS="${POLAR_TBITS:-3}"
POLAR_VBITS="${POLAR_VBITS:-2}"

CHAT_ARGS=()
if [ "${APPLY_CHAT_TEMPLATE}" = "1" ]; then
  CHAT_ARGS+=(--apply_chat_template)
else
  CHAT_ARGS+=(--no-apply_chat_template)
fi

run_ruler() {
  local method="$1"
  shift
  CUDA_VISIBLE_DEVICES="${GPU}" python "${SCRIPT_DIR}/run_ruler_evaluation.py" \
    --model "${MODEL}" --method "${method}" \
    --tasks "${TASKS}" --max_seq_lengths "${MAX_SEQ_LENGTHS}" \
    --max_length "${MAX_LENGTH}" --batch_size "${BATCH_SIZE}" \
    --max_gen_toks "${MAX_GEN_TOKS}" --limit "${LIMIT}" \
    "${CHAT_ARGS[@]}" "$@"
}

for method in "${METHODS[@]}"; do
  case "${method}" in
    quantpc)
      for cfg in "${CONFIGS[@]}"; do
        read -r K V EXP ZR <<< "${cfg}"
        QUANT_ARGS=(
          --k-bits "${K}" --v-bits "${V}"
          --exp-bits "${EXP}" --eas "${EAS}" --zerorope "${ZR}"
          --decoder-ridge "${DECODER_RIDGE}"
          --online-decoder-refit "${ONLINE_DECODER_REFIT}"
        )
        [ "${WHITEN}" = "1" ] && QUANT_ARGS+=(--whiten)
        run_ruler quantpc "${QUANT_ARGS[@]}" "${OFFLINE_ARGS[@]}"
      done
      ;;
    polar)
      run_ruler polar \
        --polar-mode "${POLAR_MODE}" --polar-rbits "${POLAR_RBITS}" \
        --polar-tbits "${POLAR_TBITS}" --polar-vbits "${POLAR_VBITS}"
      ;;
    kivi)
      run_ruler kivi --k-bits "${KIVI_K}" --v-bits "${KIVI_V}" \
        --group-size "${KIVI_GROUP}" --residual-length "${KIVI_RESIDUAL}"
      ;;
    oscar)
      OSCAR_ARGS=(--num-bits "${OSCAR_NUM_BITS}" --quant-mode "${OSCAR_QUANT_MODE}" --group-size "${OSCAR_GROUP}")
      [ "${OSCAR_KV_ROTATION}" != "none" ] && OSCAR_ARGS+=(--kv-rotation "${OSCAR_KV_ROTATION}")
      [ "${OSCAR_KV_NORM}" != "0" ] && OSCAR_ARGS+=(--kv-norm "${OSCAR_KV_NORM}")
      [ -n "${OSCAR_EVICT}" ] && OSCAR_ARGS+=(--residual-evict-size "${OSCAR_EVICT}")
      [ "${OSCAR_OFFLINE_V_HAD}" = "1" ] && OSCAR_ARGS+=(--offline-v-hadamard)
      run_ruler oscar "${OSCAR_ARGS[@]}"
      ;;
    flash_attention_2|sdpa|eager)
      run_ruler "${method}"
      ;;
    *)
      echo "Unknown method: ${method}" >&2
      exit 2
      ;;
  esac
done
