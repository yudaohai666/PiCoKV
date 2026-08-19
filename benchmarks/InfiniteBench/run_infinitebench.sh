#!/bin/bash
# InfiniteBench launcher for quantpc / kivi / oscar / polar / dense baselines.
# Results are auto-written by run_infinitebench.py to results/infinitebench/{MODEL}/{TAG}/
# Select methods via the METHODS list below (METHOD=oscar forces a single one).
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
cd "${REPO_ROOT}"

GPU="${GPU:-0}"
MODEL="${MODEL:-/data/Llama-3.1-8B-Instruct}"
MODEL_TAG="$(basename "${MODEL%/}")"

# ── benchmark runtime params ──
MAX_LENGTH="${MAX_LENGTH:-131072}"
BATCH_SIZE="${BATCH_SIZE:-1}"
# 1200 is longbook_sum_en's ceiling; other tasks cap themselves via task yamls.
MAX_GEN_TOKS="${MAX_GEN_TOKS:-1200}"
# LIMIT is empty by default (full test set); set to e.g. 20 for a smoke run.
LIMIT="${LIMIT:-200}"
TASKS="${TASKS:-infinitebench_passkey,infinitebench_kv_retrieval,infinitebench_number_string,infinitebench_code_run,infinitebench_code_debug,infinitebench_math_find,infinitebench_longdialogue_qa_en,infinitebench_longbook_qa_en,infinitebench_longbook_sum_en,infinitebench_longbook_choice_en,infinitebench_longbook_qa_chn}"
APPLY_CHAT_TEMPLATE="${APPLY_CHAT_TEMPLATE:-1}"

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
# quantpc; default = offline + nowhiten + vrefit
K_BITS="${K_BITS:-320}"; V_BITS="${V_BITS:-320}"
SINK="${SINK:-64}"; LOCAL="${LOCAL:-256}"
EXP_BITS="${EXP_BITS:-4}"; EAS="${EAS:-0.01}"; ZEROROPE="${ZEROROPE:-20}"
USE_OFFLINE="${USE_OFFLINE:-1}"
WHITEN="${WHITEN:-0}"
ONLINE_DECODER_REFIT="${ONLINE_DECODER_REFIT:-v}"
DECODER_RIDGE="${DECODER_RIDGE:-1e-2}"
OFFLINE_PROJ_PATH="${OFFLINE_PROJ_PATH:-}"
# kivi (2/4-bit KV)
KIVI_K="${KIVI_K:-2}"; KIVI_V="${KIVI_V:-2}"; KIVI_GROUP="${KIVI_GROUP:-32}"; KIVI_RESIDUAL="${KIVI_RESIDUAL:-128}"
# oscar INT2 recipe (group MUST be 32; int2 V-param packer hardcodes 32-wide groups)
OSCAR_NUM_BITS="${OSCAR_NUM_BITS:-2}"; OSCAR_QUANT_MODE="${OSCAR_QUANT_MODE:-k-channel}"; OSCAR_GROUP="${OSCAR_GROUP:-32}"
OSCAR_KV_ROTATION="${OSCAR_KV_ROTATION:-hadamard}"; OSCAR_KV_NORM="${OSCAR_KV_NORM:-1}"; OSCAR_EVICT="${OSCAR_EVICT:-256}"; OSCAR_OFFLINE_V_HAD="${OSCAR_OFFLINE_V_HAD:-1}"
# polar (src/PolarQuant)
POLAR_MODE="${POLAR_MODE:-mixed}"; POLAR_RBITS="${POLAR_RBITS:-3}"; POLAR_TBITS="${POLAR_TBITS:-3}"; POLAR_VBITS="${POLAR_VBITS:-2}"

# Build METHOD_ARGS[] and TAG for a method. Needs REPO_ROOT, MODEL_TAG + the
# per-method defaults above. quantpc TAG matches run_infinitebench.py's python
# run_tag(): quantpc_{mode}_k{K}_v{V}_{whiten}[_krefit|_vrefit|_refit] (informational
# only here -- the python runner auto-writes the results dir).
build_method_args() {
  METHOD_ARGS=(); TAG="$1"
  case "$1" in
    quantpc)
      METHOD_ARGS=(--k-bits "${K_BITS}" --v-bits "${V_BITS}" --sink-size "${SINK}" --local-size "${LOCAL}"
        --exp-bits "${EXP_BITS}" --eas "${EAS}" --zerorope "${ZEROROPE}"
        --decoder-ridge "${DECODER_RIDGE}" --online-decoder-refit "${ONLINE_DECODER_REFIT}")
      local mode proj whiten_tag
      if [ "${USE_OFFLINE}" = "1" ] || [ -n "${OFFLINE_PROJ_PATH}" ]; then
        mode="offline"; METHOD_ARGS+=(--offline); proj="${OFFLINE_PROJ_PATH}"
        # offline + whiten -> the folded _whiten.pt (no runtime --whiten flag)
        if [ -z "${proj}" ] && [ "${WHITEN}" = "1" ]; then
          proj="${REPO_ROOT}/offline_projection/projections/${MODEL_TAG}_k${K_BITS}_v${V_BITS}_whiten.pt"
        fi
        [ -n "${proj}" ] && METHOD_ARGS+=(--offline-proj-path "${proj}")
      else
        mode="online"; [ "${WHITEN}" = "1" ] && METHOD_ARGS+=(--whiten)
      fi
      [ "${WHITEN}" = "1" ] && whiten_tag="whiten" || whiten_tag="nowhiten"
      TAG="quantpc_${mode}_k${K_BITS}_v${V_BITS}_${whiten_tag}"
      # refit suffix applies only to the offline path.
      if [ "${mode}" = "offline" ]; then
        case "${ONLINE_DECODER_REFIT}" in k) TAG+="_krefit" ;; v) TAG+="_vrefit" ;; both) TAG+="_refit" ;; esac
      fi
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
LIMIT_ARGS=()
[ -n "${LIMIT}" ] && LIMIT_ARGS=(--limit "${LIMIT}")

for method in "${METHODS[@]}"; do
  build_method_args "${method}"
  echo "=== InfiniteBench: method=${method} (${TAG}) ==="
  CUDA_VISIBLE_DEVICES="${GPU}" python "${SCRIPT_DIR}/run_infinitebench.py" \
    --model "${MODEL}" --method "${method}" \
    --tasks "${TASKS}" \
    --max_length "${MAX_LENGTH}" --batch_size "${BATCH_SIZE}" \
    --max_gen_toks "${MAX_GEN_TOKS}" \
    "${LIMIT_ARGS[@]}" "${CHAT_ARGS[@]}" "${METHOD_ARGS[@]}"
done

echo "All done."
