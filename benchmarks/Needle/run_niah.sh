#!/bin/bash
# NIAH (needle-in-a-haystack): quantpc / kivi / oscar / dense baselines, then visualize.
# Switch method via env, e.g. METHOD=kivi bash run_niah.sh. Output under results/niah/.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

GPU="${GPU:-0}"
MODEL="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct"
MODEL_NAME=$(basename "${MODEL}")
BASE_RESULT_DIR="results/niah"
VIS_DIR="${BASE_RESULT_DIR}/vis"

LENGTHS="8192,16384,24576,32768,40960,49152,57344,65536,73728,81920,90112,98304,106496,114688,122880,131072"
DEPTHS="0,11,22,33,44,56,67,78,89,100"
EXPECTED_ANSWER="eat a sandwich and sit in Dolores Park on a sunny day."  # must match show_image_paper.py

METHOD="${METHOD:-quantpc}"          # quantpc | kivi | oscar | flash_attention_2 | sdpa | eager

# quantpc. Offline whitening is baked into the folded artifact (never pass
# --whiten for offline); online whitening rotates K/V at prefill.
CONFIGS=("320 256")                  # K/V sweep: add e.g. "320 256"
SINK="${SINK:-64}"; LOCAL="${LOCAL:-256}"
EXP_BITS="${EXP_BITS:-4}"; EAS="${EAS:-0.01}"; ZEROROPE="${ZEROROPE:-20}"
USE_OFFLINE="${USE_OFFLINE:-1}"      # 1=load pre-fitted projections; 0=online QR
WHITEN="${WHITEN:-1}"                # 1=whiten K+V; k=whiten K only (offline); 0=no
OFFLINE_PROJ_PATH="${OFFLINE_PROJ_PATH:-}"  # override the derived offline file

# KIVI (2/4-bit KV only).
KIVI_K="${KIVI_K:-2}"; KIVI_V="${KIVI_V:-2}"; KIVI_GROUP="${KIVI_GROUP:-32}"; KIVI_RESIDUAL="${KIVI_RESIDUAL:-128}"

# OScaR INT2. group_size MUST be 32: the int2 V-param packer hardcodes 32-wide groups.
OSCAR_NUM_BITS="${OSCAR_NUM_BITS:-2}"; OSCAR_QUANT_MODE="${OSCAR_QUANT_MODE:-k-channel}"; OSCAR_GROUP="${OSCAR_GROUP:-32}"
OSCAR_KV_ROTATION="${OSCAR_KV_ROTATION:-hadamard}"; OSCAR_KV_NORM="${OSCAR_KV_NORM:-1}"
OSCAR_EVICT="${OSCAR_EVICT:-256}"; OSCAR_OFFLINE_V_HAD="${OSCAR_OFFLINE_V_HAD:-1}"

# run_niah SUBDIR RUN_ARGS... - run one config and plot.
run_niah() {
  local subdir="$1"; shift
  mkdir -p "${subdir}"
  echo "=== NIAH: method=${METHOD} -> ${subdir}/results.json ==="
  CUDA_VISIBLE_DEVICES=${GPU} python "${SCRIPT_DIR}/run_bin.py" \
    --model "${MODEL}" --method "${METHOD}" \
    --lengths "${LENGTHS}" --depths "${DEPTHS}" \
    --output "${subdir}/results.json" "$@"
  python "${SCRIPT_DIR}/show_image_paper.py" \
    --model "${MODEL}" --eval_path "${subdir}" \
    --save_dir "${VIS_DIR}" --expected_answer "${EXPECTED_ANSWER}"
}

case "${METHOD}" in
  quantpc)
    for cfg in "${CONFIGS[@]}"; do
      read -r K V <<< "$cfg"
      ARGS=()
      if [ "${USE_OFFLINE}" = "1" ]; then
        case "${WHITEN}" in
          1) SUFFIX="folded";         TAG="_offline_whiten"   ;;  # K+V whitened
          k) SUFFIX="folded_whitenK"; TAG="_offline_whitenK"  ;;  # K-only whitened
          *) SUFFIX="nowhiten";       TAG="_offline_nowhiten" ;;
        esac
        PROJ="${OFFLINE_PROJ_PATH:-${REPO_ROOT}/offline_projection/projections/${MODEL_NAME}_k${K}_v${V}_block32_${SUFFIX}.pt}"
        ARGS+=(--offline-proj-path "${PROJ}")
      else
        [ "${WHITEN}" = "1" ] && { ARGS+=(--whiten); TAG="_online_whiten"; } || TAG="_online"
      fi
      run_niah "${BASE_RESULT_DIR}/${MODEL_NAME}_k${K}_v${V}${TAG}" \
        --k-bits "${K}" --v-bits "${V}" --sink-size "${SINK}" --local-size "${LOCAL}" \
        --exp-bits "${EXP_BITS}" --eas "${EAS}" --zerorope "${ZEROROPE}" "${ARGS[@]}"
    done
    ;;
  kivi)
    run_niah "${BASE_RESULT_DIR}/${MODEL_NAME}_kivi_k${KIVI_K}_v${KIVI_V}" \
      --k-bits "${KIVI_K}" --v-bits "${KIVI_V}" --group-size "${KIVI_GROUP}" --residual-length "${KIVI_RESIDUAL}"
    ;;
  oscar)
    ARGS=(--num-bits "${OSCAR_NUM_BITS}" --quant-mode "${OSCAR_QUANT_MODE}" --group-size "${OSCAR_GROUP}")
    [ "${OSCAR_KV_ROTATION}" != "none" ] && ARGS+=(--kv-rotation "${OSCAR_KV_ROTATION}")
    [ "${OSCAR_KV_NORM}" != "0" ]       && ARGS+=(--kv-norm "${OSCAR_KV_NORM}")
    [ -n "${OSCAR_EVICT}" ]             && ARGS+=(--residual-evict-size "${OSCAR_EVICT}")
    [ "${OSCAR_OFFLINE_V_HAD}" = "1" ]  && ARGS+=(--offline-v-hadamard)
    run_niah "${BASE_RESULT_DIR}/${MODEL_NAME}_oscar_int${OSCAR_NUM_BITS}" "${ARGS[@]}"
    ;;
  *)
    run_niah "${BASE_RESULT_DIR}/${MODEL_NAME}_${METHOD}"
    ;;
esac

echo "All done. Images saved to: ${VIS_DIR}/${MODEL_NAME}/"
