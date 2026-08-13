#!/bin/bash
# NIAH (needle-in-a-haystack): quantpc / kivi / oscar / dense baselines, then visualize.
# Switch method via env: METHOD=kivi bash run_niah.sh
# Output: results/niah/<MODEL>/<TAG>/results.json ; images in results/niah/vis/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

GPU="${GPU:-0}"
MODEL="${MODEL:-/data/Llama-3.1-8B-Instruct}"
BASE_RESULT_DIR="results/niah"
VIS_DIR="${BASE_RESULT_DIR}/vis"
MODEL_NAME=$(basename "${MODEL}")

# Context lengths / needle depths (comma-separated).
# LENGTHS="81920,90112,98304,106496,114688,122880,131072"
LENGTHS="8192,16384,24576,32768,40960,49152,57344,65536,73728,81920,90112,98304,106496,114688,122880,131072"
DEPTHS="0,11,22,33,44,56,67,78,89,100"

# Expected answer (must match show_image_paper.py default).
EXPECTED_ANSWER="eat a sandwich and sit in Dolores Park on a sunny day."

# Method: quantpc | kivi | oscar | polar | flash_attention_2 | sdpa | eager
METHOD="${METHOD:-quantpc}"

# quantpc K/V sweep + full-precision sink/local windows.
CONFIGS=(
  "320 320"
  # "320 256"
  # "320 320"
)
SINK="${SINK:-64}"
LOCAL="${LOCAL:-256}"

# quantpc offline encoders: USE_OFFLINE=1 skips the online QR at prefill
# (default path offline_projection/projections/{model}_k{K}_v{V}_nowhiten.pt).
# Keep NIAH online by default; set USE_OFFLINE=1 to load the fitted PT.
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
# KIVI params (2/4-bit KV only).
KIVI_K="${KIVI_K:-2}"
KIVI_V="${KIVI_V:-2}"
KIVI_GROUP="${KIVI_GROUP:-32}"
KIVI_RESIDUAL="${KIVI_RESIDUAL:-128}"

# OScaR INT2 recipe. group_size MUST be 32: the int2 V-param interleaved packer
# hardcodes 32-wide groups (4 groups); group 128 under-allocates v_params -> OOB write.
OSCAR_NUM_BITS="${OSCAR_NUM_BITS:-2}"
OSCAR_QUANT_MODE="${OSCAR_QUANT_MODE:-k-channel}"
OSCAR_GROUP="${OSCAR_GROUP:-32}"
OSCAR_KV_ROTATION="${OSCAR_KV_ROTATION:-hadamard}"
OSCAR_KV_NORM="${OSCAR_KV_NORM:-1}"
OSCAR_EVICT="${OSCAR_EVICT:-256}"
OSCAR_OFFLINE_V_HAD="${OSCAR_OFFLINE_V_HAD:-1}"

# PolarQuant params (src/PolarQuant), kept in sync with quantpc_v2.
POLAR_MODE="${POLAR_MODE:-mixed}"
POLAR_RBITS="${POLAR_RBITS:-3}"
POLAR_TBITS="${POLAR_TBITS:-3}"
POLAR_VBITS="${POLAR_VBITS:-2}"

# run_niah SUBDIR RUN_ARGS...  - run one config and plot.
run_niah() {
  local subdir="$1"; shift
  mkdir -p "${subdir}"
  echo "=== NIAH: method=${METHOD} -> ${subdir}/results.json ==="
  CUDA_VISIBLE_DEVICES=${GPU} python "${SCRIPT_DIR}/run_bin.py" \
    --model "${MODEL}" --method "${METHOD}" \
    --lengths "${LENGTHS}" --depths "${DEPTHS}" \
    --output "${subdir}/results.json" --decoder-ridge "${DECODER_RIDGE}" "$@"
  python "${SCRIPT_DIR}/show_image_paper.py" \
    --model "${MODEL}" --eval_path "${subdir}" \
    --save_dir "${VIS_DIR}" --expected_answer "${EXPECTED_ANSWER}"
}

case "${METHOD}" in
    quantpc)
      for cfg in "${CONFIGS[@]}"; do
        K=$(echo "$cfg" | awk '{print $1}'); V=$(echo "$cfg" | awk '{print $2}')
        if [ ${#OFFLINE_ARGS[@]} -gt 0 ]; then MODE_TAG="offline"; else MODE_TAG="online"; fi
        if [ "${WHITEN}" = "1" ]; then WHITEN_TAG="whiten"; else WHITEN_TAG="nowhiten"; fi
        TAG="${MODE_TAG}_k${K}_v${V}_${WHITEN_TAG}"
        case "${ONLINE_DECODER_REFIT}" in
          k) TAG+="_krefit" ;;
          v) TAG+="_vrefit" ;;
          both) TAG+="_refit" ;;
        esac
        QUANT_ARGS=(--k-bits "${K}" --v-bits "${V}" --sink-size "${SINK}" --local-size "${LOCAL}"
          --exp-bits 4 --eas 0.01 --zerorope 20
          --online-decoder-refit "${ONLINE_DECODER_REFIT}")
        [ "${WHITEN}" = "1" ] && QUANT_ARGS+=(--whiten)
        run_niah "${BASE_RESULT_DIR}/${MODEL_NAME}/${TAG}" "${QUANT_ARGS[@]}" "${OFFLINE_ARGS[@]}"
      done
      ;;
  kivi)
    run_niah "${BASE_RESULT_DIR}/${MODEL_NAME}/kivi_k${KIVI_K}_v${KIVI_V}" \
      --k-bits "${KIVI_K}" --v-bits "${KIVI_V}" \
      --group-size "${KIVI_GROUP}" --residual-length "${KIVI_RESIDUAL}"
    ;;
  oscar)
    OSCAR_ARGS=(--num-bits "${OSCAR_NUM_BITS}" --quant-mode "${OSCAR_QUANT_MODE}" --group-size "${OSCAR_GROUP}")
    [ "${OSCAR_KV_ROTATION}" != "none" ] && OSCAR_ARGS+=(--kv-rotation "${OSCAR_KV_ROTATION}")
    [ "${OSCAR_KV_NORM}" != "0" ]       && OSCAR_ARGS+=(--kv-norm "${OSCAR_KV_NORM}")
    [ -n "${OSCAR_EVICT}" ]             && OSCAR_ARGS+=(--residual-evict-size "${OSCAR_EVICT}")
    [ "${OSCAR_OFFLINE_V_HAD}" = "1" ]  && OSCAR_ARGS+=(--offline-v-hadamard)
    run_niah "${BASE_RESULT_DIR}/${MODEL_NAME}/oscar_int${OSCAR_NUM_BITS}" "${OSCAR_ARGS[@]}"
    ;;
  polar)
    run_niah "${BASE_RESULT_DIR}/${MODEL_NAME}/polar_${POLAR_MODE}_${POLAR_RBITS}${POLAR_TBITS}${POLAR_VBITS}" \
      --polar-mode "${POLAR_MODE}" --polar-rbits "${POLAR_RBITS}" \
      --polar-tbits "${POLAR_TBITS}" --polar-vbits "${POLAR_VBITS}"
    ;;
  *)
    run_niah "${BASE_RESULT_DIR}/${MODEL_NAME}/${METHOD}"
    ;;
esac

echo "All done. Images saved to: ${VIS_DIR}/${MODEL_NAME}/"
