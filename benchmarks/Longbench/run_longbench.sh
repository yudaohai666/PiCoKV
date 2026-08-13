#!/bin/bash
# Run LongBench subsets sequentially, then score with eval.py.
# Output: results/longbench/<MODEL_NAME>/<TAG>/{<dataset>.jsonl,result.json}
# Multiple methods per run: edit METHODS below (or METHOD=oscar to force one).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

GPU="${GPU:-7}"
MODEL="${MODEL:-/data/Llama-3.1-8B-Instruct}"
DATA_DIR="${DATA_DIR:-}"
BASE_RESULT_DIR="results/longbench"
MODEL_NAME=$(basename "${MODEL}")

if [ -z "${DATA_DIR}" ]; then
  echo "ERROR: set DATA_DIR to the directory containing the LongBench *.jsonl files" >&2
  echo "Example: DATA_DIR=/path/to/longbench/data bash ${SCRIPT_DIR}/run_longbench.sh" >&2
  exit 2
fi

# Methods to run. quantpc/kivi/oscar/polar = quantized KV; flash_attention_2/sdpa/eager = dense baselines.
METHODS=(
  quantpc
  # kivi
  # oscar
  # flash_attention_2
  # sdpa
)
# Back-compat: METHOD=oscar forces a single method.
if [ -n "${METHOD:-}" ]; then METHODS=("${METHOD}"); fi

# quantpc K/V sweep (quantpc only).
CONFIGS=(
  "320 320"
)

# quantpc offline encoders: USE_OFFLINE=1 skips the online QR at prefill
# (default path offline_projection/projections/{model}_k{K}_v{V}_nowhiten.pt).
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
# KIVI params (2/4-bit KV, single run).
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

# run_lb METHOD TAG RUN_ARGS...
run_lb() {
  local method="$1"; local tag="$2"; shift 2
  local out_dir="${BASE_RESULT_DIR}/${MODEL_NAME}/${tag}"
  echo "=== LongBench: method=${method} -> ${out_dir} ==="
  CUDA_VISIBLE_DEVICES=${GPU} python "${SCRIPT_DIR}/pred.py" \
    --model "${MODEL}" --method "${method}" --save_dir "${BASE_RESULT_DIR}" \
    --output-tag "${tag}" --decoder-ridge "${DECODER_RIDGE}" \
    --data-dir "${DATA_DIR}" "$@"
  python "${SCRIPT_DIR}/eval.py" --path "${out_dir}/" --eval_avg
}

for METHOD in "${METHODS[@]}"; do
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
        QUANT_ARGS=(--k-bits "${K}" --v-bits "${V}" --exp-bits 4 --eas 0.01 --zerorope 20
          --online-decoder-refit "${ONLINE_DECODER_REFIT}")
        [ "${WHITEN}" = "1" ] && QUANT_ARGS+=(--whiten)
        run_lb quantpc "${TAG}" "${QUANT_ARGS[@]}" "${OFFLINE_ARGS[@]}"
      done
      ;;
    kivi)
      run_lb kivi "kivi_k${KIVI_K}_v${KIVI_V}" \
        --k-bits "${KIVI_K}" --v-bits "${KIVI_V}" \
        --group-size "${KIVI_GROUP}" --residual-length "${KIVI_RESIDUAL}"
      ;;
    oscar)
      OSCAR_ARGS=(--num-bits "${OSCAR_NUM_BITS}" --quant-mode "${OSCAR_QUANT_MODE}" --group-size "${OSCAR_GROUP}")
      [ "${OSCAR_KV_ROTATION}" != "none" ] && OSCAR_ARGS+=(--kv-rotation "${OSCAR_KV_ROTATION}")
      [ "${OSCAR_KV_NORM}" != "0" ]       && OSCAR_ARGS+=(--kv-norm "${OSCAR_KV_NORM}")
      [ -n "${OSCAR_EVICT}" ]             && OSCAR_ARGS+=(--residual-evict-size "${OSCAR_EVICT}")
      [ "${OSCAR_OFFLINE_V_HAD}" = "1" ]  && OSCAR_ARGS+=(--offline-v-hadamard)
      run_lb oscar "oscar_int${OSCAR_NUM_BITS}" "${OSCAR_ARGS[@]}"
      ;;
    polar)
      run_lb polar "polar_${POLAR_MODE}_${POLAR_RBITS}${POLAR_TBITS}${POLAR_VBITS}" \
        --polar-mode "${POLAR_MODE}" --polar-rbits "${POLAR_RBITS}" \
        --polar-tbits "${POLAR_TBITS}" --polar-vbits "${POLAR_VBITS}"
      ;;
    *)
      run_lb "${METHOD}" "${METHOD}"
      ;;
  esac
done

echo "All done."
