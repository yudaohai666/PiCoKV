#!/bin/bash
# Run LongBench subsets sequentially, then score with eval.py.
# Output: results/longbench/<MODEL_NAME>/<TAG>/{<dataset>.jsonl,result.json}
# Multiple methods per run: edit METHODS below (or METHOD=oscar to force one).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

GPU="${GPU:-7}"
MODEL="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct"
BASE_RESULT_DIR="results/longbench"
MODEL_NAME=$(basename "${MODEL}")
# Directory holding the unzipped LongBench *.jsonl files (passed to pred.py).
DATA_DIR="${DATA_DIR:-/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/LongBench/data}"

# Methods to run. quantpc/kivi/oscar/polar = quantized KV; flash_attention_2/sdpa/eager = dense baselines.
METHODS=(
  quantpc
  # kivi
  # oscar
  # polar
  # flash_attention_2
  # sdpa
)
# Back-compat: METHOD=oscar forces a single method.
if [ -n "${METHOD:-}" ]; then METHODS=("${METHOD}"); fi

# quantpc K/V sweep (quantpc only). Override a single pair via env, e.g.
# KV="384 320" bash run_longbench.sh; otherwise use the list below.
if [ -n "${KV:-}" ]; then
  CONFIGS=("${KV}")
else
  CONFIGS=(
    "320 256"
  )
fi

# quantpc. Offline whitening is baked into the folded artifact (never pass
# --whiten for offline); online whitening rotates K/V at prefill. Mirrors the
# offline/whiten switches in Needle/run_niah.sh so both benchmarks drive the
# same projection sweeps.
SINK="${SINK:-64}"; LOCAL="${LOCAL:-256}"
EXP_BITS="${EXP_BITS:-4}"; EAS="${EAS:-0.01}"; ZEROROPE="${ZEROROPE:-20}"
USE_OFFLINE="${USE_OFFLINE:-1}"      # 1=load pre-fitted projections; 0=online QR
WHITEN="${WHITEN:-1}"                # 1=whiten K+V; k=whiten K only (offline); 0=no
OFFLINE_PROJ_PATH="${OFFLINE_PROJ_PATH:-}"  # override the derived offline file

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

# PolarQuant params (src/PolarQuant). K polar-quant + (mixed) V KIVI-quant. Passed
# to pred.py as --polar-* flags (threaded to common.py::_load_polar_model).
#   POLAR_MODE : mixed (K polar + V KIVI) | group (K-only) | snapkv
#   RBITS/TBITS: K radius/angle bits (rbits+tbits<=8; default 33)
#   VBITS      : V bits, mixed only (KIVI kernel supports 2 or 4; default 2)
POLAR_MODE="${POLAR_MODE:-mixed}"
POLAR_RBITS="${POLAR_RBITS:-3}"; POLAR_TBITS="${POLAR_TBITS:-3}"; POLAR_VBITS="${POLAR_VBITS:-2}"

# run_lb METHOD TAG RUN_ARGS...
run_lb() {
  local method="$1"; local tag="$2"; shift 2
  local out_dir="${BASE_RESULT_DIR}/${MODEL_NAME}/${tag}"
  echo "=== LongBench: method=${method} -> ${out_dir} ==="
  CUDA_VISIBLE_DEVICES=${GPU} python "${SCRIPT_DIR}/pred.py" \
    --model "${MODEL}" --method "${method}" --save_dir "${BASE_RESULT_DIR}" \
    --data-dir "${DATA_DIR}" --output-tag "${tag}" "$@"
  python "${SCRIPT_DIR}/eval.py" --path "${out_dir}/" --eval_avg
}

for METHOD in "${METHODS[@]}"; do
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
        run_lb quantpc "quantpc_k${K}_v${V}${TAG}" \
          --k-bits "${K}" --v-bits "${V}" --sink-size "${SINK}" --local-size "${LOCAL}" \
          --exp-bits "${EXP_BITS}" --eas "${EAS}" --zerorope "${ZEROROPE}" "${ARGS[@]}"
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
      TAG="polar_${POLAR_MODE}_${POLAR_RBITS}${POLAR_TBITS}${POLAR_VBITS}"
      run_lb polar "${TAG}" \
        --polar-mode "${POLAR_MODE}" --polar-rbits "${POLAR_RBITS}" \
        --polar-tbits "${POLAR_TBITS}" --polar-vbits "${POLAR_VBITS}"
      ;;
    *)
      run_lb "${METHOD}" "${METHOD}"
      ;;
  esac
done

echo "All done."
