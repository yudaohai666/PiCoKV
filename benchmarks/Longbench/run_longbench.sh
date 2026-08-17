#!/bin/bash
# Run LongBench subsets sequentially, then score with eval.py.
# Output: results/longbench/<MODEL>/<TAG>/{<dataset>.jsonl,result.json}
# Select methods via the METHODS list below (METHOD=oscar forces a single one).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

GPU="${GPU:-7}"
MODEL="${MODEL:-/data/Llama-3.1-8B-Instruct}"
MODEL_TAG="$(basename "${MODEL%/}")"
BASE_RESULT_DIR="results/longbench"

# ── benchmark runtime params ──
DATA_DIR="${DATA_DIR:-}"        # dir holding the unzipped LongBench *.jsonl files
DATASETS="${DATASETS:-}"        # optional comma-separated subset (default: config list)
LIMIT="${LIMIT:-}"              # optional first-N samples per dataset (smoke run)
if [ -z "${DATA_DIR}" ]; then
  echo "ERROR: set DATA_DIR to the directory containing the LongBench *.jsonl files" >&2
  echo "Example: DATA_DIR=/path/to/longbench/data bash ${SCRIPT_DIR}/run_longbench.sh" >&2
  exit 2
fi

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
# per-method defaults above. quantpc TAG matches run_ruler/run_infinitebench's
# python run_tag(): quantpc_{mode}_k{K}_v{V}_{whiten}[_krefit|_vrefit|_refit].
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

EXTRA_ARGS=()
[ -n "${DATASETS}" ] && EXTRA_ARGS+=(--datasets "${DATASETS}")
[ -n "${LIMIT}" ] && EXTRA_ARGS+=(--limit "${LIMIT}")

for method in "${METHODS[@]}"; do
  build_method_args "${method}"
  out_dir="${BASE_RESULT_DIR}/${MODEL_TAG}/${TAG}"
  echo "=== LongBench: method=${method} -> ${out_dir} ==="
  CUDA_VISIBLE_DEVICES=${GPU} python "${SCRIPT_DIR}/pred.py" \
    --model "${MODEL}" --method "${method}" \
    --save_dir "${BASE_RESULT_DIR}" --output-tag "${TAG}" \
    --data-dir "${DATA_DIR}" "${EXTRA_ARGS[@]}" "${METHOD_ARGS[@]}"
  python "${SCRIPT_DIR}/eval.py" --path "${out_dir}/" --eval_avg
done

echo "All done."
