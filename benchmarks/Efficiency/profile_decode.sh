#!/bin/bash
# Decode-stage torch.profiler analysis: quantpc / kivi / oscar / dense baselines.
# Prefill once, time the decode steps, export key_averages table + chrome trace.
# Switch method via env: METHOD=kivi bash profile_decode.sh
# Output: results/efficiency/prof/<method>_<length>.json (chrome trace)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

GPU="${GPU:-6}"
MODEL="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct"
PROF_DIR="results/efficiency/prof"
mkdir -p "${PROF_DIR}"

# Profiling params (aligned with run_efficiency.py).
LENGTH="${LENGTH:-32768}"   # prefill length
WARMUP="${WARMUP:-2}"       # untimed decode steps before profiling (= --warmup)
ACTIVE="${ACTIVE:-64}"      # profiled decode steps (= --decode-tokens)
ROWS="${ROWS:-30}"          # key_averages table rows

# Method: quantpc | kivi | oscar | flash_attention_2 | sdpa | eager
METHOD="${METHOD:-quantpc}"

# quantpc KV-cache params (aligned with run_efficiency.py defaults).
K_BITS="${K_BITS:-256}"
V_BITS="${V_BITS:-256}"
SINK="${SINK:-64}"
LOCAL="${LOCAL:-256}"
EXP_BITS="${EXP_BITS:-4}"
EAS="${EAS:-0.01}"
ZEROROPE="${ZEROROPE:-20}"

# quantpc offline block-32 encoders: USE_OFFLINE=1 skips the online QR at prefill
# (default path offline_projection/{model}_k{K}_v{V}_block32.pt). OFFLINE_PROJ_PATH
USE_OFFLINE="${USE_OFFLINE:-0}"
OFFLINE_PROJ_PATH="${OFFLINE_PROJ_PATH:-}"

# KIVI params (2/4-bit KV).
KIVI_K="${KIVI_K:-2}"
KIVI_V="${KIVI_V:-2}"
KIVI_GROUP="${KIVI_GROUP:-32}"
KIVI_RESIDUAL="${KIVI_RESIDUAL:-128}"

# OScaR INT2 recipe (group_size MUST be 32, else int2 V-param packer writes OOB).
OSCAR_NUM_BITS="${OSCAR_NUM_BITS:-2}"
OSCAR_QUANT_MODE="${OSCAR_QUANT_MODE:-k-channel}"
OSCAR_GROUP="${OSCAR_GROUP:-32}"
OSCAR_KV_ROTATION="${OSCAR_KV_ROTATION:-hadamard}"
OSCAR_KV_NORM="${OSCAR_KV_NORM:-1}"
OSCAR_EVICT="${OSCAR_EVICT:-256}"
OSCAR_OFFLINE_V_HAD="${OSCAR_OFFLINE_V_HAD:-1}"

case "${METHOD}" in
  quantpc)
    METHOD_ARGS=(--k-bits "${K_BITS}" --v-bits "${V_BITS}" --sink-size "${SINK}" --local-size "${LOCAL}" \
                 --exp-bits "${EXP_BITS}" --eas "${EAS}" --zerorope "${ZEROROPE}")
    if [ "${USE_OFFLINE}" = "1" ] || [ -n "${OFFLINE_PROJ_PATH}" ]; then
      METHOD_ARGS+=(--offline)
      [ -n "${OFFLINE_PROJ_PATH}" ] && METHOD_ARGS+=(--offline-proj-path "${OFFLINE_PROJ_PATH}")
    fi
    ;;
  kivi)
    METHOD_ARGS=(--k-bits "${KIVI_K}" --v-bits "${KIVI_V}" \
                 --group-size "${KIVI_GROUP}" --residual-length "${KIVI_RESIDUAL}")
    ;;
  oscar)
    METHOD_ARGS=(--num-bits "${OSCAR_NUM_BITS}" --quant-mode "${OSCAR_QUANT_MODE}" --group-size "${OSCAR_GROUP}")
    [ "${OSCAR_KV_ROTATION}" != "none" ] && METHOD_ARGS+=(--kv-rotation "${OSCAR_KV_ROTATION}")
    [ "${OSCAR_KV_NORM}" != "0" ]       && METHOD_ARGS+=(--kv-norm "${OSCAR_KV_NORM}")
    [ -n "${OSCAR_EVICT}" ]             && METHOD_ARGS+=(--residual-evict-size "${OSCAR_EVICT}")
    [ "${OSCAR_OFFLINE_V_HAD}" = "1" ]  && METHOD_ARGS+=(--offline-v-hadamard)
    ;;
  *)
    METHOD_ARGS=()
    ;;
esac

TRACE="${PROF_DIR}/${METHOD}_${LENGTH}.json"
echo "=== Profiling decode: method=${METHOD} length=${LENGTH} -> ${TRACE} ==="
CUDA_VISIBLE_DEVICES=${GPU} python "${SCRIPT_DIR}/profile_decode.py" \
  --model "${MODEL}" --method "${METHOD}" \
  --length "${LENGTH}" --warmup "${WARMUP}" --active "${ACTIVE}" --rows "${ROWS}" \
  --trace "${TRACE}" "${METHOD_ARGS[@]}"
