#!/bin/bash
# Prefill vs decode efficiency: dense baseline / quantpc / oscar / kivi.
# Same runner (run_efficiency.py) per method -> one JSON each in results/efficiency/,
# then --compare aggregates a table grouped by prompt_len.
# Inputs are random token ids (latency depends on length/shape, not content).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

GPU=6
MODEL="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct"
RESULT_DIR="results/efficiency"
mkdir -p "${RESULT_DIR}"

LENGTHS="131072"
DECODE_TOKENS=64
BATCH_SIZE=1
WARMUP=1
REPEATS=3

# quantpc offline block-32 encoders: USE_OFFLINE=1 skips the online QR at prefill
# (default path offline_projection/{model}_k{K}_v{V}_block32.pt). OFFLINE_PROJ_PATH
USE_OFFLINE="${USE_OFFLINE:-1}"
OFFLINE_PROJ_PATH="${OFFLINE_PROJ_PATH:-}"
OFFLINE_ARGS=()
if [ "${USE_OFFLINE}" = "1" ] || [ -n "${OFFLINE_PROJ_PATH}" ]; then
  OFFLINE_ARGS=(--offline)
  [ -n "${OFFLINE_PROJ_PATH}" ] && OFFLINE_ARGS+=(--offline-proj-path "${OFFLINE_PROJ_PATH}")
fi
# Output-tag suffix so offline runs don't overwrite online ones (matches output_tag).
OFFLINE_TAG=""
if [ ${#OFFLINE_ARGS[@]} -gt 0 ]; then
  OFFLINE_TAG="_offline"
fi

# run_one METHOD TAG RUN_ARGS...
run_one() {
  local method="$1"; shift
  local tag="$1"; shift
  echo "=== Efficiency: method=${method} -> ${RESULT_DIR}/${tag}.json ==="
  CUDA_VISIBLE_DEVICES=${GPU} python "${SCRIPT_DIR}/run_efficiency.py" \
    --model "${MODEL}" --method "${method}" \
    --lengths "${LENGTHS}" --decode-tokens "${DECODE_TOKENS}" \
    --batch-size "${BATCH_SIZE}" --warmup "${WARMUP}" --repeats "${REPEATS}" \
    --output "${RESULT_DIR}/${tag}.json" "$@"
}

# dense baseline
# run_one flash_attention_2 flash_attention_2
run_one sdpa sdpa


# quantpc (CUDA backend quantpc.quanpc)
run_one quantpc "quantpc_k256_v256${OFFLINE_TAG}" --k-bits 256 --v-bits 256 "${OFFLINE_ARGS[@]}"

# oscar INT2 recipe. Requires oscar_cuda built (bash src/OScaR-KV-Quant/build_oscar_cuda.sh).
# group_size MUST be 32: the int2 V-param interleaved packer hardcodes 32-wide groups
# (4 groups); group 128 under-allocates v_params -> OOB write.
# run_one oscar oscar_int2 \
#   --num-bits 2 --quant-mode k-channel --group-size 32 \
#   --kv-rotation hadamard --kv-norm 1 --residual-evict-size 256 --offline-v-hadamard

# kivi (2-bit K/V)
# run_one kivi kivi_k2_v2 --k-bits 2 --v-bits 2 --group-size 32 --residual-length 128

echo "=== Comparison table ==="
python "${SCRIPT_DIR}/run_efficiency.py" --compare "${RESULT_DIR}"
