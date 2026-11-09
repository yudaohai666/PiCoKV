#!/bin/bash
# picokv online K=320 V=320, zerorope=0, re-validation at 64k and 128k, both models.
# Same 4-job layout as run_val_on320_64k128k.sh but ZEROROPE=0 on GPUs 0,1,2,3.
# Results (separate dir, tag suffixed _zr0):
#   results/val320/ruler/{model}/picokv_online_k320_v320_zr0/{shard_64k,shard_128k}/
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
# shellcheck disable=SC1091
source "${REPO_ROOT}/.venv/bin/activate"
export http_proxy="${http_proxy:-http://agent.baidu.com:8891}"
export https_proxy="${https_proxy:-http://agent.baidu.com:8891}"

DATA_ROOT="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data"
RESULTS_ROOT="results/val320"
LOG_ROOT="logs/val320"
mkdir -p "${LOG_ROOT}"

# "gpu;model;lengths;shard"
JOBS=(
  "0;Llama-3.1-8B-Instruct;65536;shard_64k"
  "1;Llama-3.1-8B-Instruct;131072;shard_128k"
  "2;Qwen3-8B;65536;shard_64k"
  "3;Qwen3-8B;131072;shard_128k"
)

for m in Llama-3.1-8B-Instruct Qwen3-8B; do
  [ -d "${DATA_ROOT}/${m}" ] || { echo "ERROR: model not found: ${DATA_ROOT}/${m}" >&2; exit 1; }
done

echo "=== picokv online k320v320 zr0 | RULER | limit=100 | 4 jobs on GPUs 4,5,6,7 ==="
pids=(); specs=()
for spec in "${JOBS[@]}"; do
  IFS=';' read -r gpu model lengths shard <<< "${spec}"
  out="${RESULTS_ROOT}/ruler/${model}/picokv_online_k320_v320_zr0/${shard}/"
  log="${LOG_ROOT}/${model}_zr0_${shard}.log"
  mkdir -p "${out}"
  echo "  [gpu ${gpu}] model=${model} lengths=${lengths} zr=0 -> ${out}"
  GPU="${gpu}" MODEL="${DATA_ROOT}/${model}" METHOD=picokv \
    K_BITS=320 V_BITS=320 USE_OFFLINE=0 ZEROROPE=0 \
    MAX_SEQ_LENGTHS="${lengths}" LIMIT=100 \
    OUTPUT_PATH="${out}" \
    bash "${SCRIPT_DIR}/run_ruler.sh" > "${log}" 2>&1 &
  pids+=("$!"); specs+=("${spec}")
done

rc_total=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "  done: ${specs[$i]}"; else echo "  FAILED: ${specs[$i]}" >&2; rc_total=1; fi
done
echo "=== all 4 zr0 jobs finished (rc=${rc_total}) ==="
exit "${rc_total}"
