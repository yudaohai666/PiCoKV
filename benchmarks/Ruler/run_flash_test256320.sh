#!/bin/bash
# flash_attention_2 (dense baseline) at limit=20, RULER, matched to the
# picokv 256/320 limit-20 runs so the comparison is apples-to-apples.
# 2 models x 2 length groups = 4 jobs on GPUs 0,2,4,6 (one per card):
#   0: Llama 8-64k   2: Llama 128k   4: Qwen 8-64k   6: Qwen 128k
# Results: results/test256320/ruler/{model}/dense_flash_attention_2/{shard}/
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
# shellcheck disable=SC1091
source "${REPO_ROOT}/.venv/bin/activate"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export http_proxy="${http_proxy:-http://agent.baidu.com:8891}"
export https_proxy="${https_proxy:-http://agent.baidu.com:8891}"

DATA_ROOT="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data"
LIMIT=20
RULER_TASKS="niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multiquery,niah_multivalue,ruler_vt,ruler_cwe,ruler_fwe,ruler_qa_squad,ruler_qa_hotpot"
RESULTS_ROOT="results/test256320"
LOG_ROOT="logs/test256320"
mkdir -p "${LOG_ROOT}"

# "gpu;model;lengths;shard"
JOBS=(
  "0;Llama-3.1-8B-Instruct;8192,16384,32768,65536;shard_8to64k"
  "2;Llama-3.1-8B-Instruct;131072;shard_128k"
  "4;Qwen3-8B;8192,16384,32768,65536;shard_8to64k"
  "6;Qwen3-8B;131072;shard_128k"
)

pids=(); specs=()
for spec in "${JOBS[@]}"; do
  IFS=';' read -r gpu model lengths shard <<< "${spec}"
  out="${RESULTS_ROOT}/ruler/${model}/dense_flash_attention_2/${shard}/"
  log="${LOG_ROOT}/${model}_flash_${shard}.log"
  mkdir -p "${out}"
  echo "  [gpu ${gpu}] flash model=${model} lengths=${lengths} -> ${out}"
  CUDA_VISIBLE_DEVICES="${gpu}" python "${SCRIPT_DIR}/run_ruler_evaluation.py" \
    --model "${DATA_ROOT}/${model}" --method flash_attention_2 \
    --tasks "${RULER_TASKS}" \
    --max_seq_lengths "${lengths}" --max_length 131072 \
    --batch_size 1 --max_gen_toks 128 --limit "${LIMIT}" \
    --apply_chat_template \
    --output_path "${out}" > "${log}" 2>&1 &
  pids+=("$!"); specs+=("${spec}")
done

rc_total=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "  done: ${specs[$i]}"; else echo "  FAILED: ${specs[$i]}" >&2; rc_total=1; fi
done
echo "=== flash limit20: all 4 jobs finished (rc=${rc_total}) ==="
exit "${rc_total}"
