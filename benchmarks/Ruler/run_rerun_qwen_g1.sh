#!/bin/bash
# Rerun the single missing groups-ablation job: Qwen3-8B online group_bits=1.
# Original run was killed at 54% (no traceback) and the eval script only writes
# results.json/predictions.jsonl at the very end, so the result dir is empty.
# Reuses the proven wave2 dispatch pattern: wait for a GPU with >=THRESH_MIB
# free, run, and retry (up to MAX_RETRY) if the job dies without producing
# results.json (it was killed externally once before).
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
MODEL="Qwen3-8B"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
LIMIT="${LIMIT:-100}"
K_BITS=320; V_BITS=320; SINK=64; LOCAL=256; EXP_BITS=4; EAS=0.01; ZEROROPE=20
THRESH_MIB="${THRESH_MIB:-55000}"   # min free MiB before we place the job
POLL="${POLL:-30}"
MAX_RETRY="${MAX_RETRY:-3}"
TASKS="niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multiquery,niah_multivalue,ruler_vt,ruler_cwe,ruler_fwe,ruler_qa_squad,ruler_qa_hotpot"
TAG="picokv_online_k${K_BITS}_v${V_BITS}_g1"
OUT="results/test_groups/${MODEL}/${TAG}/"
LOG_ROOT="logs/test_eas005_groups"
LOG="${LOG_ROOT}/${MODEL}_${TAG}_rerun.log"
mkdir -p "${OUT}" "${LOG_ROOT}"

IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' '; }

run_once() {  # $1=gpu
  local gpu="$1"
  echo "  [gpu ${gpu}] ${MODEL} group=1 -> ${OUT} ($(date '+%F %T'))"
  CUDA_VISIBLE_DEVICES="${gpu}" PICOKV_ONLINE_GROUP_BITS="1" \
    python "${SCRIPT_DIR}/run_ruler_evaluation.py" \
    --model "${DATA_ROOT}/${MODEL}" --method picokv \
    --k-bits "${K_BITS}" --v-bits "${V_BITS}" --sink-size "${SINK}" --local-size "${LOCAL}" \
    --exp-bits "${EXP_BITS}" --eas "${EAS}" --zerorope "${ZEROROPE}" \
    --tasks "${TASKS}" \
    --max_seq_lengths 131072 --max_length 131072 \
    --batch_size 1 --max_gen_toks 128 --limit "${LIMIT}" \
    --apply_chat_template \
    --output_path "${OUT}" > "${LOG}" 2>&1
}

echo "=== rerun ${MODEL} g1 | free>=${THRESH_MIB}MiB | GPUs [${GPUS}] | retries<=${MAX_RETRY} ==="
rc=1
for attempt in $(seq 1 "${MAX_RETRY}"); do
  [ -s "${OUT}/results.json" ] && { echo "results.json already present, done."; exit 0; }
  echo "--- attempt ${attempt} ---"
  placed=""
  while [ -z "${placed}" ]; do
    for gpu in "${GPU_ARR[@]}"; do
      f="$(free_mib "$gpu")"; [ -z "${f}" ] && continue
      if [ "${f}" -ge "${THRESH_MIB}" ]; then placed="${gpu}"; break; fi
    done
    [ -z "${placed}" ] && { echo "  no GPU free (best: $(free_mib 0)MiB on gpu0), sleep ${POLL}s"; sleep "${POLL}"; }
  done
  run_once "${placed}"
  rc=$?
  if [ -s "${OUT}/results.json" ]; then
    echo "  done: results.json written (rc=${rc})"; exit 0
  fi
  echo "  FAILED on gpu ${placed} (rc=${rc}), no results.json" >&2
done
echo "=== gave up after ${MAX_RETRY} attempts ===" >&2
exit 1
