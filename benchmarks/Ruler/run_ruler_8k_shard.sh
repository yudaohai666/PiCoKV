#!/bin/bash
# RULER 8k verification for picokv offline, sharded one-task-group-per-GPU.
#
# A single length (8192) cannot be split across GPUs the way run_ruler.sh's
# multi-length sweep can, so the 13 tasks are split into NGPU groups instead.
# Each shard gets its own OUTPUT_PATH: run_ruler_evaluation.py derives its
# results dir from (model, tag) alone, so without the override every shard
# would write the same results.json and the last writer would win.
#
# Params match the stored results/ruler-97-1 8k runs so the numbers are
# comparable: offline k320/v320, sink 64, local 256, exp 4, eas 0.01, zr 20.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
# See run_ruler.sh: fragmentation fix for the block-32 solve's fp32 code-matrix
# churn. Must precede torch's allocator init, so it cannot move into python.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# lm_eval's ruler qa_squad task downloads dev-v2.0.json from rajpurkar.github.io
# (qa_utils.py:46) with no local-file escape hatch; requests reads these.
export http_proxy="${http_proxy:-http://agent.baidu.com:8891}"
export https_proxy="${https_proxy:-http://agent.baidu.com:8891}"

DATA_ROOT="${DATA_ROOT:-/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data}"
MODELS="${MODELS:-Llama-3.1-8B-Instruct,Qwen3-8B}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"

K_BITS="${K_BITS:-320}"; V_BITS="${V_BITS:-320}"
SINK="${SINK:-64}"; LOCAL="${LOCAL:-256}"
EXP_BITS="${EXP_BITS:-4}"; EAS="${EAS:-0.01}"; ZEROROPE="${ZEROROPE:-20}"
LIMIT="${LIMIT:-100}"; MAX_GEN_TOKS="${MAX_GEN_TOKS:-128}"
SEQ_LEN="${SEQ_LEN:-8192}"
METHOD="${METHOD:-picokv}"
RESULTS_ROOT="${RESULTS_ROOT:-results/ruler_8k}"
LOG_ROOT="${LOG_ROOT:-logs/ruler_8k}"

# 13 tasks -> 8 groups. ruler_qa_squad sits alone in its group so only one
# process hits the SQuAD download.
SHARDS=(
  "niah_single_1,niah_single_2"
  "niah_single_3,niah_multikey_1"
  "niah_multikey_2,niah_multikey_3"
  "niah_multiquery,niah_multivalue"
  "ruler_vt"
  "ruler_cwe"
  "ruler_fwe"
  "ruler_qa_squad,ruler_qa_hotpot"
)

IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
IFS=',' read -r -a MODEL_ARR <<< "${MODELS}"
NGPU="${#GPU_ARR[@]}"
NSHARD="${#SHARDS[@]}"
if [ "${NSHARD}" -gt "${NGPU}" ]; then
  echo "ERROR: ${NSHARD} shards but only ${NGPU} GPUs; shards would collide." >&2
  exit 1
fi

if [ "${METHOD}" = "picokv" ]; then
  METHOD_ARGS=(--offline --k-bits "${K_BITS}" --v-bits "${V_BITS}"
    --sink-size "${SINK}" --local-size "${LOCAL}"
    --exp-bits "${EXP_BITS}" --eas "${EAS}" --zerorope "${ZEROROPE}")
  TAG="picokv_offline_k${K_BITS}_v${V_BITS}"
else
  METHOD_ARGS=()
  TAG="dense_${METHOD}"
fi

mkdir -p "${LOG_ROOT}"
rc_total=0

for model_tag in "${MODEL_ARR[@]}"; do
  model_path="${DATA_ROOT}/${model_tag}"
  if [ ! -d "${model_path}" ]; then
    echo "ERROR: model not found: ${model_path}" >&2
    rc_total=1
    continue
  fi
  echo "=== RULER ${SEQ_LEN} | ${model_tag} | ${TAG} | ${NSHARD} shards on GPU ${GPUS} ==="
  pids=()
  for i in "${!SHARDS[@]}"; do
    gpu="${GPU_ARR[$i]}"
    tasks="${SHARDS[$i]}"
    out="${RESULTS_ROOT}/${model_tag}/${TAG}/shard$((i + 1))"
    log="${LOG_ROOT}/${model_tag}_shard$((i + 1)).log"
    mkdir -p "${out}"
    echo "  shard$((i + 1)) gpu=${gpu} tasks=${tasks}"
    CUDA_VISIBLE_DEVICES="${gpu}" python "${SCRIPT_DIR}/run_ruler_evaluation.py" \
      --model "${model_path}" --method "${METHOD}" \
      --tasks "${tasks}" \
      --max_seq_lengths "${SEQ_LEN}" --max_length "${SEQ_LEN}" \
      --batch_size 1 --max_gen_toks "${MAX_GEN_TOKS}" --limit "${LIMIT}" \
      --apply_chat_template \
      --output_path "${out}/" \
      "${METHOD_ARGS[@]}" > "${log}" 2>&1 &
    pids+=("$!")
  done
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      echo "  FAILED: ${model_tag} shard$((i + 1)) -- see ${LOG_ROOT}/${model_tag}_shard$((i + 1)).log" >&2
      rc_total=1
    fi
  done
  echo "=== ${model_tag} done ==="
done

exit "${rc_total}"
