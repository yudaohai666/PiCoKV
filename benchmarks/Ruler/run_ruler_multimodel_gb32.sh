#!/bin/bash
# RULER multi-model sweep: picokv ONLINE mode, PICOKV_ONLINE_GROUP_BITS=32
# (the safe default -- see run_ruler_ablation_groupbits_lengths.sh's postmortem:
# group_bits > head_dim(128) silently breaks the online Householder-QR step).
# K_BITS=V_BITS=256, sink=64 local=256 exp=4 eas=0.01 zr=20 (repo defaults
# except k/v bits, per this ablation's request).
#
# 8 GPUs, 8 configs:
#   Llama-3.1-8B-Instruct: 8k+16k+32k (1 task, 1 GPU)
#                           64k        (1 task, 1 GPU)
#                           128k split into 3 task-groups of the 13 RULER
#                             tasks (3 tasks, 3 GPUs) -- 128k on a single
#                             process with all 13 tasks would be by far the
#                             slowest config in this sweep, so it is sharded
#                             across GPUs by task list instead of by length.
#   Qwen3-8B:               128k split into 3 task-groups, same scheme
#                             (8k/16k/32k/64k for Qwen3-8B were already run in
#                             results/ablation/ruler_lengths_groupbits/ at
#                             gb=32 -- not repeated here).
#
# Qwen3-8B's native context is 40960; 128k is served via the YaRN rope scaling
# already wired into picokv/model_config.load_model_config() (factor=4.0,
# original_max_position_embeddings=32768, extended max_position_embeddings=
# 131072), applied automatically for qwen3 configs in benchmarks/common.py.
# Llama-3.1-8B-Instruct's own config.json already ships rope_scaling (llama3
# type, factor=8.0) with max_position_embeddings=131072, so 128k needs no
# override there.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck disable=SC1091
source "${REPO_ROOT}/.venv/bin/activate"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
# See run_ruler.sh: fragmentation fix for the block-32 online solve's fp32
# code-matrix churn. Must precede torch's allocator init.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# lm_eval's ruler_qa_squad task downloads dev-v2.0.json; requests reads these.
export http_proxy="${http_proxy:-http://agent.baidu.com:8891}"
export https_proxy="${https_proxy:-http://agent.baidu.com:8891}"
export PICOKV_ONLINE_GROUP_BITS=32

DATA_ROOT="${DATA_ROOT:-/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"

K_BITS="${K_BITS:-256}"; V_BITS="${V_BITS:-256}"
SINK="${SINK:-64}"; LOCAL="${LOCAL:-256}"
EXP_BITS="${EXP_BITS:-4}"; EAS="${EAS:-0.01}"; ZEROROPE="${ZEROROPE:-20}"
LIMIT="${LIMIT:-100}"; MAX_GEN_TOKS="${MAX_GEN_TOKS:-128}"
RESULTS_ROOT="${RESULTS_ROOT:-results/ablation/ruler_multimodel_gb32}"
LOG_ROOT="${LOG_ROOT:-logs/ablation/ruler_multimodel_gb32}"

ALL_TASKS=(niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multiquery niah_multivalue ruler_vt ruler_cwe ruler_fwe ruler_qa_squad ruler_qa_hotpot)
ALL_TASKS_CSV="$(IFS=,; echo "${ALL_TASKS[*]}")"
# 13 tasks -> groups of 5/4/4.
TASKGROUP_0="${ALL_TASKS[0]},${ALL_TASKS[1]},${ALL_TASKS[2]},${ALL_TASKS[3]},${ALL_TASKS[4]}"
TASKGROUP_1="${ALL_TASKS[5]},${ALL_TASKS[6]},${ALL_TASKS[7]},${ALL_TASKS[8]}"
TASKGROUP_2="${ALL_TASKS[9]},${ALL_TASKS[10]},${ALL_TASKS[11]},${ALL_TASKS[12]}"

for m in Llama-3.1-8B-Instruct Qwen3-8B; do
  if [ ! -d "${DATA_ROOT}/${m}" ]; then
    echo "ERROR: model not found: ${DATA_ROOT}/${m}" >&2
    exit 1
  fi
done
if [ $((K_BITS % 32)) -ne 0 ] || [ $((V_BITS % 32)) -ne 0 ]; then
  echo "ERROR: group_bits=32 does not divide K_BITS=${K_BITS}/V_BITS=${V_BITS}" >&2
  exit 1
fi

IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
NGPU="${#GPU_ARR[@]}"

# Build the 8 (model,tag,tasks,lengths,max_length) configs.
# Fields are semicolon-joined since tasks/lengths already use commas.
CONFIGS=(
  "Llama-3.1-8B-Instruct;llama_gb32_short;${ALL_TASKS_CSV};8192,16384,32768;32768"
  "Llama-3.1-8B-Instruct;llama_gb32_long64k;${ALL_TASKS_CSV};65536;65536"
  "Llama-3.1-8B-Instruct;llama_gb32_128k_tg0;${TASKGROUP_0};131072;131072"
  "Llama-3.1-8B-Instruct;llama_gb32_128k_tg1;${TASKGROUP_1};131072;131072"
  "Llama-3.1-8B-Instruct;llama_gb32_128k_tg2;${TASKGROUP_2};131072;131072"
  "Qwen3-8B;qwen3_gb32_128k_tg0;${TASKGROUP_0};131072;131072"
  "Qwen3-8B;qwen3_gb32_128k_tg1;${TASKGROUP_1};131072;131072"
  "Qwen3-8B;qwen3_gb32_128k_tg2;${TASKGROUP_2};131072;131072"
)
NCFG="${#CONFIGS[@]}"

mkdir -p "${LOG_ROOT}"
echo "=== RULER multi-model gb32 sweep | picokv online k${K_BITS}v${V_BITS} | ${NCFG} configs on ${NGPU} GPUs ==="

rc_total=0
batch_start=0
batch_num=1
while [ "${batch_start}" -lt "${NCFG}" ]; do
  pids=()
  idxs=()
  for ((i = 0; i < NGPU && batch_start + i < NCFG; i++)); do
    idx=$((batch_start + i))
    gpu="${GPU_ARR[$i]}"
    IFS=';' read -r model tag tasks lengths max_length <<< "${CONFIGS[$idx]}"
    model_path="${DATA_ROOT}/${model}"
    out="${RESULTS_ROOT}/${model}/${tag}/"
    log="${LOG_ROOT}/${model}_${tag}.log"
    mkdir -p "${out}"
    echo "  batch${batch_num} gpu=${gpu} model=${model} tag=${tag} lengths=${lengths} max_length=${max_length}"
    CUDA_VISIBLE_DEVICES="${gpu}" python "${SCRIPT_DIR}/run_ruler_evaluation.py" \
      --model "${model_path}" --method picokv \
      --k-bits "${K_BITS}" --v-bits "${V_BITS}" \
      --sink-size "${SINK}" --local-size "${LOCAL}" \
      --exp-bits "${EXP_BITS}" --eas "${EAS}" --zerorope "${ZEROROPE}" \
      --tasks "${tasks}" \
      --max_seq_lengths "${lengths}" --max_length "${max_length}" \
      --batch_size 1 --max_gen_toks "${MAX_GEN_TOKS}" --limit "${LIMIT}" \
      --apply_chat_template \
      --output_path "${out}" > "${log}" 2>&1 &
    pids+=("$!")
    idxs+=("${idx}")
  done
  for i in "${!pids[@]}"; do
    idx="${idxs[$i]}"
    IFS=';' read -r model tag tasks lengths max_length <<< "${CONFIGS[$idx]}"
    if ! wait "${pids[$i]}"; then
      echo "  FAILED: model=${model} tag=${tag} -- see ${LOG_ROOT}/${model}_${tag}.log" >&2
      rc_total=1
    fi
  done
  echo "=== batch${batch_num} done (${#pids[@]} configs) ==="
  batch_start=$((batch_start + NGPU))
  batch_num=$((batch_num + 1))
done

exit "${rc_total}"
