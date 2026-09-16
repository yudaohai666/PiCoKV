#!/bin/bash
# RULER length ablation: picokv ONLINE mode, sweep PICOKV_ONLINE_GROUP_BITS in
# {16,32,64,128} (all <= head_dim=128, the safe range -- see constants.py and
# run_ruler_ablation_groupbits.sh's postmortem: group_bits > head_dim silently
# breaks the online Householder-QR step) crossed with two length "buckets":
# short (8k,16k,32k run together as one process, one length after another) and
# long (64k alone, since it is by far the slowest and should not block/compete
# with the short lengths on the same process).
# 4 group_bits x 2 length-buckets = 8 configs -> 8 GPUs, one config per GPU.
# K_BITS=V_BITS=256, sink=64 local=256 exp=4 eas=0.01 zr=20 (repo defaults
# except k/v bits, per this ablation's request).
# Qwen3-8B's native context is 40960; 64k is served via the YaRN rope scaling
# already wired into picokv/model_config.load_model_config() (factor=4.0,
# original_max_position_embeddings=32768, extended max_position_embeddings=
# 131072), applied automatically for qwen3 configs in benchmarks/common.py.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
# See run_ruler.sh: fragmentation fix for the block-32 online solve's fp32
# code-matrix churn. Must precede torch's allocator init.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# lm_eval's ruler_qa_squad task downloads dev-v2.0.json; requests reads these.
export http_proxy="${http_proxy:-http://agent.baidu.com:8891}"
export https_proxy="${https_proxy:-http://agent.baidu.com:8891}"

DATA_ROOT="${DATA_ROOT:-/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data}"
MODEL_TAG="${MODEL_TAG:-Qwen3-8B}"
MODEL_PATH="${DATA_ROOT}/${MODEL_TAG}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"

K_BITS="${K_BITS:-256}"; V_BITS="${V_BITS:-256}"
SINK="${SINK:-64}"; LOCAL="${LOCAL:-256}"
EXP_BITS="${EXP_BITS:-4}"; EAS="${EAS:-0.01}"; ZEROROPE="${ZEROROPE:-20}"
LIMIT="${LIMIT:-100}"; MAX_GEN_TOKS="${MAX_GEN_TOKS:-128}"
TASKS="${TASKS:-niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multiquery,niah_multivalue,ruler_vt,ruler_cwe,ruler_fwe,ruler_qa_squad,ruler_qa_hotpot}"
RESULTS_ROOT="${RESULTS_ROOT:-results/ablation/ruler_lengths_groupbits}"
LOG_ROOT="${LOG_ROOT:-logs/ablation/ruler_lengths_groupbits}"

GROUP_BITS_VALUES=(16 32 64 128)
# length-bucket name -> comma-separated max_seq_lengths list; MAX_LENGTH is
# per-bucket so the model's attention/kv buffers are only sized for what that
# process actually needs.
SHORT_LENGTHS="8192,16384,32768"
SHORT_MAX_LENGTH=32768
LONG_LENGTHS="65536"
LONG_MAX_LENGTH=65536
BUCKETS=(short long)

if [ ! -d "${MODEL_PATH}" ]; then
  echo "ERROR: model not found: ${MODEL_PATH}" >&2
  exit 1
fi
for gb in "${GROUP_BITS_VALUES[@]}"; do
  if [ $((K_BITS % gb)) -ne 0 ] || [ $((V_BITS % gb)) -ne 0 ]; then
    echo "ERROR: group_bits=${gb} does not divide K_BITS=${K_BITS}/V_BITS=${V_BITS}" >&2
    exit 1
  fi
  if [ "${gb}" -gt 128 ]; then
    echo "ERROR: group_bits=${gb} > head_dim(128) is unsafe in online mode (silent QR truncation, see constants.py/online.py)" >&2
    exit 1
  fi
done

IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
NGPU="${#GPU_ARR[@]}"

# Build all 8 (group_bits, bucket) combos: 4 group_bits x 2 length buckets.
CONFIGS=()
for gb in "${GROUP_BITS_VALUES[@]}"; do
  for bucket in "${BUCKETS[@]}"; do
    CONFIGS+=("${gb},${bucket}")
  done
done
NCFG="${#CONFIGS[@]}"

mkdir -p "${LOG_ROOT}"
echo "=== RULER length ablation | ${MODEL_TAG} | picokv online k${K_BITS}v${V_BITS} | ${NCFG} group_bits/length-bucket combos on ${NGPU} GPUs ==="

rc_total=0
batch_start=0
batch_num=1
while [ "${batch_start}" -lt "${NCFG}" ]; do
  pids=()
  idxs=()
  for ((i = 0; i < NGPU && batch_start + i < NCFG; i++)); do
    idx=$((batch_start + i))
    gpu="${GPU_ARR[$i]}"
    IFS=',' read -r gb bucket <<< "${CONFIGS[$idx]}"
    if [ "${bucket}" = "short" ]; then
      seq_lengths="${SHORT_LENGTHS}"; max_length="${SHORT_MAX_LENGTH}"
    else
      seq_lengths="${LONG_LENGTHS}"; max_length="${LONG_MAX_LENGTH}"
    fi
    tag="picokv_online_k${K_BITS}_v${V_BITS}_gb${gb}_${bucket}"
    out="${RESULTS_ROOT}/${MODEL_TAG}/${tag}/"
    log="${LOG_ROOT}/${MODEL_TAG}_${tag}.log"
    mkdir -p "${out}"
    echo "  batch${batch_num} gpu=${gpu} group_bits=${gb} bucket=${bucket} lengths=${seq_lengths} tag=${tag}"
    CUDA_VISIBLE_DEVICES="${gpu}" PICOKV_ONLINE_GROUP_BITS="${gb}" python "${SCRIPT_DIR}/run_ruler_evaluation.py" \
      --model "${MODEL_PATH}" --method picokv \
      --k-bits "${K_BITS}" --v-bits "${V_BITS}" \
      --sink-size "${SINK}" --local-size "${LOCAL}" \
      --exp-bits "${EXP_BITS}" --eas "${EAS}" --zerorope "${ZEROROPE}" \
      --tasks "${TASKS}" \
      --max_seq_lengths "${seq_lengths}" --max_length "${max_length}" \
      --batch_size 1 --max_gen_toks "${MAX_GEN_TOKS}" --limit "${LIMIT}" \
      --apply_chat_template \
      --output_path "${out}" > "${log}" 2>&1 &
    pids+=("$!")
    idxs+=("${idx}")
  done
  for i in "${!pids[@]}"; do
    idx="${idxs[$i]}"
    IFS=',' read -r gb bucket <<< "${CONFIGS[$idx]}"
    if ! wait "${pids[$i]}"; then
      echo "  FAILED: group_bits=${gb} bucket=${bucket} -- see ${LOG_ROOT}/${MODEL_TAG}_picokv_online_k${K_BITS}_v${V_BITS}_gb${gb}_${bucket}.log" >&2
      rc_total=1
    fi
  done
  echo "=== batch${batch_num} done (${#pids[@]} configs) ==="
  batch_start=$((batch_start + NGPU))
  batch_num=$((batch_num + 1))
done

exit "${rc_total}"
