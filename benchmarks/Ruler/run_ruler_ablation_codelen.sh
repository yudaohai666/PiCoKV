#!/bin/bash
# RULER 32k ablation: picokv online, sweep K_BITS x V_BITS in {192,256,320,384}
# (16 configs) on Qwen3-8B. 8 GPUs, one config per GPU, 2 sequential batches.
# Other picokv params fixed at repo defaults: sink=64 local=256 exp=4 eas=0.01 zr=20.
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

SINK="${SINK:-64}"; LOCAL="${LOCAL:-256}"
EXP_BITS="${EXP_BITS:-4}"; EAS="${EAS:-0.01}"; ZEROROPE="${ZEROROPE:-20}"
LIMIT="${LIMIT:-100}"; MAX_GEN_TOKS="${MAX_GEN_TOKS:-128}"
SEQ_LEN="${SEQ_LEN:-32768}"
TASKS="${TASKS:-niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multiquery,niah_multivalue,ruler_vt,ruler_cwe,ruler_fwe,ruler_qa_squad,ruler_qa_hotpot}"
RESULTS_ROOT="${RESULTS_ROOT:-results/ablation/ruler32k_codelen}"
LOG_ROOT="${LOG_ROOT:-logs/ablation/ruler32k_codelen}"

BIT_VALUES=(192 256 320 384)

if [ ! -d "${MODEL_PATH}" ]; then
  echo "ERROR: model not found: ${MODEL_PATH}" >&2
  exit 1
fi

IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
NGPU="${#GPU_ARR[@]}"

# Build all 16 (k,v) combos.
CONFIGS=()
for k in "${BIT_VALUES[@]}"; do
  for v in "${BIT_VALUES[@]}"; do
    CONFIGS+=("${k},${v}")
  done
done
NCFG="${#CONFIGS[@]}"

mkdir -p "${LOG_ROOT}"
echo "=== RULER ${SEQ_LEN} ablation | ${MODEL_TAG} | picokv online | ${NCFG} k/v combos on ${NGPU} GPUs ==="

rc_total=0
batch_start=0
batch_num=1
while [ "${batch_start}" -lt "${NCFG}" ]; do
  pids=()
  idxs=()
  for ((i = 0; i < NGPU && batch_start + i < NCFG; i++)); do
    idx=$((batch_start + i))
    gpu="${GPU_ARR[$i]}"
    IFS=',' read -r k_bits v_bits <<< "${CONFIGS[$idx]}"
    tag="picokv_online_k${k_bits}_v${v_bits}"
    out="${RESULTS_ROOT}/${MODEL_TAG}/${tag}/"
    log="${LOG_ROOT}/${MODEL_TAG}_${tag}.log"
    mkdir -p "${out}"
    echo "  batch${batch_num} gpu=${gpu} k=${k_bits} v=${v_bits} tag=${tag}"
    CUDA_VISIBLE_DEVICES="${gpu}" python "${SCRIPT_DIR}/run_ruler_evaluation.py" \
      --model "${MODEL_PATH}" --method picokv \
      --k-bits "${k_bits}" --v-bits "${v_bits}" \
      --sink-size "${SINK}" --local-size "${LOCAL}" \
      --exp-bits "${EXP_BITS}" --eas "${EAS}" --zerorope "${ZEROROPE}" \
      --tasks "${TASKS}" \
      --max_seq_lengths "${SEQ_LEN}" --max_length "${SEQ_LEN}" \
      --batch_size 1 --max_gen_toks "${MAX_GEN_TOKS}" --limit "${LIMIT}" \
      --apply_chat_template \
      --output_path "${out}" > "${log}" 2>&1 &
    pids+=("$!")
    idxs+=("${idx}")
  done
  for i in "${!pids[@]}"; do
    idx="${idxs[$i]}"
    IFS=',' read -r k_bits v_bits <<< "${CONFIGS[$idx]}"
    if ! wait "${pids[$i]}"; then
      echo "  FAILED: k=${k_bits} v=${v_bits} -- see ${LOG_ROOT}/${MODEL_TAG}_picokv_online_k${k_bits}_v${v_bits}.log" >&2
      rc_total=1
    fi
  done
  echo "=== batch${batch_num} done (${#pids[@]} configs) ==="
  batch_start=$((batch_start + NGPU))
  batch_num=$((batch_num + 1))
done

exit "${rc_total}"
