#!/bin/bash
# RULER 32k ablation: picokv ONLINE mode, sweep the algorithm's per-group solve
# width PICOKV_ONLINE_GROUP_BITS in {16,32,64,320} on Qwen3-8B, K_BITS=V_BITS=320.
# 320 is a multiple of each value (20/10/5/1 groups respectively) so the
# bits % ONLINE_GROUP_BITS == 0 check in online.py always passes.
# Other picokv params fixed at repo defaults: sink=64 local=256 exp=4 eas=0.01 zr=20.
# Offline mode is out of scope here: the fused CUDA decode kernel hardcodes
# group width 32 and the shipped offline_projection/*.pt files were fit at
# group width 32, so this knob only has an effect (and is only safe) online.
# 4 configs -> single batch, 4 GPUs used, GPUs 4-7 idle.
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

K_BITS="${K_BITS:-320}"; V_BITS="${V_BITS:-320}"
SINK="${SINK:-64}"; LOCAL="${LOCAL:-256}"
EXP_BITS="${EXP_BITS:-4}"; EAS="${EAS:-0.01}"; ZEROROPE="${ZEROROPE:-20}"
LIMIT="${LIMIT:-100}"; MAX_GEN_TOKS="${MAX_GEN_TOKS:-128}"
SEQ_LEN="${SEQ_LEN:-32768}"
TASKS="${TASKS:-niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multiquery,niah_multivalue,ruler_vt,ruler_cwe,ruler_fwe,ruler_qa_squad,ruler_qa_hotpot}"
RESULTS_ROOT="${RESULTS_ROOT:-results/ablation/ruler32k_groupbits}"
LOG_ROOT="${LOG_ROOT:-logs/ablation/ruler32k_groupbits}"

GROUP_BITS_VALUES=(16 32 64 320)

if [ ! -d "${MODEL_PATH}" ]; then
  echo "ERROR: model not found: ${MODEL_PATH}" >&2
  exit 1
fi
for gb in "${GROUP_BITS_VALUES[@]}"; do
  if [ $((K_BITS % gb)) -ne 0 ] || [ $((V_BITS % gb)) -ne 0 ]; then
    echo "ERROR: group_bits=${gb} does not divide K_BITS=${K_BITS}/V_BITS=${V_BITS}" >&2
    exit 1
  fi
done

IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
NGPU="${#GPU_ARR[@]}"
NCFG="${#GROUP_BITS_VALUES[@]}"

mkdir -p "${LOG_ROOT}"
echo "=== RULER ${SEQ_LEN} ablation | ${MODEL_TAG} | picokv online k${K_BITS}v${V_BITS} | ${NCFG} group_bits values on ${NGPU} GPUs ==="

rc_total=0
batch_start=0
batch_num=1
while [ "${batch_start}" -lt "${NCFG}" ]; do
  pids=()
  idxs=()
  for ((i = 0; i < NGPU && batch_start + i < NCFG; i++)); do
    idx=$((batch_start + i))
    gpu="${GPU_ARR[$i]}"
    gb="${GROUP_BITS_VALUES[$idx]}"
    tag="picokv_online_k${K_BITS}_v${V_BITS}_gb${gb}"
    out="${RESULTS_ROOT}/${MODEL_TAG}/${tag}/"
    log="${LOG_ROOT}/${MODEL_TAG}_${tag}.log"
    mkdir -p "${out}"
    echo "  batch${batch_num} gpu=${gpu} group_bits=${gb} tag=${tag}"
    CUDA_VISIBLE_DEVICES="${gpu}" PICOKV_ONLINE_GROUP_BITS="${gb}" python "${SCRIPT_DIR}/run_ruler_evaluation.py" \
      --model "${MODEL_PATH}" --method picokv \
      --k-bits "${K_BITS}" --v-bits "${V_BITS}" \
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
    gb="${GROUP_BITS_VALUES[$idx]}"
    if ! wait "${pids[$i]}"; then
      echo "  FAILED: group_bits=${gb} -- see ${LOG_ROOT}/${MODEL_TAG}_picokv_online_k${K_BITS}_v${V_BITS}_gb${gb}.log" >&2
      rc_total=1
    fi
  done
  echo "=== batch${batch_num} done (${#pids[@]} configs) ==="
  batch_start=$((batch_start + NGPU))
  batch_num=$((batch_num + 1))
done

exit "${rc_total}"
