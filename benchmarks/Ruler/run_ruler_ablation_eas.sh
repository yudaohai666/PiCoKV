#!/bin/bash
# RULER 32k ablation: picokv online+offline, sweep EAS in {0.01,0.001,1e-7}
# (2 modes x 3 eas = 6 configs) on Qwen3-8B, K_BITS=320 V_BITS=320.
# Other picokv params fixed at repo defaults: sink=64 local=256 exp=4 zr=20.
# 8 GPUs available but only 6 configs -> single batch, GPUs 6/7 idle.
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
EXP_BITS="${EXP_BITS:-4}"; ZEROROPE="${ZEROROPE:-20}"
LIMIT="${LIMIT:-100}"; MAX_GEN_TOKS="${MAX_GEN_TOKS:-128}"
SEQ_LEN="${SEQ_LEN:-32768}"
TASKS="${TASKS:-niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multiquery,niah_multivalue,ruler_vt,ruler_cwe,ruler_fwe,ruler_qa_squad,ruler_qa_hotpot}"
RESULTS_ROOT="${RESULTS_ROOT:-results/ablation/ruler32k_eas}"
LOG_ROOT="${LOG_ROOT:-logs/ablation/ruler32k_eas}"

EAS_VALUES=(0.01 0.001 1e-7)
MODES=(online offline)

if [ ! -d "${MODEL_PATH}" ]; then
  echo "ERROR: model not found: ${MODEL_PATH}" >&2
  exit 1
fi

IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
NGPU="${#GPU_ARR[@]}"

# Build all 6 (mode,eas) combos: 2 modes x 3 eas values.
CONFIGS=()
for mode in "${MODES[@]}"; do
  for eas in "${EAS_VALUES[@]}"; do
    CONFIGS+=("${mode},${eas}")
  done
done
NCFG="${#CONFIGS[@]}"

mkdir -p "${LOG_ROOT}"
echo "=== RULER ${SEQ_LEN} ablation | ${MODEL_TAG} | picokv k${K_BITS}v${V_BITS} | ${NCFG} mode/eas combos on ${NGPU} GPUs ==="

rc_total=0
batch_start=0
batch_num=1
while [ "${batch_start}" -lt "${NCFG}" ]; do
  pids=()
  idxs=()
  for ((i = 0; i < NGPU && batch_start + i < NCFG; i++)); do
    idx=$((batch_start + i))
    gpu="${GPU_ARR[$i]}"
    IFS=',' read -r mode eas <<< "${CONFIGS[$idx]}"
    tag="picokv_${mode}_k${K_BITS}_v${V_BITS}_eas${eas}"
    out="${RESULTS_ROOT}/${MODEL_TAG}/${tag}/"
    log="${LOG_ROOT}/${MODEL_TAG}_${tag}.log"
    mkdir -p "${out}"
    mode_args=()
    [ "${mode}" = "offline" ] && mode_args+=(--offline)
    echo "  batch${batch_num} gpu=${gpu} mode=${mode} eas=${eas} tag=${tag}"
    CUDA_VISIBLE_DEVICES="${gpu}" python "${SCRIPT_DIR}/run_ruler_evaluation.py" \
      --model "${MODEL_PATH}" --method picokv \
      --k-bits "${K_BITS}" --v-bits "${V_BITS}" \
      --sink-size "${SINK}" --local-size "${LOCAL}" \
      --exp-bits "${EXP_BITS}" --eas "${eas}" --zerorope "${ZEROROPE}" \
      "${mode_args[@]}" \
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
    IFS=',' read -r mode eas <<< "${CONFIGS[$idx]}"
    if ! wait "${pids[$i]}"; then
      echo "  FAILED: mode=${mode} eas=${eas} -- see ${LOG_ROOT}/${MODEL_TAG}_picokv_${mode}_k${K_BITS}_v${V_BITS}_eas${eas}.log" >&2
      rc_total=1
    fi
  done
  echo "=== batch${batch_num} done (${#pids[@]} configs) ==="
  batch_start=$((batch_start + NGPU))
  batch_num=$((batch_num + 1))
done

exit "${rc_total}"
