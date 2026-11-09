#!/bin/bash
# BITS ablation, ONLINE picokv, RULER @128k (all 13 tasks), limit 100.
# 8 jobs = {Llama, Qwen} x {k320v256, k256v320, k256v256, k192v192}, one per GPU.
# Fixed: sink=64 local=256 exp_bits=4 eas=0.01 zerorope=20.
# Results: results/test_bits/{model}/picokv_online_k{K}_v{V}/
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
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
LIMIT="${LIMIT:-100}"
LENGTHS="131072"
TASKS="niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multiquery,niah_multivalue,ruler_vt,ruler_cwe,ruler_fwe,ruler_qa_squad,ruler_qa_hotpot"
SINK=64; LOCAL=256; EXP_BITS=4; EAS=0.01; ZEROROPE=20
RESULTS_ROOT="results/test_bits"
LOG_ROOT="logs/test_bits"
mkdir -p "${LOG_ROOT}"

# "model;k_bits;v_bits"
JOBS=(
  "Llama-3.1-8B-Instruct;320;256"
  "Llama-3.1-8B-Instruct;256;320"
  "Llama-3.1-8B-Instruct;256;256"
  "Llama-3.1-8B-Instruct;192;192"
  "Qwen3-8B;320;256"
  "Qwen3-8B;256;320"
  "Qwen3-8B;256;256"
  "Qwen3-8B;192;192"
)

IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
NGPU="${#GPU_ARR[@]}"; NJOB="${#JOBS[@]}"
[ "${NGPU}" -ge "${NJOB}" ] || { echo "ERROR: need ${NJOB} GPUs, got ${NGPU} (GPUS=${GPUS})" >&2; exit 1; }
for m in Llama-3.1-8B-Instruct Qwen3-8B; do
  [ -d "${DATA_ROOT}/${m}" ] || { echo "ERROR: model not found: ${DATA_ROOT}/${m}" >&2; exit 1; }
done

echo "=== BITS ablation online | RULER @128k | limit=${LIMIT} | ${NJOB} jobs on GPUs [${GPUS}] ==="
pids=(); specs=()
for i in "${!JOBS[@]}"; do
  gpu="${GPU_ARR[$i]}"
  IFS=';' read -r model kb vb <<< "${JOBS[$i]}"
  tag="picokv_online_k${kb}_v${vb}"
  out="${RESULTS_ROOT}/${model}/${tag}/"
  log="${LOG_ROOT}/${model}_${tag}.log"
  mkdir -p "${out}"
  echo "  [gpu ${gpu}] model=${model} k=${kb} v=${vb} -> ${out}"
  CUDA_VISIBLE_DEVICES="${gpu}" python "${SCRIPT_DIR}/run_ruler_evaluation.py" \
    --model "${DATA_ROOT}/${model}" --method picokv \
    --k-bits "${kb}" --v-bits "${vb}" --sink-size "${SINK}" --local-size "${LOCAL}" \
    --exp-bits "${EXP_BITS}" --eas "${EAS}" --zerorope "${ZEROROPE}" \
    --tasks "${TASKS}" \
    --max_seq_lengths "${LENGTHS}" --max_length 131072 \
    --batch_size 1 --max_gen_toks 128 --limit "${LIMIT}" \
    --apply_chat_template \
    --output_path "${out}" > "${log}" 2>&1 &
  pids+=("$!"); specs+=("${JOBS[$i]}")
done

rc_total=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "  done: ${specs[$i]}"; else echo "  FAILED: ${specs[$i]}" >&2; rc_total=1; fi
done
echo "=== all ${NJOB} jobs finished (rc=${rc_total}) ==="
exit "${rc_total}"
