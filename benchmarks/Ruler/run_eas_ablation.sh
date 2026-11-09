#!/bin/bash
# EAS ablation @ k320 v320, ONLINE picokv, RULER @128k (all 13 tasks), limit 100.
# Default eas=0.01 already exists; ablate {0.02, 0.001, 0.0001, 1e-7}.
# 8 jobs = {Llama, Qwen} x 4 eas values, one per GPU.
# Fixed: sink=64 local=256 exp_bits=4 zerorope=20 k=320 v=320.
# Results: results/test_eas/{model}/picokv_online_k320_v320_eas{eas}/
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
K_BITS=320; V_BITS=320
SINK=64; LOCAL=256; EXP_BITS=4; ZEROROPE=20
RESULTS_ROOT="results/test_eas"
LOG_ROOT="logs/test_eas"
mkdir -p "${LOG_ROOT}"

# "model;eas"  (default eas=0.01 skipped; already run)
JOBS=(
  "Llama-3.1-8B-Instruct;0.02"
  "Llama-3.1-8B-Instruct;0.001"
  "Llama-3.1-8B-Instruct;0.0001"
  "Llama-3.1-8B-Instruct;1e-7"
  "Qwen3-8B;0.02"
  "Qwen3-8B;0.001"
  "Qwen3-8B;0.0001"
  "Qwen3-8B;1e-7"
)

IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
NGPU="${#GPU_ARR[@]}"; NJOB="${#JOBS[@]}"
[ "${NGPU}" -ge "${NJOB}" ] || { echo "ERROR: need ${NJOB} GPUs, got ${NGPU} (GPUS=${GPUS})" >&2; exit 1; }
for m in Llama-3.1-8B-Instruct Qwen3-8B; do
  [ -d "${DATA_ROOT}/${m}" ] || { echo "ERROR: model not found: ${DATA_ROOT}/${m}" >&2; exit 1; }
done

echo "=== EAS ablation k${K_BITS}v${V_BITS} online | RULER @128k | limit=${LIMIT} | ${NJOB} jobs on GPUs [${GPUS}] ==="
pids=(); specs=()
for i in "${!JOBS[@]}"; do
  gpu="${GPU_ARR[$i]}"
  IFS=';' read -r model eas <<< "${JOBS[$i]}"
  tag="picokv_online_k${K_BITS}_v${V_BITS}_eas${eas}"
  out="${RESULTS_ROOT}/${model}/${tag}/"
  log="${LOG_ROOT}/${model}_${tag}.log"
  mkdir -p "${out}"
  echo "  [gpu ${gpu}] model=${model} eas=${eas} -> ${out}"
  CUDA_VISIBLE_DEVICES="${gpu}" python "${SCRIPT_DIR}/run_ruler_evaluation.py" \
    --model "${DATA_ROOT}/${model}" --method picokv \
    --k-bits "${K_BITS}" --v-bits "${V_BITS}" --sink-size "${SINK}" --local-size "${LOCAL}" \
    --exp-bits "${EXP_BITS}" --eas "${eas}" --zerorope "${ZEROROPE}" \
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
