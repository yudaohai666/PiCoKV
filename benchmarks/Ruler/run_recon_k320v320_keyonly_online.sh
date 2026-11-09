#!/bin/bash
# picokv_recon ablation: K=320 V=320, quant_target=key, ONLINE only (offline
# projections are baked at zerorope=20, so offline+zr0 would be a train/infer
# mismatch -- online learns directions from the current keys, so zr0/zr20 are
# both self-consistent). Llama + Qwen, RULER @128k, limit 100.
# 4 jobs = {Llama, Qwen} x {zerorope=20 (default), zerorope=0}, one per GPU.
# Results: results/test_recon/{model}/picokv_recon_online_k320_v320_key[_zr0]/
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
GPUS="${GPUS:-0,1,2,3}"
LIMIT="${LIMIT:-100}"
LENGTHS="131072"
TASKS="ruler_cwe,niah_multivalue,niah_multikey_2,niah_multikey_3,ruler_fwe"
K_BITS=320; V_BITS=320
RESULTS_ROOT="results/test_recon"
LOG_ROOT="logs/test_recon"
mkdir -p "${LOG_ROOT}"

# "model;zerorope"
JOBS=(
  "Llama-3.1-8B-Instruct;20"
  "Llama-3.1-8B-Instruct;0"
  "Qwen3-8B;20"
  "Qwen3-8B;0"
)

IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
NGPU="${#GPU_ARR[@]}"; NJOB="${#JOBS[@]}"
[ "${NGPU}" -ge "${NJOB}" ] || { echo "ERROR: need ${NJOB} GPUs, got ${NGPU} (GPUS=${GPUS})" >&2; exit 1; }
for m in Llama-3.1-8B-Instruct Qwen3-8B; do
  [ -d "${DATA_ROOT}/${m}" ] || { echo "ERROR: model not found: ${DATA_ROOT}/${m}" >&2; exit 1; }
done

echo "=== picokv_recon key-only online k${K_BITS}v${V_BITS} | RULER @128k | limit=${LIMIT} | ${NJOB} jobs on GPUs [${GPUS}] ==="
pids=(); specs=()
for i in "${!JOBS[@]}"; do
  gpu="${GPU_ARR[$i]}"
  IFS=';' read -r model zr <<< "${JOBS[$i]}"
  tag="picokv_recon_online_k${K_BITS}_v${V_BITS}_key"
  [ "${zr}" != "20" ] && tag="${tag}_zr${zr}"
  out="${RESULTS_ROOT}/${model}/${tag}/"
  log="${LOG_ROOT}/${model}_${tag}.log"
  mkdir -p "${out}"
  echo "  [gpu ${gpu}] model=${model} zerorope=${zr} -> ${out}"
  CUDA_VISIBLE_DEVICES="${gpu}" python "${SCRIPT_DIR}/run_ruler_evaluation.py" \
    --model "${DATA_ROOT}/${model}" --method picokv_recon --quant-target key \
    --k-bits "${K_BITS}" --v-bits "${V_BITS}" --sink-size 64 --local-size 256 \
    --exp-bits 4 --eas 0.01 --zerorope "${zr}" \
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
