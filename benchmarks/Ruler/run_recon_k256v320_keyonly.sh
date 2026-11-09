#!/bin/bash
# picokv_recon ablation: K=256 V=320, quant_target=key (only K quantized+recon,
# dequant+flash, no CUDA kernel). Llama-3.1-8B-Instruct, RULER @128k, limit 100.
# 4 jobs = {online, offline} x {zerorope=20 (default), zerorope=0}, one per GPU.
# Results: results/test_recon/{model}/picokv_recon_{mode}_k256_v320_key[_zr0]/
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
MODEL="${MODEL:-Llama-3.1-8B-Instruct}"
GPUS="${GPUS:-0,1,2,3}"
LIMIT="${LIMIT:-100}"
LENGTHS="131072"
TASKS="ruler_cwe,niah_multivalue,niah_multikey_2,niah_multikey_3,ruler_fwe"
RESULTS_ROOT="results/test_recon"
LOG_ROOT="logs/test_recon"
mkdir -p "${LOG_ROOT}"

[ -d "${DATA_ROOT}/${MODEL}" ] || { echo "ERROR: model not found: ${DATA_ROOT}/${MODEL}" >&2; exit 1; }

# "mode;zerorope"
JOBS=(
  "online;20"
  "online;0"
  "offline;20"
  "offline;0"
)

IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
NGPU="${#GPU_ARR[@]}"; NJOB="${#JOBS[@]}"
[ "${NGPU}" -ge "${NJOB}" ] || { echo "ERROR: need ${NJOB} GPUs, got ${NGPU} (GPUS=${GPUS})" >&2; exit 1; }

echo "=== picokv_recon key-only k256v320 | RULER @128k | limit=${LIMIT} | ${NJOB} jobs on GPUs [${GPUS}] ==="
pids=(); specs=()
for i in "${!JOBS[@]}"; do
  gpu="${GPU_ARR[$i]}"
  IFS=';' read -r mode zr <<< "${JOBS[$i]}"
  mode_args=(); [ "${mode}" = "offline" ] && mode_args+=(--offline)
  tag="picokv_recon_${mode}_k256_v320_key"
  [ "${zr}" != "20" ] && tag="${tag}_zr${zr}"
  out="${RESULTS_ROOT}/${MODEL}/${tag}/"
  log="${LOG_ROOT}/${MODEL}_${tag}.log"
  mkdir -p "${out}"
  echo "  [gpu ${gpu}] mode=${mode} zerorope=${zr} -> ${out}"
  CUDA_VISIBLE_DEVICES="${gpu}" python "${SCRIPT_DIR}/run_ruler_evaluation.py" \
    --model "${DATA_ROOT}/${MODEL}" --method picokv_recon --quant-target key \
    --k-bits 256 --v-bits 320 --sink-size 64 --local-size 256 \
    --exp-bits 4 --eas 0.01 --zerorope "${zr}" "${mode_args[@]}" \
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
