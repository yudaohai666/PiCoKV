#!/bin/bash
# picokv K=256 V=320, RULER, online + offline, both models, all 5 lengths in one
# process each. 2 models x 2 modes = 4 jobs, one per GPU (GPUS env, positional).
# Follows run_ruler.sh defaults: eas=0.01, zerorope=20, exp_bits=4, sink=64,
# local=256, limit=100. Offline auto-resolves
# offline_projection/projections/{model}_k256_v320.pt (no explicit path needed).
# Results auto-written by run_ruler_evaluation.py to:
#   online  -> results/ruler/{model}/picokv_online_k256_v320/
#   offline -> results/ruler/{model}/picokv_offline_k256_v320/
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
GPUS="${GPUS:-0,2,4,6}"
LIMIT="${LIMIT:-100}"
LENGTHS="8192,16384,32768,65536,131072"
TASKS="niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multiquery,niah_multivalue,ruler_vt,ruler_cwe,ruler_fwe,ruler_qa_squad,ruler_qa_hotpot"
LOG_ROOT="logs/k256v320_onoff"
mkdir -p "${LOG_ROOT}"

# "model;mode"
JOBS=(
  "Llama-3.1-8B-Instruct;online"
  "Llama-3.1-8B-Instruct;offline"
  "Qwen3-8B;online"
  "Qwen3-8B;offline"
)

IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
NGPU="${#GPU_ARR[@]}"; NJOB="${#JOBS[@]}"
[ "${NGPU}" -ge "${NJOB}" ] || { echo "ERROR: need ${NJOB} GPUs, got ${NGPU} (GPUS=${GPUS})" >&2; exit 1; }
for m in Llama-3.1-8B-Instruct Qwen3-8B; do
  [ -d "${DATA_ROOT}/${m}" ] || { echo "ERROR: model not found: ${DATA_ROOT}/${m}" >&2; exit 1; }
done

echo "=== picokv k256v320 online+offline | RULER | limit=${LIMIT} | ${NJOB} jobs on GPUs [${GPUS}] ==="
pids=(); specs=()
for i in "${!JOBS[@]}"; do
  gpu="${GPU_ARR[$i]}"
  IFS=';' read -r model mode <<< "${JOBS[$i]}"
  mode_args=(); [ "${mode}" = "offline" ] && mode_args+=(--offline)
  tag="picokv_${mode}_k256_v320"
  out="results/ruler/${model}/${tag}/"
  log="${LOG_ROOT}/${model}_${tag}.log"
  mkdir -p "${out}"
  echo "  [gpu ${gpu}] model=${model} mode=${mode} -> ${out}"
  CUDA_VISIBLE_DEVICES="${gpu}" python "${SCRIPT_DIR}/run_ruler_evaluation.py" \
    --model "${DATA_ROOT}/${model}" --method picokv \
    --k-bits 256 --v-bits 320 --sink-size 64 --local-size 256 \
    --exp-bits 4 --eas 0.01 --zerorope 20 "${mode_args[@]}" \
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
