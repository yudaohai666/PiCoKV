#!/bin/bash
# zerorope sweep {10,30} ablation @ k320 v320, ONLINE, RULER @128k (all 13 tasks), limit 100.
# 8 jobs = {Llama, Qwen} x {dense_zr, key} x {zr10, zr30}, one per GPU:
#   dense_zr -> dense, NO quant, zerorope only on global K (pure zerorope ablation)
#   key      -> quantize K only (V exact, V-quant only feeds EAS)
# Results: results/test_key/{model}/picokv_recon_online_k320_v320_{tag}/
#   dense_zr zr10 -> ..._dense_zr_zr10   | dense_zr zr30 -> ..._dense_zr_zr30
#   key      zr10 -> ..._key_zr10        | key      zr30 -> ..._key_zr30
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
RESULTS_ROOT="results/test_key"
LOG_ROOT="logs/test_key"
mkdir -p "${LOG_ROOT}"

# "model;quant_target;zerorope"
JOBS=(
  "Llama-3.1-8B-Instruct;dense_zr;10"
  "Llama-3.1-8B-Instruct;dense_zr;30"
  "Llama-3.1-8B-Instruct;key;10"
  "Llama-3.1-8B-Instruct;key;30"
  "Qwen3-8B;dense_zr;10"
  "Qwen3-8B;dense_zr;30"
  "Qwen3-8B;key;10"
  "Qwen3-8B;key;30"
)

IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
NGPU="${#GPU_ARR[@]}"; NJOB="${#JOBS[@]}"
[ "${NGPU}" -ge "${NJOB}" ] || { echo "ERROR: need ${NJOB} GPUs, got ${NGPU} (GPUS=${GPUS})" >&2; exit 1; }
for m in Llama-3.1-8B-Instruct Qwen3-8B; do
  [ -d "${DATA_ROOT}/${m}" ] || { echo "ERROR: model not found: ${DATA_ROOT}/${m}" >&2; exit 1; }
done

echo "=== zr{10,30} ablation k${K_BITS}v${V_BITS} online | RULER @128k | limit=${LIMIT} | ${NJOB} jobs on GPUs [${GPUS}] ==="
pids=(); specs=()
for i in "${!JOBS[@]}"; do
  gpu="${GPU_ARR[$i]}"
  IFS=';' read -r model qt zr <<< "${JOBS[$i]}"
  tag="picokv_recon_online_k${K_BITS}_v${V_BITS}_${qt}"
  [ "${zr}" != "20" ] && tag="${tag}_zr${zr}"
  out="${RESULTS_ROOT}/${model}/${tag}/"
  log="${LOG_ROOT}/${model}_${tag}.log"
  mkdir -p "${out}"
  echo "  [gpu ${gpu}] model=${model} quant_target=${qt} zerorope=${zr} -> ${out}"
  CUDA_VISIBLE_DEVICES="${gpu}" python "${SCRIPT_DIR}/run_ruler_evaluation.py" \
    --model "${DATA_ROOT}/${model}" --method picokv_recon --quant-target "${qt}" \
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
