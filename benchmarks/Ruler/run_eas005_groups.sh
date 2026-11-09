#!/bin/bash
# EAS=0.005 ablation (online+offline) + online GROUPS ablation, RULER @128k, limit 100.
# 14 jobs total, run in two waves of <=8 on the 8 GPUs ("分两批"):
#   EAS0.005 (4): {Llama,Qwen} x {online,offline}, eas=0.005 -> results/test_eas/
#   GROUPS   (10): {Llama,Qwen} x online x PICOKV_ONLINE_GROUP_BITS{1,8,16,64,80}
#                  (default 32 already exists; online-only) -> results/test_groups/
# All 5 group bits divide code-bits 320 (online.py requires bits % group == 0).
# Fixed: k320 v320 sink64 local256 exp4 zerorope20; eas=0.01 for groups runs.
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
K_BITS=320; V_BITS=320; SINK=64; LOCAL=256; EXP_BITS=4; ZEROROPE=20
LOG_ROOT="logs/test_eas005_groups"
mkdir -p "${LOG_ROOT}"

# "model;mode;eas;group;results_root;tag"   (group empty => default 32)
JOBS=(
  "Llama-3.1-8B-Instruct;online;0.005;;results/test_eas;picokv_online_k320_v320_eas0.005"
  "Llama-3.1-8B-Instruct;offline;0.005;;results/test_eas;picokv_offline_k320_v320_eas0.005"
  "Qwen3-8B;online;0.005;;results/test_eas;picokv_online_k320_v320_eas0.005"
  "Qwen3-8B;offline;0.005;;results/test_eas;picokv_offline_k320_v320_eas0.005"
  "Llama-3.1-8B-Instruct;online;0.01;1;results/test_groups;picokv_online_k320_v320_g1"
  "Llama-3.1-8B-Instruct;online;0.01;8;results/test_groups;picokv_online_k320_v320_g8"
  "Llama-3.1-8B-Instruct;online;0.01;16;results/test_groups;picokv_online_k320_v320_g16"
  "Llama-3.1-8B-Instruct;online;0.01;64;results/test_groups;picokv_online_k320_v320_g64"
  "Qwen3-8B;online;0.01;1;results/test_groups;picokv_online_k320_v320_g1"
  "Qwen3-8B;online;0.01;8;results/test_groups;picokv_online_k320_v320_g8"
  "Qwen3-8B;online;0.01;16;results/test_groups;picokv_online_k320_v320_g16"
  "Qwen3-8B;online;0.01;64;results/test_groups;picokv_online_k320_v320_g64"
  "Llama-3.1-8B-Instruct;online;0.01;80;results/test_groups;picokv_online_k320_v320_g80"
  "Qwen3-8B;online;0.01;80;results/test_groups;picokv_online_k320_v320_g80"
)

IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
NGPU="${#GPU_ARR[@]}"; NJOB="${#JOBS[@]}"
for m in Llama-3.1-8B-Instruct Qwen3-8B; do
  [ -d "${DATA_ROOT}/${m}" ] || { echo "ERROR: model not found: ${DATA_ROOT}/${m}" >&2; exit 1; }
  [ -f "offline_projection/projections/${m}_k${K_BITS}_v${V_BITS}.pt" ] || { echo "ERROR: offline projection missing: ${m}_k${K_BITS}_v${V_BITS}.pt" >&2; exit 1; }
done

run_job() {  # $1=gpu $2=jobspec
  local gpu="$1" spec="$2" model mode eas group rroot tag out log offline_flag=""
  IFS=';' read -r model mode eas group rroot tag <<< "${spec}"
  out="${rroot}/${model}/${tag}/"; log="${LOG_ROOT}/${model}_${tag}.log"
  mkdir -p "${out}"
  [ "${mode}" = "offline" ] && offline_flag="--offline"
  echo "  [gpu ${gpu}] ${model} mode=${mode} eas=${eas} group=${group:-32} -> ${out}"
  CUDA_VISIBLE_DEVICES="${gpu}" PICOKV_ONLINE_GROUP_BITS="${group:-32}" \
    python "${SCRIPT_DIR}/run_ruler_evaluation.py" \
    --model "${DATA_ROOT}/${model}" --method picokv ${offline_flag} \
    --k-bits "${K_BITS}" --v-bits "${V_BITS}" --sink-size "${SINK}" --local-size "${LOCAL}" \
    --exp-bits "${EXP_BITS}" --eas "${eas}" --zerorope "${ZEROROPE}" \
    --tasks "${TASKS}" \
    --max_seq_lengths "${LENGTHS}" --max_length 131072 \
    --batch_size 1 --max_gen_toks 128 --limit "${LIMIT}" \
    --apply_chat_template \
    --output_path "${out}" > "${log}" 2>&1
}

echo "=== EAS0.005+GROUPS | RULER @128k | limit=${LIMIT} | ${NJOB} jobs, ${NGPU} GPUs [${GPUS}], waves of ${NGPU} ==="
rc_total=0; wave=0
for ((base=0; base<NJOB; base+=NGPU)); do
  wave=$((wave+1)); echo "--- wave ${wave} ---"
  pids=(); specs=()
  for ((j=0; j<NGPU && base+j<NJOB; j++)); do
    gpu="${GPU_ARR[$j]}"; spec="${JOBS[$((base+j))]}"
    run_job "${gpu}" "${spec}" & pids+=("$!"); specs+=("${spec}")
  done
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then echo "  done: ${specs[$i]}"; else echo "  FAILED: ${specs[$i]}" >&2; rc_total=1; fi
  done
done
echo "=== all ${NJOB} jobs finished (rc=${rc_total}) ==="
exit "${rc_total}"
