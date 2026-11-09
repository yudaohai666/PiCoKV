#!/bin/bash
# Wave 2 of the groups ablation: the 6 online-groups jobs NOT covered by wave 1.
# (wave 1 ran: EAS0.005 x4 + Llama groups g1/g8/g16/g64; Llama g1 still finishing.)
# Dispatches each job onto a GPU only once that GPU has >=THRESH_MIB free, so it
# coexists with external tenants + the still-running Llama-g1 without OOM.
# All jobs: online picokv, k320 v320, eas=0.01, RULER @128k, limit 100.
# Results: results/test_groups/{model}/picokv_online_k320_v320_g{group}/
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
K_BITS=320; V_BITS=320; SINK=64; LOCAL=256; EXP_BITS=4; EAS=0.01; ZEROROPE=20
THRESH_MIB="${THRESH_MIB:-55000}"   # min free MiB before we place a job on a GPU
POLL="${POLL:-30}"
RESULTS_ROOT="results/test_groups"
LOG_ROOT="logs/test_eas005_groups"
mkdir -p "${LOG_ROOT}"

# "model;group"
JOBS=(
  "Qwen3-8B;1"
  "Qwen3-8B;8"
  "Qwen3-8B;16"
  "Qwen3-8B;64"
  "Llama-3.1-8B-Instruct;80"
  "Qwen3-8B;80"
)

IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
declare -A GPU_PID   # gpu -> pid of the job we placed there (may be finished)
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' '; }

launch() {  # $1=gpu $2=model $3=group
  local gpu="$1" model="$2" group="$3"
  local tag="picokv_online_k${K_BITS}_v${V_BITS}_g${group}"
  local out="${RESULTS_ROOT}/${model}/${tag}/" log="${LOG_ROOT}/${model}_${tag}.log"
  mkdir -p "${out}"
  echo "  [gpu ${gpu}] ${model} group=${group} -> ${out}"
  CUDA_VISIBLE_DEVICES="${gpu}" PICOKV_ONLINE_GROUP_BITS="${group}" \
    python "${SCRIPT_DIR}/run_ruler_evaluation.py" \
    --model "${DATA_ROOT}/${model}" --method picokv \
    --k-bits "${K_BITS}" --v-bits "${V_BITS}" --sink-size "${SINK}" --local-size "${LOCAL}" \
    --exp-bits "${EXP_BITS}" --eas "${EAS}" --zerorope "${ZEROROPE}" \
    --tasks "${TASKS}" \
    --max_seq_lengths "${LENGTHS}" --max_length 131072 \
    --batch_size 1 --max_gen_toks 128 --limit "${LIMIT}" \
    --apply_chat_template \
    --output_path "${out}" > "${log}" 2>&1 &
}

echo "=== wave2 groups | ${#JOBS[@]} jobs | dispatch when GPU free>=${THRESH_MIB}MiB | GPUs [${GPUS}] ==="
pids=(); specs=()
for spec in "${JOBS[@]}"; do
  IFS=';' read -r model group <<< "${spec}"
  placed=0
  while [ "${placed}" -eq 0 ]; do
    for gpu in "${GPU_ARR[@]}"; do
      # skip a GPU whose previously-placed job is still running
      if [ -n "${GPU_PID[$gpu]:-}" ] && kill -0 "${GPU_PID[$gpu]}" 2>/dev/null; then continue; fi
      f="$(free_mib "$gpu")"; [ -z "${f}" ] && continue
      if [ "${f}" -ge "${THRESH_MIB}" ]; then
        launch "${gpu}" "${model}" "${group}"
        GPU_PID[$gpu]="$!"; pids+=("$!"); specs+=("${spec}")
        placed=1; sleep 20   # let the model allocate before rechecking free mem
        break
      fi
    done
    [ "${placed}" -eq 0 ] && sleep "${POLL}"
  done
done

rc_total=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "  done: ${specs[$i]}"; else echo "  FAILED: ${specs[$i]}" >&2; rc_total=1; fi
done
echo "=== wave2 (${#JOBS[@]} jobs) finished (rc=${rc_total}) ==="
exit "${rc_total}"
