#!/bin/bash
# picokv K=256 V=320, online mode, RULER, limit 20.
# Two models (Llama-3.1-8B-Instruct, Qwen3-8B) x two zerorope settings (20 default, 0).
# Each (model,zerorope) is split into two length groups so the slow 128k run does
# not block the shorter lengths:
#   shard_8to64k : 8k,16k,32k,64k  (one process, lengths run sequentially)
#   shard_128k   : 128k            (one process)
# => 2 models x 2 zerorope x 2 groups = 8 jobs.
# Results: results/test256320/ruler/{model}/{tag}/{shard}/ where
#   tag = picokv_online_k256_v320       (zerorope=20, run_tag() default)
#   tag = picokv_online_k256_v320_zr0   (zerorope=0)
# Jobs are queued across the GPU pool (GPUS env, default free GPUs 4,7).
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

DATA_ROOT="${DATA_ROOT:-/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data}"
GPUS="${GPUS:-4,7}"

K_BITS=256; V_BITS=320
SINK=64; LOCAL=256; EXP_BITS=4; EAS=0.01
LIMIT="${LIMIT:-20}"
MAX_GEN_TOKS=128
RULER_TASKS="niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multiquery,niah_multivalue,ruler_vt,ruler_cwe,ruler_fwe,ruler_qa_squad,ruler_qa_hotpot"

RESULTS_ROOT="${RESULTS_ROOT:-results/test256320}"
LOG_ROOT="${LOG_ROOT:-logs/test256320}"
mkdir -p "${LOG_ROOT}"

MODELS=(Llama-3.1-8B-Instruct Qwen3-8B)
for m in "${MODELS[@]}"; do
  [ -d "${DATA_ROOT}/${m}" ] || { echo "ERROR: model not found: ${DATA_ROOT}/${m}" >&2; exit 1; }
done

# 8 jobs: "model;zerorope;lengths;shard"
JOBS=(
  "Llama-3.1-8B-Instruct;20;8192,16384,32768,65536;shard_8to64k"
  "Llama-3.1-8B-Instruct;20;131072;shard_128k"
  "Llama-3.1-8B-Instruct;0;8192,16384,32768,65536;shard_8to64k"
  "Llama-3.1-8B-Instruct;0;131072;shard_128k"
  "Qwen3-8B;20;8192,16384,32768,65536;shard_8to64k"
  "Qwen3-8B;20;131072;shard_128k"
  "Qwen3-8B;0;8192,16384,32768,65536;shard_8to64k"
  "Qwen3-8B;0;131072;shard_128k"
)

IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
NGPU="${#GPU_ARR[@]}"
NJOB="${#JOBS[@]}"

launch_job() {
  local gpu="$1" spec="$2"
  IFS=';' read -r model zr lengths shard <<< "${spec}"
  local model_path="${DATA_ROOT}/${model}"
  local tag="picokv_online_k${K_BITS}_v${V_BITS}"
  [ "${zr}" != "20" ] && tag="${tag}_zr${zr}"
  local out="${RESULTS_ROOT}/ruler/${model}/${tag}/${shard}/"
  local log="${LOG_ROOT}/${model}_${tag}_${shard}.log"
  mkdir -p "${out}"
  echo "  [gpu ${gpu}] model=${model} zr=${zr} lengths=${lengths} -> ${out}"
  CUDA_VISIBLE_DEVICES="${gpu}" python "${SCRIPT_DIR}/run_ruler_evaluation.py" \
    --model "${model_path}" --method picokv \
    --k-bits "${K_BITS}" --v-bits "${V_BITS}" \
    --sink-size "${SINK}" --local-size "${LOCAL}" \
    --exp-bits "${EXP_BITS}" --eas "${EAS}" --zerorope "${zr}" \
    --tasks "${RULER_TASKS}" \
    --max_seq_lengths "${lengths}" --max_length 131072 \
    --batch_size 1 --max_gen_toks "${MAX_GEN_TOKS}" --limit "${LIMIT}" \
    --apply_chat_template \
    --output_path "${out}" > "${log}" 2>&1
}

echo "=== picokv k${K_BITS}v${V_BITS} online | RULER | limit=${LIMIT} | ${NJOB} jobs on GPUs [${GPUS}] ==="

# GPU-pool scheduler: keep one job per GPU running; launch the next queued job on
# a GPU as soon as its previous job exits. Tracks pass/fail per job.
declare -A GPU_PID GPU_SPEC
next=0
rc_total=0
declare -a FAILED=()

start_on_gpu() {
  local gpu="$1"
  if [ "${next}" -lt "${NJOB}" ]; then
    local spec="${JOBS[$next]}"
    launch_job "${gpu}" "${spec}" &
    GPU_PID["${gpu}"]="$!"
    GPU_SPEC["${gpu}"]="${spec}"
    next=$((next + 1))
  else
    GPU_PID["${gpu}"]=""
  fi
}

# prime each GPU with a job
for gpu in "${GPU_ARR[@]}"; do start_on_gpu "${gpu}"; done

# poll: when a GPU's job finishes, record result and launch the next
running=1
while [ "${running}" -eq 1 ]; do
  running=0
  for gpu in "${GPU_ARR[@]}"; do
    pid="${GPU_PID[${gpu}]:-}"
    [ -z "${pid}" ] && continue
    if kill -0 "${pid}" 2>/dev/null; then
      running=1
    else
      wait "${pid}"; rc=$?
      spec="${GPU_SPEC[${gpu}]}"
      if [ "${rc}" -ne 0 ]; then
        echo "  FAILED (rc=${rc}): ${spec}" >&2
        FAILED+=("${spec}")
        rc_total=1
      else
        echo "  done: ${spec}"
      fi
      start_on_gpu "${gpu}"
      [ -n "${GPU_PID[${gpu}]:-}" ] && running=1
    fi
  done
  [ "${running}" -eq 1 ] && sleep 20
done

echo "=== all ${NJOB} jobs finished ==="
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "FAILED jobs (${#FAILED[@]}):"
  for f in "${FAILED[@]}"; do echo "  - ${f}"; done
fi
exit "${rc_total}"
