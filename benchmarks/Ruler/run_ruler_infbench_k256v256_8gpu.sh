#!/bin/bash
# picokv K_BITS=V_BITS=256 sweep: online (eas=0.01) + offline (eas=1e-7),
# on Llama-3.1-8B-Instruct + Qwen3-8B, over RULER (5 lengths: 8k/16k/32k/64k/128k)
# and InfiniteBench (single max_length=131072). 8 configs total, one per GPU:
#   RULER:         2 models x 2 modes = 4 configs, each runs all 5 lengths
#                  sequentially inside one process (--max_seq_lengths CSV).
#   InfiniteBench: 2 models x 2 modes = 4 configs, single process each.
# offline mode uses the prebuilt block-32 encoders in
# offline_projection/projections/{MODEL}_k256_v256.pt (resolved automatically
# by benchmarks/common.py from --offline + --k-bits/--v-bits, no explicit
# --offline-proj-path needed).
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck disable=SC1091
source "${REPO_ROOT}/.venv/bin/activate"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
# See run_ruler.sh: fragmentation fix for the block-32 online solve's fp32
# code-matrix churn. Must precede torch's allocator init.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# lm_eval's ruler_qa_squad task downloads dev-v2.0.json; requests reads these.
export http_proxy="${http_proxy:-http://agent.baidu.com:8891}"
export https_proxy="${https_proxy:-http://agent.baidu.com:8891}"

DATA_ROOT="${DATA_ROOT:-/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"

K_BITS="${K_BITS:-256}"; V_BITS="${V_BITS:-256}"
SINK="${SINK:-64}"; LOCAL="${LOCAL:-256}"
EXP_BITS="${EXP_BITS:-4}"; ZEROROPE="${ZEROROPE:-20}"
EAS_ONLINE="${EAS_ONLINE:-0.01}"; EAS_OFFLINE="${EAS_OFFLINE:-0.0000001}"
LIMIT="${LIMIT:-100}"; MAX_GEN_TOKS_RULER="${MAX_GEN_TOKS_RULER:-128}"
MAX_GEN_TOKS_INF="${MAX_GEN_TOKS_INF:-1200}"; LIMIT_INF="${LIMIT_INF:-100}"

RULER_LENGTHS="${RULER_LENGTHS:-8192,16384,32768,65536,131072}"
RULER_TASKS="${RULER_TASKS:-niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multiquery,niah_multivalue,ruler_vt,ruler_cwe,ruler_fwe,ruler_qa_squad,ruler_qa_hotpot}"

RESULTS_ROOT="${RESULTS_ROOT:-results/ablation/k256v256_sweep}"
LOG_ROOT="${LOG_ROOT:-logs/ablation/k256v256_sweep}"

MODELS=(Llama-3.1-8B-Instruct Qwen3-8B)
for m in "${MODELS[@]}"; do
  if [ ! -d "${DATA_ROOT}/${m}" ]; then
    echo "ERROR: model not found: ${DATA_ROOT}/${m}" >&2
    exit 1
  fi
done

IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
NGPU="${#GPU_ARR[@]}"

# 8 configs: (benchmark;model;mode;eas)
CONFIGS=(
  "ruler;Llama-3.1-8B-Instruct;online;${EAS_ONLINE}"
  "ruler;Llama-3.1-8B-Instruct;offline;${EAS_OFFLINE}"
  "ruler;Qwen3-8B;online;${EAS_ONLINE}"
  "ruler;Qwen3-8B;offline;${EAS_OFFLINE}"
  "infinitebench;Llama-3.1-8B-Instruct;online;${EAS_ONLINE}"
  "infinitebench;Llama-3.1-8B-Instruct;offline;${EAS_OFFLINE}"
  "infinitebench;Qwen3-8B;online;${EAS_ONLINE}"
  "infinitebench;Qwen3-8B;offline;${EAS_OFFLINE}"
)
NCFG="${#CONFIGS[@]}"

if [ "${NGPU}" -lt "${NCFG}" ]; then
  echo "ERROR: ${NCFG} configs need ${NCFG} GPUs, only ${NGPU} given (GPUS=${GPUS})" >&2
  exit 1
fi

mkdir -p "${LOG_ROOT}"
echo "=== picokv k${K_BITS}v${V_BITS} sweep | online eas=${EAS_ONLINE} / offline eas=${EAS_OFFLINE} | ${NCFG} configs on ${NGPU} GPUs ==="

rc_total=0
pids=(); idxs=()
for ((i = 0; i < NCFG; i++)); do
  gpu="${GPU_ARR[$i]}"
  IFS=';' read -r bench model mode eas <<< "${CONFIGS[$i]}"
  model_path="${DATA_ROOT}/${model}"
  tag="picokv_${mode}_k${K_BITS}_v${V_BITS}"
  mode_args=()
  [ "${mode}" = "offline" ] && mode_args+=(--offline)

  if [ "${bench}" = "ruler" ]; then
    out="${RESULTS_ROOT}/ruler/${model}/${tag}/"
    log="${LOG_ROOT}/ruler_${model}_${tag}.log"
    mkdir -p "${out}"
    echo "  gpu=${gpu} bench=ruler model=${model} mode=${mode} eas=${eas} tag=${tag} lengths=${RULER_LENGTHS}"
    CUDA_VISIBLE_DEVICES="${gpu}" python "${SCRIPT_DIR}/run_ruler_evaluation.py" \
      --model "${model_path}" --method picokv \
      --k-bits "${K_BITS}" --v-bits "${V_BITS}" \
      --sink-size "${SINK}" --local-size "${LOCAL}" \
      --exp-bits "${EXP_BITS}" --eas "${eas}" --zerorope "${ZEROROPE}" \
      "${mode_args[@]}" \
      --tasks "${RULER_TASKS}" \
      --max_seq_lengths "${RULER_LENGTHS}" --max_length 131072 \
      --batch_size 1 --max_gen_toks "${MAX_GEN_TOKS_RULER}" --limit "${LIMIT}" \
      --apply_chat_template \
      --output_path "${out}" > "${log}" 2>&1 &
  else
    out="${RESULTS_ROOT}/infinitebench/${model}/${tag}/"
    log="${LOG_ROOT}/infinitebench_${model}_${tag}.log"
    mkdir -p "${out}"
    echo "  gpu=${gpu} bench=infinitebench model=${model} mode=${mode} eas=${eas} tag=${tag}"
    CUDA_VISIBLE_DEVICES="${gpu}" python "${REPO_ROOT}/benchmarks/InfiniteBench/run_infinitebench.py" \
      --model "${model_path}" --method picokv \
      --k-bits "${K_BITS}" --v-bits "${V_BITS}" \
      --sink-size "${SINK}" --local-size "${LOCAL}" \
      --exp-bits "${EXP_BITS}" --eas "${eas}" --zerorope "${ZEROROPE}" \
      "${mode_args[@]}" \
      --max_length 131072 \
      --batch_size 1 --max_gen_toks "${MAX_GEN_TOKS_INF}" --limit "${LIMIT_INF}" \
      --apply_chat_template \
      --output_path "${out}" > "${log}" 2>&1 &
  fi
  pids+=("$!")
  idxs+=("${i}")
done

for i in "${!pids[@]}"; do
  idx="${idxs[$i]}"
  IFS=';' read -r bench model mode eas <<< "${CONFIGS[$idx]}"
  if ! wait "${pids[$i]}"; then
    echo "  FAILED: bench=${bench} model=${model} mode=${mode} -- see log" >&2
    rc_total=1
  fi
done

echo "=== all ${NCFG} configs done ==="
exit "${rc_total}"
