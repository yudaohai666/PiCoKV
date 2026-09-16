#!/bin/bash
# k256v256 sweep: 2 models x {online,offline} x {Longbench,Needle} = 8 tasks on 8 GPUs.
set -uo pipefail

REPO_ROOT="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/ICLR2027/PiCoKV"
OUT_ROOT="${REPO_ROOT}/results/ablation/k256v256_sweep"
LOG_DIR="${REPO_ROOT}/logs/k256v256_sweep"
LONGBENCH_DATA_DIR="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/LongBench/data"

export http_proxy=http://agent.baidu.com:8891
export https_proxy=http://agent.baidu.com:8891

source "${REPO_ROOT}/.venv/bin/activate"
cd "${REPO_ROOT}"

LLAMA=/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct
QWEN=/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Qwen3-8B

# common picokv knobs for this sweep
K_BITS=256; V_BITS=256; SINK=64; LOCAL=256; EXP_BITS=4; ZEROROPE=20
EAS_ONLINE=0.01
EAS_OFFLINE=0.0000001

run_longbench() {
  local model="$1" mode="$2" gpu="$3" tag="$4"
  local extra=()
  [ "${mode}" = "offline" ] && extra+=(USE_OFFLINE=1)
  local eas="${EAS_ONLINE}"; [ "${mode}" = "offline" ] && eas="${EAS_OFFLINE}"
  local out_path="${OUT_ROOT}/longbench/${tag}"
  mkdir -p "${out_path}"
  GPU="${gpu}" MODEL="${model}" DATA_DIR="${LONGBENCH_DATA_DIR}" \
    K_BITS="${K_BITS}" V_BITS="${V_BITS}" SINK="${SINK}" LOCAL="${LOCAL}" \
    EXP_BITS="${EXP_BITS}" EAS="${eas}" ZEROROPE="${ZEROROPE}" \
    OUTPUT_PATH="${out_path}" "${extra[@]}" \
    bash "${REPO_ROOT}/benchmarks/Longbench/run_longbench.sh" \
    > "${LOG_DIR}/longbench_${tag}.log" 2>&1
}

run_needle() {
  local model="$1" mode="$2" gpu="$3" tag="$4"
  local extra=()
  [ "${mode}" = "offline" ] && extra+=(USE_OFFLINE=1)
  local eas="${EAS_ONLINE}"; [ "${mode}" = "offline" ] && eas="${EAS_OFFLINE}"
  local out_path="${OUT_ROOT}/needle/${tag}"
  mkdir -p "${out_path}"
  GPU="${gpu}" MODEL="${model}" METHOD=picokv \
    K_BITS="${K_BITS}" V_BITS="${V_BITS}" SINK="${SINK}" LOCAL="${LOCAL}" \
    EXP_BITS="${EXP_BITS}" EAS="${eas}" ZEROROPE="${ZEROROPE}" \
    OUTPUT_PATH="${out_path}" "${extra[@]}" \
    bash "${REPO_ROOT}/benchmarks/Needle/run_niah.sh" \
    > "${LOG_DIR}/needle_${tag}.log" 2>&1
}

pids=()

run_longbench "${LLAMA}" online  0 "Llama-3.1-8B-Instruct_online"  & pids+=($!)
run_longbench "${LLAMA}" offline 1 "Llama-3.1-8B-Instruct_offline" & pids+=($!)
run_needle    "${LLAMA}" online  2 "Llama-3.1-8B-Instruct_online"  & pids+=($!)
run_needle    "${LLAMA}" offline 3 "Llama-3.1-8B-Instruct_offline" & pids+=($!)

run_longbench "${QWEN}" online  4 "Qwen3-8B_online"  & pids+=($!)
run_longbench "${QWEN}" offline 5 "Qwen3-8B_offline" & pids+=($!)
run_needle    "${QWEN}" online  6 "Qwen3-8B_online"  & pids+=($!)
run_needle    "${QWEN}" offline 7 "Qwen3-8B_offline" & pids+=($!)

echo "Launched PIDs: ${pids[*]}"
wait "${pids[@]}"
echo "All 8 tasks finished."
