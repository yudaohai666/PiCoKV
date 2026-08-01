#!/bin/bash
# Prefill vs decode 效率对比：dense 基线 / quantpc / oscar / kivi 四类方法横比。
#
# 用同一个 runner run_efficiency.py，每个方法各产出一份 per-method JSON 到
# results/efficiency/，最后 --compare 聚合成按 prompt_len 分组的对比表。
# 数据用随机 token id（延迟只与序列长度/形状有关，与内容无关）。
#
# 结果目录结构：
#   results/efficiency/<tag>.json     （tag 由 method + KV bits 决定）

# ── 路径引导 ──────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

# ── 基础配置 ──────────────────────────────────────────────
GPU=6
MODEL="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct"
RESULT_DIR="results/efficiency"
mkdir -p "${RESULT_DIR}"

# prompt 长度（与 NIAH 保持同量级，便于横比）
LENGTHS="131072"
DECODE_TOKENS=8
BATCH_SIZE=1
WARMUP=1
REPEATS=3

run_one() {
  # $1=method  $2=tag  剩余参数透传给 run_efficiency.py
  local method="$1"; shift
  local tag="$1"; shift
  echo "========================================"
  echo "Efficiency: method=${method}  ->  ${RESULT_DIR}/${tag}.json"
  echo "========================================"
  CUDA_VISIBLE_DEVICES=${GPU} \
  python "${SCRIPT_DIR}/run_efficiency.py" \
    --model "${MODEL}" \
    --method "${method}" \
    --lengths "${LENGTHS}" \
    --decode-tokens "${DECODE_TOKENS}" \
    --batch-size "${BATCH_SIZE}" \
    --warmup "${WARMUP}" \
    --repeats "${REPEATS}" \
    --output "${RESULT_DIR}/${tag}.json" \
    "$@"
  echo ""
}

# ── dense 基线 ────────────────────────────────────────────
run_one flash_attention_2 flash_attention_2

# ── quantpc（CUDA 后端 quantpc.quanpc，默认 KV bits）──────
run_one quantpc quantpc_k320_v320 \
  --k-bits 320 --v-bits 320 --cuda-v1-exp8

# ── oscar（int4 k-channel，group_size 128）────────────────
# 需先在 src/OScaR-KV-Quant 里 build 好 oscar_cuda：
#   bash src/OScaR-KV-Quant/build_oscar_cuda.sh
run_one oscar oscar_int4 \
  --num-bits 4 --quant-mode k-channel --group-size 128

# # ── kivi（K/V 各 2bit）────────────────────────────────────
# run_one kivi kivi_k2_v2 \
#   --k-bits 2 --v-bits 2 --group-size 32 --residual-length 128

# ── 聚合对比表 ────────────────────────────────────────────
echo "========================================"
echo "Comparison table"
echo "========================================"
python "${SCRIPT_DIR}/run_efficiency.py" --compare "${RESULT_DIR}"
