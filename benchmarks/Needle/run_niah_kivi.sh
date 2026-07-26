#!/bin/bash
# NIAH（大海捞针）KIVI 量化 KV cache 评测。
#
# 与 run_niah_fa.sh（dense 基线）/ run_niah.sh（quantpc 消融）对应，
# 用同一个 runner run_bin_v4.py，仅通过 --method kivi 切换到 KIVI 方法。
# KIVI 自带内部量化 KV cache（不打 modify、不覆盖 past_key_values）。
#
# 结果目录结构：
#   results/niah/<MODEL>_kivi_k<K>_v<V>/results_kivi_k<K>_v<V>.json
# 图片输出到：
#   results/niah/vis/

# ── 路径引导 ──────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

# ── 基础配置 ──────────────────────────────────────────────
GPU=4
MODEL="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct"
BASE_RESULT_DIR="results/niah"
VIS_DIR="${BASE_RESULT_DIR}/vis"
MODEL_NAME=$(basename "${MODEL}")

# context lengths / needle depths（与 run_niah.sh 保持一致，便于横比）
LENGTHS="81920,90112,98304,106496,114688,122880,131072"
DEPTHS="0,11,22,33,44,56,67,78,89,100"

# 期望答案（与 show_image_paper.py 默认值保持一致）
EXPECTED_ANSWER="eat a sandwich and sit in Dolores Park on a sunny day."

# ── KIVI 量化配置 ─────────────────────────────────────────
# KIVI 仅支持 K/V 各 2 或 4 bit
K_BITS=2
V_BITS=2
GROUP_SIZE=32
RESIDUAL_LENGTH=128

TAG="kivi_k${K_BITS}_v${V_BITS}"
SUBDIR="${BASE_RESULT_DIR}/${MODEL_NAME}_${TAG}"
mkdir -p "${SUBDIR}"

echo "========================================"
echo "Running NIAH: method=kivi (K=${K_BITS}bit, V=${V_BITS}bit, group=${GROUP_SIZE}, residual=${RESIDUAL_LENGTH})"
echo "Output : ${SUBDIR}/results_${TAG}.json"
echo "========================================"

CUDA_VISIBLE_DEVICES=${GPU} \
python "${SCRIPT_DIR}/run_bin_v4.py" \
  --model "${MODEL}" \
  --method kivi \
  --k-bits "${K_BITS}" \
  --v-bits "${V_BITS}" \
  --group-size "${GROUP_SIZE}" \
  --residual-length "${RESIDUAL_LENGTH}" \
  --lengths "${LENGTHS}" \
  --depths "${DEPTHS}" \
  --output "${SUBDIR}/results_${TAG}.json"

echo "Done: method=kivi"
echo ""

# ── 可视化 ─────────────────────────────────────────────────
echo "========================================"
echo "Generating visualizations..."
echo "========================================"

python "${SCRIPT_DIR}/show_image_paper.py" \
  --model "${MODEL}" \
  --eval_path "${SUBDIR}" \
  --save_dir "${VIS_DIR}" \
  --expected_answer "${EXPECTED_ANSWER}"

echo ""
echo "All done. Images saved to: ${VIS_DIR}/${MODEL_NAME}/"
