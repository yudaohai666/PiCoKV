#!/bin/bash
# NIAH（大海捞针）dense 基线评测：flash_attention_2 / sdpa。
#
# 与 run_niah.sh（quantpc 消融）对应，用同一个 runner run_bin_v4.py，
# 仅通过 --method 切换到 dense 全精度基线（不打 modify、用模型默认 KV cache）。
#
# 结果目录结构：
#   results/niah/<MODEL>_<flash_attention_2|sdpa>/results_<method>.json
# 图片输出到：
#   results/niah/vis/

# ── 路径引导 ──────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

# ── 基础配置 ──────────────────────────────────────────────
GPU=0,1,2,3,4,5
MODEL="/home/zxh/zxh/models/Llama-3.1-8B-Instruct"
BASE_RESULT_DIR="results/niah"
VIS_DIR="${BASE_RESULT_DIR}/vis"
MODEL_NAME=$(basename "${MODEL}")

# context lengths / needle depths（与 run_niah.sh 保持一致，便于横比）
LENGTHS="8192,16384,24576,32768,40960,49152,57344,65536,73728,81920,90112,98304,106496,114688,122880,131072"
DEPTHS="0,11,22,33,44,56,67,78,89,100"

# 期望答案（与 show_image_paper.py 默认值保持一致）
EXPECTED_ANSWER="eat a sandwich and sit in Dolores Park on a sunny day."

# ── dense 基线方法列表 ─────────────────────────────────────
# flash_attention_2 = flash-attn-2 全精度基线
# sdpa              = torch SDPA 全精度基线
METHODS=(
  flash_attention_2
  # sdpa
)

# ── 逐方法实验 ────────────────────────────────────────────
for method in "${METHODS[@]}"; do
  SUBDIR="${BASE_RESULT_DIR}/${MODEL_NAME}_${method}"
  mkdir -p "${SUBDIR}"

  echo "========================================"
  echo "Running NIAH: method=${method} (dense baseline)"
  echo "Output : ${SUBDIR}/results_${method}.json"
  echo "========================================"

  CUDA_VISIBLE_DEVICES=${GPU} \
  python "${SCRIPT_DIR}/run_bin_v4.py" \
    --model "${MODEL}" \
    --method "${method}" \
    --lengths "${LENGTHS}" \
    --depths "${DEPTHS}" \
    --output "${SUBDIR}/results_${method}.json"

  echo "Done: method=${method}"
  echo ""
done

# ── 批量可视化 ─────────────────────────────────────────────
echo "========================================"
echo "Generating visualizations..."
echo "========================================"

for method in "${METHODS[@]}"; do
  SUBDIR="${BASE_RESULT_DIR}/${MODEL_NAME}_${method}"

  python "${SCRIPT_DIR}/show_image_paper.py" \
    --model "${MODEL}" \
    --eval_path "${SUBDIR}" \
    --save_dir "${VIS_DIR}" \
    --expected_answer "${EXPECTED_ANSWER}"

  echo "Saved image for method=${method}"
done

echo ""
echo "All done. Images saved to: ${VIS_DIR}/${MODEL_NAME}/"
