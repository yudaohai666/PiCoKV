#!/bin/bash
# 顺序跑多种 k-bits / v-bits 组合，跑完后批量可视化
#
# 结果目录结构：
#   results/niah/<MODEL>_k{K}_v{V}/results_cuda.json
# 图片输出到：
#   results/niah/vis/

# ── 路径引导 ──────────────────────────────────────────────
# 本脚本位于 <repo>/benchmarks/Needle/，把仓库根加入 PYTHONPATH，
# 这样 runner 内的 `import quantpc` 可从任意 CWD 解析（runner 保持干净）。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

# ── 基础配置 ──────────────────────────────────────────────
GPU=6
MODEL="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct"
BASE_RESULT_DIR="results/niah"
VIS_DIR="${BASE_RESULT_DIR}/vis"
MODEL_NAME=$(basename "${MODEL}")

# context lengths（逗号分隔，传给 --lengths）
LENGTHS="81920,90112,98304,106496,114688,122880,131072"

# needle depths（逗号分隔，传给 --depths）
DEPTHS="0,11,22,33,44,56,67,78,89,100"

# 是否启用解码矩阵重拟合（refit_decoder）。
# 置 1 开启；可用环境变量覆盖，例如：REFIT_DECODER=1 bash run_niah.sh
# 注意：开启后 API prefill 会回退 torch（quanpc_api），pinv 有额外开销。
REFIT_DECODER="${REFIT_DECODER:-1}"

# 期望答案（与 show_image_paper.py 默认值保持一致）
EXPECTED_ANSWER="eat a sandwich and sit in Dolores Park on a sunny day."

# ── K/V 组合列表（按需修改）────────────────────────────────
CONFIGS=(
  "256 256"
  # "256 320"
  # "320 256"
  # "320 320"
)

# ── 逐组实验 ──────────────────────────────────────────────
for cfg in "${CONFIGS[@]}"; do
  K=$(echo $cfg | awk '{print $1}')
  V=$(echo $cfg | awk '{print $2}')

  # 目录名格式：<ModelName>_k<K>_v<V>；开启 refit 时加后缀便于区分结果
  SUBDIR="${BASE_RESULT_DIR}/${MODEL_NAME}_k${K}_v${V}"
  REFIT_FLAG=()
  if [ "${REFIT_DECODER}" = "1" ]; then
    # 固定 pinv 后缀，断点续跑按目录匹配
    SUBDIR="${SUBDIR}_refit_pinv"
    REFIT_FLAG=(--refit-decoder)
  fi
  mkdir -p "${SUBDIR}"

  echo "========================================"
  echo "Running: k-bits=${K}  v-bits=${V}  refit=${REFIT_DECODER}"
  echo "Output : ${SUBDIR}/results_cuda.json"
  echo "========================================"

  CUDA_VISIBLE_DEVICES=${GPU} \
  python "${SCRIPT_DIR}/run_bin_v4.py" \
    --lengths "${LENGTHS}" \
    --depths "${DEPTHS}" \
    --output "${SUBDIR}/results_cuda.json" \
    --k-bits "${K}" \
    --v-bits "${V}" \
    --projected-query-bits 8 \
    --exp-bits 4 \
    --eas 0.01 \
    --zerorope 20 \
    --cuda-v1-exp8 \
    "${REFIT_FLAG[@]}"

  echo "Done: k-bits=${K}  v-bits=${V}"
  echo ""
done

# ── 批量可视化 ─────────────────────────────────────────────
echo "========================================"
echo "Generating visualizations..."
echo "========================================"

for cfg in "${CONFIGS[@]}"; do
  K=$(echo $cfg | awk '{print $1}')
  V=$(echo $cfg | awk '{print $2}')
  SUBDIR="${BASE_RESULT_DIR}/${MODEL_NAME}_k${K}_v${V}"
  if [ "${REFIT_DECODER}" = "1" ]; then
    SUBDIR="${SUBDIR}_refit_pinv"
  fi

  python "${SCRIPT_DIR}/show_image_paper.py" \
    --model "${MODEL}" \
    --eval_path "${SUBDIR}" \
    --save_dir "${VIS_DIR}" \
    --expected_answer "${EXPECTED_ANSWER}"

  echo "Saved image for k-bits=${K}  v-bits=${V}"
done

echo ""
echo "All done. Images saved to: ${VIS_DIR}/${MODEL_NAME}/"
