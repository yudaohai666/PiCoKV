#!/bin/bash
# 顺序跑 LongBench 子数据集，跑完后用 eval.py 打分。
# 参考 benchmarks/Needle/run_niah.sh 的路径引导与参数风格。
#
# 结果目录结构：
#   results/longbench/<MODEL_NAME>/<TAG>/<dataset>.jsonl
#   results/longbench/<MODEL_NAME>/<TAG>/result.json   (eval.py 输出)

# ── 路径引导 ──────────────────────────────────────────────
# 本脚本位于 <repo>/benchmarks/Longbench/，把仓库根加入 PYTHONPATH，
# 这样 pred.py 内的 `import quantpc` 可从任意 CWD 解析。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

# ── 基础配置 ──────────────────────────────────────────────
GPU=7
MODEL="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct"
BASE_RESULT_DIR="results/longbench"
MODEL_NAME=$(basename "${MODEL}")

# 评测方法：quantpc | flash_attention_2 | sdpa | eager
METHOD="${METHOD:-quantpc}"

# 是否启用解码矩阵重拟合（refit_decoder）。置 1 开启。
REFIT_DECODER="${REFIT_DECODER:-1}"

# ── K/V 组合列表（按需修改）────────────────────────────────
CONFIGS=(
  "320 256"
)

# ── 逐组实验 ──────────────────────────────────────────────
for cfg in "${CONFIGS[@]}"; do
  K=$(echo $cfg | awk '{print $1}')
  V=$(echo $cfg | awk '{print $2}')

  # 目录标签需与 pred.py main() 中的 tag 规则保持一致，便于断点续跑与打分。
  if [ "${METHOD}" = "quantpc" ]; then
    TAG="quantpc_k${K}_v${V}"
    EXTRA=(--k-bits "${K}" --v-bits "${V}" --projected-query-bits 8 \
           --exp-bits 4 --eas 0.01 --zerorope 20 --cuda-v1-exp8)
    if [ "${REFIT_DECODER}" = "1" ]; then
      TAG="${TAG}_refit"
      EXTRA+=(--refit-decoder)
    fi
  else
    TAG="${METHOD}"
    EXTRA=()
  fi
  OUT_DIR="${BASE_RESULT_DIR}/${MODEL_NAME}/${TAG}"

  echo "========================================"
  echo "Running LongBench: method=${METHOD}  k=${K}  v=${V}  refit=${REFIT_DECODER}"
  echo "Output : ${OUT_DIR}"
  echo "========================================"

  CUDA_VISIBLE_DEVICES=${GPU} \
  python "${SCRIPT_DIR}/pred.py" \
    --model "${MODEL}" \
    --method "${METHOD}" \
    --save_dir "${BASE_RESULT_DIR}" \
    "${EXTRA[@]}"

  echo "Scoring: ${OUT_DIR}"
  python "${SCRIPT_DIR}/eval.py" --path "${OUT_DIR}/" --eval_avg

  echo "Done: method=${METHOD}  k=${K}  v=${V}"
  echo ""
done

echo "All done."
