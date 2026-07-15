#!/bin/bash
# RULER 评测：支持 QuanPC 方法与 dense 基线
#
# 结果目录结构（相对仓库根，脚本会 cd 到 REPO_ROOT 后再跑）：
#   quantpc:  results/ruler/<MODEL>_quantpc_k{K}_v{V}/
#   baseline: results/ruler/<MODEL>_{flash_attention_2|sdpa}/
#
# 参考 benchmarks/Needle/run_niah.sh 的组织方式：把仓库根加入 PYTHONPATH，
# 这样 runner 内的 `import src.lmeval` / `import quantpc` 可从任意 CWD 解析。

# ── 路径引导 ──────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

# 切到仓库根，使 runner 的相对输出路径 results/ruler/... 落在 QuantPC/results 下。
cd "${REPO_ROOT}"

# hotpot QA 数据本地文件（CMU 源常年 502，改用本地副本）。
# 由 SCRIPT_DIR 拼绝对路径，避免依赖运行时 CWD。
export HOTPOTQA_JSON="${SCRIPT_DIR}/ruler_qa/hotpot_dev_distractor_v1.json"

# ── 基础配置 ──────────────────────────────────────────────
GPU=5
MODEL="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct"

# RULER 参数
MAX_SEQ_LENGTHS="32768"   # 逗号分隔，可写 "4096,8192,16384,32768,65536,131072"
MAX_LENGTH=131072
BATCH_SIZE=1
MAX_GEN_TOKS=128
LIMIT=100
# All RULER tasks - use "ruler" to run all tasks at once
TASKS="ruler"
# Or specify individual tasks:
# TASKS="niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multiquery,niah_multivalue,ruler_vt,ruler_cwe,ruler_fwe,ruler_qa_squad,ruler_qa_hotpot"
# TASKS="ruler_qa_hotpot"

# ── 方法列表 ──────────────────────────────────────────────
# quantpc           = 我的量化 KV cache 方法（走下方 CONFIGS 的 K/V 组合）
# flash_attention_2 = dense 全精度基线（flash-attn-2）
# sdpa              = dense 全精度基线（torch SDPA）
METHODS=(
  # quantpc
  flash_attention_2
  # sdpa
)

# quantpc 的 K/V 组合列表（仅 method=quantpc 时生效）
CONFIGS=(
  "256 256"
)

# ── 逐方法实验 ────────────────────────────────────────────
for method in "${METHODS[@]}"; do
  if [ "${method}" = "quantpc" ]; then
    for cfg in "${CONFIGS[@]}"; do
      K=$(echo "$cfg" | awk '{print $1}')
      V=$(echo "$cfg" | awk '{print $2}')

      echo "========================================"
      echo "Running RULER: method=quantpc  k-bits=${K}  v-bits=${V}"
      echo "========================================"

      CUDA_VISIBLE_DEVICES=${GPU} \
      python "${SCRIPT_DIR}/run_ruler_evaluation.py" \
        --model "${MODEL}" \
        --method quantpc \
        --tasks "${TASKS}" \
        --max_seq_lengths "${MAX_SEQ_LENGTHS}" \
        --max_length "${MAX_LENGTH}" \
        --batch_size "${BATCH_SIZE}" \
        --max_gen_toks "${MAX_GEN_TOKS}" \
        --limit "${LIMIT}" \
        --k-bits "${K}" \
        --v-bits "${V}" \
        --projected-query-bits 8 \
        --exp-bits 4 \
        --eas 0.01 \
        --zerorope 20 \
        --cuda-v1-exp8 \
        --apply_chat_template

      echo "Done: method=quantpc  k-bits=${K}  v-bits=${V}"
      echo ""
    done
  else
    echo "========================================"
    echo "Running RULER: method=${method} (dense baseline)"
    echo "========================================"

    CUDA_VISIBLE_DEVICES=${GPU} \
    python "${SCRIPT_DIR}/run_ruler_evaluation.py" \
      --model "${MODEL}" \
      --method "${method}" \
      --tasks "${TASKS}" \
      --max_seq_lengths "${MAX_SEQ_LENGTHS}" \
      --max_length "${MAX_LENGTH}" \
      --batch_size "${BATCH_SIZE}" \
      --max_gen_toks "${MAX_GEN_TOKS}" \
      --limit "${LIMIT}" \
      --apply_chat_template

    echo "Done: method=${method}"
    echo ""
  fi
done
