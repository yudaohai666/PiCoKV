#!/usr/bin/env bash
# 运行 benchmark_llama_layer_vs_sdpa.py：
# 捕获真实 Llama 某一层的 decode 单步 QuanPC 注意力，与 torch SDPA 对比延迟/加速比。
# 单配置（一个 K:V bits 组合）。从仓库根目录执行本脚本即可。
set -euo pipefail

# ============ 需要你确认/修改的项 ============
# 模型路径（系统默认的 /data/Llama-3.1-8B-Instruct 不存在，请改成真实路径）
MODEL="${MODEL:-/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct}"
# 用哪些 GPU（本机 0-7 共 8 张 H800）。
#   单卡: GPUS=0            —— 计时干净，100k 单张 H800(80GB) 通常够
#   多卡: GPUS=0,1,2,3      —— 用 device_map=auto 分摊显存加载模型/prefill，避免 OOM
# 注意：无论几张卡，最终 decode 计时只落在目标层所在的那一张卡上（单卡算子，公平对比）。
# 目标层(layer1)会落在列表里靠前的卡，所以把最空闲的卡放第一个。
GPUS="${GPUS:-${GPU:-0,1,2,3,4,5,6,7}}"
# ============================================

# ---- benchmark 参数（可用环境变量覆盖）----
LENGTH="${LENGTH:-128000}"        # 上下文长度
LAYER_IDX="${LAYER_IDX:-1}"       # 0-based 层号，1=第二层（与结果文件一致）
K_BITS="${K_BITS:-256}"
V_BITS="${V_BITS:-256}"
Q_BITS="${Q_BITS:-8}"             # projected-query-bits
EXP_BITS="${EXP_BITS:-4}"
EAS="${EAS:-0.01}"
ZEROROPE="${ZEROROPE:-20}"
PRUNE_MODE="${PRUNE_MODE:-3}"
PRUNE_LOG_EPS="${PRUNE_LOG_EPS:--10.0}"
WARMUP="${WARMUP:-20}"
ITERS="${ITERS:-100}"

# ---- 路径 ----
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
OUT_DIR="results/benchmarks"
OUT_JSON="${OUT_JSON:-$OUT_DIR/vs_sdpa_L${LENGTH}_layer${LAYER_IDX}_k${K_BITS}_v${V_BITS}.json}"
mkdir -p "$OUT_DIR"

# ---- 前置检查 ----
if [ ! -d "$MODEL" ]; then
  echo "[错误] 模型路径不存在: $MODEL"
  echo "       请用 MODEL=/你的/模型路径 $0 重新运行。"
  exit 1
fi

echo "==== 运行配置 ===="
echo "MODEL=$MODEL  GPUS=$GPUS  LENGTH=$LENGTH  LAYER_IDX=$LAYER_IDX"
echo "K:V bits=$K_BITS:$V_BITS  q_bits=$Q_BITS  exp_bits=$EXP_BITS  prune_mode=$PRUNE_MODE  log_eps=$PRUNE_LOG_EPS"
echo "warmup=$WARMUP  iters=$ITERS"
echo "输出 JSON: $OUT_JSON"
echo "=================="

CUDA_VISIBLE_DEVICES="$GPUS" \
python quantpc/quanpc_cuda_extension/benchmark_llama_layer_vs_sdpa.py \
  --model "$MODEL" \
  --length "$LENGTH" \
  --layer-idx "$LAYER_IDX" \
  --k-bits "$K_BITS" \
  --v-bits "$V_BITS" \
  --projected-query-bits "$Q_BITS" \
  --exp-bits "$EXP_BITS" \
  --eas "$EAS" \
  --zerorope "$ZEROROPE" \
  --prune-mode "$PRUNE_MODE" \
  --prune-log-eps "$PRUNE_LOG_EPS" \
  --warmup "$WARMUP" \
  --iters "$ITERS" \
  --output-json "$OUT_JSON"

echo "==== 完成，结果写入: $OUT_JSON ===="
