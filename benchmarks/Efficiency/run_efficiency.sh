#!/bin/bash
# Prefill vs decode efficiency: dense baseline / quantpc / kivi / oscar / polar.
# Each method -> one JSON in results/efficiency/, then --compare aggregates a
# table grouped by prompt_len. Inputs are random token ids (latency depends on
# length/shape, not content). Select methods via the METHODS list below.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
# Fragmentation, not live memory: the block-32 online solve churns a 126 MiB
# fp32 copy of its +-1 code matrix per group across two CUDA streams, and the
# default allocator opens fresh segments for that size class instead of reusing
# them. Measured on Qwen3-8B at 131k: nvidia-smi peak 35 -> 32 GB with
# max_memory_allocated unchanged. Must precede torch's allocator init, so it
# cannot move into python. Keep identical across all benchmark launchers.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Safe for the numbers this launcher publishes: run_efficiency.py:66 reports
# max_memory_allocated, which the allocator mode does not move (measured shift
# ~1 MiB of 1450), so arms recorded before and after this line stay comparable.

GPU="${GPU:-0}"
MODEL="${MODEL:-/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct}"
MODEL_TAG="$(basename "${MODEL%/}")"
RESULT_DIR="results/efficiency"
mkdir -p "${RESULT_DIR}"

# ── benchmark runtime params ──
# LENGTHS="${LENGTHS:-131072}"
LENGTHS="${LENGTHS:-131072}"

DECODE_TOKENS="${DECODE_TOKENS:-64}"
BATCH_SIZE="${BATCH_SIZE:-1}"
WARMUP="${WARMUP:-2}"
REPEATS="${REPEATS:-4}"
# Optional real-text dataset (.jsonl with "text" fields). Empty = use random token ids.
DATASET="${DATASET:-${SCRIPT_DIR}/datasets/PaulGrahamEssays.jsonl}"
# Override the auto result file (results/efficiency/{TAG}.json). Needed when one
# method is split across processes (e.g. one GPU per length): otherwise every
# process writes the same file. --compare still scans the whole RESULT_DIR.
OUTPUT_PATH="${OUTPUT_PATH:-}"

# ── methods to run (comment out to skip; METHOD=oscar forces a single one) ──
METHODS=(
  # flash_attention_2
  quantpc
  # kivi
  # oscar
  # polar
  # sdpa
  # eager
)
[ -n "${METHOD:-}" ] && METHODS=("${METHOD}")

# ── per-method defaults (identical across all benchmark launchers) ──
# quantpc; default = offline + nowhiten + vrefit
K_BITS="${K_BITS:-320}"; V_BITS="${V_BITS:-320}"
SINK="${SINK:-64}"; LOCAL="${LOCAL:-256}"
EXP_BITS="${EXP_BITS:-4}"; EAS="${EAS:-0.01}"; ZEROROPE="${ZEROROPE:-20}"
NO_PRUNING="${NO_PRUNING:-0}"
USE_OFFLINE="${USE_OFFLINE:-1}"
WHITEN="${WHITEN:-0}"
ONLINE_DECODER_REFIT="${ONLINE_DECODER_REFIT:-none}"
# 1e-3 == constants.ONLINE_DECODER_RIDGE, i.e. the shipped path. Do NOT raise this
# back to 1e-2: a nominal 1e-2 swamps the -9.3e-4 negative ridge the bf16 gram used
# to contribute, making it the one setting under which the fp32-gram fix is
# unmeasurable (16 paired cells: 6 up / 7 down / 3 tie, mean -0.004).
DECODER_RIDGE="${DECODER_RIDGE:-1e-3}"
OFFLINE_PROJ_PATH="${OFFLINE_PROJ_PATH:-}"
# kivi (2/4-bit KV)
KIVI_K="${KIVI_K:-2}"; KIVI_V="${KIVI_V:-2}"; KIVI_GROUP="${KIVI_GROUP:-32}"; KIVI_RESIDUAL="${KIVI_RESIDUAL:-128}"
# oscar INT2 recipe (group MUST be 32; int2 V-param packer hardcodes 32-wide groups)
OSCAR_NUM_BITS="${OSCAR_NUM_BITS:-2}"; OSCAR_QUANT_MODE="${OSCAR_QUANT_MODE:-k-channel}"; OSCAR_GROUP="${OSCAR_GROUP:-32}"
OSCAR_KV_ROTATION="${OSCAR_KV_ROTATION:-hadamard}"; OSCAR_KV_NORM="${OSCAR_KV_NORM:-1}"; OSCAR_EVICT="${OSCAR_EVICT:-256}"; OSCAR_OFFLINE_V_HAD="${OSCAR_OFFLINE_V_HAD:-1}"
# polar (src/PolarQuant)
POLAR_MODE="${POLAR_MODE:-mixed}"; POLAR_RBITS="${POLAR_RBITS:-3}"; POLAR_TBITS="${POLAR_TBITS:-3}"; POLAR_VBITS="${POLAR_VBITS:-2}"

# Build METHOD_ARGS[] and TAG for a method. Needs REPO_ROOT, MODEL_TAG + the
# per-method defaults above. quantpc TAG matches run_ruler/run_infinitebench's
# python run_tag(): quantpc_{mode}_k{K}_v{V}_{whiten}[_krefit|_vrefit|_refit].
build_method_args() {
  METHOD_ARGS=(); TAG="$1"
  case "$1" in
    quantpc)
      METHOD_ARGS=(--k-bits "${K_BITS}" --v-bits "${V_BITS}" --sink-size "${SINK}" --local-size "${LOCAL}"
        --exp-bits "${EXP_BITS}" --eas "${EAS}" --zerorope "${ZEROROPE}"
        --decoder-ridge "${DECODER_RIDGE}" --online-decoder-refit "${ONLINE_DECODER_REFIT}")
      local mode proj whiten_tag
      if [ "${USE_OFFLINE}" = "1" ] || [ -n "${OFFLINE_PROJ_PATH}" ]; then
        mode="offline"; METHOD_ARGS+=(--offline); proj="${OFFLINE_PROJ_PATH}"
        # offline + whiten -> the folded _whiten.pt (no runtime --whiten flag)
        if [ -z "${proj}" ] && [ "${WHITEN}" = "1" ]; then
          proj="${REPO_ROOT}/offline_projection/projections/${MODEL_TAG}_k${K_BITS}_v${V_BITS}_whiten.pt"
        fi
        [ -n "${proj}" ] && METHOD_ARGS+=(--offline-proj-path "${proj}")
      else
        mode="online"; [ "${WHITEN}" = "1" ] && METHOD_ARGS+=(--whiten)
      fi
      [ "${WHITEN}" = "1" ] && whiten_tag="whiten" || whiten_tag="nowhiten"
      TAG="quantpc_${mode}_k${K_BITS}_v${V_BITS}_${whiten_tag}"
      # refit suffix applies only to the offline path.
      if [ "${mode}" = "offline" ]; then
        case "${ONLINE_DECODER_REFIT}" in k) TAG+="_krefit" ;; v) TAG+="_vrefit" ;; both) TAG+="_refit" ;; esac
      fi
      # Knobs that change the result but were missing from the tag. EAS is the
      # one that bit us on NIAH: two EAS arms shared a directory. Here TAG names
      # the result *file*, so a second arm would overwrite the first outright.
      # Suffix only on deviation from the default, matching
      # run_ruler_evaluation.run_tag(), so existing file names stay valid.
      # String compare, not numeric: EAS=0.010 would read as non-default here
      # whereas run_tag()'s float compare would not.
      [ "${EXP_BITS}" != "4" ]  && TAG+="_exp${EXP_BITS}"
      [ "${ZEROROPE}" != "20" ] && TAG+="_zr${ZEROROPE}"
      [ "${EAS}" != "0.01" ]    && TAG+="_eas${EAS}"
      [ "${NO_PRUNING}" = "1" ] && METHOD_ARGS+=(--no-pruning) && TAG+="_noprune"
      # A false `[ ]` above leaves the branch's status at 1, which would trip
      # `set -e` at the call site if this script ever gains it. `:` resets it.
      :
      ;;
    kivi)
      METHOD_ARGS=(--k-bits "${KIVI_K}" --v-bits "${KIVI_V}" --group-size "${KIVI_GROUP}" --residual-length "${KIVI_RESIDUAL}")
      TAG="kivi_k${KIVI_K}_v${KIVI_V}"
      ;;
    oscar)
      METHOD_ARGS=(--num-bits "${OSCAR_NUM_BITS}" --quant-mode "${OSCAR_QUANT_MODE}" --group-size "${OSCAR_GROUP}")
      [ "${OSCAR_KV_ROTATION}" != "none" ] && METHOD_ARGS+=(--kv-rotation "${OSCAR_KV_ROTATION}")
      [ "${OSCAR_KV_NORM}" != "0" ]       && METHOD_ARGS+=(--kv-norm "${OSCAR_KV_NORM}")
      [ -n "${OSCAR_EVICT}" ]             && METHOD_ARGS+=(--residual-evict-size "${OSCAR_EVICT}")
      [ "${OSCAR_OFFLINE_V_HAD}" = "1" ]  && METHOD_ARGS+=(--offline-v-hadamard)
      TAG="oscar_int${OSCAR_NUM_BITS}"
      ;;
    polar)
      METHOD_ARGS=(--polar-mode "${POLAR_MODE}" --polar-rbits "${POLAR_RBITS}" --polar-tbits "${POLAR_TBITS}" --polar-vbits "${POLAR_VBITS}")
      TAG="polar_${POLAR_MODE}_${POLAR_RBITS}${POLAR_TBITS}${POLAR_VBITS}"
      ;;
  esac
}

for method in "${METHODS[@]}"; do
  build_method_args "${method}"
  out_file="${OUTPUT_PATH:-${RESULT_DIR}/${TAG}.json}"
  echo "=== Efficiency: method=${method} -> ${out_file} ==="
  CUDA_VISIBLE_DEVICES=${GPU} python "${SCRIPT_DIR}/run_efficiency.py" \
    --model "${MODEL}" --method "${method}" \
    --lengths "${LENGTHS}" --decode-tokens "${DECODE_TOKENS}" \
    --batch-size "${BATCH_SIZE}" --warmup "${WARMUP}" --repeats "${REPEATS}" \
    ${DATASET:+--dataset "${DATASET}"} \
    --output "${out_file}" "${METHOD_ARGS[@]}"
done

echo "=== Comparison table ==="
python "${SCRIPT_DIR}/run_efficiency.py" --compare "${RESULT_DIR}"
