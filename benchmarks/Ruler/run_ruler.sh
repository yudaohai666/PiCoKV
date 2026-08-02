#!/bin/bash
# RULER eval: QuanPC methods + dense baselines.
# Output (relative to REPO_ROOT, cd'd below):
#   quantpc:  results/ruler/<MODEL>_quantpc_k{K}_v{V}_exp{E}_zr{ZR}/
#   baseline: results/ruler/<MODEL>_{flash_attention_2|sdpa}/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

# cd to repo root so runner's relative results/ruler/... lands under QuantPC/results.
cd "${REPO_ROOT}"

# hotpot QA local file (CMU source is often 502; use local copy). Absolute path.
export HOTPOTQA_JSON="${SCRIPT_DIR}/ruler_qa/hotpot_dev_distractor_v1.json"

GPU=3
MODEL="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct"

# RULER params
MAX_SEQ_LENGTHS="32768"   # comma-separated, e.g. "4096,8192,16384,32768,65536,131072"
MAX_LENGTH=131072
BATCH_SIZE=1
MAX_GEN_TOKS=128
LIMIT=100
# Use "ruler" for all tasks; here only cwe (fast, 100 samples).
TASKS="ruler_cwe"
# All tasks:
# TASKS="niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multiquery,niah_multivalue,ruler_vt,ruler_cwe,ruler_fwe,ruler_qa_squad,ruler_qa_hotpot"

# Methods to run. quantpc/kivi/oscar = quantized KV; flash_attention_2/sdpa/eager = dense baselines.
METHODS=(
  quantpc
  # kivi
  # oscar
  # flash_attention_2
  # sdpa
)

# KIVI params (2/4-bit KV, single run).
KIVI_K="${KIVI_K:-2}"
KIVI_V="${KIVI_V:-2}"
KIVI_GROUP="${KIVI_GROUP:-32}"
KIVI_RESIDUAL="${KIVI_RESIDUAL:-128}"

# OScaR INT2 recipe. group_size MUST be 32: the int2 V-param interleaved packer
# hardcodes 32-wide groups (4 groups); group 128 under-allocates v_params -> OOB write.
OSCAR_NUM_BITS="${OSCAR_NUM_BITS:-2}"
OSCAR_QUANT_MODE="${OSCAR_QUANT_MODE:-k-channel}"
OSCAR_GROUP="${OSCAR_GROUP:-32}"
OSCAR_KV_ROTATION="${OSCAR_KV_ROTATION:-hadamard}"
OSCAR_KV_NORM="${OSCAR_KV_NORM:-1}"
OSCAR_EVICT="${OSCAR_EVICT:-256}"
OSCAR_OFFLINE_V_HAD="${OSCAR_OFFLINE_V_HAD:-1}"

# quantpc offline block-32 encoders: USE_OFFLINE=1 skips the online QR at prefill
# (default path offline_projection/{model}_k{K}_v{V}_block32.pt). OFFLINE_PROJ_PATH
# overrides the path; OFFLINE_PROJ_MODE = both|key|value (default both).
USE_OFFLINE="${USE_OFFLINE:-0}"
OFFLINE_PROJ_PATH="${OFFLINE_PROJ_PATH:-}"
OFFLINE_PROJ_MODE="${OFFLINE_PROJ_MODE:-both}"
OFFLINE_ARGS=()
if [ "${USE_OFFLINE}" = "1" ] || [ -n "${OFFLINE_PROJ_PATH}" ]; then
  OFFLINE_ARGS=(--offline --offline-proj-mode "${OFFLINE_PROJ_MODE}")
  [ -n "${OFFLINE_PROJ_PATH}" ] && OFFLINE_ARGS+=(--offline-proj-path "${OFFLINE_PROJ_PATH}")
fi
# Output-dir suffix so offline runs don't overwrite online ones (matches output_tag).
OFFLINE_TAG=""
if [ ${#OFFLINE_ARGS[@]} -gt 0 ]; then
  OFFLINE_TAG="_offline"; [ "${OFFLINE_PROJ_MODE}" != "both" ] && OFFLINE_TAG="_offline_${OFFLINE_PROJ_MODE}"
fi

# quantpc ablation sweep (quantpc only). Fields: "K V EXP_BITS ZEROROPE".
#   K/V      = per-token binary projection components (storage precision)
#   EXP_BITS = softmax weight quant bits, CUDA supports {1,2,4,8}
#   ZEROROPE = RoPE dims zeroed before quantizing key, 0 <= ZR <= head_dim/2 (=64)
# Output dir encodes all vars so same K/V with different exp/zr don't overwrite.
CONFIGS=(
  # K   V   exp  zr    (anchor = current default)
  "320 320  4   20"
  # exp-bits ablation (fixed K320 V320 zr20)
  "320 320  2   20"
  "320 320  6   20"
  "320 320  8   20"
  # zerorope ablation (fixed K320 V320 exp4)
  "320 320  4   0"
  "320 320  4   5"
  "320 320  4   10"
  "320 320  4   15"
  "320 320  4   25"
  "320 320  4   30"
)

# run_ruler METHOD OUT_PATH RUN_ARGS...
run_ruler() {
  local method="$1"; shift
  local out="$1"; shift
  local out_arg=()
  [ -n "${out}" ] && out_arg=(--output_path "${out}")
  CUDA_VISIBLE_DEVICES=${GPU} python "${SCRIPT_DIR}/run_ruler_evaluation.py" \
    --model "${MODEL}" --method "${method}" --tasks "${TASKS}" \
    --max_seq_lengths "${MAX_SEQ_LENGTHS}" --max_length "${MAX_LENGTH}" \
    --batch_size "${BATCH_SIZE}" --max_gen_toks "${MAX_GEN_TOKS}" --limit "${LIMIT}" \
    "${out_arg[@]}" --apply_chat_template "$@"
}

MODEL_NAME="$(basename "${MODEL}")"

for method in "${METHODS[@]}"; do
  case "${method}" in
    quantpc)
      for cfg in "${CONFIGS[@]}"; do
        read -r K V EXP ZR <<< "$cfg"
        OUT="results/ruler/${MODEL_NAME}_quantpc_k${K}_v${V}_exp${EXP}_zr${ZR}${OFFLINE_TAG}/"
        echo "=== RULER: quantpc k=${K} v=${V} exp=${EXP} zr=${ZR} -> ${OUT} ==="
        run_ruler quantpc "${OUT}" \
          --k-bits "${K}" --v-bits "${V}" --exp-bits "${EXP}" --eas 0.01 --zerorope "${ZR}" "${OFFLINE_ARGS[@]}"
      done
      ;;
    kivi)
      echo "=== RULER: kivi k=${KIVI_K} v=${KIVI_V} group=${KIVI_GROUP} residual=${KIVI_RESIDUAL} ==="
      run_ruler kivi "" \
        --k-bits "${KIVI_K}" --v-bits "${KIVI_V}" \
        --group-size "${KIVI_GROUP}" --residual-length "${KIVI_RESIDUAL}"
      ;;
    oscar)
      echo "=== RULER: oscar int${OSCAR_NUM_BITS} group=${OSCAR_GROUP} rot=${OSCAR_KV_ROTATION} norm=${OSCAR_KV_NORM} evict=${OSCAR_EVICT} offlineV=${OSCAR_OFFLINE_V_HAD} ==="
      OSCAR_ARGS=(--num-bits "${OSCAR_NUM_BITS}" --quant-mode "${OSCAR_QUANT_MODE}" --group-size "${OSCAR_GROUP}")
      [ "${OSCAR_KV_ROTATION}" != "none" ] && OSCAR_ARGS+=(--kv-rotation "${OSCAR_KV_ROTATION}")
      [ "${OSCAR_KV_NORM}" != "0" ]       && OSCAR_ARGS+=(--kv-norm "${OSCAR_KV_NORM}")
      [ -n "${OSCAR_EVICT}" ]             && OSCAR_ARGS+=(--residual-evict-size "${OSCAR_EVICT}")
      [ "${OSCAR_OFFLINE_V_HAD}" = "1" ]  && OSCAR_ARGS+=(--offline-v-hadamard)
      run_ruler oscar "" "${OSCAR_ARGS[@]}"
      ;;
    *)
      echo "=== RULER: ${method} (dense baseline) ==="
      run_ruler "${method}" ""
      ;;
  esac
done
