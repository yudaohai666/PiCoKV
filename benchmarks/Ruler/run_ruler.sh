#!/bin/bash
# RULER eval: QuanPC methods + dense baselines.
# Mirrors the offline/whiten switches in Needle/run_niah.sh and
# Longbench/run_longbench.sh so all three benchmarks drive the same projection
# sweeps.
# Output (relative to REPO_ROOT, cd'd below):
#   quantpc:  results/ruler/<MODEL>_quantpc_k{K}_v{V}_exp{E}_zr{ZR}{TAG}/
#   baseline: results/ruler/<MODEL>_{flash_attention_2|sdpa}/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

# cd to repo root so runner's relative results/ruler/... lands under QuantPC/results.
cd "${REPO_ROOT}"

# hotpot QA local file (CMU source is often 502; use local copy). Absolute path.
export HOTPOTQA_JSON="${SCRIPT_DIR}/ruler_qa/hotpot_dev_distractor_v1.json"

GPU="${GPU:-3}"
MODEL="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct"
MODEL_NAME="$(basename "${MODEL}")"

# RULER params
MAX_SEQ_LENGTHS="${MAX_SEQ_LENGTHS:-8192,16384,32768,65536,131072}"   # comma-separated, e.g. "4096,8192,16384,32768,65536,131072"
MAX_LENGTH="${MAX_LENGTH:-131072}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-128}"
LIMIT="${LIMIT:-100}"
# Use "ruler" for all tasks; here only cwe (fast, 100 samples).
# TASKS="${TASKS:-ruler_cwe}"
# All tasks:
TASKS="niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multiquery,niah_multivalue,ruler_vt,ruler_cwe,ruler_fwe,ruler_qa_squad,ruler_qa_hotpot"

# Methods to run. quantpc/kivi/oscar = quantized KV; flash_attention_2/sdpa/eager = dense baselines.
METHODS=(
  quantpc
  # kivi
  # oscar
  # flash_attention_2
  # sdpa
)
# Back-compat: METHOD=oscar forces a single method.
if [ -n "${METHOD:-}" ]; then METHODS=("${METHOD}"); fi

# quantpc K/V sweep (quantpc only). Override a single pair via env, e.g.
# KV="320 256" bash run_ruler.sh; otherwise use the list below.
if [ -n "${KV:-}" ]; then
  CONFIGS=("${KV}")
else
  CONFIGS=(
    "320 256"
  )
fi

# quantpc. Offline whitening is baked into the folded artifact (never pass
# --whiten for offline); online whitening rotates K/V at prefill. Mirrors the
# offline/whiten switches in Needle/run_niah.sh so all benchmarks drive the
# same projection sweeps.
SINK="${SINK:-64}"; LOCAL="${LOCAL:-256}"
EXP_BITS="${EXP_BITS:-4}"; EAS="${EAS:-0.01}"; ZEROROPE="${ZEROROPE:-20}"
USE_OFFLINE="${USE_OFFLINE:-1}"      # 1=load pre-fitted projections; 0=online QR
WHITEN="${WHITEN:-1}"                # 1=whiten K+V; k=whiten K only (offline); 0=no
OFFLINE_PROJ_PATH="${OFFLINE_PROJ_PATH:-}"  # override the derived offline file

# exp-bits / zerorope ablation: the offline folded pair is calibrated at
# zerorope 20, so vary these only on the online path (USE_OFFLINE=0), e.g.
#   for zr in 0 5 10 15 25 30; do USE_OFFLINE=0 ZEROROPE=$zr bash run_ruler.sh; done
#   for e  in 2 4 6 8;         do USE_OFFLINE=0 EXP_BITS=$e  bash run_ruler.sh; done

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

for METHOD in "${METHODS[@]}"; do
  case "${METHOD}" in
    quantpc)
      for cfg in "${CONFIGS[@]}"; do
        read -r K V <<< "$cfg"
        ARGS=()
        if [ "${USE_OFFLINE}" = "1" ]; then
          case "${WHITEN}" in
            1) SUFFIX="folded";         TAG="_offline_whiten"   ;;  # K+V whitened
            k) SUFFIX="folded_whitenK"; TAG="_offline_whitenK"  ;;  # K-only whitened
            *) SUFFIX="nowhiten";       TAG="_offline_nowhiten" ;;
          esac
          PROJ="${OFFLINE_PROJ_PATH:-${REPO_ROOT}/offline_projection/projections/${MODEL_NAME}_k${K}_v${V}_block32_${SUFFIX}.pt}"
          ARGS+=(--offline-proj-path "${PROJ}")
        else
          [ "${WHITEN}" = "1" ] && { ARGS+=(--whiten); TAG="_online_whiten"; } || TAG="_online"
        fi
        OUT="results/ruler/${MODEL_NAME}_quantpc_k${K}_v${V}_exp${EXP_BITS}_zr${ZEROROPE}${TAG}/"
        echo "=== RULER: quantpc k=${K} v=${V} exp=${EXP_BITS} zr=${ZEROROPE} -> ${OUT} ==="
        run_ruler quantpc "${OUT}" \
          --k-bits "${K}" --v-bits "${V}" --sink-size "${SINK}" --local-size "${LOCAL}" \
          --exp-bits "${EXP_BITS}" --eas "${EAS}" --zerorope "${ZEROROPE}" "${ARGS[@]}"
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
      echo "=== RULER: ${METHOD} (dense baseline) ==="
      run_ruler "${METHOD}" ""
      ;;
  esac
done

echo "All done."
