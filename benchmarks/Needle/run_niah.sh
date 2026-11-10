#!/bin/bash
set -euo pipefail
# NIAH (needle-in-a-haystack): picokv / kivi / oscar / polar / dense, then visualize.
# Select methods via the METHODS list below (METHOD=kivi forces a single one).
# Output: results/niah/<MODEL>/<TAG>/results.json ; PNG+PDF in results/niah/vis/<MODEL>/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPU="${GPU:-0}"
MODEL="${MODEL:-/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct}"
MODEL_TAG="$(basename "${MODEL%/}")"
# Overridable for the same reason PROF_DIR is in profile_decode.sh: a
# self-contained campaign (e.g. a rebuttal) lands in its own tree instead of
# mixing with results/niah's historical runs. VIS_DIR follows it so the heatmap
# does not get orphaned from the results.json it was rendered from.
BASE_RESULT_DIR="${BASE_RESULT_DIR:-results/niah}"
VIS_DIR="${VIS_DIR:-${BASE_RESULT_DIR}/vis}"

# ── benchmark runtime params ──
LENGTHS="${LENGTHS:-8192,16384,24576,32768,40960,49152,57344,65536,73728,81920,90112,98304,106496,114688,122880,131072}"
# LENGTHS="${LENGTHS:-131072}"

DEPTHS="${DEPTHS:-0,11,22,33,44,56,67,78,89,100}"
# Expected answer (must match show_image_paper.py default).
EXPECTED_ANSWER="${EXPECTED_ANSWER:-eat a sandwich and sit in Dolores Park on a sunny day.}"
# Override the auto results dir (results/niah/{MODEL}/{TAG}/). Needed when one
# method is split across processes (e.g. one GPU per length band): run_bin.py
# resumes from results.json by (length, depth), so a shared dir would let two
# processes interleave into one file.
OUTPUT_PATH="${OUTPUT_PATH:-}"

# ── methods to run (comment out to skip; METHOD=oscar forces a single one) ──
METHODS=(
  picokv
  # kivi
  # oscar
  # polar
  flash_attention_2
  # sdpa
  # eager
)
[ -n "${METHOD:-}" ] && METHODS=("${METHOD}")

# ── per-method defaults (identical across all benchmark launchers) ──
# picokv; default = online
K_BITS="${K_BITS:-320}"; V_BITS="${V_BITS:-320}"
SINK="${SINK:-64}"; LOCAL="${LOCAL:-256}"
EXP_BITS="${EXP_BITS:-4}"; EAS="${EAS:-0.01}"; ZEROROPE="${ZEROROPE:-20}"
USE_OFFLINE="${USE_OFFLINE:-0}"
OFFLINE_PROJ_PATH="${OFFLINE_PROJ_PATH:-}"
# kivi (2/4-bit KV)
KIVI_K="${KIVI_K:-2}"; KIVI_V="${KIVI_V:-2}"; KIVI_GROUP="${KIVI_GROUP:-32}"; KIVI_RESIDUAL="${KIVI_RESIDUAL:-128}"
# oscar INT2 recipe (group MUST be 32; int2 V-param packer hardcodes 32-wide groups)
OSCAR_NUM_BITS="${OSCAR_NUM_BITS:-2}"; OSCAR_QUANT_MODE="${OSCAR_QUANT_MODE:-k-channel}"; OSCAR_GROUP="${OSCAR_GROUP:-32}"
OSCAR_KV_ROTATION="${OSCAR_KV_ROTATION:-hadamard}"; OSCAR_KV_NORM="${OSCAR_KV_NORM:-1}"; OSCAR_EVICT="${OSCAR_EVICT:-256}"; OSCAR_OFFLINE_V_HAD="${OSCAR_OFFLINE_V_HAD:-1}"
# polar (src/PolarQuant)
POLAR_MODE="${POLAR_MODE:-mixed}"; POLAR_RBITS="${POLAR_RBITS:-3}"; POLAR_TBITS="${POLAR_TBITS:-3}"; POLAR_VBITS="${POLAR_VBITS:-2}"
# bpc (src/BPC); default = offline (mixlen-mixdata projection)
BPC_MASK_OUT="${BPC_MASK_OUT:-0.98}"; BPC_MIN_REMAIN="${BPC_MIN_REMAIN:-128}"; BPC_ERR_RATIO="${BPC_ERR_RATIO:-0.1}"; BPC_FIX_LAYERS="${BPC_FIX_LAYERS:-0,1}"
BPC_OFFLINE_PROJ_PATH="${BPC_OFFLINE_PROJ_PATH:-${REPO_ROOT}/src/BPC/ckp/projections/llama3-1-8b-ins-projection-mixlen-mixdata.pt}"
# bitdecoding (src/BitDecoding); 4-bit/g128 is the only numerically valid combo
BD_BITS="${BD_BITS:-4}"; BD_GROUP="${BD_GROUP:-128}"; BD_ALLOW_UNVALIDATED="${BD_ALLOW_UNVALIDATED:-0}"

# Build METHOD_ARGS[] and TAG for a method. Needs REPO_ROOT, MODEL_TAG + the
# per-method defaults above. picokv TAG matches run_ruler/run_infinitebench's
# python run_tag(): picokv_{mode}_k{K}_v{V}.
build_method_args() {
  METHOD_ARGS=(); TAG="$1"
  case "$1" in
    picokv)
      METHOD_ARGS=(--k-bits "${K_BITS}" --v-bits "${V_BITS}" --sink-size "${SINK}" --local-size "${LOCAL}"
        --exp-bits "${EXP_BITS}" --eas "${EAS}" --zerorope "${ZEROROPE}")
      local mode proj
      if [ "${USE_OFFLINE}" = "1" ] || [ -n "${OFFLINE_PROJ_PATH}" ]; then
        mode="offline"; METHOD_ARGS+=(--offline); proj="${OFFLINE_PROJ_PATH}"
        [ -n "${proj}" ] && METHOD_ARGS+=(--offline-proj-path "${proj}")
      else
        mode="online"
      fi
      TAG="picokv_${mode}_k${K_BITS}_v${V_BITS}"
      # Knobs that change the result but were missing from the tag. EAS is the
      # one that bit us: an EAS=0.01 and an EAS=0.001 run shared a directory, and
      # since run_bin.py resumes by (length, depth) from that one results.json,
      # the second arm skipped the cells the first had written instead of
      # recomputing them -- one heatmap silently made of two configs.
      # Suffix only on deviation from the default, matching
      # run_ruler_evaluation.run_tag(), so existing directory names stay valid.
      # String compare, not numeric: EAS=0.010 would read as non-default here
      # whereas run_tag()'s float compare would not.
      [ "${EXP_BITS}" != "4" ]  && TAG+="_exp${EXP_BITS}"
      [ "${ZEROROPE}" != "20" ] && TAG+="_zr${ZEROROPE}"
      [ "${EAS}" != "0.01" ]    && TAG+="_eas${EAS}"
      # A false `[ ]` above leaves the branch's status at 1, which propagates out
      # of the function and trips `set -e` at the call site. `:` resets it.
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
    bpc)
      METHOD_ARGS=(--bpc-mask-out "${BPC_MASK_OUT}" --bpc-min-remain "${BPC_MIN_REMAIN}"
        --bpc-err-ratio "${BPC_ERR_RATIO}" --bpc-fix-layers "${BPC_FIX_LAYERS}")
      TAG="bpc"
      if [ -n "${BPC_OFFLINE_PROJ_PATH}" ]; then
        METHOD_ARGS+=(--bpc-offline-proj-path "${BPC_OFFLINE_PROJ_PATH}")
        TAG="bpc_offline"
      fi
      ;;
    bitdecoding)
      # Without this case bitdecoding fell to the default branch and got
      # TAG="bitdecoding", so a 2-bit and a 4-bit arm shared one results.json.
      # run_bin.py resumes by (length, depth), so the second arm would skip the
      # first's cells instead of recomputing them -- one heatmap made of two
      # configs. Mirrors common.output_tag()'s bitdecoding_{bits}bit_g{group}.
      METHOD_ARGS=(--num-bits "${BD_BITS}" --group-size "${BD_GROUP}")
      [ "${BD_ALLOW_UNVALIDATED}" = "1" ] && METHOD_ARGS+=(--bitdecoding-allow-unvalidated)
      TAG="bitdecoding_${BD_BITS}bit_g${BD_GROUP}"
      :
      ;;
  esac
}

for method in "${METHODS[@]}"; do
  build_method_args "${method}"
  subdir="${OUTPUT_PATH:-${BASE_RESULT_DIR}/${MODEL_TAG}/${TAG}}"
  mkdir -p "${subdir}"
  echo "=== NIAH: method=${method} -> ${subdir}/results.json ==="
  CUDA_VISIBLE_DEVICES=${GPU} python "${SCRIPT_DIR}/run_bin.py" \
    --model "${MODEL}" --method "${method}" \
    --lengths "${LENGTHS}" --depths "${DEPTHS}" \
    --output "${subdir}/results.json" "${METHOD_ARGS[@]}"
  # A shard's heatmap is meaningless (it holds a subset of LENGTHS), and every
  # shard would overwrite the same PNG under VIS_DIR. Merge the shard
  # results.json files into one dir first, then run show_image_paper.py on that.
  if [ -n "${OUTPUT_PATH}" ]; then
    echo "=== NIAH: OUTPUT_PATH set -> skipping visualization for this shard ==="
    continue
  fi
  python "${SCRIPT_DIR}/show_image_paper.py" \
    --model "${MODEL}" --eval_path "${subdir}" \
    --save_dir "${VIS_DIR}" --expected_answer "${EXPECTED_ANSWER}"
done

echo "All done. PNG+PDF saved to: ${VIS_DIR}/${MODEL_TAG}/"
