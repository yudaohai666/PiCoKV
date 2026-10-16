#!/bin/bash
# Decode-ITL benchmark wrapper around scripts/bench_itl.py.
#
# usage: run_itl.sh <gpu> <tag> <backend: picokv|flashinfer> <plen> <batch> <gen> <mode: eager|bcg|full> [VAR=val ...]
#   e.g. run_itl.sh 0 pk_b16_8k picokv 8192 16 1024 bcg
#        run_itl.sh 0 fi_b16_8k flashinfer 8192 16 1024 bcg
#
# Assumes the sibling layout <WS>/{PiCoKV,sglang}. Overridable env:
#   PY           python with sglang + torch      (default: python)
#   OUT          log directory                   (default: current dir)
#   BENCH_MODEL  model dir, relative to <WS> ok  (default: llama31_8b_rand)
#   PICOKV_PREBUILT_DIR  prebuilt extension dir  (default: PiCoKV/build/picokv_extensions)
#   ITL_REPS / ITL_MEMF / ITL_TIMEOUT            (default: 3 / 0.70 / 1500 s)
# bcg  = breakable decode CUDA graph (recommended for picokv)
# full = full decode graph (picokv raises NotImplementedError; flashinfer only)
set -u
OUT=$(mkdir -p "${OUT:-$PWD}" && cd "${OUT:-$PWD}" && pwd)
WS=$(cd "$(dirname "$0")/../.." && pwd)
PY=${PY:-python}
cd "$WS"
gpu=$1; tag=$2; be=$3; plen=$4; bs=$5; gen=$6; mode=$7; shift 7
G=""
[ "$mode" = bcg ] && G="BENCH_GRAPHS=1 BENCH_DECODE_GRAPH=breakable"
[ "$mode" = full ] && G="BENCH_GRAPHS=1 BENCH_DECODE_GRAPH=full"
log=$OUT/itl_$tag.log
q() { nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader -i "$gpu" | tr -d ' '; }
pre=$(q)
env PYTHONPATH="$WS/sglang/python:$WS/PiCoKV${PYTHONPATH:+:$PYTHONPATH}" \
  PICOKV_PREBUILT_DIR="${PICOKV_PREBUILT_DIR:-$WS/PiCoKV/build/picokv_extensions}" \
  CUDA_VISIBLE_DEVICES="$gpu" $G "$@" \
  timeout "${ITL_TIMEOUT:-1500}" "$PY" -u "$WS/PiCoKV/sglang_integration/scripts/bench_itl.py" \
  "$be" "$plen" "$bs" "$gen" "${ITL_REPS:-3}" "${ITL_MEMF:-0.70}" < /dev/null > "$log" 2>&1
rc=$?
post=$(q)
# Other GPUs with >1 GB in use: a busy neighbour inflates ITL, check before trusting numbers.
others=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | awk -F', ' '$2+0>1000{printf "%s:%s ", $1, $2}')
echo "[$tag] rc=$rc pre=$pre post=$post busy=[$others] $(grep -o 'step_ms_med=.*' "$log") |" \
  "$(grep '^REP' "$log" | awk '{printf "%s/%s ", $4, $5}')" \
  "maxrun=$(grep 'Decode batch' "$log" | grep -o '#running-req: [0-9]*' | awk '{print $2}' | sort -n | tail -1)" \
  "$(grep -m1 -E '^[A-Za-z]*Error|out of memory' "$log")"
