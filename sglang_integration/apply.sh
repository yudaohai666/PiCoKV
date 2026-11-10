#!/bin/bash
# Install the PiCoKV attention backend into an sglang source tree.
#
# usage: apply.sh <sglang_dir> [--link]
#   <sglang_dir>  root of an sglang source tree (the dir containing python/sglang)
#   --link        symlink the new files (picokv_backend.py, picokv_radix_cache.py, picokv_mla/*.py,
#                 picokv_mla_sm90/decode_i8.cuh) instead of copying (edits in
#                 PiCoKV/sglang_integration/ then take effect without re-running)
#
# The patch was generated against upstream sgl-project/sglang commit BASE below;
# other commits may still apply but are untested. Idempotent: an already-applied
# patch is detected and skipped.
set -euo pipefail
BASE=567d5925fe896a7b70043271f5621d6fe6e650d4
HERE=$(cd "$(dirname "$0")" && pwd -P)
P=$HERE/sglang_picokv.patch
[ $# -ge 1 ] || { sed -n 4,7p "$0"; exit 2; }
SG=$(cd "$1" && pwd -P)
MODE=${2:-}
[ -d "$SG/python/sglang/srt" ] || { echo "[apply] $SG has no python/sglang/srt" >&2; exit 1; }

if [ "$(git -C "$SG" rev-parse --show-toplevel 2>/dev/null)" = "$SG" ]; then
  head=$(git -C "$SG" rev-parse HEAD)
  [ "$head" = "$BASE" ] || echo "[apply] WARNING: $SG is at $head; patch was made against $BASE" >&2
  chk()  { git -C "$SG" apply --check "$P"; }
  rchk() { git -C "$SG" apply -R --check "$P" 2>/dev/null; }
  doap() { git -C "$SG" apply "$P"; }
else
  # not a git checkout (e.g. a `git archive` export): plain patch -p1
  chk()  { patch -d "$SG" -p1 -f -s --dry-run < "$P" >/dev/null; }
  rchk() { patch -d "$SG" -p1 -f -s -R --dry-run < "$P" >/dev/null 2>&1; }
  doap() { patch -d "$SG" -p1 -f -s < "$P"; }
fi

if rchk; then
  echo "[apply] patch already applied, skipping"
elif chk; then
  doap
  echo "[apply] patched $(grep -c '^diff --git' "$P") sglang files"
else
  echo "[apply] patch does not apply cleanly to $SG" >&2; exit 1
fi

# new files (not in the patch): the GQA backend + its PICOKV_RADIX=host radix cache, the GLM / MLA
# backend (picokv_mla) and its decode kernel; <path under sglang_integration>:<path under python/sglang>
for e in picokv_backend.py:srt/layers/attention/picokv_backend.py \
         picokv_radix_cache.py:srt/mem_cache/picokv_radix_cache.py \
         picokv_mla/__init__.py:srt/layers/attention/picokv_mla/__init__.py \
         picokv_mla/backend.py:srt/layers/attention/picokv_mla/backend.py \
         picokv_mla/codec.py:srt/layers/attention/picokv_mla/codec.py \
         picokv_mla/decode_i8.py:srt/layers/attention/picokv_mla/decode_i8.py \
         picokv_mla/pool.py:srt/layers/attention/picokv_mla/pool.py \
         picokv_mla/radix.py:srt/layers/attention/picokv_mla/radix.py \
         picokv_mla_sm90/decode_i8.cuh:kernels/jit/csrc/picokv_mla_sm90/decode_i8.cuh; do
  src=$HERE/${e%%:*}; dst=$SG/python/sglang/${e#*:}
  mkdir -p "$(dirname "$dst")"
  rm -f "$dst"
  if [ "$MODE" = --link ]; then
    ln -s "$src" "$dst"; echo "[apply] linked $dst"
  else
    cp "$src" "$dst"; echo "[apply] copied $dst"
  fi
done
