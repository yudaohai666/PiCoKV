#!/bin/bash
# Install the PiCoKV attention backend into an sglang source tree.
#
# usage: apply.sh <sglang_dir> [--link]
#   <sglang_dir>  root of an sglang source tree (the dir containing python/sglang)
#   --link        symlink picokv_backend.py instead of copying it (edits in
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

dst=$SG/python/sglang/srt/layers/attention/picokv_backend.py
rm -f "$dst"
if [ "$MODE" = --link ]; then
  ln -s "$HERE/picokv_backend.py" "$dst"; echo "[apply] linked $dst"
else
  cp "$HERE/picokv_backend.py" "$dst"; echo "[apply] copied $dst"
fi
