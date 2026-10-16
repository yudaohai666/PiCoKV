#!/bin/bash
# Maintainer helper: refresh sglang_picokv.patch + picokv_backend.py from a
# working sglang checkout (the inverse of apply.sh).
#
# usage: export_from_sglang.sh <sglang_git_checkout>
set -euo pipefail
BASE=567d5925fe896a7b70043271f5621d6fe6e650d4
HERE=$(cd "$(dirname "$0")" && pwd -P)
[ $# -ge 1 ] || { sed -n 5p "$0"; exit 2; }
SG=$(cd "$1" && pwd -P)
FILES=(
  python/sglang/srt/layers/attention/attention_registry.py
  python/sglang/srt/mem_cache/kv_cache_configurator.py
  python/sglang/srt/model_executor/forward_batch_info.py
  python/sglang/srt/model_executor/pool_configurator.py
  python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py
  python/sglang/srt/model_executor/runner_backend/breakable_cuda_graph_backend.py
)
git -C "$SG" diff "$BASE" -- "${FILES[@]}" > "$HERE/sglang_picokv.patch"
cp "$SG/python/sglang/srt/layers/attention/picokv_backend.py" "$HERE/picokv_backend.py"
# Anything else modified vs BASE is NOT exported -- list it so it is not lost silently.
extra=$(git -C "$SG" diff --name-only "$BASE" -- python | grep -vxF -f <(printf '%s\n' "${FILES[@]}") || true)
[ -z "$extra" ] || { echo "[export] WARNING: also modified vs base (not exported):"; echo "$extra"; }
echo "[export] $(grep -c '^diff --git' "$HERE/sglang_picokv.patch") files -> sglang_picokv.patch; backend copied"
