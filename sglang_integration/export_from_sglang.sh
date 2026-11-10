#!/bin/bash
# Maintainer helper: refresh sglang_picokv.patch + the new files (not in the patch: picokv_backend.py,
# picokv_radix_cache.py, picokv_mla/*.py, picokv_mla_sm90/decode_i8.cuh) from a working sglang checkout
# (the inverse of apply.sh).
# usage: export_from_sglang.sh <sglang_git_checkout>
set -euo pipefail
BASE=567d5925fe896a7b70043271f5621d6fe6e650d4
HERE=$(cd "$(dirname "$0")" && pwd -P)
[ $# -ge 1 ] || { sed -n 5p "$0"; exit 2; }
SG=$(cd "$1" && pwd -P)
FILES=(
  python/sglang/srt/arg_groups/attention_hook.py
  python/sglang/srt/arg_groups/choices.py
  python/sglang/srt/arg_groups/pipeline.py
  python/sglang/srt/beam_search/coordinator.py
  python/sglang/srt/layers/attention/attention_registry.py
  python/sglang/srt/managers/schedule_batch.py
  python/sglang/srt/managers/schedule_policy.py
  python/sglang/srt/managers/scheduler_components/batch_result_processor.py
  python/sglang/srt/mem_cache/kv_cache_configurator.py
  python/sglang/srt/mem_cache/registry.py
  python/sglang/srt/model_executor/forward_batch_info.py
  python/sglang/srt/model_executor/model_runner.py
  python/sglang/srt/model_executor/pool_configurator.py
  python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py
  python/sglang/srt/model_executor/runner_backend/breakable_cuda_graph_backend.py
  python/sglang/srt/model_executor/runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py
  python/sglang/srt/models/deepseek_common/attention_backend_handler.py
  python/sglang/srt/models/deepseek_common/utils.py
  python/sglang/srt/models/utils.py
)
# new files: sglang path -> path under sglang_integration/ (see apply.sh)
NEW=(
  python/sglang/srt/layers/attention/picokv_backend.py:picokv_backend.py
  python/sglang/srt/mem_cache/picokv_radix_cache.py:picokv_radix_cache.py
  python/sglang/srt/layers/attention/picokv_mla/__init__.py:picokv_mla/__init__.py
  python/sglang/srt/layers/attention/picokv_mla/backend.py:picokv_mla/backend.py
  python/sglang/srt/layers/attention/picokv_mla/codec.py:picokv_mla/codec.py
  python/sglang/srt/layers/attention/picokv_mla/decode_i8.py:picokv_mla/decode_i8.py
  python/sglang/srt/layers/attention/picokv_mla/pool.py:picokv_mla/pool.py
  python/sglang/srt/layers/attention/picokv_mla/radix.py:picokv_mla/radix.py
  python/sglang/kernels/jit/csrc/picokv_mla_sm90/decode_i8.cuh:picokv_mla_sm90/decode_i8.cuh
)
git -C "$SG" diff "$BASE" -- "${FILES[@]}" > "$HERE/sglang_picokv.patch"
for e in "${NEW[@]}"; do
  mkdir -p "$(dirname "$HERE/${e#*:}")"; cp "$SG/${e%%:*}" "$HERE/${e#*:}"
done
# Anything else modified or added vs BASE is NOT exported -- list it so it is not lost silently.
extra=$( { git -C "$SG" diff --name-only "$BASE" -- python; git -C "$SG" ls-files --others --exclude-standard -- python; } \
  | grep -vxF -f <(printf '%s\n' "${FILES[@]}" "${NEW[@]%%:*}") || true)
[ -z "$extra" ] || { echo "[export] WARNING: also modified vs base (not exported):"; echo "$extra"; }
echo "[export] $(grep -c '^diff --git' "$HERE/sglang_picokv.patch") files -> sglang_picokv.patch; ${#NEW[@]} new files copied"
