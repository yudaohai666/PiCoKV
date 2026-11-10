"""PiCoKV for MLA latent pools (GLM-4.7-Flash): ``codec`` (b1280 encode / fill kernels), ``pool`` (page-4 arena pool and
allocator), ``backend`` (fa3 subclass ``picokv_mla``), ``radix`` (R-exact radix tree). The package itself imports no
Triton code."""

import os

PICOKV_MLA_BACKEND = "picokv_mla"
SINK = int(os.environ.get("SGLANG_PICOKV_MLA_SINK", 64))
LOCAL = int(os.environ.get("SGLANG_PICOKV_MLA_LOCAL", 256))
CHUNK = int(os.environ.get("SGLANG_PICOKV_MLA_CHUNK", 64))
# decode over the codes (path B, decode_i8.py) instead of the bf16 scratch (path A)
PATH_B = os.environ.get("SGLANG_PICOKV_MLA_PATH_B", "1") == "1"
DW = SINK + LOCAL + CHUNK  # dense rows of a decode row at most: sink + local window + one slide chunk
# path B: max running requests = rows of the decode workspace
MAX_RUNNING = int(os.environ.get("SGLANG_PICOKV_MLA_MAX_RUNNING", 256))
# radix cache through the R-exact tree (radix.py); 0: picokv_mla needs --disable-radix-cache
RADIX = os.environ.get("SGLANG_PICOKV_MLA_RADIX", "1") == "1"
# a prefill chunk that would cross the fire point F (a teacher-forced prompt, a retracted request's prompt + output)
# ends at F, so every token at or past F reads the compressed rows (a chunk crossing F would fire after the tokens past
# F read them exact)
FIRE_CAP = os.environ.get("SGLANG_PICOKV_MLA_FIRE_CAP", "1") == "1"
_CAP_ON = []


def picokv_mla_ws_rows() -> int:
    """Rows of the path B decode workspace: max_running_requests capped at ``MAX_RUNNING``."""
    from sglang.srt.runtime_context import get_schedule

    n = get_schedule().max_running_requests
    return min(n, MAX_RUNNING) if n else MAX_RUNNING


def picokv_mla_radix_on() -> bool:
    from sglang.srt.runtime_context import get_memory

    return RADIX and not get_memory().disable_radix_cache


def picokv_mla_tree_cache(ctx):
    """The R-exact radix tree when picokv_mla runs with the radix cache on, else None."""
    from sglang.srt.runtime_context import attention_backends

    if PICOKV_MLA_BACKEND not in attention_backends():
        return None
    if ctx.disable_radix_cache or not RADIX:
        return None
    from sglang.srt.layers.attention.picokv_mla.radix import PicoKVMLARadixCache

    return PicoKVMLARadixCache(ctx.params)


def picokv_mla_cell_size_bytes(layers: int) -> int:
    from sglang.srt.layers.attention.picokv_mla.pool import picokv_mla_cell_size_bytes as f

    return f(layers)


def picokv_mla_fixed_bytes(layers: int) -> int:
    from sglang.srt.layers.attention.picokv_mla.pool import picokv_mla_fixed_bytes as f

    return f(layers)


def picokv_mla_chunk_cap(req, prefix_len: int, limit: int) -> int:
    """Prefill chunk limit of req: F - prefix_len when picokv_mla runs and its fire point F lies within this chunk
    before the end of the fill (a teacher-forced prompt, or a retracted request's prompt + output), else ``limit``."""
    if not _CAP_ON:
        from sglang.srt.runtime_context import attention_backends

        _CAP_ON.append(FIRE_CAP and PICOKV_MLA_BACKEND in attention_backends())
    if not _CAP_ON[0]:
        return limit
    f = picokv_mla_fire_point(req)
    k = f - prefix_len
    return k if 0 < k < limit and f < len(req.full_untruncated_fill_ids) else limit


def picokv_mla_fire_point(req) -> int:
    from sglang.srt.layers.attention.picokv_mla.backend import picokv_mla_fire_point as f

    return f(req)
