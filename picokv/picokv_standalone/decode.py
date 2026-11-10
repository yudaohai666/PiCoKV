"""Single-token CUDA attention over dense-retained and packed-global KV."""

from __future__ import annotations

import math
import os
import re
import time
from typing import Callable, Optional

import torch

from .cache import PreparedKVCache
from .constants import PROJECTED_QUERY_BITS

# Optional decode-step breakdown profiling (env PICOKV_PROFILE=1). Accumulates
# per-op wall time (sync'd) to bound how much the exact dense-window attention
# costs relative to the compressed-global CUDA kernel -- i.e. option A's ceiling.
_PROF = {"dense": 0.0, "proj": 0.0, "kernel": 0.0, "n": 0}

# Option A: serve the exact dense-retained-window attention with flashinfer's
# fused single-query kernel instead of torch SDPA / the matmul fallback, when
# PICOKV_EXACT=flashinfer. Profiling shows this window (not the compressed
# kernel) dominates decode, so this is where a faster exact kernel can help.
_EXACT_BACKEND = os.environ.get("PICOKV_EXACT", "sdpa").lower()
_LN2 = 0.6931471805599453  # flashinfer returns log2 LSE; tail kernel wants ln.
# Value projection of the split tail: cuBLAS GEMM (default) or the scalar
# fused kernel (PICOKV_PROJ_GEMM=0), the same switch the fused tail op reads.
# Parsed like the op's getenv (C atoi: "00" / "off" -> 0) so both agree.
_PROJ_GEMM_ENV = re.match(r"\s*([+-]?\d+)", os.environ.get("PICOKV_PROJ_GEMM", "1"))
_PROJ_GEMM = 1 if "PICOKV_PROJ_GEMM" not in os.environ or (
    _PROJ_GEMM_ENV is not None and int(_PROJ_GEMM_ENV.group(1)) != 0) else 0
# Shared projections (shared_proj.py): a cache whose key_projection_t /
# value_projection are stride-0 batch broadcasts of one copy runs the query
# and value projections as Triton kernels around the global-only tail op.
# PICOKV_SHARED_PROJ=0: callers keep per-row replicas (cuBLAS path).
_SHARED_PROJ = os.environ.get("PICOKV_SHARED_PROJ", "1") != "0"
# Deterministic shared tail: the tensor-core global kernel writes per-block
# partials that a fixed-order Triton reduction sums (instead of float
# atomics, whose order varies run to run). PICOKV_DET=1, or set by the
# sglang backend; needs picokv_decode_tail_global_part_fn.
DETERMINISTIC = os.environ.get("PICOKV_DET", "0") == "1"
# One-row caches (batch 1) take the shared Triton projections too (their
# projections are trivially shared): one launch fewer per layer than the
# cuBLAS GEMM + epilogue pair and ~3 us faster (t57). PICOKV_SHARED_ONE=0
# keeps the cuBLAS path (bitwise the pre-t57 batch-1 output).
_SHARED_ONE = os.environ.get("PICOKV_SHARED_ONE", "1") == "1"
# Shared tail without the prep op: the Triton qproj also zeroes the
# accumulators and the tensor-core tail quantizes + packs each block's query
# heads itself from the bf16 projected query (picokv_decode_tail_global_q_fn;
# bitwise the prep op's codes, one launch fewer per layer, t62).
# PICOKV_TAIL_QUANT=0 keeps the prep op; PICOKV_DET always does.
_TAIL_QUANT = os.environ.get("PICOKV_TAIL_QUANT", "1") == "1"
# ... only up to this many tail blocks (sched entries x kv heads): each block
# pays the quantizer prologue (~1.2 us), which past a wave of blocks costs
# more than the launch it saves (t62/mb/tail_long.py: break-even ~1024
# working blocks; 8 rows: -1.5 us per layer at 4k, +9.8 at 32k).
_TAIL_QUANT_BLOCKS = int(os.environ.get("PICOKV_TAIL_QUANT_BLOCKS", "1024"))
# fork_large (the sglang backend's PICOKV_GRAPH_FORK=auto with FA3): the
# dense branch forks off on a tail grid above that gate only at B <= lo or
# B >= hi (PICOKV_FORK_B="lo,hi"). t68 bench_itl, ms/step serial -> forked:
# 122880x1 8.19 -> 8.10, 16384x8 8.39 -> 8.34, 4096x48 9.95 -> 9.92,
# 4096x64 10.89 -> 10.85, r160 B 97-160 p50 -1.5%; but 8192x16 8.49 ->
# 8.53, 8192x24 9.44 -> 9.45, 8192x32 10.23 -> 10.29.
_FORK_B = tuple(int(x) for x in os.environ.get("PICOKV_FORK_B", "8,48").split(","))


def _c_atoi(s: Optional[str]) -> int:
    """C atoi (leading blanks, sign, digits; else 0): the parse of the
    extension's getenv switches, so this module and the kernels agree."""
    m = re.match(r"\s*([+-]?\d+)", s or "")
    return int(m.group(1)) if m else 0


# the extension's picokv_tc_tail_enabled(): off iff PICOKV_NO_TC is nonzero
# or PICOKV_TC_TAIL is set and zero ("false", "off" also parse as 0)
_TC_ON = (_c_atoi(os.environ.get("PICOKV_NO_TC")) == 0
          and (os.environ.get("PICOKV_TC_TAIL") is None
               or _c_atoi(os.environ.get("PICOKV_TC_TAIL")) != 0))
# Programmatic dependent launch in the shared tail (sm90, Triton gdc):
# the query projection runs first and lets the dense append start during
# it, and the value projection starts during the tail kernel, waiting for
# it before any access (the launch gaps of two kernels per layer; same
# kernels and arithmetic, bitwise the serial order). PICOKV_PDL=0: off.
_PDL = _c_atoi(os.environ.get("PICOKV_PDL", "1")) != 0
_pdl_ok = None


def pdl_enabled() -> bool:
    """PICOKV_PDL and supported (Triton gdc intrinsics, sm90+ device)."""
    global _pdl_ok
    if _pdl_ok is None:
        ok = _PDL and torch.cuda.is_available()
        if ok:
            from .shared_proj import PDL_OK
            ok = PDL_OK and torch.cuda.get_device_capability()[0] >= 9
        _pdl_ok = bool(ok)
    return _pdl_ok


_shared_ok = None
_tail_q_ok = None
_fi_prefill = None
_fi_wrapper = None
_fi_ws = None


def _get_ragged_wrapper(device):
    """Persistent flashinfer batched-ragged prefill wrapper (single workspace)."""
    global _fi_wrapper, _fi_ws
    if _fi_wrapper is None:
        import flashinfer
        _fi_ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
        _fi_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(_fi_ws, kv_layout="NHD")
    return _fi_wrapper


def _dense_attention_flashinfer(query, key, value, scale, dense_lens):
    """Exact dense-window attention via flashinfer, returning (out[B,H,S,D],
    lse[B,H,S]). Step 1 of the FA3/FA4 plan: for a batch (B>1) use flashinfer's
    BATCHED ragged kernel in ONE launch (was a per-request loop that regressed
    batched throughput); B==1 keeps the single-call fast path."""
    global _fi_prefill
    batch, q_heads, seq, hd = query.shape
    kv_heads = key.shape[1]
    kv_len = key.shape[2]
    dev = query.device

    if batch == 1:
        if _fi_prefill is None:
            from flashinfer import single_prefill_with_kv_cache as _fp
            _fi_prefill = _fp
        lb = int(dense_lens[0]) if dense_lens is not None else kv_len
        q = query[0].transpose(0, 1).contiguous()
        k = key[0, :, :lb].transpose(0, 1).contiguous()
        v = value[0, :, :lb].transpose(0, 1).contiguous()
        o, lse2 = _fi_prefill(q, k, v, causal=False, sm_scale=float(scale), return_lse=True)
        return (o.transpose(0, 1).unsqueeze(0),
                (lse2 * _LN2).transpose(0, 1).unsqueeze(0).contiguous())

    # Batched: one ragged-KV launch across all requests.
    q_flat = query.permute(0, 2, 1, 3).reshape(batch * seq, q_heads, hd).contiguous()
    qo_indptr = torch.arange(0, batch + 1, device=dev, dtype=torch.int32) * seq
    if dense_lens is None:
        lens = torch.full((batch,), kv_len, device=dev, dtype=torch.int32)
        k_flat = key.permute(0, 2, 1, 3).reshape(batch * kv_len, kv_heads, hd).contiguous()
        v_flat = value.permute(0, 2, 1, 3).reshape(batch * kv_len, kv_heads, hd).contiguous()
    else:
        # One .tolist() sync instead of per-request int(lens[b]) syncs.
        lens_list = dense_lens.tolist()
        lens = dense_lens.to(device=dev, dtype=torch.int32)
        ks, vs = [], []
        for b in range(batch):
            lb = lens_list[b]
            ks.append(key[b, :, :lb].transpose(0, 1))
            vs.append(value[b, :, :lb].transpose(0, 1))
        k_flat = torch.cat(ks, 0).contiguous()
        v_flat = torch.cat(vs, 0).contiguous()
    kv_indptr = torch.zeros(batch + 1, device=dev, dtype=torch.int32)
    kv_indptr[1:] = torch.cumsum(lens, 0)
    w = _get_ragged_wrapper(dev)
    w.plan(qo_indptr, kv_indptr, q_heads, kv_heads, hd, causal=False,
           sm_scale=float(scale), q_data_type=query.dtype, kv_data_type=key.dtype)
    o, lse2 = w.run(q_flat, k_flat, v_flat, return_lse=True)   # [B*S,Hq,D], [B*S,Hq]
    output = o.reshape(batch, seq, q_heads, hd).permute(0, 2, 1, 3).contiguous()
    lse = (lse2 * _LN2).reshape(batch, seq, q_heads).permute(0, 2, 1).contiguous()
    return output, lse

try:
    from picokv.picokv_cuda_extension.picokv_cuda import load_picokv_extension
except Exception:  # pragma: no cover - reported by _decode_cuda
    load_picokv_extension = None


def _append_decode_tokens(
    cache: PreparedKVCache,
    decode_key: Optional[torch.Tensor],
    decode_value: Optional[torch.Tensor],
    update_cache: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return dense K/V including current tokens, optionally persisting them."""
    if decode_key is None and decode_value is None:
        return cache.dense_key, cache.dense_value
    if decode_key is None or decode_value is None:
        raise ValueError("decode_key and decode_value must be provided together")
    if decode_key.dim() != 4 or tuple(decode_key.shape) != tuple(decode_value.shape):
        raise ValueError("decode_key/decode_value must have the same [B,KVH,T,D] shape")
    expected = (*cache.dense_key.shape[:2], decode_key.shape[-2], cache.head_dim)
    if tuple(decode_key.shape) != tuple(expected):
        raise ValueError(f"decode K/V shape must be {tuple(expected)}; got {tuple(decode_key.shape)}")
    if decode_key.device != cache.device or decode_value.device != cache.device:
        raise ValueError("decode_key/decode_value must be on the cache device")
    if decode_key.dtype != cache.dtype or decode_value.dtype != cache.dtype:
        raise TypeError("decode_key/decode_value dtype must match the cache")

    dense_key = torch.cat((cache.dense_key, decode_key.contiguous()), dim=-2)
    dense_value = torch.cat((cache.dense_value, decode_value.contiguous()), dim=-2)
    if update_cache:
        cache.dense_key = dense_key
        cache.dense_value = dense_value
    return dense_key, dense_value


def _repeat_kv_heads(states: torch.Tensor, query_heads: int) -> torch.Tensor:
    """Expand KV heads only for the non-FlashAttention compatibility path."""
    kv_heads = int(states.shape[1])
    if query_heads == kv_heads:
        return states
    if query_heads % kv_heads:
        raise ValueError(f"QH must be divisible by KVH; got QH={query_heads}, KVH={kv_heads}")
    return states.repeat_interleave(query_heads // kv_heads, dim=1)


def _dense_attention_fallback(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    dense_lens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute dense retained attention and LSE without FlashAttention.

    ``dense_lens`` (int32 ``[B]``) marks each batch element's real dense length;
    positions >= that length are masked (used by the ragged/varlen batched path
    where the dense window is padded to a common length).

    GQA runs as a broadcast batched matmul ([B,KVH,G,1,D] @ [B,KVH,1,D,L]) so
    KV heads are never materialized by ``repeat_interleave``; fp32 is confined
    to the small [B,QH,1,L] scores tensor for softmax/LSE stability."""
    batch, q_heads, seq, head_dim = query.shape
    kv_heads = int(key.shape[1])
    group = q_heads // kv_heads
    q = (query.reshape(batch, kv_heads, group, seq, head_dim) * float(scale)).to(query.dtype)
    scores = torch.matmul(q, key.transpose(-1, -2).unsqueeze(2))  # [B,KVH,G,1,L]
    if dense_lens is not None:
        dense_len = int(scores.shape[-1])
        pos = torch.arange(dense_len, device=scores.device)
        keep = pos[None, :] < dense_lens.to(scores.device)[:, None]  # [B, L]
        scores = scores.masked_fill(
            ~keep[:, None, None, None, :], float("-inf")
        )
    scores32 = scores.float()
    lse = torch.logsumexp(scores32, dim=-1)  # [B,KVH,G,1]
    weights = torch.softmax(scores32, dim=-1).to(query.dtype)  # [B,KVH,G,1,L]
    out = torch.matmul(weights, value.unsqueeze(2))  # [B,KVH,G,1,D]
    return (
        out.reshape(batch, q_heads, seq, head_dim),
        lse.reshape(batch, q_heads, seq).contiguous(),
    )


def _dense_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    dense_lens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact dense-retained output and log-sum-exp normalizer."""
    # flashinfer helps only single-stream (batch==1): the dense window is tiny
    # (~sink+local), so at batch>1 SDPA already amortizes its launch over the
    # batch and flashinfer's per-step plan overhead makes it parity/worse. So
    # "flashinfer" => b1 only; "flashinfer_batch" forces it (measurement).
    if _EXACT_BACKEND in ("flashinfer", "flashinfer_batch") and query.is_cuda:
        if query.shape[0] == 1 or _EXACT_BACKEND == "flashinfer_batch":
            return _dense_attention_flashinfer(query, key, value, scale, dense_lens)
    if dense_lens is not None:
        # Truly ragged dense window (mixed-length batch): the broadcast-GEMM
        # fallback has no cat/plan() overhead and no GPU sync, so it stays the
        # default; flashinfer's ragged kernel remains reachable for measurement
        # via PICOKV_EXACT=flashinfer_batch (it launches 1 kernel but its
        # per-step plan() costs more than it saves at these small windows).
        return _dense_attention_fallback(query, key, value, scale, dense_lens)
    if not hasattr(
        torch.ops.aten, "_scaled_dot_product_flash_attention"
    ):
        return _dense_attention_fallback(query, key, value, scale, dense_lens)
    output, lse, *_ = torch.ops.aten._scaled_dot_product_flash_attention(
        query,
        key,
        value,
        0.0,
        False,
        False,
        scale=float(scale),
    )
    return output, lse.contiguous()


def _project_query_into_workspace(
    query: torch.Tensor,
    cache: PreparedKVCache,
    output: torch.Tensor,
) -> None:
    """Project GQA query heads with one strided batched GEMM and no allocation."""
    batch, query_heads, seq, head_dim = query.shape
    kv_heads = cache.kv_heads
    group_size = query_heads // kv_heads
    grouped_query = query.reshape(
        batch * kv_heads,
        group_size * seq,
        head_dim,
    )
    grouped_projection = cache.key_projection_t.reshape(
        batch * kv_heads,
        head_dim,
        cache.key_bits,
    )
    grouped_output = output.reshape(
        batch * kv_heads,
        group_size * seq,
        cache.key_bits,
    )
    torch.bmm(grouped_query, grouped_projection, out=grouped_output)


def shared_tail_supported() -> bool:
    """Whether decode can consume shared (stride-0 broadcast) projections:
    PICOKV_SHARED_PROJ on, triton importable, and an extension exporting the
    global-only tail op (picokv_decode_tail_global_fn)."""
    global _shared_ok
    if _shared_ok is None:
        _shared_ok = False
        if _SHARED_PROJ and load_picokv_extension is not None:
            try:
                from . import shared_proj  # noqa: F401
                _shared_ok = hasattr(load_picokv_extension(), "picokv_decode_tail_global_fn")
            except Exception as e:  # e.g. no triton; say why the op looks missing
                import logging

                logging.getLogger(__name__).warning(
                    "picokv: shared projections unavailable: %r", e)
                _shared_ok = False
    return _shared_ok


def _shared_cache(cache: PreparedKVCache, allow: bool = True) -> bool:
    """The cache's projections are shared broadcasts (see shared_proj.py).
    Without shared-tail support (or ``allow``) they are materialized here:
    the cuBLAS value GEMM indexes value_projection as B dense copies."""
    kpt, vp = cache.key_projection_t, cache.value_projection
    if (DETERMINISTIC or _SHARED_ONE) and allow and kpt.shape[0] == 1 and vp.shape[0] == 1:
        # one row: its projections are trivially "shared" (and the fused
        # cuBLAS tail op has no deterministic variant)
        if kpt[0].is_contiguous() and vp[0].is_contiguous() and shared_tail_supported():
            return True
    if kpt.shape[0] == 1 or (kpt.stride(0) != 0 and vp.stride(0) != 0):
        return False
    if (allow and kpt.stride(0) == 0 and vp.stride(0) == 0 and kpt[0].is_contiguous()
            and vp[0].is_contiguous() and shared_tail_supported()):
        return True
    cache.key_projection_t = kpt.contiguous()
    cache.value_projection = vp.contiguous()
    return False


def _tail_quant(extension, cache, projected_query, sr=None) -> bool:
    """The shared tail can skip the prep op (see _TAIL_QUANT): the fused op
    exists, the shape is the tensor-core kernel's (GQA-4, Ck = Cv = 320,
    bf16 projected query, single-token decode) and the grid is small
    (_TAIL_QUANT_BLOCKS; ``sr``: the sched table, else 512-token blocks
    of the packed width per row)."""
    global _tail_q_ok
    if DETERMINISTIC:  # read every call: the sglang backend sets it after import
        return False
    if _tail_q_ok is None:
        _tail_q_ok = (_TAIL_QUANT and _TC_ON
                      and hasattr(extension, "picokv_decode_tail_global_q_fn"))
    B, QH, S, C = projected_query.shape
    KVH = int(cache.packed_key.shape[1])
    if not (_tail_q_ok and projected_query.dtype == torch.bfloat16 and S == 1
            and C == 320 and cache.value_bits == 320 and cache.exp_bits >= 4
            and QH == 4 * KVH):
        return False
    return _tail_blocks(cache, projected_query, sr) <= _TAIL_QUANT_BLOCKS


def _tail_blocks(cache, projected_query, sr=None) -> int:
    """Tail grid size the fused-quantizer gate compares with
    _TAIL_QUANT_BLOCKS: ``sr`` (the sched table) entries, else 512-token
    blocks of the packed width per row; times the KV heads."""
    gx = (sr.numel() if sr is not None and sr.numel() > 0
          else projected_query.shape[0] * ((int(cache.packed_key.shape[-1]) + 511) // 512))
    return gx * int(cache.packed_key.shape[1])


def shared_tail_prep(extension, cache, query, workspace, sr=None, pdl=False):
    """The shared tail's query side (needs only q): Triton query projection,
    then either the prep op (quantize+pack, zero accumulators) or, under
    _tail_quant, the accumulator zeroing inside the projection launch.
    Returns whether the global kernel must quantize the query itself.
    ``pdl``: the projection triggers the next kernel's programmatic launch."""
    from .shared_proj import shared_query_projection
    projected_query, packed_query, query_scale, global_code, global_den, _ = workspace
    if _tail_quant(extension, cache, projected_query, sr):
        shared_query_projection(query, cache.key_projection_t, projected_query,
                                zero=(global_code, global_den), pdl=pdl)
        return True
    shared_query_projection(query, cache.key_projection_t, projected_query)
    extension.picokv_decode_tail_prep_fn(
        projected_query, packed_query, query_scale, global_code, global_den)
    return False


def _shared_tail_rest(extension, cache, query, workspace, dense_output, dense_lse,
                      output, scale, gl, sr, st, projected=None):
    """Shared-projection tail: the query side (shared_tail_prep, unless done:
    ``projected`` is then its result), the global-only kernel, then the
    Triton value projection + epilogue into ``output``."""
    from .shared_proj import shared_value_projection
    projected_query, packed_query, query_scale, global_code, global_den, _ = workspace
    if projected is None:
        projected = shared_tail_prep(extension, cache, query, workspace, sr)
    if projected:  # fused quantizer: prep skipped
        e = torch.empty(0, dtype=torch.int32, device=query.device)
        rows = cache.row_tables if cache.row_tables is not None else (e, e, e)
        extension.picokv_decode_tail_global_q_fn(
            projected_query, packed_query, cache.packed_key, cache.packed_value,
            query_scale, dense_lse.contiguous(), global_code, global_den,
            cache.exp_bits, 256, 1, float(scale), gl, sr, st, *rows)
    elif DETERMINISTIC:
        _det_tail_global(extension, cache, packed_query, query_scale, dense_lse,
                         global_code, global_den, scale, gl, sr, st)
    elif cache.row_tables is not None:  # per-row packed storage
        extension.picokv_decode_tail_global_rows_fn(
            packed_query, cache.packed_key, cache.packed_value, query_scale,
            dense_lse.contiguous(), global_code, global_den, cache.exp_bits,
            256, 1, float(scale), gl, sr, st, *cache.row_tables)
    else:
        extension.picokv_decode_tail_global_fn(
            packed_query, cache.packed_key, cache.packed_value, query_scale,
            dense_lse.contiguous(), global_code, global_den, cache.exp_bits,
            256, 1, float(scale), gl, sr, st)
    shared_value_projection(dense_output, global_code, global_den,
                            cache.value_projection, output, pdl=pdl_enabled())
    return output


def _det_tail_global(extension, cache, packed_query, query_scale, dense_lse,
                     global_code, global_den, scale, gl, sr, st):
    """Deterministic global tail: per-block partials (scratch from the
    caching allocator / the capturing graph's pool) + det_tail_reduce, which
    writes every glob_code / glob_den entry."""
    from .shared_proj import det_tail_reduce
    B, QH, S = global_code.shape[:3]
    KVH = QH // 4
    L = int(cache.packed_key.shape[3])  # transposed keys [B, KVH, KB, L]
    dev = global_code.device
    sched = sr.numel() > 0
    if sched:
        gx, gz, blk = int(sr.numel()), KVH, 512
    else:
        blk = 256 * int(extension.picokv_tail_block_tiles_fn(L, B * KVH))
        gx, gz = (L + blk - 1) // blk, B * KVH
    slots = S * gz * gx
    part_code = torch.empty(slots * 4 * 320, dtype=torch.float32, device=dev)
    part_den = torch.empty(slots * 4, dtype=torch.float32, device=dev)
    part_rs = torch.empty(B if sched else 0, dtype=torch.int32, device=dev)
    rows = cache.row_tables
    if rows is None:
        e = torch.empty(0, dtype=torch.int32, device=dev)
        rows = (e, e, e)
    extension.picokv_decode_tail_global_part_fn(
        packed_query, cache.packed_key, cache.packed_value, query_scale,
        dense_lse.contiguous(), global_code, global_den, cache.exp_bits,
        256, 1, float(scale), gl, sr, st, *rows, part_code, part_den, part_rs)
    det_tail_reduce(part_code, part_den, part_rs if sched else None, gl,
                    global_code, global_den, gx, L, blk)


def _get_workspace(
    cache: PreparedKVCache,
    query: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Reuse all decode intermediates for a stable steady-state token path."""
    batch, query_heads, seq, head_dim = query.shape
    key = (
        query.device.index,
        query.dtype,
        batch,
        query_heads,
        seq,
        head_dim,
        cache.key_bits,
        cache.value_bits,
    )
    workspace = cache._workspace.get(key)
    if workspace is None:
        workspace = (
            # Projected query before int8 quantization.
            torch.empty(
                batch,
                query_heads,
                seq,
                cache.key_bits,
                device=query.device,
                dtype=query.dtype,
            ),
            # Eight packed bitplanes for each group of 64 projected dimensions.
            torch.empty(
                batch,
                query_heads,
                seq,
                cache.key_bits // 64,
                PROJECTED_QUERY_BITS,
                device=query.device,
                dtype=torch.long,
            ),
            # Per-query scale produced by projected-query quantization.
            torch.empty(batch, query_heads, seq, device=query.device, dtype=torch.float32),
            # Accumulated value code and global softmax denominator.
            torch.empty(
                batch,
                query_heads,
                seq,
                cache.value_bits,
                device=query.device,
                dtype=torch.float32,
            ),
            torch.empty(batch, query_heads, seq, device=query.device, dtype=torch.float32),
            # Final normalized attention output.
            torch.empty(
                batch,
                query_heads,
                seq,
                head_dim,
                device=query.device,
                dtype=query.dtype,
            ),
        )
        cache._workspace[key] = workspace
    return workspace


def _decode_cuda(
    query: torch.Tensor,
    cache: PreparedKVCache,
    dense_key: torch.Tensor,
    dense_value: torch.Tensor,
    *,
    scale: float,
    use_pruning: bool,
    prune_mode: int,
    log_eps: float,
    global_lens: Optional[torch.Tensor] = None,
    dense_lens: Optional[torch.Tensor] = None,
    sched_req: Optional[torch.Tensor] = None,
    sched_tile: Optional[torch.Tensor] = None,
    dense_attn: Optional[tuple] = None,
    out: Optional[torch.Tensor] = None,
    dense_fn: Optional[Callable[[], tuple]] = None,
    dense_stream: Optional[torch.cuda.Stream] = None,
    query_first: bool = False,
    fork_large: bool = False,
) -> torch.Tensor:
    """Launch the measured GQA-4 CUDA decode path."""
    if load_picokv_extension is None:
        raise RuntimeError("PiCoKV decode CUDA extension is unavailable")
    extension = load_picokv_extension()
    projected_query, packed_query, query_scale, global_code, global_den, output = (
        _get_workspace(cache, query)
    )
    if out is not None:
        output = out
    shared = _shared_cache(cache, allow=not use_pruning)
    if cache.row_tables is not None and not shared:
        raise RuntimeError("per-row packed storage needs the shared-projection tail")

    if use_pruning:
        if dense_fn is not None or out is not None:
            raise ValueError("dense_fn/out are only supported without pruning")
        if not hasattr(
            extension,
            "picokv_decode_tail_v1_exp8_tiled_gqa4_atomic_pruned_fn",
        ):
            raise RuntimeError("loaded PiCoKV CUDA extension lacks the pruning kernel")

        # Dense retained attention and query projection are independent.  Their
        # existing two-stream overlap is kept verbatim because it is part of
        # steady-state decode latency, unlike the removed async prepare path.
        current_stream = torch.cuda.current_stream(query.device)
        if cache._dense_stream is None:
            cache._dense_stream = torch.cuda.Stream(device=query.device)
        cache._dense_stream.wait_stream(current_stream)
        with torch.cuda.stream(cache._dense_stream):
            dense_output, dense_lse = _dense_attention(
                query,
                dense_key,
                dense_value,
                scale,
            )
        _project_query_into_workspace(query, cache, projected_query)
        current_stream.wait_stream(cache._dense_stream)

        extension.picokv_decode_tail_v1_exp8_tiled_gqa4_atomic_pruned_fn(
            projected_query,
            packed_query,
            cache.packed_key,
            cache.packed_value,
            query_scale,
            dense_lse.contiguous(),
            dense_output.contiguous(),
            global_code,
            global_den,
            cache.value_projection,
            output,
            cache.exp_bits,
            256,
            1,
            float(scale),
            int(prune_mode),
            float(log_eps),
        )
        return output

    gl = (
        global_lens
        if global_lens is not None
        else torch.empty(0, dtype=torch.int32, device=query.device)
    )
    empty_i32 = torch.empty(0, dtype=torch.int32, device=query.device)
    sr = sched_req if sched_req is not None else empty_i32
    st = sched_tile if sched_tile is not None else empty_i32

    if dense_fn is not None and not hasattr(extension, "picokv_decode_tail_main_fn"):
        dense_attn = dense_fn()  # older extension: no split tail, run serially
        dense_fn = None
    if dense_fn is not None:
        # Split tail: the dense-window attention (dense_fn, e.g. append +
        # window kernel) runs on dense_stream while this stream projects,
        # quantizes and packs the query; the global kernel joins on both.
        # Same kernels and order per stream as the fused tail op.
        # query_first (no dense_stream, shared projections, fused quantizer):
        # the query side, then dense_fn(pdl=True), launched PDL-chained after
        # the query projection (it triggers at entry). Else dense_fn() first:
        # the prep op cannot trigger, so the order buys nothing there.
        # fork_large: dense_stream only on a grid above the fused quantizer's
        # gate (projection + prep op, B160 ~13 us per layer, hide under the
        # dense branch) at the batch sizes of _FORK_B; else the serial order
        # (below the gate: the PDL chain) is faster.
        current_stream = torch.cuda.current_stream(query.device)
        ws = (projected_query, packed_query, query_scale, global_code, global_den, None)
        if fork_large and (_FORK_B[0] < query.shape[0] < _FORK_B[1]
                           or _tail_blocks(cache, projected_query, sr) <= _TAIL_QUANT_BLOCKS):
            dense_stream = None
        qf = (query_first and dense_stream is None and shared and pdl_enabled()
              and _tail_quant(extension, cache, projected_query, sr))

        def query_side():
            if shared:
                return shared_tail_prep(extension, cache, query, ws, sr, pdl=qf)
            _project_query_into_workspace(query, cache, projected_query)
            extension.picokv_decode_tail_prep_fn(
                projected_query, packed_query, query_scale, global_code, global_den)
            return False

        if qf:
            fused_q = query_side()
        if dense_stream is not None:
            dense_stream.wait_stream(current_stream)
            with torch.cuda.stream(dense_stream):
                dense_output, dense_lse = dense_fn()
        else:
            dense_output, dense_lse = dense_fn(pdl=True) if qf else dense_fn()
        if not qf:
            fused_q = query_side()
        if dense_stream is not None:
            current_stream.wait_stream(dense_stream)
            if not torch.cuda.is_current_stream_capturing():
                dense_output.record_stream(current_stream)
                dense_lse.record_stream(current_stream)
        if shared:
            return _shared_tail_rest(
                extension, cache, query, ws,
                dense_output, dense_lse, output, scale, gl, sr, st, projected=fused_q)
        extension.picokv_decode_tail_main_fn(
            packed_query,
            cache.packed_key,
            cache.packed_value,
            query_scale,
            dense_lse.contiguous(),
            dense_output.contiguous(),
            global_code,
            global_den,
            cache.value_projection,
            output,
            cache.exp_bits,
            256,
            1,
            float(scale),
            gl,
            sr,
            st,
            _PROJ_GEMM,
        )
        return output

    # The unpruned reference path is intentionally serial and uses the matching
    # non-pruned CUDA tail kernel for accuracy/performance comparisons.
    _prof = os.environ.get("PICOKV_PROFILE")

    def _mark():
        if _prof:
            torch.cuda.synchronize()
            return time.perf_counter()
        return 0.0

    _a = _mark()
    if dense_attn is not None:
        # Precomputed exact dense attention (output [B,QH,1,D], ln-LSE
        # [B,QH,1]) from the caller, e.g. the backend's paged flashinfer
        # path; skips the in-here dense computation entirely.
        dense_output, dense_lse = dense_attn
    else:
        dense_output, dense_lse = _dense_attention(query, dense_key, dense_value, scale, dense_lens)
    _b = _mark()
    if shared:
        return _shared_tail_rest(
            extension, cache, query,
            (projected_query, packed_query, query_scale, global_code, global_den, None),
            dense_output, dense_lse, output, scale, gl, sr, st)
    _project_query_into_workspace(query, cache, projected_query)
    _c = _mark()
    extension.picokv_decode_tail_v1_exp8_tiled_gqa4_atomic_fn(
        projected_query,
        packed_query,
        cache.packed_key,
        cache.packed_value,
        query_scale,
        dense_lse.contiguous(),
        dense_output.contiguous(),
        global_code,
        global_den,
        cache.value_projection,
        output,
        cache.exp_bits,
        256,
        1,
        float(scale),
        gl,
        sr,
        st,
    )
    if _prof:
        _d = _mark()
        _PROF["dense"] += _b - _a
        _PROF["proj"] += _c - _b
        _PROF["kernel"] += _d - _c
        _PROF["n"] += 1
        if _PROF["n"] % 64 == 0:
            n = _PROF["n"]
            tot = _PROF["dense"] + _PROF["proj"] + _PROF["kernel"]
            print(f"[PICOKV_PROF] per-layer-call n={n} "
                  f"dense={1e3*_PROF['dense']/n:.4f}ms({100*_PROF['dense']/tot:.1f}%) "
                  f"proj={1e3*_PROF['proj']/n:.4f}ms({100*_PROF['proj']/tot:.1f}%) "
                  f"kernel={1e3*_PROF['kernel']/n:.4f}ms({100*_PROF['kernel']/tot:.1f}%)",
                  flush=True)
    return output


def decode_attention(
    query: torch.Tensor,
    cache: PreparedKVCache,
    *,
    decode_key: Optional[torch.Tensor] = None,
    decode_value: Optional[torch.Tensor] = None,
    update_cache: bool = False,
    use_pruning: bool = False,
    prune_mode: int = 3,
    log_eps: float = -10.0,
    attention_scale: Optional[float] = None,
    global_lens: Optional[torch.Tensor] = None,
    dense_lens: Optional[torch.Tensor] = None,
    sched_req: Optional[torch.Tensor] = None,
    sched_tile: Optional[torch.Tensor] = None,
    dense_attn: Optional[tuple] = None,
    out: Optional[torch.Tensor] = None,
    dense_fn: Optional[Callable[[], tuple]] = None,
    dense_stream: Optional[torch.cuda.Stream] = None,
    query_first: bool = False,
    fork_large: bool = False,
) -> torch.Tensor:
    """Run one bf16 CUDA decode-attention step over ``cache``.

    ``query`` must be ``[B, 4*KVH, 1, D]``.  Optional current-token K/V can be
    included for this call and, with ``update_cache=True``, appended to the
    mutable dense portion for later decode steps.

    ``global_lens`` (optional, int32 ``[B]`` on the cache device) gives each
    batch element's real number of compressed global tokens for the varlen
    (ragged) path: the cache's packed tensors are padded to a common global
    length and the kernel skips padding per request. ``None`` => uniform length.

    ``out`` (optional, contiguous ``[B, QH, 1, D]``) receives the result
    instead of the cache's reused workspace. ``dense_fn`` (optional) computes
    the exact dense attention ``(out [B,QH,1,D], ln-LSE [B,QH,1])`` in place
    of ``dense_attn``; with ``dense_stream`` it runs there, overlapped with
    the query projection/quantization (unpruned path only); without it,
    ``query_first`` (PDL, shared projections, fused quantizer) runs the query
    side, then ``dense_fn(pdl=True)``, which must then launch as a
    programmatic dependent of the query projection; else dense first.
    ``fork_large``: use ``dense_stream`` only on a tail grid above the fused
    quantizer's gate (_TAIL_QUANT_BLOCKS) at the batch sizes of _FORK_B,
    else the serial order above.
    """
    if query.dim() != 4 or query.shape[-2] != 1:
        raise ValueError(f"query must be [B,QH,1,D]; got {tuple(query.shape)}")
    if not query.is_cuda or query.device != cache.device:
        raise ValueError("query must be a CUDA tensor on the cache device")
    if query.dtype != torch.bfloat16 or query.dtype != cache.dtype:
        raise TypeError("query and cache must use bfloat16")
    if query.shape[0] != cache.dense_key.shape[0] or query.shape[-1] != cache.head_dim:
        raise ValueError("query batch/head dimension does not match the cache")
    if query.shape[1] != cache.kv_heads * 4:
        raise ValueError(
            f"standalone CUDA decode requires GQA=4; got QH={query.shape[1]}, "
            f"KVH={cache.kv_heads}"
        )

    dense_key, dense_value = _append_decode_tokens(
        cache,
        decode_key,
        decode_value,
        bool(update_cache),
    )
    scale = (
        1.0 / math.sqrt(float(query.shape[-1]))
        if attention_scale is None
        else float(attention_scale)
    )
    return _decode_cuda(
        query,
        cache,
        dense_key,
        dense_value,
        scale=scale,
        use_pruning=bool(use_pruning),
        prune_mode=int(prune_mode),
        log_eps=float(log_eps),
        global_lens=global_lens,
        dense_lens=dense_lens,
        sched_req=sched_req,
        sched_tile=sched_tile,
        dense_attn=dense_attn,
        out=out,
        dense_fn=dense_fn,
        dense_stream=dense_stream,
        query_first=bool(query_first),
        fork_large=bool(fork_large),
    )


__all__ = ["decode_attention"]
