"""Single-token CUDA attention over dense-retained and packed-global KV."""

from __future__ import annotations

import math
import os
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
_PROJ_GEMM = 0 if os.environ.get("PICOKV_PROJ_GEMM", "1") == "0" else 1
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
        current_stream = torch.cuda.current_stream(query.device)
        if dense_stream is not None:
            dense_stream.wait_stream(current_stream)
            with torch.cuda.stream(dense_stream):
                dense_output, dense_lse = dense_fn()
        else:
            dense_output, dense_lse = dense_fn()
        _project_query_into_workspace(query, cache, projected_query)
        extension.picokv_decode_tail_prep_fn(
            projected_query, packed_query, query_scale, global_code, global_den)
        if dense_stream is not None:
            current_stream.wait_stream(dense_stream)
            if not torch.cuda.is_current_stream_capturing():
                dense_output.record_stream(current_stream)
                dense_lse.record_stream(current_stream)
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
    the query projection/quantization (unpruned path only).
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
    )


__all__ = ["decode_attention"]
