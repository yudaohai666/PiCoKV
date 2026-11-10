"""Dense-window decode attention as Triton kernels (default mode only).

The tensor-core tail reads only the window's ln-LSE; the window's output is
first read by the value projection after the tail. So on the forked dense
branch the window attention splits in two launches: ``window_lse`` (QK^T +
logsumexp, no V) gates the tail, and ``window_out_from_lse`` (sum
exp(s - lse) * v, the same LSE: no online rescaling) runs on the dense
stream under the tail. ``window_attn`` is the one-pass variant (output +
LSE, online softmax): the serial path's opt-in (PICOKV_WIN_TRITON_SERIAL=1;
in-server FA3 measured faster there) and the fork's above the split size.

Layout as FA3's flash_attn_with_kvcache over the token-major pools: q
[B, QH, D] (any batch / head strides, dense last dim), k / v [B, cap, KVH, D]
contiguous, lens [B] int32 (this step's token included); lse [B, QH] fp32
(ln), out [B, QH, D] bf16. One program per (row, KV head) covers that head's
G query heads. PICOKV_DET=1 keeps FA3 (the sglang backend decides)."""

import os

import torch
import triton
import triton.language as tl

try:
    from triton.language.extra.cuda import gdc_wait
    _PDL_OK = True
except ImportError:  # older Triton
    gdc_wait = None
    _PDL_OK = False

_LOG2E = 1.4426950408889634
_LN2 = 0.6931471805599453


@triton.jit
def _scores(q, k_base, t, tm, d, RS: tl.constexpr, qk_scale):
    # q [GP, D] bf16; returns s [GP, BN] fp32 in log2 units (masked -inf)
    k = tl.load(k_base + t[None, :].to(tl.int64) * RS + d[:, None], mask=tm[None, :], other=0.0)
    s = tl.dot(q, k) * qk_scale
    return tl.where(tm[None, :], s, float("-inf"))


@triton.jit(do_not_specialize=["cap", "sqb", "sqh"])
def _win_lse_kernel(q_ptr, k_ptr, len_ptr, lse_ptr, m_ptr, cap, sqb, sqh, qk_scale,
                    KVH: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
                    GP: tl.constexpr, BN: tl.constexpr, PDL: tl.constexpr,
                    HAS_M: tl.constexpr):
    # HAS_M: also the row max (log2 units) for _win_out_lse_kernel
    if PDL:
        gdc_wait()
    pid = tl.program_id(0)
    b = pid // KVH
    h = pid % KVH
    g = tl.arange(0, GP)
    gm = g < G
    d = tl.arange(0, D)
    q = tl.load(q_ptr + b * sqb + (h * G + g)[:, None] * sqh + d[None, :],
                mask=gm[:, None], other=0.0)
    L = tl.load(len_ptr + b)
    k_base = k_ptr + (b.to(tl.int64) * cap * KVH + h) * D
    m = tl.full((GP,), float("-inf"), tl.float32)
    l = tl.zeros((GP,), dtype=tl.float32)
    for s0 in range(0, L, BN):
        t = s0 + tl.arange(0, BN)
        s = _scores(q, k_base, t, t < L, d, KVH * D, qk_scale)
        mn = tl.maximum(m, tl.max(s, 1))
        l = l * tl.exp2(m - mn) + tl.sum(tl.exp2(s - mn[:, None]), 1)
        m = mn
    lse = (m + tl.log2(l)) * 0.6931471805599453
    tl.store(lse_ptr + b * (KVH * G) + h * G + g, lse, mask=gm)
    if HAS_M:
        tl.store(m_ptr + b * (KVH * G) + h * G + g, m, mask=gm)


@triton.jit(do_not_specialize=["cap", "sqb", "sqh"])
def _win_out_lse_kernel(q_ptr, k_ptr, v_ptr, len_ptr, lse_ptr, m_ptr, o_ptr, cap, sqb, sqh,
                        qk_scale, KVH: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
                        GP: tl.constexpr, BN: tl.constexpr, HAS_M: tl.constexpr):
    # out = sum_t exp(s_t - lse) v_t with the LSE window_lse wrote. HAS_M:
    # as FA3, P = exp(s_t - max) (its largest entry exactly 1 in bf16), the
    # sum scaled by exp(max - lse) = 1/l at the end; normalized bf16 P
    # measured 1.24x FA3's mean output error against an fp64 reference
    pid = tl.program_id(0)
    b = pid // KVH
    h = pid % KVH
    g = tl.arange(0, GP)
    gm = g < G
    d = tl.arange(0, D)
    q = tl.load(q_ptr + b * sqb + (h * G + g)[:, None] * sqh + d[None, :],
                mask=gm[:, None], other=0.0)
    L = tl.load(len_ptr + b)
    row = b * (KVH * G) + h * G + g
    lse2 = tl.load(lse_ptr + row, mask=gm, other=0.0) * 1.4426950408889634
    if HAS_M:
        ref = tl.load(m_ptr + row, mask=gm, other=0.0)
    else:
        ref = lse2
    off = (b.to(tl.int64) * cap * KVH + h) * D
    acc = tl.zeros((GP, D), dtype=tl.float32)
    for s0 in range(0, L, BN):
        t = s0 + tl.arange(0, BN)
        tm = t < L
        s = _scores(q, k_ptr + off, t, tm, d, KVH * D, qk_scale)
        p = tl.exp2(s - ref[:, None])
        v = tl.load(v_ptr + off + t[:, None].to(tl.int64) * (KVH * D) + d[None, :],
                    mask=tm[:, None], other=0.0)
        acc = tl.dot(p.to(v.dtype), v, acc)
    if HAS_M:
        acc = acc * tl.exp2(ref - lse2)[:, None]
    tl.store(o_ptr + row[:, None] * D + d[None, :], acc.to(o_ptr.dtype.element_ty),
             mask=gm[:, None])


@triton.jit(do_not_specialize=["cap", "sqb", "sqh"])
def _win_attn_kernel(q_ptr, k_ptr, v_ptr, len_ptr, lse_ptr, o_ptr, cap, sqb, sqh,
                     qk_scale, KVH: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
                     GP: tl.constexpr, BN: tl.constexpr, PDL: tl.constexpr):
    # one pass: online softmax, output + LSE
    if PDL:
        gdc_wait()
    pid = tl.program_id(0)
    b = pid // KVH
    h = pid % KVH
    g = tl.arange(0, GP)
    gm = g < G
    d = tl.arange(0, D)
    q = tl.load(q_ptr + b * sqb + (h * G + g)[:, None] * sqh + d[None, :],
                mask=gm[:, None], other=0.0)
    L = tl.load(len_ptr + b)
    off = (b.to(tl.int64) * cap * KVH + h) * D
    m = tl.full((GP,), float("-inf"), tl.float32)
    l = tl.zeros((GP,), dtype=tl.float32)
    acc = tl.zeros((GP, D), dtype=tl.float32)
    for s0 in range(0, L, BN):
        t = s0 + tl.arange(0, BN)
        tm = t < L
        s = _scores(q, k_ptr + off, t, tm, d, KVH * D, qk_scale)
        mn = tl.maximum(m, tl.max(s, 1))
        alpha = tl.exp2(m - mn)
        p = tl.exp2(s - mn[:, None])
        l = l * alpha + tl.sum(p, 1)
        v = tl.load(v_ptr + off + t[:, None].to(tl.int64) * (KVH * D) + d[None, :],
                    mask=tm[:, None], other=0.0)
        acc = tl.dot(p.to(v.dtype), v, acc * alpha[:, None])
        m = mn
    row = b * (KVH * G) + h * G + g
    ls = tl.where(l > 0, l, 1.0)  # an empty window: zero output, as FA3
    tl.store(o_ptr + row[:, None] * D + d[None, :], (acc / ls[:, None]).to(o_ptr.dtype.element_ty),
             mask=gm[:, None])
    tl.store(lse_ptr + row, (m + tl.log2(l)) * 0.6931471805599453, mask=gm)


def _geom(q, k):
    B, QH, D = q.shape
    KVH = int(k.shape[2])
    G = QH // KVH
    assert q.stride(-1) == 1 and k.is_contiguous() and QH == KVH * G
    return B, QH, D, KVH, G, max(16, triton.next_power_of_2(G))


# t: CUDA-graph microbench vs FA3 (us/call, cap 384-1024, L 320-384):
# B1 FA3 8.05, lse 4.2-4.4, out 5.3-5.5, one-pass 5.8-6.0; B8 FA3 9.0,
# 4.5 / 5.8 / 6.4; B16 FA3 9.1, 4.8 / 6.3 / 6.8 (BN 64 or 2 stages: +1-3 us)
_BN = 128
_STAGES = 3
_WARPS = 4
# window_out_from_lse runs beside the tail, so its shared memory keeps tail
# blocks off the SMs it holds: BN 64 with no pipelining (in-server, 32768x8
# ITL 9.48-9.54 ms vs 9.76-9.79 with the BN 128 / 3-stage tile above, FA3
# 9.60-9.62; the standalone time is not what counts here)
_OUT_BN = int(os.environ.get("PICOKV_WIN_OUT_BN", "64"))
_OUT_STAGES = int(os.environ.get("PICOKV_WIN_OUT_STAGES", "1"))
_WARMED = set()


def _args(q, k, scale):
    return int(k.shape[1]), q.stride(0), q.stride(1), float(scale) * _LOG2E


def warm(q, k, v, lens, lse, out, scale, mx=None):
    """Compile (no launch) every variant on the first eager call: the split
    pair is first reached inside a graph capture, where a cold compile
    stalls the step."""
    B, QH, D, KVH, G, GP = _geom(q, k)
    key = (KVH, G, D, q.device.index)
    if key in _WARMED or torch.cuda.is_current_stream_capturing():
        return
    _WARMED.add(key)
    a, grid = _args(q, k, scale), (B * KVH,)
    kw = dict(KVH=KVH, G=G, D=D, GP=GP, BN=_BN, num_warps=_WARPS, num_stages=_STAGES, grid=grid)
    for p in (False, True) if _PDL_OK else (False,):
        _win_lse_kernel.warmup(q, k, lens, lse, lse if mx is None else mx, *a, PDL=p,
                               launch_pdl=p, HAS_M=mx is not None, **kw)
        _win_attn_kernel.warmup(q, k, v, lens, lse, out, *a, PDL=p, launch_pdl=p, **kw)
    kw.update(BN=_OUT_BN, num_stages=_OUT_STAGES)
    _win_out_lse_kernel.warmup(q, k, v, lens, lse, lse if mx is None else mx, out, *a,
                               HAS_M=mx is not None, **kw)


def window_lse(q, k, lens, lse, scale, pdl=False, mx=None):
    """ln-LSE [B, QH] of q against the first lens[b] keys of row b
    (``mx`` [B, QH] fp32: also the row max, for window_out_from_lse)."""
    B, QH, D, KVH, G, GP = _geom(q, k)
    pdl = pdl and _PDL_OK
    _win_lse_kernel[(B * KVH,)](q, k, lens, lse, lse if mx is None else mx,
                                *_args(q, k, scale), KVH=KVH, G=G, D=D, GP=GP, BN=_BN,
                                PDL=pdl, HAS_M=mx is not None, num_warps=_WARPS,
                                num_stages=_STAGES, launch_pdl=pdl)
    return lse


def window_out_from_lse(q, k, v, lens, lse, out, scale, mx=None):
    """out [B, QH, D] = sum_t exp(s_t - lse) v_t (lse, and ``mx`` if
    given, from window_lse)."""
    B, QH, D, KVH, G, GP = _geom(q, k)
    _win_out_lse_kernel[(B * KVH,)](q, k, v, lens, lse, lse if mx is None else mx, out,
                                    *_args(q, k, scale), KVH=KVH, G=G, D=D, GP=GP,
                                    BN=_OUT_BN, HAS_M=mx is not None, num_warps=_WARPS,
                                    num_stages=_OUT_STAGES)
    return out


def window_attn(q, k, v, lens, lse, out, scale, pdl=False):
    """One pass: out [B, QH, D] and ln-LSE [B, QH]. ``pdl``: launched as a
    programmatic dependent (waits for the previous kernel at entry)."""
    B, QH, D, KVH, G, GP = _geom(q, k)
    pdl = pdl and _PDL_OK
    _win_attn_kernel[(B * KVH,)](q, k, v, lens, lse, out, *_args(q, k, scale), KVH=KVH, G=G,
                                 D=D, GP=GP, BN=_BN, PDL=pdl, num_warps=_WARPS,
                                 num_stages=_STAGES, launch_pdl=pdl)
    return out, lse
