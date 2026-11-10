"""Shared-projection decode tail: the query projection and the value
projection + epilogue as Triton kernels that read ONE projection matrix per
KV head.

A batched decode cache built by the sglang backend carries the offline
projections as stride-0 batch broadcasts of a single ``[1, KVH, ...]`` copy
(they are identical for every request). The cuBLAS strided-batched GEMMs of
the replicated path need ``B`` materialized copies and re-read one per row;
these kernels instead gather the ``B*G`` query rows of each KV head into one
tile against that head's matrix. Same bf16 operands, fp32 accumulation, same
epilogue arithmetic as ``picokv_proj_gemm_epilogue_kernel``."""

import os

import torch
import triton
import triton.language as tl

try:  # programmatic dependent launch (sm90+, Triton >= 3.4)
    from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait
    PDL_OK = True
except ImportError:
    gdc_launch_dependents = gdc_wait = None
    PDL_OK = False
try:
    from triton.language.extra import libdevice
except ImportError:  # older Triton
    from triton.language.extra.cuda import libdevice


@triton.jit(do_not_specialize=["R"])  # one variant for every batch size
def _qproj_kernel(q_ptr, p_ptr, o_ptr, R, sqb, sqh, QH: tl.constexpr, G: tl.constexpr,
                  D: tl.constexpr, C: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                  zc_ptr=None, zd_ptr=None, ZERO: tl.constexpr = False,
                  PDL: tl.constexpr = False):
    # o[row, n] = q[row, :] @ P_t[h, :, n] over the G query heads of KV head h;
    # ZERO: also zero the tail accumulators code[row, n] (Cv == C) / den[row]
    # (the prep op's job when the tail quantizes the query itself).
    # PDL: let the next (PDL-launched) kernel start now (the dense append,
    # which reads nothing this one writes, and waits for it before exiting)
    if PDL:
        gdc_launch_dependents()
    h = tl.program_id(0)
    r = tl.program_id(1) * BM + tl.arange(0, BM)
    n = tl.program_id(2) * BN + tl.arange(0, BN)
    rm = r < R
    nm = n < C
    b, hq = r // G, h * G + r % G
    row = b * QH + hq  # flat [B*QH] output row of (b, g)
    d = tl.arange(0, D)
    # q is read through its (batch, head) strides: a view of the fused qkv
    # projection needs no staging copy
    q = tl.load(q_ptr + (b * sqb + hq * sqh)[:, None] + d[None, :], mask=rm[:, None], other=0.0)
    p = tl.load(p_ptr + h * D * C + d[:, None] * C + n[None, :], mask=nm[None, :], other=0.0)
    acc = tl.dot(q, p)
    tl.store(o_ptr + row[:, None] * C + n[None, :], acc.to(o_ptr.dtype.element_ty),
             mask=rm[:, None] & nm[None, :])
    if ZERO:
        tl.store(zc_ptr + row[:, None] * C + n[None, :], tl.zeros((BM, BN), dtype=tl.float32),
                 mask=rm[:, None] & nm[None, :])
        if tl.program_id(2) == 0:
            tl.store(zd_ptr + row, tl.zeros((BM,), dtype=tl.float32), mask=rm)


@triton.jit(do_not_specialize=["R"])  # one variant for every batch size
def _qproj_quant_kernel(q_ptr, p_ptr, o_ptr, pk_ptr, qs_ptr, zc_ptr, zd_ptr, R, sqb, sqh,
                        QH: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
                        C: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr,
                        CPP: tl.constexpr):
    # _qproj_kernel (BK-column chunks of one D-wide dot each) plus the prep
    # op (picokv_quantize_pack_zero_projected_query) in the same launch, its
    # arithmetic bit for bit: scale = max(max|bf16 o|, 2^-23) / 127 (IEEE
    # division), v = clamp(rint(o / scale), +-127), per 64 columns 8 int64
    # bit-plane words (plane 0: v < 0; plane p: bit p-1 of |v|; bit j:
    # column kb*64 + j), written here as their little-endian bytes (pk_ptr:
    # the packed buffer as uint8); zeroes the tail accumulators code[row, :C]
    # (Cv == C) / den[row]. Program (h, rows, z) owns chunks z*CPP.. and
    # computes the row max over all of them (the dots are cheap).
    h = tl.program_id(0)
    z = tl.program_id(2)
    r = tl.program_id(1) * BM + tl.arange(0, BM)
    rm = r < R
    b, hq = r // G, h * G + r % G
    row = b * QH + hq
    d = tl.arange(0, D)
    j = tl.arange(0, BK)
    q = tl.load(q_ptr + (b * sqb + hq * sqh)[:, None] + d[None, :], mask=rm[:, None], other=0.0)
    m = tl.zeros((BM, BK), dtype=tl.float32)
    for kb in tl.static_range(C // BK):
        p = tl.load(p_ptr + h * D * C + d[:, None] * C + (kb * BK + j)[None, :])
        x = tl.dot(q, p).to(o_ptr.dtype.element_ty).to(tl.float32)
        m = tl.maximum(m, tl.abs(x))
    scale = libdevice.div_rn(tl.maximum(tl.max(m, axis=1), 1.1920928955078125e-7), 127.0)
    if z == 0:
        tl.store(qs_ptr + row, scale, mask=rm)
        tl.store(zd_ptr + row, tl.zeros((BM,), dtype=tl.float32), mask=rm)
    pl = tl.arange(0, 8)[None, :, None, None]
    sh = tl.maximum(pl - 1, 0)
    jj = tl.arange(0, 8)[None, None, None, :]
    kk = tl.arange(0, 8)
    for c in tl.static_range(CPP):
        kb = z * CPP + c
        n = kb * BK + j
        p = tl.load(p_ptr + h * D * C + d[:, None] * C + n[None, :])
        xo = tl.dot(q, p).to(o_ptr.dtype.element_ty)
        tl.store(o_ptr + row[:, None] * C + n[None, :], xo, mask=rm[:, None])
        tl.store(zc_ptr + row[:, None] * C + n[None, :], tl.zeros((BM, BK), dtype=tl.float32),
                 mask=rm[:, None])
        v = libdevice.rint(libdevice.div_rn(xo.to(tl.float32), scale[:, None]))
        v = tl.minimum(tl.maximum(v, -127.0), 127.0).to(tl.int32)
        v4 = tl.reshape(v, (BM, 1, BK // 8, 8))  # [row, -, byte k, bit jj]
        a4 = tl.abs(v4)
        bit = tl.where(pl == 0, (v4 < 0).to(tl.int32), (a4 >> sh) & 1)
        byte = tl.sum(bit << jj, axis=3)  # [row, plane, k]; disjoint bits: sum == or
        off = ((row * (C // BK) + kb) * 8)[:, None, None] + tl.arange(0, 8)[None, :, None]
        tl.store(pk_ptr + off * 8 + kk[None, None, :], byte.to(tl.uint8), mask=rm[:, None, None])


@triton.jit(do_not_specialize=["R"])  # one variant for every batch size
def _vproj_kernel(code_ptr, den_ptr, dense_ptr, p_ptr, o_ptr, R, QH: tl.constexpr,
                  G: tl.constexpr, D: tl.constexpr, C: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  PDL: tl.constexpr = False):
    # o[row] = (dense[row] + bf16(code[row]) @ P_v[h]) * (1 / (1 + den[row]))
    # PDL (launched as a programmatic dependent of the tail): wait for the
    # tail (and, transitively, everything before it) before any access
    if PDL:
        gdc_wait()
    h = tl.program_id(0)
    r = tl.program_id(1) * BM + tl.arange(0, BM)
    n = tl.program_id(2) * BN + tl.arange(0, BN)
    rm = r < R
    row = (r // G) * QH + h * G + r % G
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, C, BK):
        k = k0 + tl.arange(0, BK)
        km = k < C
        c = tl.load(code_ptr + row[:, None] * C + k[None, :],
                    mask=rm[:, None] & km[None, :], other=0.0)
        p = tl.load(p_ptr + h * C * D + k[:, None] * D + n[None, :], mask=km[:, None], other=0.0)
        acc = tl.dot(c.to(tl.bfloat16), p, acc)
    den = tl.load(den_ptr + row, mask=rm, other=0.0)
    x = tl.load(dense_ptr + row[:, None] * D + n[None, :], mask=rm[:, None], other=0.0)
    y = (x.to(tl.float32) + acc) * (1.0 / (1.0 + den))[:, None]
    tl.store(o_ptr + row[:, None] * D + n[None, :], y.to(o_ptr.dtype.element_ty),
             mask=rm[:, None])


def is_shared(x: torch.Tensor) -> bool:
    """Batch dim is a broadcast of one contiguous [KVH, ...] matrix set."""
    return (x.shape[0] == 1 or x.stride(0) == 0) and x[0].is_contiguous()


# Tile shapes by batch size (Llama-3 8B: G=4, D=128, C=320; sm90; us per
# call, t57/mb/proj_cand.py). Small batches want narrow column tiles (more
# programs), large ones tall row tiles (the per-head matrix is re-read once
# per row tile: B=160 qproj 7.4 -> 4.0, vproj 7.9 -> 4.8). The reduction
# (qproj: one D-wide dot; vproj: BK=64 chunks in order) is the same for
# every shape, so the output is bitwise identical (t57/mb/proj_tune.py).
# PICOKV_PROJ_TILES=0: the fixed pre-t57 tiles (16 x 64, BK 64) at every B.
_TILES = os.environ.get("PICOKV_PROJ_TILES", "1") == "1"


def _qproj_tiles(B):
    if not _TILES:
        return 16, 64, 4
    if B <= 4:
        return 16, 32, 4
    if B <= 48:
        return 64, 64, 4
    if B <= 192:
        return 128, 64, 8
    return 64, 128, 4


def _vproj_tiles(B):
    if not _TILES:
        return 16, 64, 64
    if B <= 16:
        return 16, 32, 64
    if B <= 32:
        return 32, 32, 64
    if B <= 64:
        return 16, 128, 64
    if B <= 128:
        return 32, 128, 64
    return 64, 128, 64


# Every tile variant is compiled on the first call (outside stream capture)
# with that call's own inputs on a one-program grid into a scratch output
# (``out`` is never written by it: it may alias an input): the variant is
# picked by batch size, and a cold compile (160-510 ms each) inside a
# serving step once the batch grows into a new range stalls that step
# (r160: a 1 s decode step). Key: the constexpr geometry (a later call
# with other stride / pointer alignment specializations compiles on use).
# Both PDL variants when the decode path uses PDL: the per-layer graphs
# launch the projections' PDL variants and capture with no eager step
# before them (and a compile inside a capture skips this warm).
_SAMPLE_B = (1, 8, 24, 40, 64, 100, 160, 256)
_WARMED = set()


def _warm(key, cur, tiles, out, launch):
    if key in _WARMED or torch.cuda.is_current_stream_capturing():
        return
    _WARMED.add(key)
    from .decode import pdl_enabled
    scratch = torch.empty_like(out)
    for p in (False, True) if pdl_enabled() else (False,):
        for cfg in sorted({tiles(b) for b in _SAMPLE_B}):
            if (cfg, p) != cur:
                launch(cfg, scratch, p)


def shared_query_projection(query, key_projection_t, out, zero=None, pdl=False):
    """query [B,QH,1,D] bf16, any batch/head strides with a dense last dim;
    key_projection_t [*,KVH,D,C] (shared); out [B,QH,1,C] contiguous,
    written in place. ``zero``: (code [B,QH,1,C], den [B,QH,1]) contiguous
    fp32 tail accumulators, zeroed by the same launch. ``pdl``: trigger the
    next kernel's programmatic launch at entry."""
    B, QH, _, D = query.shape
    P = key_projection_t[0]
    KVH, C = int(P.shape[0]), int(P.shape[-1])
    G = QH // KVH
    q = query if query.stride(-1) == 1 else query.contiguous()
    zc, zd = zero if zero is not None else (None, None)

    def launch(cfg, o, p, grid=(1, 1, 1), zc=None, zd=None):
        BM, BN, nw = cfg
        _qproj_kernel[grid](q, P, o, B * G, q.stride(0), q.stride(1),
                            QH=QH, G=G, D=D, C=C, BM=BM, BN=BN,
                            zc_ptr=zc, zd_ptr=zd, ZERO=zc is not None, PDL=p,
                            num_warps=nw)

    cfg = _qproj_tiles(B)
    # Warm both accumulator modes on the first eager call: the tail's grid
    # gate (decode._tail_quant) can move a batch from one to the other at a
    # recapture (its schedule grew), which has no eager step before it.
    def launch_z(c, o, p):  # scratch accumulators: only reached while warming
        launch(c, o, p, zc=torch.empty(o.shape, dtype=torch.float32, device=o.device),
               zd=torch.empty(o.shape[:-1], dtype=torch.float32, device=o.device))

    _warm(("q", QH, G, D, C, q.device.index), (cfg, pdl) if zc is None else None,
          _qproj_tiles, out, launch)
    _warm(("qz", QH, G, D, C, q.device.index), (cfg, pdl) if zc is not None else None,
          _qproj_tiles, out, launch_z)
    # and the fused prep variant (shared_query_projection_quant), which a
    # batch first reaches at a recapture (the tail grid outgrew the gate)
    if C % 64 == 0:
        def launch_qq(c, o, p):
            if p:
                return
            f32 = dict(dtype=torch.float32, device=o.device)
            _qproj_quant_kernel[(1, 1, 1)](
                q, P, o, torch.empty(o.shape[:-1] + (C // 64, 8), dtype=torch.int64,
                                     device=o.device).view(torch.uint8),
                torch.empty(o.shape[:-1], **f32), torch.empty(o.shape, **f32),
                torch.empty(o.shape[:-1], **f32), B * G, q.stride(0), q.stride(1),
                QH=QH, G=G, D=D, C=C, BM=16, BK=64, CPP=1, num_warps=4)

        _warm(("qq", QH, G, D, C, q.device.index), None, lambda b: (16, 64, 1), out, launch_qq)
    launch(cfg, out, pdl, (KVH, triton.cdiv(B * G, cfg[0]), triton.cdiv(C, cfg[1])), zc, zd)
    return out


def shared_query_projection_quant(query, key_projection_t, out, packed, query_scale,
                                  code, den):
    """shared_query_projection then the prep op (quantize + pack ``out``
    into ``packed`` [B,QH,1,C/64,8] int64 / ``query_scale`` [B,QH,1] fp32,
    zero ``code`` [B,QH,1,C] / ``den`` [B,QH,1]) as one launch, bitwise
    both. On the forked dense branch the prep op, launched after the
    projection, waited for FA3 to free the SMs; this kernel runs before it."""
    B, QH, _, D = query.shape
    P = key_projection_t[0]
    KVH, C = int(P.shape[0]), int(P.shape[-1])
    G = QH // KVH
    q = query if query.stride(-1) == 1 else query.contiguous()
    # one 64-column chunk per program (CPP=1, 4 warps; B=8 4.6 us vs 5.1 for
    # projection + prep op, B=1 5.5 vs 4.8; past 32 rows the pair is faster)
    BM, CPP = 16, 1
    _qproj_quant_kernel[(KVH, triton.cdiv(B * G, BM), C // 64 // CPP)](
        q, P, out, packed.view(torch.uint8), query_scale, code, den, B * G,
        q.stride(0), q.stride(1), QH=QH, G=G, D=D, C=C, BM=BM, BK=64, CPP=CPP,
        num_warps=4)
    return out


def shared_value_projection(dense_output, glob_code, glob_den, value_projection, out,
                            pdl=False):
    """dense_output [B,QH,1,D] bf16, glob_code [B,QH,1,C] fp32, glob_den
    [B,QH,1] fp32, value_projection [*,KVH,C,D] (shared); out [B,QH,1,D].
    ``pdl``: launched as a programmatic dependent (starts during the
    previous kernel, waits for it before reading)."""
    B, QH, _, D = dense_output.shape
    P = value_projection[0]
    KVH, C = int(P.shape[0]), int(P.shape[1])
    G = QH // KVH
    dense = dense_output.contiguous()

    def launch(cfg, o, p, grid=(1, 1, 1)):
        BM, BN, BK = cfg
        _vproj_kernel[grid](glob_code, glob_den, dense, P, o, B * G,
                            QH=QH, G=G, D=D, C=C, BM=BM, BN=BN, BK=BK, PDL=p,
                            num_warps=4, launch_pdl=p)

    def tiles(b):  # column tiles within D: the kernel's n axis is unmasked
        BM, BN, BK = _vproj_tiles(b)
        return BM, min(BN, D), BK

    cfg = tiles(B)
    assert D % cfg[1] == 0, f"shared value projection: head dim {D} not a multiple of {cfg[1]}"
    _warm(("v", QH, G, D, C, dense.device.index), (cfg, pdl), tiles, out, launch)
    launch(cfg, out, pdl, (KVH, triton.cdiv(B * G, cfg[0]), D // cfg[1]))
    return out


@triton.jit(do_not_specialize=["GX", "L", "S"])
def _det_reduce_kernel(pc_ptr, pd_ptr, rs_ptr, gl_ptr, code_ptr, den_ptr, GX, L, S, BLK,
                       KVH: tl.constexpr, QH: tl.constexpr, CV: tl.constexpr,
                       SCHED: tl.constexpr, HAS_GL: tl.constexpr,
                       BC: tl.constexpr, TT: tl.constexpr, NCB: tl.constexpr):
    # code[b, kvh*4+h, s, c] = sum over row b's tiles t (ascending, fixed
    # tree) of part_code[slot(b, s, kvh, t), h, c]; den likewise. Each
    # program covers NCB column blocks of BC.
    bs = tl.program_id(0)
    kvh = tl.program_id(1)
    b = bs // S
    s = bs % S
    if HAS_GL:
        lb = tl.load(gl_ptr + b)
    else:
        lb = L
    ntile = (lb + BLK - 1) // BLK
    if SCHED:  # grid (entries, S, KVH); row b's tiles from entry rs[b]
        base = (s * KVH + kvh).to(tl.int64) * GX + tl.load(rs_ptr + b)
    else:  # grid (tiles, S, B*KVH)
        nb = tl.num_programs(0) // S
        base = ((s * nb + b) * KVH + kvh).to(tl.int64) * GX
    hd = tl.arange(0, 4)
    for j in tl.static_range(NCB):
        cb = tl.program_id(2) * NCB + j
        col = cb * BC + tl.arange(0, BC)  # flat (head, code) of 4 * CV
        acc = tl.zeros((TT, BC), dtype=tl.float32)
        dacc = tl.zeros((TT, 4), dtype=tl.float32)
        for t0 in range(0, ntile, TT):
            t = t0 + tl.arange(0, TT)
            m = t < ntile
            slot = base + t
            acc += tl.load(pc_ptr + slot[:, None] * (4 * CV) + col[None, :], mask=m[:, None], other=0.0)
            if cb == 0:
                dacc += tl.load(pd_ptr + slot[:, None] * 4 + hd[None, :], mask=m[:, None], other=0.0)
        h = col // CV
        row = (b * QH + kvh * 4 + h) * S + s
        tl.store(code_ptr + row * CV + col % CV, tl.sum(acc, axis=0))
        if cb == 0:
            tl.store(den_ptr + (b * QH + kvh * 4 + hd) * S + s, tl.sum(dacc, axis=0))


def det_tail_reduce(part_code, part_den, part_rs, gl, glob_code, glob_den, gx, L, blk):
    """Fixed-order reduction of the deterministic tail's per-block partials
    (picokv_decode_tail_global_part_fn) into glob_code [B,QH,S,CV] /
    glob_den [B,QH,S] (every entry written; no prep zeroing needed).
    part_rs: int32 [B] first sched entry per row, or None (unscheduled grid
    of ``gx`` blocks per (b, kvh)); gl: int32 [B] lengths or None (all L);
    blk: tokens per partial (512 sched, 256 * tail BT unscheduled)."""
    B, QH, S, CV = glob_code.shape
    KVH = QH // 4
    BC = 256
    assert (4 * CV) % BC == 0 and QH == 4 * KVH
    sched = part_rs is not None
    has_gl = gl is not None and gl.numel() > 0
    dummy = glob_den
    # Launch shape from the mean tiles per row (fixed for a given batch, so
    # the summation tree is too): few tiles per row = many short programs,
    # bandwidth-bound with one 2-warp program per (row, kv head) over all
    # columns (B=160 x 1k: 23 -> 4 us per layer); long rows split the
    # columns 5 ways with a 32-tile batch per load.
    nb = triton.cdiv(gx, B) if sched else gx
    if nb <= 8:
        tt, ncb, nw = 4, (4 * CV) // BC, 2
    else:
        tt, ncb, nw = (16 if nb <= 16 else 32), 1, 4
    grid = (B * S, KVH, (4 * CV) // BC // ncb)
    _det_reduce_kernel[grid](part_code, part_den, part_rs if sched else dummy,
                             gl if has_gl else dummy, glob_code, glob_den, gx, L, S, blk,
                             KVH=KVH, QH=QH, CV=CV, SCHED=sched, HAS_GL=has_gl,
                             BC=BC, TT=tt, NCB=ncb, num_warps=nw)
    return glob_code, glob_den
