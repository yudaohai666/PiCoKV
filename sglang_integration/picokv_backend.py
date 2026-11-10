"""PiCoKV attention backend for SGLang.

Layers (bottom-up):

* ``PicoKVRequestCache`` — one request's per-layer compressed KV state, built
  directly on PiCoKV's **pure** ``prepare_kv_cache`` / ``decode_attention`` API
  (no ``transformers`` import — so it is compatible with whatever transformers
  version the serving engine pins, and unit-testable standalone).
* ``PicoKVCore`` — single-request (batch=1) engine in sglang's flattened
  ``[num_tokens, heads*head_dim]`` layout: ``extend`` (prefill: exact attention
  + compress) and ``decode`` (one PiCoKV CUDA decode step).
* ``PicoKVBatchDecoder`` — step 3: a continuous-batch decoder holding one
  ``PicoKVRequestCache`` per running request at *independent* context lengths,
  so a ragged batch decodes correctly. Fuses per-request results; the current
  build launches the verified single-request kernel per request (a fused ragged
  kernel is the remaining optimization).
* ``PicoKVAttnBackend`` — the ``AttentionBackend`` glue (defined only when
  sglang imports). Registered as backend name ``picokv``.

Decode fast path (all branches unified): a persistent token-major dense pool
``[B, cap, KVH, D]`` per layer, written in place each step at device-side
positions; exact dense-window attention via flashinfer's paged decode kernel
(one plan per step, page_size = cap so every kernel input lives at a stable
address); then the fused PiCoKV tail. When shapes are stable (fixed batch
membership), the whole per-layer op sequence is replayed through a CUDA graph
(``PICOKV_GRAPHS=0`` disables).

Hard requirements of the PiCoKV decode kernel: GQA=4, head_dim=128, bf16.
"""

from __future__ import annotations

import array
import contextlib
import itertools
import logging
import operator
import os
import re
import sys
import weakref
from typing import Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Pure PiCoKV API — imports NO transformers/numpy (verified), unlike the HF
# adapter in picokv_standalone.transformers.
from picokv.picokv_standalone import (
    PreparedKVCache,
    StandaloneConfig,
    decode_attention,
    prepare_kv_cache,
)

# Incremental chunk-append path (eas=0 + offline projections). Both come from
# the standalone/prefill extension; when unavailable, continuations fall back
# to re-prepare (old behavior) instead of failing.
try:
    from picokv.picokv_prefill_cuda_extension.picokv_prefill_cuda import (
        blockk_fixed_cuda,
    )
    from picokv.picokv_standalone.prepare import _prepare_fixed_block32_cached
except Exception:  # pragma: no cover - append path degrades to re-prepare
    blockk_fixed_cuda = None
    _prepare_fixed_block32_cached = None

# Fused decode append (PICOKV_FUSED_APPEND=0 disables): one launch writes this
# step's K and V rows into the dense pools, replacing two index_put kernels.
try:
    import triton
    import triton.language as tl

    try:  # programmatic dependent launch (sm90, Triton >= 3.4)
        from triton.language.extra.cuda import gdc_wait as _gdc_wait
    except ImportError:
        _gdc_wait = None

    @triton.jit
    def _append_kv_kernel(kbuf, vbuf, k, v, wpos, cap, k_s0, v_s0,
                          ROW: tl.constexpr, PDL: tl.constexpr = False):
        # one program per request; a row is KVH*D contiguous elements.
        # PDL (a programmatic dependent of the shared query projection,
        # which touches none of these): runs during it, then waits for it
        # before exiting, so the kernels after this one follow both
        b = tl.program_id(0).to(tl.int64)
        off = tl.arange(0, ROW)
        dst = (b * cap + tl.load(wpos + b).to(tl.int64)) * ROW + off
        tl.store(kbuf + dst, tl.load(k + b * k_s0 + off))
        tl.store(vbuf + dst, tl.load(v + b * v_s0 + off))
        if PDL:
            _gdc_wait()

    # Decode slide (P1-4): per row b, n[b] in [0, S] tokens leave the dense
    # local window and are appended to the compressed global segment at
    # offset gl[b] (S = 64 staging slots per row). Bitwise == prepare over
    # the whole sequence (t35/bench_slide.py T2/T3).
    @triton.jit
    def _slide_vmerge_kernel(tok_ptr, pv_ptr, gl_ptr, n_ptr, H, BS, G, CV: tl.constexpr,
                             VB: tl.constexpr, S: tl.constexpr):
        # tok: [H, B*S, VB] per-token words (code c = j*64+s at word j bit s)
        # pv : [B, H, G, CV] group words (bit i = token g*64+i's code bit c);
        # bits past gl must be zero (OR-merge).
        pid = tl.program_id(0)          # b*H + h
        j = tl.program_id(1)            # word j -> codes j*64..j*64+63
        b = pid // H
        h = pid % H
        gl = tl.load(gl_ptr + b)
        n = tl.load(n_ptr + b)
        i = tl.arange(0, S)
        s = tl.arange(0, 64)
        w = tl.load(tok_ptr + (h.to(tl.int64) * BS + b * S + i) * VB + j)
        w = tl.where(i < n, w, 0)
        bits = (w[:, None] >> s[None, :].to(tl.int64)) & 1           # [S,64]
        word = tl.sum(bits << i[:, None].to(tl.int64), axis=0)        # [64]
        rem = (gl % 64).to(tl.int64)
        g0 = gl // 64
        base = ((b.to(tl.int64) * H + h) * G) * CV + j * 64 + s
        lo_ptr = pv_ptr + base + g0.to(tl.int64) * CV
        old = tl.load(lo_ptr, mask=g0 < G, other=0)
        tl.store(lo_ptr, old | (word << rem), mask=g0 < G)
        hi = ((word >> 1) >> (63 - rem)) & ((tl.full([64], 1, tl.int64) << rem) - 1)
        tl.store(lo_ptr + CV, hi, mask=(rem + n > 64) & (g0 + 1 < G))

    @triton.jit
    def _slide_kmerge_kernel(pkn_ptr, pk_ptr, gl_ptr, n_ptr, H, BS, L, KB: tl.constexpr,
                             S: tl.constexpr):
        # pkn: [H, KB, B*S] new per-token key columns; pk: [B, H, KB, L]
        pid = tl.program_id(0)
        b = pid // H
        h = pid % H
        gl = tl.load(gl_ptr + b)
        n = tl.load(n_ptr + b)
        i = tl.arange(0, S)
        for k in tl.static_range(KB):
            v = tl.load(pkn_ptr + (h.to(tl.int64) * KB + k) * BS + b * S + i)
            dst = pk_ptr + ((b.to(tl.int64) * H + h) * KB + k) * L + gl + i
            tl.store(dst, v, mask=(i < n) & (gl + i < L))

    # PICOKV_ROWS: the same merges into per-row storage (_RowStore): row b's
    # keys [H, KB, cap] / values [H, cap/64, CV] at the addresses kp[b] /
    # vp[b], cap = ct[b] (vacant rows: cap 0 -> every access masked off)
    @triton.jit
    def _slide_vmerge_rows_kernel(tok_ptr, vp_ptr, ct_ptr, gl_ptr, n_ptr, H, BS,
                                  CV: tl.constexpr, VB: tl.constexpr, S: tl.constexpr):
        pid = tl.program_id(0)
        j = tl.program_id(1)
        b = pid // H
        h = pid % H
        gl = tl.load(gl_ptr + b)
        n = tl.load(n_ptr + b)
        G = tl.load(ct_ptr + b) // 64
        row = tl.load(vp_ptr + b).to(tl.pointer_type(tl.int64))
        i = tl.arange(0, S)
        s = tl.arange(0, 64)
        w = tl.load(tok_ptr + (h.to(tl.int64) * BS + b * S + i) * VB + j)
        w = tl.where(i < n, w, 0)
        bits = (w[:, None] >> s[None, :].to(tl.int64)) & 1           # [S,64]
        word = tl.sum(bits << i[:, None].to(tl.int64), axis=0)        # [64]
        rem = (gl % 64).to(tl.int64)
        g0 = gl // 64
        lo_ptr = row + (h.to(tl.int64) * G + g0) * CV + j * 64 + s
        old = tl.load(lo_ptr, mask=g0 < G, other=0)
        tl.store(lo_ptr, old | (word << rem), mask=g0 < G)
        hi = ((word >> 1) >> (63 - rem)) & ((tl.full([64], 1, tl.int64) << rem) - 1)
        tl.store(lo_ptr + CV, hi, mask=(rem + n > 64) & (g0 + 1 < G))

    @triton.jit
    def _slide_kmerge_rows_kernel(pkn_ptr, kp_ptr, ct_ptr, gl_ptr, n_ptr, H, BS,
                                  KB: tl.constexpr, S: tl.constexpr):
        pid = tl.program_id(0)
        b = pid // H
        h = pid % H
        gl = tl.load(gl_ptr + b)
        n = tl.load(n_ptr + b)
        L = tl.load(ct_ptr + b)
        row = tl.load(kp_ptr + b).to(tl.pointer_type(tl.int64))
        i = tl.arange(0, S)
        for k in tl.static_range(KB):
            v = tl.load(pkn_ptr + (h.to(tl.int64) * KB + k) * BS + b * S + i)
            dst = row + (h.to(tl.int64) * KB + k) * L + gl + i
            tl.store(dst, v, mask=(i < n) & (gl + i < L))

    @triton.jit
    def _slide_shift_kernel(buf_ptr, n_ptr, cap, ROWW: tl.constexpr, SINK: tl.constexpr,
                            LOCAL: tl.constexpr, T: tl.constexpr, DC: tl.constexpr):
        # buf [B, cap, ROWW]: rows [SINK+n, SINK+n+LOCAL) -> [SINK, SINK+LOCAL)
        # in place; each program owns a disjoint column chunk and walks rows
        # upward (dst < src: every load precedes any store that could clobber it).
        b = tl.program_id(0)
        c = tl.program_id(1)
        n = tl.load(n_ptr + b)
        if n > 0:
            cols = c * DC + tl.arange(0, DC)
            r = tl.arange(0, T)
            base = buf_ptr + b.to(tl.int64) * cap * ROWW
            for t in tl.static_range(LOCAL // T):
                src = base + (SINK + n + t * T + r)[:, None].to(tl.int64) * ROWW + cols[None, :]
                x = tl.load(src)
                tl.debug_barrier()  # all loads of this tile before any store
                dst = base + (SINK + t * T + r)[:, None].to(tl.int64) * ROWW + cols[None, :]
                tl.store(dst, x)
                tl.debug_barrier()

    # Batched prepare (PICOKV_PREP_BATCH): per-request outputs of ONE blockk
    # launch over the concatenated global tokens of several prompts. Codes
    # are per-token, so each request's slice is its own prepare's codes.
    @triton.jit
    def _prep_vgroup_kernel(tok_ptr, out_ptr, tab_ptr, Tg, VW: tl.constexpr):
        # tok: [H, Tg, VW] per-token words; tab row k = (src token, n <= 64,
        # dst offset, head stride) of one 64-token group. Out word c = j*64+s
        # of the group: bit i = token i's code bit c (== the gather's
        # pack_value_from_token_packed_tile_kernel).
        k = tl.program_id(0)
        j = tl.program_id(1)
        h = tl.program_id(2).to(tl.int64)
        src = tl.load(tab_ptr + k * 4)
        n = tl.load(tab_ptr + k * 4 + 1)
        dst = tl.load(tab_ptr + k * 4 + 2)
        hs = tl.load(tab_ptr + k * 4 + 3)
        i = tl.arange(0, 64)
        w = tl.load(tok_ptr + (h * Tg + src + i) * VW + j, mask=i < n, other=0)
        bits = (w[:, None] >> i[None, :].to(tl.int64)) & 1           # [token, s]
        word = tl.sum(bits << i[:, None].to(tl.int64), axis=0)        # [s]
        tl.store(out_ptr + dst + h * hs + j * 64 + i, word)

    @triton.jit
    def _prep_kcopy_kernel(src_ptr, dst_ptr, tab_ptr, Tg, BLOCK: tl.constexpr):
        # src: [H*KW, Tg] key columns; tab row r = (column start, gl, dst
        # offset) of one request -> dst[off : off + H*KW*gl] as [H*KW, gl]
        r = tl.program_id(0)
        w = tl.program_id(1).to(tl.int64)
        c0 = tl.load(tab_ptr + r * 3)
        gl = tl.load(tab_ptr + r * 3 + 1)
        d0 = tl.load(tab_ptr + r * 3 + 2)
        o = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
        m = o < gl
        x = tl.load(src_ptr + w * Tg + c0 + o, mask=m)
        tl.store(dst_ptr + d0 + w * gl + o, x, mask=m)

    # Radix code reuse (PICOKV_RADIX_CODES): ONE launch per layer and staged
    # row instead of a copy per run. tab = page code addresses (ck, cv) x NP,
    # then per global token j: its mirror page index * BLK + slot offset, or
    # -1. Tokens j < c read their codes from the mirrors, the rest from the
    # tail blockk (tk [H,KW,m] / tv [H,m,VW], token j - c) and go to their
    # mirrors (e >= 0). Out: packed key [H,KW,KS >= glen] and the value words
    # grouped as _prep_vgroup_kernel (pv: head stride VS >= G*CV), programs
    # G..G+NWQ copy the dense window of g ([T,2,H,D], strides gs0..2) to dw
    # [2,H,SINK+LOCAL,D], the rest zero both past the content (PICOKV_PREALLOC
    # full-capacity buffers).
    @triton.jit
    def _code_asm_kernel(tab_ptr, NP, tk_ptr, tv_ptr, m, c, glen, l, pka_ptr, pv_ptr, KS, VS,
                         g_ptr, gs0, gs1, gs2, s, n, dw_ptr, BLK, G,
                         H: tl.constexpr, KW: tl.constexpr, VW: tl.constexpr, D: tl.constexpr,
                         SINK: tl.constexpr, LOCAL: tl.constexpr, BT: tl.constexpr,
                         NWQ: tl.constexpr):
        q = tl.program_id(0)
        r = tl.program_id(1)
        h = tl.program_id(2).to(tl.int64)
        i = tl.arange(0, 64)
        if q < G:
            j = q.to(tl.int64) * 64 + i
            ok = j < glen
            e = tl.load(tab_ptr + 2 * NP + j, mask=ok, other=-1)
            hp = e >= 0
            mir = ok & (j < c)
            tail = ok & (j >= c)
            put = tail & hp
            pi = tl.where(hp, e // BLK, 0)
            a = tl.where(hp, e % BLK, 0)
            t = tl.where(tail, j - c, 0)
            if r < KW:
                mp = tl.load(tab_ptr + 2 * pi, mask=hp, other=0).to(tl.pointer_type(tl.int64))
                mo = ((l * H + h) * KW + r) * BLK + a
                x = tl.load(mp + mo, mask=mir, other=0)
                y = tl.load(tk_ptr + (h * KW + r) * m + t, mask=tail, other=0)
                x = tl.where(mir, x, y)
                tl.store(pka_ptr + (h * KW + r) * KS + j, x, mask=ok)
                tl.store(mp + mo, x, mask=put)
            elif r < KW + VW:
                w = r - KW
                mp = tl.load(tab_ptr + 2 * pi + 1, mask=hp, other=0).to(tl.pointer_type(tl.int64))
                mo = ((l * H + h) * BLK + a) * VW + w
                x = tl.load(mp + mo, mask=mir, other=0)
                y = tl.load(tv_ptr + (h * m + t) * VW + w, mask=tail, other=0)
                x = tl.where(mir, x, y)
                tl.store(mp + mo, x, mask=put)
                bits = (x[:, None] >> i[None, :].to(tl.int64)) & 1           # [token, s]
                word = tl.sum(bits << i[:, None].to(tl.int64), axis=0)        # [s]
                tl.store(pv_ptr + h * VS + q * (VW * 64) + w * 64 + i, word)
        elif q < G + NWQ:
            if r < 2:  # dense window rows (q - G) * BT.. of K (r 0) / V (r 1), head h
                tt = (q - G) * BT + tl.arange(0, BT)
                mk = tt < SINK + LOCAL
                src = tl.where(tt < SINK, s + tt, s + n - LOCAL - SINK + tt).to(tl.int64)
                d = tl.arange(0, D)
                x2 = tl.load(g_ptr + src[:, None] * gs0 + r * gs1 + h * gs2 + d[None, :],
                             mask=mk[:, None])
                tl.store(dw_ptr + ((r * H + h) * (SINK + LOCAL) + tt[:, None]) * D + d[None, :],
                         x2, mask=mk[:, None])
        else:  # zero past the content: key columns glen.., value groups G..
            z = (q - G - NWQ).to(tl.int64)
            zero = tl.zeros([64], dtype=tl.int64)
            if r < KW:
                j = glen + z * 64 + i
                tl.store(pka_ptr + (h * KW + r) * KS + j, zero, mask=j < KS)
            elif r < KW + VW:
                o = (G + z) * (VW * 64)
                tl.store(pv_ptr + h * VS + o + (r - KW) * 64 + i, zero, mask=o < VS)

    # The writes alone (a fresh prompt's plan): global tokens j >= c of pka
    # [H,KW,Tk] / pva [H,Tk,VW] (columns j0 + j) to their mirrors (tab as
    # above).
    @triton.jit
    def _code_put_kernel(tab_ptr, NP, pka_ptr, pva_ptr, Tk, j0, c, glen, l, BLK,
                         H: tl.constexpr, KW: tl.constexpr, VW: tl.constexpr,
                         BLOCK: tl.constexpr):
        r = tl.program_id(1)
        h = tl.program_id(2).to(tl.int64)
        j = c + tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        ok = j < glen
        e = tl.load(tab_ptr + 2 * NP + j, mask=ok, other=-1)
        put = ok & (e >= 0)
        pi = tl.where(put, e // BLK, 0)
        a = tl.where(put, e % BLK, 0)
        if r < KW:
            mp = tl.load(tab_ptr + 2 * pi, mask=put, other=0).to(tl.pointer_type(tl.int64))
            x = tl.load(pka_ptr + (h * KW + r) * Tk + j0 + j, mask=put)
            tl.store(mp + ((l * H + h) * KW + r) * BLK + a, x, mask=put)
        else:
            w = r - KW
            mp = tl.load(tab_ptr + 2 * pi + 1, mask=put, other=0).to(tl.pointer_type(tl.int64))
            x = tl.load(pva_ptr + (h * Tk + j0 + j) * VW + w, mask=put)
            tl.store(mp + ((l * H + h) * BLK + a) * VW + w, x, mask=put)

    # Fresh decode-batch rows (build / slot change) in one launch instead of
    # four copies per row and layer. Table entry e (_FILL_F int64) = one row
    # of one layer: source / destination addresses of its packed key [H*KW,
    # g], packed value [H, G*CV], dense K / V window [H, d, D] (token / head
    # strides), the lengths, and the destination widths. Packed rows are
    # written to their full width (zero past g, as the merges need).
    @triton.jit
    def _fill_rows_kernel(tab_ptr, HKW: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                          BLOCK: tl.constexpr, BT: tl.constexpr):
        t = tab_ptr + tl.program_id(0).to(tl.int64) * 17
        p = tl.program_id(1)
        o = tl.arange(0, BLOCK)
        if p < HKW + H:  # packed key row p / packed value head p - HKW
            kq = p < HKW
            ho = tl.where(kq, p, p - HKW)
            psrc = tl.load(t + tl.where(kq, 0, 1)).to(tl.pointer_type(tl.int64))
            pdst = tl.load(t + tl.where(kq, 4, 5)).to(tl.pointer_type(tl.int64))
            n, w = tl.load(t + tl.where(kq, 8, 9)), tl.load(t + tl.where(kq, 11, 12))
            for s in range(0, w, BLOCK):
                x = tl.load(psrc + ho * n + s + o, mask=s + o < n, other=0)
                tl.store(pdst + ho * w + s + o, x, mask=s + o < w)
        else:  # dense K (kv 0) / V (kv 1) window of head h
            q = p - HKW - H
            kv = q // H
            h = q % H
            dsrc = tl.load(t + 2 + kv).to(tl.pointer_type(tl.bfloat16))
            ddst = tl.load(t + 6 + kv).to(tl.pointer_type(tl.bfloat16))
            d = tl.load(t + 10)
            st, sh = tl.load(t + 13 + 2 * kv), tl.load(t + 14 + 2 * kv)
            r = tl.arange(0, BT)
            c = tl.arange(0, D)
            for s2 in range(0, d, BT):
                tk = (s2 + r).to(tl.int64)
                m = (tk < d)[:, None]
                x2 = tl.load(dsrc + tk[:, None] * st + h * sh + c[None, :], mask=m)
                tl.store(ddst + (tk[:, None] * H + h) * D + c[None, :], x2, mask=m)
except Exception:  # pragma: no cover - falls back to index_put
    _append_kv_kernel = _gdc_wait = None
    _slide_vmerge_kernel = _slide_kmerge_kernel = _slide_shift_kernel = None
    _slide_vmerge_rows_kernel = _slide_kmerge_rows_kernel = None
    _prep_vgroup_kernel = _prep_kcopy_kernel = _fill_rows_kernel = None
    _code_asm_kernel = _code_put_kernel = None

_SLIDE_S = 64  # staging slots per row = tokens compressed per row per epoch (max)
# _RowStore: rows up to this capacity (tokens) share allocations (0: never)
_ROWS_SHARE_CAP = int(os.environ.get("PICOKV_ROWS_SHARE_CAP", "2048"))
# PICOKV_RADIX_STATS=1: one stdout line per radix extend (tokens by copy path)
_RADIX_STATS = os.environ.get("PICOKV_RADIX_STATS") == "1"
# PICOKV_RADIX_CODES_CHECK=1: every code-reusing prepare is compared with the
# full recompress (torch.equal of the cache fields; a mismatch is logged)
_RADIX_CODES_CHECK = os.environ.get("PICOKV_RADIX_CODES_CHECK") == "1"


def _round_up(x: int, m: int) -> int:
    return (int(x) + m - 1) // m * m


def _pk_full(t):
    """PICOKV_PREALLOC: the full-capacity buffer ([1, H, KW, cap] keys /
    [1, H, cap/64, CV] values, zero past the content) a request cache's
    packed tensor is a prefix view of, else None. Found through the view's
    base, so a cache whose packed tensor is replaced drops its buffer."""
    b = t._base if t is not None else None
    return b if b is not None and getattr(b, "_pk_full", False) else None


def _c_atoi(s: Optional[str]) -> int:
    """C atoi of an env value (leading blanks, sign, digits; else 0): the
    parse the extension's getenv switches use, so Python and C++ agree."""
    m = re.match(r"\s*([+-]?\d+)", s or "")
    return int(m.group(1)) if m else 0


def picokv_tc_tail_on() -> bool:
    """The extension's tensor-core tail switch (picokv_tc_tail_enabled):
    off iff PICOKV_NO_TC != 0 or PICOKV_TC_TAIL is set and == 0 (C atoi)."""
    no, tc = os.environ.get("PICOKV_NO_TC"), os.environ.get("PICOKV_TC_TAIL")
    if no is not None and _c_atoi(no) != 0:
        return False
    return tc is None or _c_atoi(tc) != 0


def picokv_code_bytes() -> int:
    """Packed bytes per token / kv-head / layer of the compressed cache: the
    key_bits + value_bits codes of the projection config (80 at 320/320)."""
    sc = StandaloneConfig()
    return (int(sc.key_bits) + int(sc.value_bits)) // 8


def picokv_slide_off_reason(sc: Optional[StandaloneConfig] = None) -> Optional[str]:
    """Why the decode slide is off (None: on). Off, decode tokens stay dense
    (bf16 K/V per token, not packed codes) and the window grows with the
    output: the per-token pool accounting and the fixed reserve assume the
    slide. The backend's own predicate; the pool sizing reads it too."""
    E = os.environ.get
    if sc is None:
        sc = StandaloneConfig(eas=float(E("PICOKV_EAS", "0.0")))
    if E("PICOKV_SLIDE", "1") != "1":
        return "PICOKV_SLIDE=0"
    if not E("PICOKV_OFFLINE_PROJ") and E("PICOKV_ONLINE", "0") == "1":
        return "online projections"
    if float(sc.eas) != 0.0:
        return f"eas={sc.eas}"
    if sc.use_pruning:
        return "pruned tail"
    if blockk_fixed_cuda is None or _slide_shift_kernel is None:
        return "kernels unavailable"
    loc = int(E("PICOKV_SLIDE_LOCAL", sc.local_size))
    if loc % 32 or loc < _SLIDE_S or loc < int(sc.local_size):
        return f"local={loc}"
    return None


def picokv_rows_off_reason(q_heads: int, kv_heads: int) -> Optional[str]:
    """Why per-row packed storage is off from the env / shape alone (None:
    on if the extension has the rows op). Off, a decode batch's packed codes
    are a rectangle of B x its longest row, which per-token accounting does
    not bound."""
    E = os.environ.get
    if E("PICOKV_ROWS", "1") != "1":
        return "PICOKV_ROWS=0"
    if E("PICOKV_FAST_BUILD", "1") != "1":
        return "PICOKV_FAST_BUILD=0"
    if E("PICOKV_SHARED_PROJ", "1") == "0":
        return "PICOKV_SHARED_PROJ=0"
    if not picokv_tc_tail_on():
        return "tensor-core tail off (PICOKV_NO_TC / PICOKV_TC_TAIL)"
    sc = StandaloneConfig()
    if int(sc.key_bits) != 320 or int(sc.value_bits) != 320:
        return f"bits {sc.key_bits}/{sc.value_bits} (rows need 320/320)"
    if int(q_heads) != 4 * int(kv_heads):
        return f"GQA {q_heads}/{kv_heads} (rows need 4)"
    return picokv_slide_off_reason()


def picokv_cell_size_bytes(kv_heads: int, head_dim: int, layers: int,
                           q_heads: Optional[int] = None) -> int:
    """Per-token bytes of the token pool's accounting (cell_size) under
    picokv: the packed codes (picokv_code_bytes) per kv-head and layer when
    the decode slide is on; dense bf16 K/V when it is off (decode tokens stay
    dense, so packed accounting would admit ~6x what fits)."""
    why = picokv_slide_off_reason()
    if why is not None:
        logger.warning(
            "picokv: decode slide off (%s): decode tokens stay dense, so the token "
            "pool is accounted at dense bf16 K/V size (no admission gain).", why)
        return int(2 * kv_heads * head_dim * 2 * layers)
    if q_heads is not None:
        rw = picokv_rows_off_reason(q_heads, kv_heads)
        if rw is not None:
            logger.warning(
                "picokv: per-row packed storage off (%s): a decode batch holds B x "
                "its longest row of packed codes, which the token pool's accounting "
                "does not bound (mixed lengths can OOM).", rw)
    return int(kv_heads * picokv_code_bytes() * layers)


def _require_op(ok: bool, op: str, knob: str) -> None:
    """A default-on feature found its extension op missing: fail with the
    loaded build instead of serving silently slower / unbounded."""
    if ok:
        return
    from picokv.picokv_cuda_extension import picokv_cuda

    raise RuntimeError(
        f"picokv: the loaded CUDA extension ({picokv_cuda.loaded_extension_path()}) "
        f"has no {op}: rebuild it (python picokv/build_cuda_extensions.py; "
        f"PICOKV_PREBUILT_DIR selects the build) or set {knob}")


def picokv_fixed_reserve_bytes(max_reqs: int, kv_heads: int, head_dim: int, layers: int,
                               dtype_bytes: int = 2, kv_bits: Optional[int] = None) -> int:
    """Decode-batch memory of up to ``max_reqs`` rows that the per-token pool
    accounting (picokv_cell_size_bytes: the key_bits + value_bits codes per
    token / kv-head / layer; kv_bits overrides both) does not cover: the
    dense windows — the bucket's rows x (sink + local + one 64-token epoch)
    of K and V, local = PICOKV_SLIDE_LOCAL, or PICOKV_DENSE_LOCAL (opt-in)
    while the batch has <= PICOKV_DENSE_LOCAL_MAX_BS rows — plus each
    row's packed headroom (<= 2 x PICOKV_SLIDE_HEADROOM + 511 tokens past its
    length; slot-mode regrows double the headroom), plus the dead slot batch
    a finished batch may keep for the next decode (<= PICOKV_SLOTS_MAX_WASTE_MB,
    its tokens already free in the pool). A bound only with the decode slide
    on (offline projections, eas=0) and per-row storage (PICOKV_ROWS=1): a
    rectangle holds B x its longest row."""
    if max_reqs <= 0:
        return 0
    E = os.environ.get
    from picokv.picokv_standalone import StandaloneConfig

    sc = StandaloneConfig()  # the backend's sink / local (it rejects SLIDE_LOCAL < local)
    sink = int(sc.sink_size)
    local = max(int(sc.local_size), int(E("PICOKV_SLIDE_LOCAL", sc.local_size)))
    dloc = int(E("PICOKV_DENSE_LOCAL", "0")) // 32 * 32
    dbs = int(E("PICOKV_DENSE_LOCAL_MAX_BS", "8")) if dloc > local else 0
    head = max(0, int(E("PICOKV_SLIDE_HEADROOM", "1024")))
    S = 1 << (int(max_reqs) - 1).bit_length()  # >= the BCG bucket of max_reqs rows
    try:
        # the largest decode bucket a batch of <= max_reqs rows pads to: the
        # graph runner caps its list at (and adds) the aligned request count
        from sglang.srt.utils.common import get_cuda_graph_max_batch_size

        S = min(S, int(get_cuda_graph_max_batch_size(int(max_reqs))))
    except Exception:  # no engine config (unit tests): the power-of-2 bound
        pass
    win = max(min(S, dbs) * (sink + dloc + _SLIDE_S), S * (sink + local + _SLIDE_S))
    dense_tok = 2 * kv_heads * head_dim * dtype_bytes * layers
    code = picokv_code_bytes() if kv_bits is None else 2 * int(kv_bits) // 8
    packed_tok = kv_heads * code * layers
    dead = (int(float(E("PICOKV_SLOTS_MAX_WASTE_MB", "1024")) * (1 << 20))
            if E("PICOKV_SLOTS", "1") == "1" else 0)
    return win * dense_tok + int(max_reqs) * (2 * head + 511) * packed_tok + dead


def _stage_mode() -> str:
    """PICOKV_CHUNK_STAGE: host (default) = exact chunked prefill through the
    pinned host stage (_ChunkStage); off = continuation chunks attend only
    themselves (pre-C1). The stage needs the ragged prefill: off under
    PICOKV_PREFILL=sdpa."""
    if os.environ.get("PICOKV_PREFILL", "fi").strip().lower() == "sdpa":
        return "off"
    return "off" if os.environ.get("PICOKV_CHUNK_STAGE", "host").strip().lower() == "off" else "host"


def picokv_radix_mode() -> bool:
    """PICOKV_RADIX=host (default off): radix prefix reuse through host K/V —
    SGLang's radix tree (mem_cache/picokv_radix_cache.py) whose nodes own the
    raw K/V of their tokens in the chunk stage's pinned pages; a hit streams
    the prefix back (_ChunkStage radix mode). Needs the chunk stage."""
    return (os.environ.get("PICOKV_RADIX", "off").strip().lower() == "host"
            and _stage_mode() == "host")


def _radix_host_gb(tp: int = 1) -> float:
    """PICOKV_RADIX_HOST_GB: the radix host K/V store's pinned-memory cap,
    across the attention-TP ranks (each holds its heads' share: one rank's
    store is GB / tp, the same tokens as at TP 1)."""
    return float(os.environ.get("PICOKV_RADIX_HOST_GB", "32")) / max(1, int(tp))


class _StageFailed(RuntimeError):
    """The chunk stage / radix host store could not take an extend (device or
    pinned host memory, on this or another attention-TP rank)."""


_CUDART = None


def _cudart():
    """(cudaMemcpy2DAsync, cudaGetLastError) of the CUDA runtime torch loaded
    (/proc/self/maps; ctypes), or False: one strided D2H per run with no
    device-side gather (PICOKV_RADIX_2D=0: torch copies)."""
    global _CUDART
    if _CUDART is None:
        _CUDART = False
        if os.environ.get("PICOKV_RADIX_2D", "1") != "0":
            try:
                import ctypes

                path = "libcudart.so"
                with open("/proc/self/maps") as f:
                    for line in f:
                        p = line.split()[-1]
                        if "/libcudart.so" in p:
                            path = p
                            break
                # PyDLL: the call keeps the GIL (a CDLL call drops it, and any
                # other runnable thread of the scheduler then holds it for up
                # to the switch interval: 5 ms) - a copy issue is ~4 us
                lib = ctypes.PyDLL(path)
                fn = lib.cudaMemcpy2DAsync
                sz, vp = ctypes.c_size_t, ctypes.c_void_p
                fn.argtypes = [vp, sz, vp, sz, sz, sz, ctypes.c_int, vp]
                fn.restype = ctypes.c_int
                lib.cudaGetLastError.restype = ctypes.c_int
                _CUDART = (fn, lib.cudaGetLastError)
            except Exception as e:
                logger.warning(f"[PicoKVAttnBackend] radix host K/V: no cudaMemcpy2DAsync "
                               f"({e!r}): torch copies")
    return _CUDART


def picokv_stage_reserve_bytes(context_len: int, chunk: Optional[int], kv_heads: int,
                               head_dim: int, dtype_bytes: int = 2) -> int:
    """GPU bytes of exact chunked prefill (_ChunkStage) the per-token pool
    does not cover: its two layer buffers of raw K/V for a whole prompt plus
    one chunk (a request's last chunk next to the next request's first). 0
    when chunked prefill never splits a prompt or PICOKV_CHUNK_STAGE=off.
    Radix mode stages every extend: a whole-prompt prefix plus one extend
    (a chunk, or a prompt without chunking)."""
    tok = 2 * kv_heads * head_dim * dtype_bytes
    if picokv_radix_mode():
        c = min(int(chunk), int(context_len)) if chunk and chunk > 0 else int(context_len)
        return 2 * (int(context_len) + c) * tok
    if _stage_mode() == "off" or not chunk or chunk <= 0 or chunk >= context_len:
        return 0
    return 2 * (int(context_len) + int(chunk)) * tok


def load_offline_projections_pure(path: str) -> dict:
    """Transformers-free loader for the shipped block-32 encoder/decoder .pt.

    Returns a dict with ``key_encoder``/``value_encoder`` (and matching
    ``*_decoder``) each shaped ``[num_layers, num_kv_heads, bits, head_dim]``.
    """
    data = torch.load(path, map_location="cpu")
    for side in ("key", "value"):
        if f"{side}_encoder" not in data and side in data:
            data[f"{side}_encoder"] = data.pop(side)
    if "key_encoder" not in data or "value_encoder" not in data:
        raise ValueError(f"{path}: missing key_encoder/value_encoder")
    for side in ("key", "value"):
        if f"{side}_decoder" not in data:
            raise ValueError(
                f"{path} stores encoders only; PiCoKV serving needs the folded "
                "encoder+decoder pair (rebuild with create_offline_projections.py)"
            )
    return data


def validate_offline_projections(data: dict, path: str, layers: int, kv_heads: Optional[int],
                                 head_dim: int, scfg: StandaloneConfig) -> None:
    """Reject a projection .pt that does not fit the served model / config
    (another model's file would otherwise run on the first ``layers`` slices,
    or on the wrong bit widths, and decode garbage). The file carries no model
    identity: a same-shape file of another checkpoint is not detectable."""
    bits = {"key": int(scfg.key_bits), "value": int(scfg.value_bits)}
    bad = []
    for side in ("key", "value"):
        for part in ("encoder", "decoder"):
            t = data[f"{side}_{part}"]
            want = (layers, kv_heads if kv_heads else t.shape[1], bits[side], head_dim)
            if t.dim() != 4 or tuple(t.shape) != want:
                bad.append(f"{side}_{part} {tuple(t.shape)} != {want} "
                           f"[layers, kv_heads, bits, head_dim]")
    meta = data.get("meta")
    if isinstance(meta, dict):
        for k, v in (("k_bits", scfg.key_bits), ("v_bits", scfg.value_bits),
                     ("zerorope", scfg.zerorope), ("sink_size", scfg.sink_size),
                     ("local_size", scfg.local_size)):
            if k in meta and int(meta[k]) != int(v):
                bad.append(f"meta {k}={meta[k]} != {v}")
    prog = data.get("progress")
    if isinstance(prog, dict):
        for k in ("key_done", "value_done"):
            t = prog.get(k)
            if torch.is_tensor(t) and not bool(t.all()):
                bad.append(f"progress {k}: calibration incomplete "
                           f"({int(t.sum())}/{t.numel()} heads)")
    if bad:
        raise ValueError(f"PICOKV offline projection {path} does not fit the model: "
                         + "; ".join(bad))


def _slice_layer(stack: Optional[torch.Tensor], layer_idx: int, heads: int):
    """Slice ``[layers, heads, bits, dim]`` to one layer, or return None."""
    if stack is None:
        return None
    if layer_idx >= stack.shape[0]:
        raise ValueError(f"projection has {stack.shape[0]} layers; got {layer_idx}")
    layer = stack[layer_idx]
    if layer.shape[0] != heads:
        raise ValueError(
            f"projection layer {layer_idx} has {layer.shape[0]} heads; model has {heads}"
        )
    return layer


class _RowStore:
    """PICOKV_ROWS: one decode-batch layer's packed global codes stored per
    row instead of as one [B, H, KW, L] rectangle padded to the longest row
    (mixed lengths: sum of the rows' capacities instead of B x max; vacant
    slots hold nothing). Row b: keys k[b] [1, H, KW, cap_b] and values v[b]
    [1, H, cap_b/64, CV], cap_b a multiple of 512, zero past the row's gl
    (the merges OR into it); None = vacant (cap 0). The tail kernel and the
    merges reach the rows through device tables kp / vp (int64 addresses)
    and ct (int32 caps) whose own addresses never change: a row move,
    regrow or refill rewrites them in place (sync) and captured graphs
    stay valid. Rows change only through set / permute / alloc, which keep
    host mirrors of the three tables (sync: one H2D of them, no per-row
    tensor calls; kp, vp and ct are rows of one device buffer, ct its int32
    view)."""

    def __init__(self, B, H, KW, CV, dtype, device, tabs=None):
        self.k, self.v = [None] * B, [None] * B
        self.H, self.KW, self.CV, self.dtype, self.device = H, KW, CV, dtype, device
        self._B = B
        gi = tabs.take(B) if tabs is not None else None
        self._grp, self._gi = (tabs, gi) if gi is not None else (None, None)
        if gi is None:
            self._h = np.zeros((3, B), dtype=np.int64)  # host mirror: k ptr, v ptr, caps (int32)
            self._tab = torch.zeros(3, B, dtype=torch.int64, device=device)
        else:  # rows of the batch's shared mirrors / tables (_RowTabs)
            self._h, self._tab = tabs.h[gi], tabs.t[gi]
        self._hc = self._h[2].view(np.int32)[:B]
        self.ct = self._tab[2].view(torch.int32)[:B]
        self.tables = (self._tab[0], self._tab[1], self.ct)

    def caps(self):
        return self._hc.tolist()

    def set(self, i, k, v):
        """Row i's storage = (k, v) (None, None: vacant)."""
        self.k[i], self.v[i] = k, v
        h = self._h
        if k is None:
            h[0, i] = h[1, i] = 0
            self._hc[i] = 0
        else:
            h[0, i], h[1, i] = k.data_ptr(), v.data_ptr()
            self._hc[i] = k.shape[-1]

    @staticmethod
    def plan(src, n):
        """permute's index arrays for ``src`` / ``n``: computed once per
        membership change and shared by every layer's store."""
        keep = [i for i, j in enumerate(src) if j is not None]
        return ([-1 if j is None else j for j in src] + [-1] * (n - len(src)),
                np.asarray(keep, dtype=np.int64),
                np.asarray([src[i] for i in keep], dtype=np.int64))

    def permute(self, src, n, plan=None):
        """Row i <- old row src[i] (None: vacant) for i < len(src); rows
        len(src)..n-1 vacant. ``plan``: plan(src, n), precomputed."""
        g, keep, frm = plan or self.plan(src, n)
        k, v = self.k + [None], self.v + [None]  # index -1: vacant
        self.k, self.v = [k[j] for j in g], [v[j] for j in g]
        h2, hc = self._h[:2, frm], self._hc[frm]  # gathered copies
        self._h[:] = 0
        self._h[:2, keep] = h2
        self._hc[keep] = hc

    def alloc(self, rows, caps):
        """Zeroed storage for ``rows`` at ``caps`` (multiples of 512; 0 =
        vacant), one key and one value buffer PER ROW: a shared buffer would
        stay pinned by its last live row after the others left or regrew
        (a finished 120k row held by a short survivor), memory SGLang's pool
        already counts as free. Short rows (cap <= PICOKV_ROWS_SHARE_CAP
        tokens) of one capacity share one buffer per side per <= 16 rows
        instead (rows = its unbind views): a tensor per row costs ~3.5 us of
        host time (64 short rows x 32 layers: 15 ms per batch build, 4 ms
        shared), and a survivor pins at most 15 short rows. A cap of None:
        the row's storage is already set (adopted, PICOKV_PREALLOC)."""
        H, KW, CV = self.H, self.KW, self.CV
        new, small = [], {}
        for i, c in zip(rows, caps):
            if c is None:
                continue
            if not c:
                self.set(i, None, None)
            elif c <= _ROWS_SHARE_CAP:
                small.setdefault(c, []).append(i)
            else:
                self.set(i, torch.empty(1, H, KW, c, dtype=self.dtype, device=self.device),
                         torch.empty(1, H, c // 64, CV, dtype=self.dtype, device=self.device))
                new += [self.k[i], self.v[i]]
        for c, ids in small.items():
            for g in range(0, len(ids), 16):
                grp = ids[g:g + 16]
                kb = torch.empty(len(grp), 1, H, KW, c, dtype=self.dtype, device=self.device)
                vb = torch.empty(len(grp), 1, H, c // 64, CV, dtype=self.dtype, device=self.device)
                new += [kb, vb]
                # row r of a group: base + r x row stride (== unbind's views)
                kr, vr = kb.unbind(0), vb.unbind(0)
                for r, i in enumerate(grp):
                    self.k[i], self.v[i] = kr[r], vr[r]
                ix = np.asarray(grp)
                self._h[0, ix] = kb.data_ptr() + np.arange(len(grp)) * kb.stride(0) * kb.element_size()
                self._h[1, ix] = vb.data_ptr() + np.arange(len(grp)) * vb.stride(0) * vb.element_size()
                self._hc[ix] = c
        if new:
            torch._foreach_zero_(new)

    @staticmethod
    def _span(stores):
        """(tabs, a, m) if ``stores`` are rows a..a+m-1 of one _RowTabs, in
        order (their mirrors / tables are then slices of its arrays)."""
        g, a = stores[0]._grp, stores[0]._gi
        if g is None:
            return None
        for n, s in enumerate(stores):
            if s._grp is not g or s._gi != a + n:
                return None
        return g, a, len(stores)

    @staticmethod
    def permute_all(stores, src, n, plan=None):
        """permute(src, n, plan) of every store; one numpy pass over the
        host mirrors when they are a _RowTabs span (the same values)."""
        plan = plan or _RowStore.plan(src, n)
        sp = _RowStore._span(stores) if len(stores) > 1 else None
        if sp is None:
            for s in stores:
                s.permute(src, n, plan)
            return
        g, keep, frm = plan
        ig = operator.itemgetter(*g) if len(g) > 1 else None
        for s in stores:
            k, v = s.k + [None], s.v + [None]
            s.k, s.v = ((list(ig(k)), list(ig(v))) if ig is not None else
                        ([k[j] for j in g], [v[j] for j in g]))
        tabs, a, m = sp
        h = tabs.h[a:a + m]
        hc = h[:, 2].view(np.int32)[:, :h.shape[2]]
        h2, c2 = h[:, :2, frm], hc[:, frm]  # gathered copies
        h[:] = 0
        h[:, :2, keep] = h2
        hc[:, keep] = c2

    @staticmethod
    def alloc_all(stores, rows, caps):
        """alloc(rows, caps[n]) of every store n (the layers of one batch).
        Short rows' storage shares one buffer per side per capacity ACROSS
        the stores, <= 16 rows x the stores per buffer (a row's layers join
        and leave together: a survivor still pins at most 15 other short rows
        per layer, as in alloc); one zero launch for all. Long rows: one
        buffer per row and layer, as in alloc. ~2 tensors per fresh short row
        instead of 2 per row and layer (32 layers: ~1.5 ms of host time per
        fresh row in a slot change)."""
        s0 = stores[0]
        geo = (s0.H, s0.KW, s0.CV, s0.dtype, s0.device)
        if len(stores) == 1 or any((s.H, s.KW, s.CV, s.dtype, s.device) != geo for s in stores):
            for s, c in zip(stores, caps):
                s.alloc(rows, c)
            return
        H, KW, CV, dt, dev = geo
        new, small = [], {}
        for r, i in enumerate(rows):
            for s, cs in zip(stores, caps):
                c = cs[r]
                if c is None:
                    continue  # adopted storage (set by the caller)
                if not c:
                    s.set(i, None, None)
                elif c <= _ROWS_SHARE_CAP:
                    small.setdefault(c, []).append((s, i))
                else:
                    s.set(i, torch.empty(1, H, KW, c, dtype=dt, device=dev),
                          torch.empty(1, H, c // 64, CV, dtype=dt, device=dev))
                    new += [s.k[i], s.v[i]]
        for c, pairs in small.items():
            grps, seen = [[]], set()  # <= 16 distinct rows per buffer
            for p in pairs:  # row-major: a row's layers are adjacent
                if p[1] not in seen:
                    if len(seen) == 16:
                        grps.append([])
                        seen = set()
                    seen.add(p[1])
                grps[-1].append(p)
            for grp in grps:
                kb = torch.empty(len(grp), 1, H, KW, c, dtype=dt, device=dev)
                vb = torch.empty(len(grp), 1, H, c // 64, CV, dtype=dt, device=dev)
                new += [kb, vb]
                kr, vr = kb.unbind(0), vb.unbind(0)
                kp, ks = kb.data_ptr(), kb.stride(0) * kb.element_size()
                vp, vs = vb.data_ptr(), vb.stride(0) * vb.element_size()
                for r, (s, i) in enumerate(grp):  # == set(i, kr[r], vr[r])
                    s.k[i], s.v[i] = kr[r], vr[r]
                    s._h[0, i], s._h[1, i] = kp + r * ks, vp + r * vs
                    s._hc[i] = c
        if new:
            torch._foreach_zero_(new)

    def sync(self):
        """Rewrite the device tables from the host mirrors (in place, async)."""
        self._tab.copy_(torch.from_numpy(self._h).pin_memory(), non_blocking=True)

    @staticmethod
    def sync_all(stores):
        """sync() of several stores from ONE pinned staging block (one copy
        when they are a _RowTabs span)."""
        sp = _RowStore._span(stores) if len(stores) > 1 else None
        if sp is not None:
            tabs, a, m = sp
            tabs.t[a:a + m].copy_(torch.from_numpy(tabs.h[a:a + m]).pin_memory(),
                                  non_blocking=True)
            return
        if len(stores) == 1 or len({s._h.shape for s in stores}) != 1:
            for s in stores:
                s.sync()
            return
        hp = torch.from_numpy(np.stack([s._h for s in stores])).pin_memory()
        for s, h in zip(stores, hp.unbind(0)):
            s._tab.copy_(h, non_blocking=True)

    def nbytes(self, held=None):
        """Bytes held (whole buffers: a row view pins its allocation).
        ``held``: a dict shared by several stores, filled instead (buffers
        alloc_all shares across layers count once)."""
        own = held is None
        held = {} if own else held
        for t in self.k + self.v:
            if t is not None:
                s = t.untyped_storage()
                held[s.data_ptr()] = s.nbytes()
        return sum(held.values()) if own else None


class _RowTabs:
    """The host mirrors and device tables of one batch's row stores (one
    per layer) as rows of ONE host array and ONE device buffer, handed out
    in store-creation order (layer order): a membership change permutes
    every layer's mirrors in one numpy pass and syncs them in one H2D copy
    (_RowStore.permute_all / sync_all) instead of one per layer. Addresses
    fixed for the batch's life, as a store's own tables."""

    def __init__(self, n, B, device):
        self.h = np.zeros((n, 3, B), dtype=np.int64)
        self.t = torch.zeros(n, 3, B, dtype=torch.int64, device=device)
        self.used = 0

    def take(self, B):
        if self.used >= self.h.shape[0] or B != self.h.shape[2]:
            return None  # more stores than layers / another row count: own tables
        self.used += 1
        return self.used - 1


class _ChunkStage:
    """C1 exact chunked prefill (PICOKV_CHUNK_STAGE=host, default). A
    continuation chunk must attend prefix ++ chunk — what one single-chunk
    prefill computes — but the compressed cache keeps no raw prefix K/V
    (PicoKVRequestCache.append merges the codes exactly; attention cannot
    read them). So every chunk that is not its request's last leaves its raw
    post-RoPE K/V in a pinned host block pool (D2H on a copy stream, per
    layer), and a continuation chunk streams layer l's prefix back into one
    of two GPU layer buffers two layers ahead (H2D overlapped with layers
    l-2 .. l-1: ~8.5 ms per layer at a 120k prefix vs ~28 ms of compute) and
    attends it with a second ragged plan (flashinfer causal = bottom-right:
    chunk token t sees kv [0, plen + t]).

    GPU: the two layer buffers G [2, context + chunk, 2, KVH, D] while any
    request is staged (1.1 GB at 128k, KVH 8; picokv_stage_reserve_bytes) —
    O(1) in layers and requests. Host: blocks of BLK tokens x every layer
    (1 GiB at BLK 8192, KVH 8), pinned once and reused (PICOKV_STAGE_PREALLOC
    =1: context-sized at init, else grown on first use, ~0.45 s per block).

    Row records of one forward (``rows``, layer 0 .. L-1): [batch index,
    token offset, chunk len, prefix len, base in G, more, rid, entry];
    continuation rows (prefix > 0) take G's first regions in batch order (one
    ragged plan over them), first chunks of chunked requests the rest."""

    def __init__(self, layers, H, D, dtype, device, ctx, chunk, blk, prealloc):
        self.L, self.H, self.D, self.dtype, self.dev = layers, H, D, dtype, device
        self.ctx, self.chunk, self.BLK = int(ctx), int(chunk), int(blk)
        self.pool, self.free = [], []
        self.ent = {}  # rid -> {"n": staged tokens, "blocks": [pool index per BLK tokens]}
        self.rows = []
        self.wplan = None  # _wplan of rows
        self.G = None
        self.cs = torch.cuda.Stream(device=device)
        self.ev_r = [torch.cuda.Event() for _ in range(layers)]  # copy side: layer l ready
        self.ev_u = [torch.cuda.Event() for _ in range(layers)]  # compute side: layer l done
        self.ev_d = [torch.cuda.Event() for _ in range(layers)]  # radix D2H of layer l done
        self._dprev = None  # (ev_d, K/V buffers) of the previous layer's D2H (_d2h_done)
        self.tree = self.rplan = None  # radix mode (attach)
        self.hot = None  # radix mode: the GPU hot tier's mirrors (attach)
        # radix mode, per extend (rbegin): the forward's stream, _d2h's layout
        # verdict (every layer's K/V views share it)
        self.cur = self._dok = None
        # attention TP: ok -> every rank ok (PicoKVAttnBackend._tp_sync); the
        # ranks' schedulers must see the same staged / lost rows and radix trees
        self.sync = None
        if prealloc:
            self._grow(-(-self.ctx // self.BLK))

    def _grow(self, n):
        while len(self.pool) < n:
            self.pool.append(torch.empty((self.L, self.BLK, 2, self.H, self.D),
                                         dtype=self.dtype, pin_memory=True))
            if self.tree is not None:
                self.tree.slots.grow(self.BLK)
            else:
                self.free.append(len(self.pool) - 1)

    def _grow_all(self, n):
        """Radix mode: _grow on every attention-TP rank or on none (a failed
        pinned allocation rolls each rank back), so the host stores keep one
        size and the trees evict alike. False: not grown."""
        k = len(self.pool)
        try:
            self._grow(n)
            ok = True
        except RuntimeError:
            ok = False
        if self.sync is not None:
            ok = self.sync(ok)
        if not ok:
            m = len(self.pool) - k
            del self.pool[k:]
            self.tree.slots.shrink(m * self.BLK)
        return ok

    # -- radix mode (PICOKV_RADIX=host) ---------------------------------------
    # The pages are the radix tree's host K/V store: slot s = page s // BLK,
    # token s % BLK (PicoKVRadixCache.slots hands them out). Every extend row
    # writes its tokens' K/V to fresh slots (noted under its request id; the
    # tree takes them on insert): per layer, D2H straight from the layer's
    # K/V on the D2H stream ds (no device copy, no G: a miss costs what it
    # costs without radix). A row with a prefix (a radix hit or its earlier
    # chunks) streams the prefix's slots into G (cs) and attends it there.
    # Rows: [batch index, token offset, len, prefix len, base in G, False,
    # req row]; ``rows`` holds the prefix rows only (G = their prefix ++
    # tokens in order). rplan: (H2D runs (page, a, n, dst), D2D runs (dst,
    # src, n) for a slot shared by several rows, D2H runs (page, a, n, token
    # offset), the D2D runs as (dst, src) device indices or None, the D2H
    # runs as (host address at layer 0, token offset, n), then the hot
    # tier's: mirror -> G runs (mirror, a, n, dst), G -> mirror fill runs of
    # the prefix slots a mirror lacks, the new tokens' mirror writes
    # (device address at layer 0, token offset, n) and the slots those two
    # make valid once the forward's last layer is queued).
    #
    # GPU hot tier (PICOKV_RADIX_HOT=1, default; PICOKV_RADIX_HOT_GB, default
    # 8 GiB per rank, bounds the mirrors): a page may have a device
    # mirror [L, BLK, 2, H, D] (same layout); a prefix slot whose mirror copy
    # is valid streams device to device, the rest over PCIe (and fills the
    # mirror on the way). Every write to a mirrored page's slots also writes
    # the mirror (_d2h), so a valid slot stays valid until its page loses the
    # mirror. A mirror holds the pool's accounting tokens of its bytes
    # (tree.pk_hot_hold, only while PICOKV_RADIX_HOT_FREE of the pool stays
    # free) and the tree counts them evictable: its evict() frees mirrors
    # first (hot_evict), so the mirrors never cost admission or a request's
    # memory. hot: page -> [mirror, accounting token indices, valid slots
    # (bool [BLK]), last-use tick].

    def cap_pages(self, cap_gb):
        page = self.L * self.BLK * 2 * self.H * self.D * torch.finfo(self.dtype).bits // 8
        return int(cap_gb * (1 << 30)) // page

    def attach(self, tree, cap_gb, cell=0, pool=None, hot_gb=float("inf"), codes=None):
        """Radix mode on ``tree``; ``cell``: the token pool's accounting bytes
        per token (0: no hot tier), ``pool``: the mirrors' allocator pool,
        ``hot_gb``: PICOKV_RADIX_HOT_GB (this rank's share), ``codes``: (sink,
        local, key words, value words, min prompt) of the compressed prepare
        when mirrors keep codes (PICOKV_RADIX_CODES), else None."""
        self.tree, self.free, self.ent = tree, [], {}
        tree.slots.grow(len(self.pool) * self.BLK)
        self.rcap = max(len(self.pool), self.cap_pages(cap_gb))
        self.rtok = self.ctx + min(self.chunk, self.ctx)
        tree.picokv_rtok = self.rtok  # admission: prefix rows per batch fit G
        self.rwarn = set()
        self.ds = torch.cuda.Stream(device=self.dev)
        self.tb = 2 * self.H * self.D * torch.finfo(self.dtype).bits // 8  # bytes / token
        self.hot, self.code = None, None
        self.cplan, self.cmarks, self.cwr, self.rsl = {}, [], {}, {}
        if cell > 0 and os.environ.get("PICOKV_RADIX_HOT", "1") != "0":
            page = self.L * self.BLK * self.tb
            if codes is not None:
                self.code = codes
                page += self.L * self.BLK * self.H * (codes[2] + codes[3]) * 8
            self.hot, self.htick, self.hpool = {}, 0, pool
            self.hn = -(-page // int(cell))  # accounting tokens per mirror
            self.hcap = int(hot_gb * (1 << 30)) // page if hot_gb != float("inf") else 1 << 30
            tree._pk_hot = self

    def _r_warn(self, what, msg):
        if what not in self.rwarn:
            self.rwarn.add(what)
            logger.warning(f"[PicoKVAttnBackend] radix host K/V: {msg}")

    def _r_alloc(self, n):
        """n host slots: free ones, then new pages up to PICOKV_RADIX_HOST_GB,
        then evicting the tree (LRU leaves), then dropping the host K/V of
        nodes only decoding requests hold (drop_host), then pages past the
        cap (the running prefills alone exceed it)."""
        sl = self.tree.slots
        x = sl.alloc(n)
        if x is None and len(self.pool) < self.rcap:
            if not self._grow_all(min(self.rcap,
                                      len(self.pool) + -(-(n - sl.nfree) // self.BLK))):
                self.rcap = len(self.pool)  # pinned memory is short: the cap is here
            x = sl.alloc(n)
        if x is None:
            self.tree.evict_host(n)
            x = sl.alloc(n)
        if x is None:  # the rest is locked by decoding requests: drop its host K/V
            self.tree.drop_host(n - sl.nfree)
            x = sl.alloc(n)
        if x is None:
            self._r_warn("cap", f"host store past PICOKV_RADIX_HOST_GB ({len(self.pool)} "
                                f"pages): {n} tokens held by running requests")
            if self._grow_all(len(self.pool) + -(-(n - sl.nfree) // self.BLK)):
                x = sl.alloc(n)
        return x

    def _hot_take(self, pages):
        """rbegin: mirrors for the pages this extend reads (prefix slots) or
        writes (new slots). A held mirror's tick moves; a page without one
        gets a new mirror while pk_hot_hold and PICOKV_RADIX_HOT_GB allow,
        else the least recently used mirror of a page this extend does not
        touch (no slot valid for its new page; the copy streams are in order
        behind every earlier use of it). New mirrors get one attention-TP
        verdict: every rank's pool holds the same accounting tokens."""
        hot = self.hot
        self.htick += 1
        tick, need = self.htick, []
        for p in pages:
            h = hot.get(p)
            if h is None:
                need.append(p)
            else:
                h[3] = tick
        if not need:
            return
        made = []
        for p in need:
            idx = self.tree.pk_hot_hold(self.hn) if len(hot) < self.hcap else None
            if idx is not None:
                hot[p] = [None, idx, np.zeros(self.BLK, dtype=bool), tick, None, None,
                          np.zeros(self.BLK, dtype=bool)]
                made.append(p)
                continue
            old = [q for q, h in hot.items() if h[3] < tick]
            if not old:
                break
            h = hot.pop(min(old, key=lambda q: hot[q][3]))
            h[2][:] = False
            h[6][:] = False
            h[3] = tick
            hot[p] = h
        if not made:
            return
        ok = True
        try:
            with (torch.cuda.use_mem_pool(self.hpool) if self.hpool is not None
                  else contextlib.nullcontext()):
                for p in made:
                    hot[p][0] = torch.empty((self.L, self.BLK, 2, self.H, self.D),
                                            dtype=self.dtype, device=self.dev)
                    if self.code is not None:
                        hot[p][4] = torch.empty((self.L, self.H, self.code[2], self.BLK),
                                                dtype=torch.int64, device=self.dev)
                        hot[p][5] = torch.empty((self.L, self.H, self.BLK, self.code[3]),
                                                dtype=torch.int64, device=self.dev)
        except RuntimeError:
            ok = False
        if self.sync is not None:
            ok = self.sync(ok)
        if not ok:
            for p in made:
                h = hot.pop(p)
                h[0] = h[4] = h[5] = None
                self.tree.pk_hot_unhold(h[1])

    def hot_evict(self, n):
        """The tree's evict() (scheduler: admission / allocation short of
        tokens): free mirrors until n accounting tokens are back in the pool,
        pages holding no live slot first, then the least recently used; their
        host K/V stays. Returns the tokens freed."""
        hot = self.hot
        if not hot or n <= 0:
            return 0
        sl, B = self.tree.slots, self.BLK
        order = sorted(hot, key=lambda p: (sl.free_in(p * B, (p + 1) * B) < B, hot[p][3]))
        k = min(len(order), -(-n // self.hn))
        self._hot_drop(order[:k])
        return k * self.hn

    def hot_reset(self):
        """The tree's reset (flush_cache): every mirror goes."""
        if self.hot:
            self._hot_drop(list(self.hot))

    def _hot_drop(self, pages):
        """Free the mirrors of ``pages``. The copy streams are drained first:
        the tokens go back to the pool now and the request they were evicted
        for allocates now, so the device memory must be reusable at once
        (record_stream would hold it until those streams catch up)."""
        if not pages:
            return
        self.cs.synchronize()
        self.ds.synchronize()
        for p in pages:
            h = self.hot.pop(p)
            h[0] = h[4] = h[5] = None
            self.tree.pk_hot_unhold(h[1])

    @staticmethod
    def _pages(arrs, B):
        """Sorted distinct pages (slot // B) of the slot arrays ``arrs``:
        consecutive repeats dropped first (paths are mostly slot runs), then
        np.unique of the rest (a sort of every slot: ~0.65 ms at 16k)."""
        q = [x // B for x in arrs if len(x)]
        q = [x[np.r_[True, x[1:] != x[:-1]]] for x in q]
        return np.unique(np.concatenate(q)).tolist() if q else []

    @staticmethod
    def _runs(a, b, brk):
        """(a0, b0, n) runs of paired int arrays where both step by 1; a new
        run at every a % brk == 0 (a page start)."""
        if len(a) == 0:
            return []
        cut = (np.diff(a) != 1) | (np.diff(b) != 1) | (a[1:] % brk == 0)
        st = np.r_[0, np.flatnonzero(cut) + 1]
        return list(zip(a[st].tolist(), b[st].tolist(), (np.r_[st[1:], len(a)] - st).tolist()))

    def rbegin(self, rows_in, eseq, plens, rstr):
        """Radix mode, layer 0 of an extend (rows_in: req rows, rstr: request
        ids): every row stages its tokens; a row with a prefix streams the
        prefix's slots (tree path + its untaken note). Returns the req rows
        whose prefix is not held (they attend only their tokens)."""
        tree, lost, cont, first = self.tree, [], [], []
        self.cur, self._dok = torch.cuda.current_stream(self.dev), None
        self.cplan, self.cmarks, self.cwr, self.rsl = {}, [], {}, {}
        pre = {}
        for i, (row, ln, plen) in enumerate(zip(rows_in, eseq, plens)):
            if plen > 0:
                p = tree.pk_prefix(rstr[i], plen)
                if p is None:
                    lost.append(row)
                    continue
                pre[i] = p
            (cont if plen > 0 else first).append([i, 0, ln, plen, 0, False, row])
        rows = cont + first
        if not rows:
            return lost
        offs = np.r_[0, np.cumsum(eseq)].tolist()
        base = 0
        for r in rows:
            r[1] = offs[r[0]]
            if r[3] > 0:
                r[4] = base
                base += r[3] + r[2]
        # G (prefix rows only), then one attention-TP verdict: a failed
        # allocation on any rank fails the extend on every rank before a host
        # slot is taken or a tree note points at slots this extend would not
        # write
        ok = True
        if cont and (self.G is None or self.G.shape[1] < base):
            if base > self.rtok:
                self._r_warn("g", f"an extend of {base} staged tokens (prefixes + new) "
                                  f"exceeds the reserved {self.rtok}: G grows past the "
                                  f"memory budget for it")
            self._set_g(None)
            try:
                self._set_g(torch.empty((2, max(base, self.rtok), 2, self.H, self.D),
                                        dtype=self.dtype, device=self.dev))
            except RuntimeError:
                ok = False
        if self.sync is not None:
            ok = self.sync(ok)
        if not ok:
            raise _StageFailed(f"no device memory for {base} staged tokens")
        new = self._r_alloc(sum(r[2] for r in rows))
        # the host slots get one verdict too: a rank whose store is short
        # would note (and later insert) no prompt while the others do, and
        # the ranks' radix trees (their prefix matches) would diverge
        ok = new is not None
        if self.sync is not None:
            ok = self.sync(ok)
        if not ok:
            if new is not None:
                tree.slots.free(new)
            raise _StageFailed(f"no pinned host memory for {sum(r[2] for r in rows)} tokens")
        noted = []
        try:
            self._rplan(rows, rstr, pre, new, noted)
            err = None
        except Exception as e:  # device memory for the plan's index tensors
            err = e
        # and the plan: every rank keeps this extend's notes, or none does
        ok = err is None
        if self.sync is not None and not self.sync(ok) and ok:
            err = _StageFailed("on another attention-TP rank")
        if err is not None:
            # undo the notes (their slots are never written: a later insert
            # would put unwritten host K/V in the tree) and free the slots
            for rid, start, s in noted:
                tree.pk_unnote(rid, start, s)
            tree.slots.free(new[sum(len(x[2]) for x in noted):])
            raise err
        return lost

    def _rplan(self, rows, rstr, pre, new, noted):
        """rbegin past its allocations: note each row's new tokens' slots
        (appended to ``noted`` as they are taken; in batch order, so the D2H
        runs span rows), build the copy plans and start the first layers'
        prefix H2D."""
        tree = self.tree
        B, k, dst, src, ss, ts = self.BLK, 0, [], [], [], []
        hot = self.hot
        if hot is not None:
            self._hot_take(self._pages([new, *pre.values()], B))
        for r in sorted(rows, key=operator.itemgetter(0)):
            s = new[k:k + r[2]]
            k += r[2]
            tree.pk_note(rstr[r[0]], r[3], s)
            noted.append((rstr[r[0]], r[3], s))
            self.rsl[r[0]] = (pre[r[0]] if r[3] > 0 else None, s)
            ss.append(s)
            ts.append(np.arange(r[1], r[1] + r[2], dtype=np.int64))
            if r[3] > 0:
                src.append(pre[r[0]].astype(np.int64))
                dst.append(np.arange(r[4], r[4] + r[3], dtype=np.int64))
        h2d, d2d, dd, m2d, fill, mark = [], [], None, [], [], []
        if src:
            S, D = np.concatenate(src), np.concatenate(dst)
            if len(src) == 1:  # one path: distinct slots (no sort)
                one, Su, Du = D, S, D
            else:
                _, fi, inv = np.unique(S, return_index=True, return_inverse=True)
                one = D[fi][inv.reshape(-1)]  # each token's first G copy
                Su, Du = S[fi], D[fi]
            if hot:
                # a slot with a valid mirror copy: device to device; the rest
                # over PCIe, filling its page's mirror (valid once queued)
                pg, hv, hp = Su // B, np.zeros(len(Su), dtype=bool), np.zeros(len(Su), dtype=bool)
                for p in self._pages([Su], B):
                    h = hot.get(p)
                    if h is not None:
                        sel = pg == p
                        hp[sel] = True
                        hv[sel] = h[2][Su[sel] % B]
                m2d = [(hot[s // B][0], s % B, n, d) for s, d, n in self._runs(Su[hv], Du[hv], B)]
                f = hp & ~hv
                fr = self._runs(Su[f], Du[f], B)
                fill = [(hot[s // B][0], s % B, n, d) for s, d, n in fr]
                mark = [(s // B, s % B, n) for s, _, n in fr]
                Su, Du = Su[~hv], Du[~hv]
            h2d = [(s // B, s % B, n, d) for s, d, n in self._runs(Su, Du, B)]
            dup = np.flatnonzero(one != D)
            d2d = [(d, s, n) for s, d, n in self._runs(one[dup], D[dup], 1 << 62)]
            # many short runs: one gather + scatter per layer; long ones (rows
            # sharing a prefix) copy run by run (no temporary, half the traffic)
            if len(d2d) > 2 and len(dup) < 256 * len(d2d):
                dd = (self._idx(D[dup]), self._idx(one[dup]))
        S, T = np.concatenate(ss).astype(np.int64), np.concatenate(ts)
        d2h = [(s // B, s % B, n, t) for s, t, n in self._runs(S, T, B)]
        tb, wm = self.tb, []
        if hot:
            # the new tokens' mirror writes; their slots are invalid until
            # the writes are queued (they held an earlier token's K/V)
            for p, a, n, t in d2h:
                h = hot.get(p)
                if h is not None:
                    h[2][a:a + n] = False
                    h[6][a:a + n] = False
                    wm.append((h[0].data_ptr() + a * tb, t, n, h[0], a))
                    mark.append((p, a, n))
        self.rplan = (h2d, d2d, d2h, dd,
                      [(self.pool[p].data_ptr() + a * tb, t, n) for p, a, n, t in d2h],
                      m2d, fill, wm, [(p, hot[p], a, n) for p, a, n in mark])
        if _RADIX_STATS:
            c = lambda runs: sum(x[2] for x in runs)
            print(f"[picokv radix] rows {len(rows)} new {len(new)} m2d {c(m2d)} fill {c(fill)} "
                  f"h2d {c(h2d)} dup {c(d2d)} wm {sum(x[2] for x in wm)} "
                  f"mirrors {0 if hot is None else len(hot)}", flush=True)
        self.rows = [r for r in rows if r[3] > 0]
        self.wplan = self._wplan(self.rows)
        if self.rows:
            # G: the previous forward is done with it; the prefix slots: their
            # D2H (an earlier extend's, on ds) is done
            self.cs.wait_stream(self.cur)
            self.cs.wait_stream(self.ds)
            with torch.cuda.stream(self.cs):
                for l in range(min(2, self.L)):
                    self._h2d(l)
                    self.ev_r[l].record(self.cs)

    def code_plan(self, i, n, staged):
        """PICOKV_RADIX_CODES, at layer 0 once row i takes a compressed
        prepare over n prompt tokens (``staged``: over its staged prefix +
        new tokens, else its new tokens alone), in batch order: its plan, or
        None. The leading run of its global tokens whose codes a mirror
        holds (valid, or written by an earlier row of this extend: every
        layer prepares the rows in this order) is read; the rest is computed
        and goes to the mirrors of their pages (valid once the last layer
        queued it: cmarks). Decided where the prepare path is, so a plan's
        writes run on every layer."""
        x = self.rsl.get(i)
        sink, local, _, _, cmin = self.code
        if x is None or (x[0] is not None) != staged or n < cmin or n <= sink + local:
            return None
        sl = x[1] if x[0] is None else np.concatenate((x[0].astype(np.int64), x[1]))
        if len(sl) != n:
            return None
        hot, B = self.hot, self.BLK
        gsl = sl[sink:n - local]
        pg, o = gsl // B, gsl % B
        ok = np.zeros(len(gsl), dtype=bool)   # codes held
        has = np.zeros(len(gsl), dtype=bool)  # on a page whose mirror keeps codes
        e = np.full(len(gsl), -1, dtype=np.int64)  # _code_asm_kernel's token entries
        ptrs, keep = [], []
        for p in self._pages([gsl], B):
            h = hot.get(p)
            if h is None or h[4] is None:
                continue
            sel = pg == p
            has[sel] = True
            e[sel] = len(keep) * B + o[sel]
            ptrs += (h[4].data_ptr(), h[5].data_ptr())
            keep.append(h)
            v = h[6][o[sel]]
            w = self.cwr.get(p)
            ok[sel] = v if w is None else v | w[o[sel]]
        c = len(ok) if ok.all() else int(np.argmin(ok))
        j = np.arange(len(gsl), dtype=np.int64)
        has[:c] = False
        wr = self._runs(gsl[has], j[has], B)
        if c == 0 and not wr:
            return None
        for a0, _, k in wr:
            p, a = a0 // B, a0 % B
            w = self.cwr.get(p)
            if w is None:
                w = self.cwr[p] = np.zeros(B, dtype=bool)
            w[a:a + k] = True
            self.cmarks.append((p, hot[p], a, k))
        tens = lambda a0: hot[a0 // B][4:6]
        # keep: the mirror entries the table's addresses point into (their
        # code tensors stay alive while the plan is)
        plan = dict(c=c, g=len(gsl), blk=B, np=len(keep), keep=[x[4:6] for x in keep],
                    tab=self._idx(np.concatenate((np.array(ptrs, dtype=np.int64), e))),
                    reads=[(*tens(a0), a0 % B, k, jj) for a0, jj, k in self._runs(gsl[:c], j[:c], B)],
                    writes=[(*tens(a0), a0 % B, k, jj) for a0, jj, k in wr])
        self.cplan[i] = plan
        return plan

    def _idx(self, a):
        """int64 host array -> device index tensor (pinned, async on the
        current stream: layer 0 issues it before cs waits for that stream)."""
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.int64)).pin_memory().to(
            self.dev, non_blocking=True)

    def _wplan(self, rows):
        """layer()'s chunk writes of several staged rows as ONE index_copy_
        per layer: (G positions, extend token indices or None = the first n
        tokens in order, n)."""
        if len(rows) < 2:
            return None
        dst = np.concatenate([np.arange(r[4] + r[3], r[4] + r[3] + r[2], dtype=np.int64)
                              for r in rows])
        src = np.concatenate([np.arange(r[1], r[1] + r[2], dtype=np.int64) for r in rows])
        ident = bool((src == np.arange(len(src))).all())
        return self._idx(dst), None if ident else self._idx(src), len(src)

    def _take(self, e, ntok):
        need = -(-ntok // self.BLK) - len(e["blocks"])
        if need > len(self.free):
            self._grow(len(self.pool) + need - len(self.free))
        for _ in range(need):
            e["blocks"].append(self.free.pop())

    def _set_g(self, g):
        if self.G is not None:
            self.G.record_stream(self.cs)  # pending copies finish before reuse
        self.G = g

    def drop(self, rid):
        e = self.ent.pop(rid, None)
        if e is not None:
            self.free.extend(e["blocks"])
        if not self.ent and self.G is not None:
            self._set_g(None)

    def begin(self, rids, eseq, plens, orig):
        """Layer 0 of an extend: this forward's staged rows. A row continues
        iff its entry holds exactly its prefix; a row whose chunk is not its
        request's last (prefix + chunk < orig) stages the chunk. Returns the
        rids whose prefix is not staged (they attend only their chunk)."""
        lost, cont, first = [], [], []
        for i, (rid, ln, plen) in enumerate(zip(rids, eseq, plens)):
            e = self.ent.get(rid)
            if e is not None and e["n"] != plen:
                self.drop(rid)  # a new request on this req row, or a missed chunk
                e = None
            if plen > 0 and e is None:
                lost.append(rid)
                continue
            more = orig is not None and plen + ln < orig[i]
            if e is None and not more:
                continue
            if e is None:
                e = self.ent[rid] = {"n": 0, "blocks": []}
            (cont if plen > 0 else first).append([i, 0, ln, plen, 0, more, rid, e])
        if not cont and not first:
            return lost
        offs = [0] * (len(eseq) + 1)
        for i, ln in enumerate(eseq):
            offs[i + 1] = offs[i] + ln
        base = 0
        for r in cont + first:
            r[1], r[4] = offs[r[0]], base
            base += r[3] + r[2]
            if r[5]:
                self._take(r[7], r[3] + r[2])
        if self.G is None or self.G.shape[1] < base:
            self._set_g(None)
            self._set_g(torch.empty((2, max(base, self.ctx + self.chunk), 2, self.H, self.D),
                                    dtype=self.dtype, device=self.dev))
        self.rows = cont + first
        self.wplan = self._wplan(self.rows)
        cur = torch.cuda.current_stream(self.dev)
        self.cs.wait_stream(cur)  # G / blocks: the previous forward is done with them
        with torch.cuda.stream(self.cs):
            for l in range(min(2, self.L)):
                self._h2d(l)
                self.ev_r[l].record(self.cs)
        return lost

    def _h2d(self, l):
        g, B = self.G[l % 2], self.BLK
        if self.tree is not None:
            h2d, d2d, _, dd, _, m2d, fill = self.rplan[:7]
            for m, a, n, d in m2d:
                g[d:d + n].copy_(m[l, a:a + n], non_blocking=True)
            for p, a, n, d in h2d:
                g[d:d + n].copy_(self.pool[p][l, a:a + n], non_blocking=True)
            for m, a, n, d in fill:
                m[l, a:a + n].copy_(g[d:d + n], non_blocking=True)
            if dd is not None:
                g.index_copy_(0, dd[0], g.index_select(0, dd[1]))
            else:
                for d, s, n in d2d:
                    g[d:d + n].copy_(g[s:s + n])
            return
        for r in self.rows:
            plen, base, bl = r[3], r[4], r[7]["blocks"]
            for j in range(-(-plen // B)):
                a = j * B
                n = min(B, plen - a)
                g[base + a:base + a + n].copy_(self.pool[bl[j]][l, :n], non_blocking=True)

    def layer(self, l, k_, v_, kv):
        """Before layer l's attention: wait for its prefix, write the chunk's
        K/V after it. Returns layer l's buffer [cap, 2, KVH, D] (radix mode:
        stages the extend's tokens first; None when no row has a prefix)."""
        if self.tree is not None:
            self._d2h(l, k_, v_, kv)
            if not self.rows:
                return None
        torch.cuda.current_stream(self.dev).wait_event(self.ev_r[l])
        g = self.G[l % 2]
        wp = self.wplan
        if wp is not None and kv is not None:
            dst, src, n = wp
            g.index_copy_(0, dst, kv[:n] if src is None else kv.index_select(0, src))
            return g
        for r in self.rows:
            off, ln, base = r[1], r[2], r[4] + r[3]
            if kv is not None:
                g[base:base + ln].copy_(kv[off:off + ln])
            else:
                g[base:base + ln, 0].copy_(k_[off:off + ln])
                g[base:base + ln, 1].copy_(v_[off:off + ln])
        return g

    def _d2h(self, l, k_, v_, kv):
        """Radix mode, layer l: the extend's tokens' K/V to their host slots,
        on ds once the layer's K/V exist: one cudaMemcpy2DAsync per slot run
        straight from k_ / v_ (sglang's qkv views: token rows a pitch apart;
        kv = both as one [T, 2, KVH, D] view), else torch copies of the runs.
        The K/V buffers go back to the allocator after these copies
        (_d2h_done)."""
        global _CUDART
        cur = self.cur
        self.ev_u[l].record(cur)
        ds = self.ds
        ds.wait_event(self.ev_u[l])
        HD = self.H * self.D
        if kv is not None:
            src, row = ((kv, 0, self.tb),), (HD, self.D, 1)
        else:
            src, row = ((k_, 0, self.tb // 2), (v_, self.tb // 2, self.tb // 2)), (self.D, 1)
        cr, ok = _cudart(), self._dok
        if ok is None or ok[0] != (kv is not None):  # once per extend
            ok = self._dok = (kv is not None, bool(cr) and not any(
                x.dtype != self.dtype or x.stride()[1:] != row
                or (x.shape[0] > 1 and x.stride(0) * x.element_size() < w)
                for x, _, w in src))
        if not ok[1]:
            return self._d2h_torch(l, k_, v_, kv)
        fn, h, tb, lo = cr[0], ds.cuda_stream, self.tb, l * self.BLK * self.tb
        wm = self.rplan[7]
        for x, o, w in src:
            # (one token: its dim-0 stride is arbitrary; any pitch >= w)
            sp = max(x.stride(0) * x.element_size(), w)
            s0, d0 = x.data_ptr(), lo + o
            # host slots (kind 2: device to host), then the hot tier's
            # mirrors of them (kind 3: device to device)
            for runs, kind in ((self.rplan[4], 2), (wm, 3)):
                for run in runs:
                    d, t, n = run[0], run[1], run[2]
                    if fn(d + d0, tb, s0 + t * sp, sp, w, n, kind, h):
                        err = cr[1]()  # clear the runtime's last error
                        logger.warning(f"[PicoKVAttnBackend] radix host K/V: cudaMemcpy2DAsync "
                                       f"failed ({err}): torch copies")
                        _CUDART = False
                        self._dok = (kv is not None, False)
                        return self._d2h_torch(l, k_, v_, kv)
        self._d2h_done(l, [x for x, _, _ in src])

    def _d2h_torch(self, l, k_, v_, kv):
        """_d2h by torch copies (on ds; device temporaries stay on ds)."""
        with torch.cuda.stream(self.ds):
            for p, a, n, t in self.rplan[2]:
                x = kv[t:t + n] if kv is not None else torch.stack((k_[t:t + n], v_[t:t + n]), 1)
                self.pool[p][l, a:a + n].copy_(x, non_blocking=True)
            for _, t, n, m, a in self.rplan[7]:
                x = kv[t:t + n] if kv is not None else torch.stack((k_[t:t + n], v_[t:t + n]), 1)
                m[l, a:a + n].copy_(x, non_blocking=True)
        self._d2h_done(l, [kv] if kv is not None else [k_, v_])

    def _d2h_done(self, l, srcs):
        """Layer l's D2H is queued from ``srcs``: keep them until the next
        layer's D2H, where the compute stream waits for this one and lets them
        go (its allocator then reuses them in stream order, a layer of slack
        for the copy); the last layer's by record_stream. (record_stream on
        every layer holds each buffer until the GPU catches up with the host,
        which runs layers ahead: about one qkv buffer per layer.)"""
        if self._dprev is not None:
            self.cur.wait_event(self._dprev[0])
            self._dprev = None
        if l == self.L - 1:
            for x in srcs:
                x.record_stream(self.ds)
            return
        self.ev_d[l].record(self.ds)
        self._dprev = (self.ev_d[l], srcs)

    def after(self, l):
        """After layer l's attention: stage its chunks (D2H), fetch layer
        l + 2's prefixes into the same buffer; the last layer closes the
        forward (entry lengths; a request's last chunk frees its entry)."""
        if self.tree is not None:  # radix mode: the D2H ran at layer()
            if self.rows and l + 2 < self.L:
                self.ev_u[l].record(self.cur)
                self.cs.wait_event(self.ev_u[l])
                with torch.cuda.stream(self.cs):
                    self._h2d(l + 2)
                    self.ev_r[l + 2].record(self.cs)
            if l == self.L - 1:
                if self.rplan is not None:
                    # every layer's fills / mirror writes are queued: their
                    # slots are valid (copy streams in order before any read)
                    for p, h, a, n in self.rplan[8]:
                        if self.hot.get(p) is h:
                            h[2][a:a + n] = True
                    for p, h, a, n in self.cmarks:  # (and the codes written)
                        if self.hot.get(p) is h:
                            h[6][a:a + n] = True
                    if _RADIX_STATS and self.cplan:
                        c = sum(x["c"] for x in self.cplan.values())
                        w = sum(x["g"] - x["c"] for x in self.cplan.values())
                        print(f"[picokv radix] codes rows {len(self.cplan)} read {c} "
                              f"computed {w}", flush=True)
                self.rows, self.rplan, self.wplan = [], None, None
                self.cplan, self.cmarks, self.cwr, self.rsl = {}, [], {}, {}
                if self.G is not None and self.G.shape[1] > self.rtok:
                    self._set_g(None)
            return
        self.ev_u[l].record(torch.cuda.current_stream(self.dev))
        self.cs.wait_event(self.ev_u[l])
        g, B = self.G[l % 2], self.BLK
        with torch.cuda.stream(self.cs):
            for r in self.rows:
                if not r[5]:
                    continue
                t, end, base, bl = r[3], r[3] + r[2], r[4], r[7]["blocks"]
                while t < end:
                    j, a = divmod(t, B)
                    n = min(B - a, end - t)
                    self.pool[bl[j]][l, a:a + n].copy_(g[base + t:base + t + n], non_blocking=True)
                    t += n
            if l + 2 < self.L:
                self._h2d(l + 2)
                self.ev_r[l + 2].record(self.cs)
        if l == self.L - 1:
            rows, self.rows, self.wplan = self.rows, [], None
            for r in rows:
                r[7]["n"] = r[3] + r[2]
                if not r[5]:
                    self.drop(r[6])


class PicoKVRequestCache:
    """One request's per-layer PiCoKV cache (pure API; mirrors the HF adapter's
    prepare/decode logic without importing transformers)."""

    def __init__(self, config: StandaloneConfig, offline: Optional[dict] = None) -> None:
        self.config = config
        self._offline = offline  # dict of full [layers,heads,bits,dim] stacks or None
        self.layers: Dict[int, PreparedKVCache] = {}
        # Layers whose prompt <= sink+local have no global segment to compress:
        # keep them fully dense and serve decode with SDPA (same math).
        self._dense: Dict[int, tuple] = {}
        self.seq_lengths: Dict[int, int] = {}
        # Set per extend at layer 0: this continuation chunk merges incrementally
        # (append) instead of re-preparing from the chunk alone.
        self._append_mode = False
        self._vtab = {}  # PICOKV_PREALLOC append: value regroup tables by (T, keep, stride)

    def _prepare_kwargs(self, heads: int) -> dict:
        kwargs = self.config.prepare_kwargs()
        if self._offline is not None:
            kwargs["key_encoders"] = _slice_layer(self._offline["key_encoder"], self._layer, heads)
            kwargs["value_encoders"] = _slice_layer(self._offline["value_encoder"], self._layer, heads)
            kwargs["key_decoders"] = _slice_layer(self._offline["key_decoder"], self._layer, heads)
            kwargs["value_decoders"] = _slice_layer(self._offline["value_decoder"], self._layer, heads)
        return kwargs

    def prepare(self, layer_idx: int, key_bhsd: torch.Tensor, value_bhsd: torch.Tensor) -> None:
        layer_idx = int(layer_idx)
        total_len = int(key_bhsd.shape[-2])
        if total_len <= self.config.sink_size + self.config.local_size:
            self._dense[layer_idx] = (key_bhsd.contiguous(), value_bhsd.contiguous())
            self.seq_lengths[layer_idx] = total_len
            return
        self._layer = layer_idx
        heads = int(key_bhsd.shape[1])
        self.layers[layer_idx] = prepare_kv_cache(
            key_bhsd, value_bhsd, **self._prepare_kwargs(heads)
        )
        self.seq_lengths[layer_idx] = total_len

    # ------------------------------------------------------------------
    # incremental chunk-append (chunked prefill continuation, eas=0)
    # ------------------------------------------------------------------

    def can_append(self, prefix_len: int) -> bool:
        """Whether a continuation chunk (``prefix_len`` tokens already cached)
        can merge incrementally instead of re-preparing from the chunk alone.

        Requirements: the recorded layer-0 length matches the engine's prefix
        length (same request, no decode tokens since — also rules out recycled
        req-pool indices), and either the dense fallback (raw prefix retained,
        exact by construction) or the compressed path with offline projections
        (deterministic fixed cascade), eas=0 (kept = every global token; no
        cross-chunk error selection) and pristine sink+local windows.
        """
        if prefix_len <= 0 or int(self.seq_lengths.get(0, -1)) != prefix_len:
            return False
        if any(c.dense_key is None or c.packed_key is None for c in self.layers.values()) \
                or any(x is None for x in self._dense.values()):
            return False  # held by a live decode batch (lazy sync): not mergeable
        if 0 in self._dense:
            return True  # dense merge: raw prefix K/V is retained exactly
        if (
            blockk_fixed_cuda is None
            or _prepare_fixed_block32_cached is None
            or self._offline is None
            or float(self.config.eas) != 0.0
            or 0 not in self.layers
        ):
            return False
        window = int(self.config.sink_size) + int(self.config.local_size)
        return all(c.dense_key.shape[-2] == window for c in self.layers.values())

    def append(self, layer_idx: int, key_bhsd: torch.Tensor, value_bhsd: torch.Tensor) -> bool:
        """Merge one continuation chunk into this layer's cache.

        Exact equivalent of re-preparing over prefix+chunk: the outgoing window
        tokens ``[L_prev-local, L_new-local)`` leave the dense window and are
        quantized with the same fixed offline projections (per-token
        deterministic cascade — slice-independent), then packed onto the global
        segment. Key pack is per-token columns (concat); value pack is 64-token
        groups (rebuild the boundary group with carry extraction). Dense-fallback
        layers keep raw K/V and promote to compressed via a full prepare once
        the sequence outgrows sink+local. Returns False when this layer's state
        does not allow a merge (caller falls back to prepare)."""
        layer_idx = int(layer_idx)
        key_bhsd = key_bhsd.contiguous()
        value_bhsd = value_bhsd.contiguous()
        C = int(key_bhsd.shape[-2])
        if C <= 0:
            return True
        cfg = self.config
        sink, local = int(cfg.sink_size), int(cfg.local_size)

        if layer_idx in self._dense:
            dk, dv = self._dense[layer_idx]
            dk = torch.cat((dk, key_bhsd), dim=-2)
            dv = torch.cat((dv, value_bhsd), dim=-2)
            total = int(dk.shape[-2])
            if total <= sink + local:
                self._dense[layer_idx] = (dk.contiguous(), dv.contiguous())
                self.seq_lengths[layer_idx] = total
                return True
            del self._dense[layer_idx]  # exact: full raw sequence available
            self.prepare(layer_idx, dk, dv)
            return True

        cache = self.layers.get(layer_idx)
        if cache is None or cache.dense_key.shape[-2] != sink + local:
            return False
        heads = int(key_bhsd.shape[1])
        dim = int(key_bhsd.shape[-1])
        device, dtype = key_bhsd.device, key_bhsd.dtype
        kb = int(cfg.key_bits)
        vb = int(cfg.value_bits)

        # Outgoing tokens [L_prev-local, L_new-local): old dense local tail
        # followed by the chunk, minus the C tokens that stay in the new window.
        tail_k = cache.dense_key[:, :, sink:]
        tail_v = cache.dense_value[:, :, sink:]
        out_k = torch.cat((tail_k, key_bhsd), dim=-2)[:, :, :C]
        out_v = torch.cat((tail_v, value_bhsd), dim=-2)[:, :, :C]

        # Quantize + pack with the same fixed projections prepare used (same
        # kernel, window at 0 — per-token results are slice-independent).
        enc_k = _prepare_fixed_block32_cached(
            _slice_layer(self._offline["key_encoder"], layer_idx, heads), 1, heads, kb, dim, device, dtype
        ).view(1, heads, kb, dim)
        dec_k = _prepare_fixed_block32_cached(
            _slice_layer(self._offline["key_decoder"], layer_idx, heads), 1, heads, kb, dim, device, dtype
        ).view(1, heads, kb, dim)
        enc_v = _prepare_fixed_block32_cached(
            _slice_layer(self._offline["value_encoder"], layer_idx, heads), 1, heads, vb, dim, device, dtype
        ).view(1, heads, vb, dim)
        dec_v = _prepare_fixed_block32_cached(
            _slice_layer(self._offline["value_decoder"], layer_idx, heads), 1, heads, vb, dim, device, dtype
        ).view(1, heads, vb, dim)
        packed_key_new, _ = blockk_fixed_cuda(
            out_k, enc_k, dec_k, 0, C,
            zerorope=int(cfg.zerorope), apply_zerorope=True, pack_key_layout=True,
        )
        packed_value_tok, _ = blockk_fixed_cuda(
            out_v, enc_v, dec_v, 0, C,
            zerorope=0, apply_zerorope=False, pack_key_layout=False,
        )

        # PICOKV_PREALLOC: a full-capacity buffer (the request's whole prompt
        # + decode headroom, _prealloc_layer) takes the chunk in place: no
        # copy regrown per chunk and layer (allocator fragmentation), same codes.
        gl_old = int(cache.packed_key.shape[-1])
        kf, vf = _pk_full(cache.packed_key), _pk_full(cache.packed_value)
        if (kf is None or vf is None or kf.shape[-1] < gl_old + C
                or kf.data_ptr() != cache.packed_key.data_ptr()
                or vf.data_ptr() != cache.packed_value.data_ptr()):
            kf = vf = None
        # Key pack [1,H,KB,gl]: per-token columns — plain concat.
        if kf is not None:
            kf[..., gl_old:gl_old + C] = packed_key_new
            cache.packed_key = kf[..., :gl_old + C]
        else:
            cache.packed_key = torch.cat((cache.packed_key, packed_key_new), dim=-1)

        # Value pack [1,H,groups,Cv]: 64-token groups. Rebuild from the last
        # partial group: extract its per-token words (carry), prepend them to
        # the new token words, re-pack whole groups.
        VB = vb // 64
        rem = gl_old % 64
        words = packed_value_tok  # [1,H,C,VB] per-token words
        if rem:
            gw = cache.packed_value[:, :, -1]  # [1,H,Cv]: bit i = token i's code bit c
            shifts_i = torch.arange(rem, device=device, dtype=torch.int64)
            bits = (gw.view(1, heads, VB, 64).unsqueeze(-1) >> shifts_i) & 1  # [1,H,VB,64,rem]
            shifts_s = torch.arange(64, device=device, dtype=torch.int64)
            carry = (bits << shifts_s.view(1, 1, 1, 64, 1)).sum(dim=3)  # [1,H,VB,rem]
            words = torch.cat((carry.permute(0, 1, 3, 2).contiguous(), words), dim=2)
        T = int(words.shape[2])
        G_tail = (T + 63) // 64
        keep = int(cache.packed_value.shape[2]) - (1 if rem else 0)
        if kf is not None and _prep_vgroup_kernel is not None:
            # the groups straight into the buffer (== the regroup below,
            # without its [1,H,T,Cv] int64 bit tensors: ~170 MB per 8k chunk)
            hs = int(vf.shape[2]) * int(vf.shape[3])
            key = (T, keep, hs)
            tab = self._vtab.get(key)
            if tab is None:
                if len(self._vtab) >= 4:
                    self._vtab.pop(next(iter(self._vtab)))
                tab = self._vtab[key] = torch.tensor(
                    [(64 * q, min(64, T - 64 * q), (keep + q) * int(vf.shape[3]), hs)
                     for q in range(G_tail)], dtype=torch.int64, pin_memory=True).to(
                    device, non_blocking=True)
            _prep_vgroup_kernel[(G_tail, VB, heads)](words.contiguous(), vf, tab, T, VW=VB)
            cache.packed_value = vf[:, :, :keep + G_tail]
        else:
            s64 = torch.arange(64, device=device, dtype=torch.int64)
            cb = ((words.unsqueeze(-1) >> s64) & 1).view(1, heads, T, vb)  # code c = j*64+s
            Tp = G_tail * 64
            if Tp != T:
                cb = torch.nn.functional.pad(cb, (0, 0, 0, Tp - T))  # missing tokens: zero codes
            cb = cb.view(1, heads, G_tail, 64, vb)
            groups = (cb << s64.view(1, 1, 1, 64, 1)).sum(dim=3)  # [1,H,G_tail,Cv]
            if kf is not None:
                vf[:, :, keep:keep + G_tail] = groups
                cache.packed_value = vf[:, :, :keep + G_tail]
            else:
                cache.packed_value = torch.cat((cache.packed_value[:, :, :keep], groups), dim=2)

        # Dense window slides by exactly C tokens: sink + last `local` of
        # [old local ++ chunk].
        comb_k = torch.cat((tail_k, key_bhsd), dim=-2)
        comb_v = torch.cat((tail_v, value_bhsd), dim=-2)
        cache.dense_key = torch.cat((cache.dense_key[:, :, :sink], comb_k[:, :, -local:]), dim=-2)
        cache.dense_value = torch.cat((cache.dense_value[:, :, :sink], comb_v[:, :, -local:]), dim=-2)
        self.seq_lengths[layer_idx] += C
        return True

    def decode(self, layer_idx: int, q_bhsd, k_bhsd, v_bhsd) -> torch.Tensor:
        layer_idx = int(layer_idx)
        if layer_idx in self._dense:
            dk, dv = self._dense[layer_idx]
            dk = torch.cat((dk, k_bhsd.contiguous()), dim=-2)
            dv = torch.cat((dv, v_bhsd.contiguous()), dim=-2)
            self._dense[layer_idx] = (dk, dv)
            out = torch.nn.functional.scaled_dot_product_attention(
                q_bhsd, dk, dv, enable_gqa=True
            )
            self.seq_lengths[layer_idx] += int(k_bhsd.shape[-2])
            return out
        cache = self.layers[layer_idx]
        # Backend dedup leaves ragged packed tensors as non-contiguous views
        # into a decode batch's copy; the per-request kernels need them dense.
        for f in ("packed_key", "packed_value"):
            if not getattr(cache, f).is_contiguous():
                setattr(cache, f, getattr(cache, f).contiguous())
        out = decode_attention(
            q_bhsd, cache, decode_key=k_bhsd, decode_value=v_bhsd,
            update_cache=True, **self.config.decode_kwargs(),
        )
        self.seq_lengths[layer_idx] += int(k_bhsd.shape[-2])
        return out

    def cache_bytes(self) -> int:
        seen, total = set(), 0

        def add(o):
            nonlocal total
            if torch.is_tensor(o):
                o = _pk_full(o) if _pk_full(o) is not None else o  # the whole buffer
                if o.is_cuda and o.data_ptr() not in seen:
                    seen.add(o.data_ptr())
                    total += o.numel() * o.element_size()
            elif isinstance(o, (list, tuple)):
                [add(x) for x in o]
            elif isinstance(o, dict):
                [add(x) for x in o.values()]
            elif hasattr(o, "__dict__"):
                [add(x) for x in vars(o).values()]

        add(self.layers)
        add(self._dense)
        return total


class _FlatMixin:
    """Shared flattened<->[1,H,S,D] helpers for sglang's per-token layout."""

    def _to_bhsd(self, x: torch.Tensor, heads: int) -> torch.Tensor:
        s = x.shape[0]
        return x.view(s, heads, self.head_dim).transpose(0, 1).unsqueeze(0).contiguous()

    def _from_bhsd(self, o: torch.Tensor) -> torch.Tensor:
        s = o.shape[2]
        return o.squeeze(0).transpose(0, 1).reshape(s, self.num_q_heads * self.head_dim)


class PicoKVCore(_FlatMixin):
    """Single-request (batch=1) PiCoKV engine in sglang's flattened layout."""

    def __init__(
        self,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        scaling: float,
        config: Optional[StandaloneConfig] = None,
        offline_proj_path: Optional[str] = None,
    ) -> None:
        if num_q_heads // num_kv_heads != 4:
            raise ValueError(f"PiCoKV needs GQA=4; got QH={num_q_heads}, KVH={num_kv_heads}")
        if head_dim != 128:
            raise ValueError(f"PiCoKV needs head_dim==128; got {head_dim}")
        self.num_q_heads, self.num_kv_heads, self.head_dim = num_q_heads, num_kv_heads, head_dim
        self.scaling = float(scaling)
        self.config = config or StandaloneConfig()
        self._offline = load_offline_projections_pure(offline_proj_path) if offline_proj_path else None
        self.req = PicoKVRequestCache(self.config, self._offline)

    def reset(self) -> None:
        self.req = PicoKVRequestCache(self.config, self._offline)

    def extend(self, layer_id, q, k, v) -> torch.Tensor:
        q4 = self._to_bhsd(q, self.num_q_heads)
        k4 = self._to_bhsd(k, self.num_kv_heads)
        v4 = self._to_bhsd(v, self.num_kv_heads)
        self.req.prepare(layer_id, k4, v4)
        o = torch.nn.functional.scaled_dot_product_attention(
            q4, k4, v4, is_causal=True, enable_gqa=True, scale=self.scaling
        )
        return self._from_bhsd(o)

    def decode(self, layer_id, q, k, v) -> torch.Tensor:
        q4 = self._to_bhsd(q, self.num_q_heads)
        k4 = self._to_bhsd(k, self.num_kv_heads)
        v4 = self._to_bhsd(v, self.num_kv_heads)
        return self._from_bhsd(self.req.decode(layer_id, q4, k4, v4))


class PicoKVBatchDecoder(_FlatMixin):
    """Step 3: continuous-batch decode over requests at INDEPENDENT context
    lengths. Each request keeps its own ``PicoKVRequestCache``; a batched decode
    step dispatches per request over the verified single-request kernel and
    stacks the results, so a ragged batch is correct. A fused ragged/paged
    kernel (one launch for the whole batch) is the remaining optimization.
    """

    def __init__(self, num_q_heads, num_kv_heads, head_dim, scaling,
                 config=None, offline_proj_path=None):
        self.num_q_heads, self.num_kv_heads, self.head_dim = num_q_heads, num_kv_heads, head_dim
        self.scaling = float(scaling)
        self.config = config or StandaloneConfig()
        self._offline = load_offline_projections_pure(offline_proj_path) if offline_proj_path else None
        self.reqs: Dict[int, PicoKVRequestCache] = {}

    def add_request(self, req_id: int) -> None:
        self.reqs[req_id] = PicoKVRequestCache(self.config, self._offline)

    def remove_request(self, req_id: int) -> None:
        self.reqs.pop(req_id, None)

    def prepare(self, req_id: int, layer_id: int, k_bhsd, v_bhsd) -> None:
        self.reqs[req_id].prepare(layer_id, k_bhsd, v_bhsd)

    def decode_batch(self, layer_id: int, req_ids: List[int], q_bhsd, k_bhsd, v_bhsd):
        """q/k/v: [num_reqs, heads, 1, D] aligned with ``req_ids``. Returns
        [num_reqs, QH, 1, D] — each row decoded against that request's cache."""
        outs = []
        for i, rid in enumerate(req_ids):
            outs.append(self.reqs[rid].decode(
                layer_id, q_bhsd[i:i + 1], k_bhsd[i:i + 1], v_bhsd[i:i + 1]))
        return torch.cat(outs, dim=0)

    def total_bytes(self) -> int:
        return sum(r.cache_bytes() for r in self.reqs.values())


# --- sglang glue: only defined when sglang is importable --------------------
try:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

    _SGLANG_AVAILABLE = True
except Exception:  # pragma: no cover - lets the core be imported standalone
    AttentionBackend = object  # type: ignore
    _SGLANG_AVAILABLE = False

try:
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
        eager_on_graph,
        is_in_breakable_cuda_graph,
    )
except Exception:  # pragma: no cover - standalone import / older sglang

    def eager_on_graph(enable, capture_stub=None):
        return lambda fn: fn

    def is_in_breakable_cuda_graph():
        return False

try:  # composed BCG replay (sglang patch): per-layer graphs as child nodes
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
        defer_flush as _bcg_flush,
        defer_graph as _bcg_defer,
        deferring as _bcg_deferring,
        enable_compose as _bcg_enable_compose,
        compose_stats as _bcg_stats,
    )
    _decode_break = eager_on_graph(True, composable=True)
except Exception:  # pragma: no cover - unpatched sglang
    _bcg_enable_compose = None
    _bcg_stats = None
    _decode_break = eager_on_graph(True)

    def _bcg_flush():
        pass

    def _bcg_defer(graph):
        return False

    def _bcg_deferring():
        return False


# FA3's post-kernel tile-semaphore reset: at::native fill of 1 int (the
# vectorized kernel's (int N, FillFunctor<int>, array<char*, 1>) signature)
_FA3_FILL_RE = re.compile(
    r"vectorized_elementwise_kernelILi[124]ENS0_11FillFunctorIiEESt5arrayIPcLm1EEEEviT0_T1_$")


def _fa3_fill_offpath(raw, sem_ptr):
    """A clone of captured graph ``raw`` (cudaGraph_t) in which the nodes
    behind FA3's semaphore reset — the 1-int zero fill of the scheduler
    metadata's tile counter at ``sem_ptr``, read only by the next FA3 call —
    depend on the FA3 kernel itself instead of the fill, so the reset leaves
    the critical path (it still completes with the graph, before the next
    layer's FA3). None when the graph has no such fill."""
    import ctypes

    from cuda.bindings import driver as dr
    from cuda.bindings import runtime as rt

    def ok(r):
        if int(r[0]) != 0:
            raise RuntimeError(repr(r[0]))
        return r[1:] if len(r) > 2 else r[1]

    def kname(n):
        if ok(rt.cudaGraphNodeGetType(n)) != rt.cudaGraphNodeType.cudaGraphNodeTypeKernel:
            return ""
        try:
            p = ok(dr.cuGraphKernelNodeGetParams(dr.CUgraphNode(int(n))))
            return ok(dr.cuFuncGetName(p.func)).decode()
        except RuntimeError:  # e.g. a library kernel without a function handle
            return ""

    clone = ok(rt.cudaGraphClone(raw))
    moved = 0
    try:
        nodes = ok(rt.cudaGraphGetNodes(clone, ok(rt.cudaGraphGetNodes(clone, 0))[1]))[0]
        names = {int(n): kname(n) for n in nodes}
        # one FA3 call per graph: no later FA3 in it could start before the reset
        if sum("FlashAttnFwd" in v for v in names.values()) != 1:
            nodes = []
        fr, to, ed, _ = ok(rt.cudaGraphGetEdges(clone, ok(rt.cudaGraphGetEdges(clone, 0))[-1]))
        edges = list(zip(fr, to, ed))
        for n in nodes:
            if not _FA3_FILL_RE.search(names[int(n)]):
                continue
            a = (ctypes.c_void_p * 3).from_address(
                int(ok(dr.cuGraphKernelNodeGetParams(dr.CUgraphNode(int(n)))).kernelParams))
            if (ctypes.c_int.from_address(a[0]).value != 1
                    or ctypes.c_int.from_address(a[1]).value != 0
                    or ctypes.c_void_p.from_address(a[2]).value != sem_ptr):
                continue
            pre = [(x, e) for x, y, e in edges if int(y) == int(n)]
            suc = [(y, e) for x, y, e in edges if int(x) == int(n)]
            if (len(pre) != 1 or pre[0][1].type != 0 or "FlashAttnFwd" not in names[int(pre[0][0])]
                    or any(e.type != 0 for _, e in suc)):
                continue
            for y, e in suc:
                ok((rt.cudaGraphRemoveDependencies(clone, [n], [y], [e], 1)[0], None))
                ok((rt.cudaGraphAddDependencies(clone, [pre[0][0]], [y], [e], 1)[0], None))
            moved += 1
    except BaseException:
        rt.cudaGraphDestroy(clone)
        raise
    if moved:
        return clone
    rt.cudaGraphDestroy(clone)
    return None


def _graph_destroy(raw):
    from cuda.bindings import runtime as rt

    rt.cudaGraphDestroy(raw)


class PicoKVAttnBackend(AttentionBackend):
    """SGLang attention backend backed by per-request PiCoKV caches.

    Supports a batch of requests at INDEPENDENT context lengths (continuous
    batching): each request slot (``forward_batch.req_pool_indices``) owns a
    :class:`PicoKVRequestCache`. ``forward_extend`` splits the concatenated
    extend tokens by ``extend_seq_lens`` and compresses each request's prompt;
    ``forward_decode`` decodes one token per running request against its own
    compressed cache. K/V are NOT written to the stock dense pool — PiCoKV keeps
    only packed codes + sink/local window.
    """

    def __init__(self, model_runner):
        super().__init__()
        import os
        from picokv.picokv_standalone import StandaloneConfig

        self.device = model_runner.device
        cfg = model_runner.model_config
        tp, tp_rank = self._attn_tp(model_runner)
        self.num_q_heads = cfg.num_attention_heads // tp
        self.num_kv_heads = cfg.get_num_kv_heads(tp)
        self.head_dim = cfg.head_dim
        self.scaling = 1.0 / (self.head_dim ** 0.5)
        self._check_model(cfg, model_runner)
        # the decode tail needs the extension: load it here, unguarded, so a
        # missing / wrong-arch build fails at startup with its path, not as a
        # silently disabled feature or a "no kernel image" mid-forward
        from picokv.picokv_cuda_extension.picokv_cuda import load_picokv_extension
        load_picokv_extension()
        # PICOKV_EXP_BITS: attention-weight bitplanes in the tail kernel
        # (values stay packed codes; only decode-side weight quantization
        # changes). 4 = default; 2 saves ~13% of the main scan (quality-gated
        # by NIAH before flipping). PICOKV_EAS: dense retention of the
        # highest-error global tokens (StandaloneConfig default 0.0; the HF
        # runner's default is 0.01). Disables incremental chunk-append when > 0.
        self._scfg = StandaloneConfig(
            exp_bits=int(os.environ.get("PICOKV_EXP_BITS", "4")),
            eas=float(os.environ.get("PICOKV_EAS", "0.0")))
        # Offline projections are the DEFAULT serving path: prefill then runs
        # the fused blockk prepare (one kernel per side) instead of per-request
        # online direction learning. Resolution order: PICOKV_OFFLINE_PROJ
        # (explicit path) > the conventional shipped .pt for this model >
        # online. PICOKV_ONLINE=1 forces the online path (A/B / fallback).
        path = os.environ.get("PICOKV_OFFLINE_PROJ") or None
        if path is None and os.environ.get("PICOKV_ONLINE", "0") != "1":
            path = self._default_offline_proj(cfg, self._scfg)  # raises if none
        self._offline = load_offline_projections_pure(path) if path else None
        if self._offline is not None:
            total = getattr(cfg, "get_total_num_kv_heads", lambda: None)()
            validate_offline_projections(
                self._offline, path, int(getattr(cfg, "num_hidden_layers", 0) or
                                         self._offline["key_encoder"].shape[0]),
                total, self.head_dim, self._scfg)
            self._offline = self._shard_heads(
                self._offline, total, self.num_kv_heads, tp, tp_rank)
            logger.info("picokv: offline projections %s", path)
        else:
            logger.warning(
                "picokv: ONLINE projections (PICOKV_ONLINE=1): per-request direction "
                "learning at prefill, decode slide off (decode tokens stay dense)")
        self._reqs = {}  # req_pool_index -> PicoKVRequestCache (ragged fallback)
        # Finished requests are pruned from _reqs once their req-pool row is
        # free again (see _prune_finished); without it a cache lived until its
        # row was reused by a later request.
        self._mr = model_runner
        # PICOKV_KV_DEDUP=1 (default): after a layer's batched copy is built,
        # each request's packed/projection tensors become views into it, so
        # decode holds the compressed cache ONCE instead of per-request + batched
        # (b16@120k OOM'd at 78.6 GB with 15 requests; ~4 GB/request vs 2.46).
        self._dedup = os.environ.get("PICOKV_KV_DEDUP", "1") == "1"
        self._batch = None       # live decode batch: rids + per-layer fixed tensors
        self._pools = None       # stashed pools from the last decode batch (carry-over)
        self._gl0_proj = {}      # per-layer projections for all-dense (gl=0) batches
        # Decode slide (P1-4; PICOKV_SLIDE=0 disables): every 64 decode steps
        # each row's dense tokens past sink+local leave the window and are
        # appended to its compressed global segment — the state prepare would
        # build over the whole sequence — so the window stays <= sink+local+64
        # and decode cost/memory stop growing with the output. The outgoing
        # 64 slots of one layer side are quantized per step (spread over the
        # epoch), then merged + shifted at the boundary. Needs the offline
        # projections (fixed cascade), eas=0 and the unpruned tail; otherwise
        # decode tokens stay dense (pre-P1-4 behavior).
        sc = self._scfg
        why = picokv_slide_off_reason(sc)  # the pool sizing reads the same predicate
        if why is None and self._offline is None:
            why = "online projections"
        # PICOKV_SLIDE_LOCAL: dense tail kept during decode (default = the
        # prefill local_size). A longer tail keeps more of the generated text
        # exact (long CoT) at +4 KB/token/layer per row of window memory.
        self._slide_local = int(os.environ.get("PICOKV_SLIDE_LOCAL", sc.local_size))
        self._slide_W = int(sc.sink_size) + self._slide_local  # max window entering an epoch
        self._dense_max = int(sc.sink_size) + int(sc.local_size)  # prompt stays dense (gl = 0)
        # PICOKV_DENSE_LOCAL (opt-in speed knob, default 0 = off): slide
        # local of a small decode batch (<= PICOKV_DENSE_LOCAL_MAX_BS rows).
        # At that size attending a long dense window costs less than
        # compressing generated tokens into the tail (and an all-short batch
        # keeps the dense-only graph), but how much of a request is
        # compressed then depends on its batch size; off, every batch slides
        # at PICOKV_SLIDE_LOCAL (_batch_W), the method's sink + local.
        # +4 KB per token per layer per row of window.
        self._dense_local = int(os.environ.get("PICOKV_DENSE_LOCAL", "0")) // 32 * 32
        if self._dense_local <= self._slide_local:
            self._dense_local = 0
        self._dense_bs = int(os.environ.get("PICOKV_DENSE_LOCAL_MAX_BS", "8"))
        self._slide = why is None
        # packed-global headroom (tokens) per row before a layer's packed copy
        # regrows (realloc + that layer's graph recapture)
        self._slide_head = max(0, int(os.environ.get("PICOKV_SLIDE_HEADROOM", "1024")))
        self._slide_proj = {}    # layer -> (enc_k, dec_k, enc_v, dec_v)
        # Incremental batch build (slide only; PICOKV_FAST_BUILD=0 = full
        # rebuild): rows carried across a batch change are gathered on device
        # from the previous batch's tensors, request caches sync lazily (only
        # rows leaving a batch), the 64-step epoch clock carries over (no
        # catch-up for carried rows) and every row shares one per-layer
        # replicated projection copy (slide => offline => identical tensors).
        self._fast = self._slide and os.environ.get("PICOKV_FAST_BUILD", "1") == "1"
        # PICOKV_FAST_BUILD2 (fast build; =0: the per-layer path): a bucket
        # change's per-build work once instead of per layer (_carry_plan),
        # kept rows' storage handed over in numpy, the carried windows
        # gathered straight into unzeroed buffers — the same batch state
        # (_fixed_layer_fast2), ~3x less host time per build.
        self._fast_build2 = self._fast and os.environ.get("PICOKV_FAST_BUILD2", "1") == "1"
        # Slot mode (BCG replay on the plan-free window path; fast build;
        # PICOKV_SLOTS=0 disables): the decode batch owns the bucket's S rows
        # (live rows first in engine order, vacant rows after: gl=0, one
        # dense token, frozen write position), so a membership change within
        # the bucket compacts/fills rows IN PLACE and keeps every tensor
        # address, table and captured per-layer graph (_slot_change) instead
        # of a rebuild + 3 eager warmup steps + 32 recaptures (~29 ms per
        # request at MATH r24). A bucket change still rebuilds.
        self._slots = self._fast and os.environ.get("PICOKV_SLOTS", "1") == "1"
        # Slot memory guard: a slot batch holds S rows at its packed width
        # (vacant rows; the width of rows that left) — ~20 KB per token-row
        # over 32 layers, GBs at long context. A build / membership change
        # whose packed bytes would exceed the unslotted batch's by more than
        # this runs unslotted / rebuilds (which re-fits the width).
        self._slot_waste = int(float(os.environ.get("PICOKV_SLOTS_MAX_WASTE_MB", "1024")) * (1 << 20))
        self._n_layers = int(getattr(cfg, "num_hidden_layers", 0) or 32)
        self._proj_rep = {}      # layer -> replicated (kp, vp, kpt) [Bcap, ...]
        self._proj_ok = set()    # layers whose request projections matched the shared ones
        logger.info(f"[PicoKVAttnBackend] decode slide "
                    f"{f'ON (64-step epochs, local {self._slide_local}' + (f'; batch <= {self._dense_bs}: local {self._dense_local})' if self._dense_local else ')') if self._slide else 'OFF (' + why + ')'}")
        self._cont_warned = False  # chunked-prefill continuation note printed once
        self.n_lost = self.n_lost_tokens = 0  # extends that lost their prefix (_count_lost)
        # PICOKV_LOST=abort (default): a request whose prompt prefix went
        # unattended (stage / radix host K/V failure) is aborted with a 503
        # at its last prefill chunk (picokv_lost_abort, called by the
        # scheduler) instead of decoding from its new tokens alone (keep)
        self._lost_abort = os.environ.get("PICOKV_LOST", "abort").strip().lower() == "abort"
        self._lost_rids = {}  # request id -> largest unseen prefix
        # flashinfer wrappers: {"wrapper":..., "planned":...} or False=disabled.
        # _fi = paged decode (dense window); _fi_pre = ragged prefill.
        self._fi = None

        # Custom dense-window kernel. PICOKV_WINDOW_KERNEL=auto (default) uses
        # it for batches of <= PICOKV_WINDOW_MAX_BS (8) requests and the
        # flashinfer split-kv tensor-core kernel above that; 1 = always,
        # 0 = never. BCG ITL @8k prompt (ms/step, custom vs fi): b1 7.16/7.41,
        # b4 8.21/8.86, b8 9.80/9.84, b16 12.46/12.37. The custom path also
        # skips the per-step fi wrapper.plan and keeps one graph signature as
        # the window grows. The choice is fixed per batch (sh["win"]).
        self._win = None  # None = not requested; False = off/failed; else fn
        _wmode = os.environ.get("PICOKV_WINDOW_KERNEL", "auto").strip().lower()
        self._win_max_bs = (1 << 30) if _wmode == "1" else int(
            os.environ.get("PICOKV_WINDOW_MAX_BS", "8"))
        if _wmode != "0":
            try:
                from picokv.picokv_cuda_extension.picokv_cuda import (
                    load_picokv_extension,
                )
                self._win = getattr(load_picokv_extension(), "picokv_window_attn_fn")
            except Exception as e:
                logger.warning(f"[PicoKVAttnBackend] window kernel unavailable ({e!r})")
                self._win = False
        if self._win:
            logger.info("[PicoKVAttnBackend] window kernel ON for batch <= "
                        f"{self._win_max_bs} (custom dense-window path, no fi plan)")
        # FA3 dense window (PICOKV_DENSE_FA3=1 default on sm90; 0 = the
        # custom-kernel / flashinfer split above): flash_attn_with_kvcache
        # reads the token-major pool [B,cap,KVH,D] and the device lens (lpl)
        # directly, so like the custom kernel it needs no fi plan, no paged
        # tables and keeps one graph signature per (B, cap); it is the
        # fastest of the three at every B measured (t39/fa3_bench.py, us per
        # layer, lens ~320: B8 9.5 vs fi 11.0 / custom 13.3; B24 13.6 vs fi
        # 20.3 / custom 27.8; B1@30k 51 vs custom 226) and returns ln-LSE.
        self._fa3 = None
        self._fa3_meta_fn = None
        if os.environ.get("PICOKV_DENSE_FA3", "1") == "1":
            try:
                if torch.cuda.get_device_capability()[0] != 9:
                    raise RuntimeError("FA3 needs sm90")
                from sgl_kernel.flash_attn import flash_attn_with_kvcache
                self._fa3 = flash_attn_with_kvcache
                # PICOKV_FA3_META=1 (default): FA3's tile-scheduler metadata
                # once per step (_fa3_sched) instead of a prepare_varlen
                # launch in every layer's call (bitwise the same output,
                # t57/mb/fa3_meta_test.py)
                self._fa3_meta_fn = None
                if os.environ.get("PICOKV_FA3_META", "1") == "1":
                    try:
                        from sgl_kernel.flash_attn import get_scheduler_metadata
                        self._fa3_meta_fn = get_scheduler_metadata
                    except ImportError:
                        pass
                self._win = self._win or True  # sh["win"]: the plan-free path
                self._win_max_bs = 1 << 30
                logger.info("[PicoKVAttnBackend] dense window: FA3 flash_attn_with_kvcache "
                            "(every batch size, no fi plan)")
            except Exception as e:
                logger.warning(f"[PicoKVAttnBackend] FA3 dense window unavailable ({e!r})")
        # Shared projections (fast build: every row's projections ARE the one
        # offline copy per layer): the batch carries key_projection_t /
        # value_projection as stride-0 broadcasts instead of B materialized
        # replicas (~2 MB per row per layer), and decode runs Triton query /
        # value projections around the global-only tail op, reading each
        # head's matrix once. Needs an extension with picokv_decode_tail_global_fn;
        # PICOKV_SHARED_PROJ=0 keeps the replicas + cuBLAS GEMMs.
        self._shared_proj = False
        from picokv.picokv_standalone.decode import shared_tail_supported
        # pruned tail has no global-only split: keep replicas
        self._shared_proj = shared_tail_supported() and not self._scfg.use_pruning
        if (self._fast and not self._scfg.use_pruning
                and os.environ.get("PICOKV_SHARED_PROJ", "1") != "0"):
            _require_op(self._shared_proj, "picokv_decode_tail_global_fn", "PICOKV_SHARED_PROJ=0")
        logger.info(f"[PicoKVAttnBackend] shared projections: {'on' if self._shared_proj else 'off'}")
        # Per-row packed storage (PICOKV_ROWS=0 keeps the rectangle): a decode
        # batch of B > 1 rows keeps each row's packed global codes at its own
        # capacity (_RowStore) instead of a [B, H, KW, L] rectangle padded to
        # the longest row; carried rows keep their storage across batches and
        # slot changes, a regrow reallocates only the overflowing row. Same
        # codes, same tensor-core tail kernel (reading row-address tables), so
        # the outputs are those of the rectangle. Needs the fast build, the
        # shared-projection tail and the tc kernel (KW 5, CV 320, GQA-4).
        self._rows = False
        if (os.environ.get("PICOKV_ROWS", "1") == "1" and self._fast and self._shared_proj
                and _slide_kmerge_rows_kernel is not None
                and picokv_tc_tail_on()
                and int(self._scfg.key_bits) == 320 and int(self._scfg.value_bits) == 320
                and self.num_q_heads == 4 * self.num_kv_heads):
            from picokv.picokv_cuda_extension.picokv_cuda import load_picokv_extension
            self._rows = hasattr(load_picokv_extension(), "picokv_decode_tail_global_rows_fn")
            # default-on and the memory bound needs it: an old build without
            # the op fails here, not as a silent rectangle (B x longest row)
            _require_op(self._rows, "picokv_decode_tail_global_rows_fn", "PICOKV_ROWS=0")
        self._row_stub = {}
        logger.info(f"[PicoKVAttnBackend] per-row packed storage: {'on' if self._rows else 'off'}")
        # PICOKV_DET=1: deterministic decode tail. The global tail kernel's
        # float atomics sum the 512-token blocks in a run-dependent order, so
        # greedy outputs can flip between identical runs; with DET each block
        # stores its partial and a fixed-order Triton reduction sums them
        # (decode.DETERMINISTIC). Needs the shared tail and the tc kernel.
        from picokv.picokv_standalone import decode as _pk_decode
        self._det = os.environ.get("PICOKV_DET", "0") == "1"
        if self._det:
            try:
                from picokv.picokv_cuda_extension.picokv_cuda import load_picokv_extension
                ok = hasattr(load_picokv_extension(), "picokv_decode_tail_global_part_fn")
            except Exception:
                ok = False
            if not (ok and self._shared_proj and picokv_tc_tail_on()
                    and int(self._scfg.key_bits) == 320 and int(self._scfg.value_bits) == 320
                    and self.num_q_heads == 4 * self.num_kv_heads):
                raise ValueError("PICOKV_DET=1 needs an extension with picokv_decode_tail_global_part_fn, "
                                 "the shared-projection tail and the tensor-core tail kernel")
        _pk_decode.DETERMINISTIC = self._det
        logger.info(f"[PicoKVAttnBackend] deterministic tail: {'on' if self._det else 'off'}")
        # PICOKV_PDL (decode.pdl_enabled): per-layer graphs launch the dense
        # append and the value projection as programmatic dependents
        self._pdl = _pk_decode.pdl_enabled() and _gdc_wait is not None
        self._pdl_warm = set()  # append ROW widths whose PDL variant is compiled
        logger.info(f"[PicoKVAttnBackend] programmatic dependent launch: {'on' if self._pdl else 'off'}")
        # PICOKV_WIN_TRITON (default on; never with PICOKV_DET, which keeps
        # FA3 and its order): the dense window as Triton kernels
        # (picokv_standalone/window_attn.py) instead of FA3, on batches up to
        # PICOKV_WIN_TRITON_MAX_BS rows and pools up to 2048 tokens (one
        # program per (row, KV head): no split over the window). The forked
        # dense branch launches the LSE alone (all the tail reads) and the
        # output from it under the tail; the serial path the one-pass kernel.
        # PICOKV_WIN_SPLIT_MAX_BS: forked batches up to this many rows split
        # (larger ones: the one-pass kernel, joined before the tail);
        # PICOKV_WIN_TRITON_SERIAL=1: the serial path too (default FA3 there:
        # in-server 512x1 the one-pass kernel, PDL after the append, measured
        # 6.59 ms/token vs FA3 6.52), PICOKV_WIN_TRITON_PDL: there a
        # programmatic dependent of the append
        self._tri_win = None
        self._tri_win_bs = int(os.environ.get("PICOKV_WIN_TRITON_MAX_BS", "16"))
        self._tri_split_bs = int(os.environ.get("PICOKV_WIN_SPLIT_MAX_BS", "16"))
        self._tri_serial = os.environ.get("PICOKV_WIN_TRITON_SERIAL", "0") == "1"
        self._tri_pdl = os.environ.get("PICOKV_WIN_TRITON_PDL", "1") == "1"
        if (self._fa3 is not None and not self._det
                and os.environ.get("PICOKV_WIN_TRITON", "1") == "1"
                and self.head_dim & (self.head_dim - 1) == 0
                and self.num_q_heads % self.num_kv_heads == 0):
            try:
                from picokv.picokv_standalone import window_attn as _wa
                self._tri_win = _wa
            except Exception as e:
                logger.warning(f"[PicoKVAttnBackend] Triton window unavailable ({e!r})")
        logger.info(f"[PicoKVAttnBackend] Triton dense window: "
                    f"{f'B<={self._tri_win_bs}' if self._tri_win is not None else 'off'}")
        # PICOKV_COMPOSE: under the breakable decode graph, a step whose every
        # layer replays an existing per-layer graph runs the engine's segments
        # and those graphs as ONE composed CUDA graph (child nodes in order:
        # the per-layer graph launches and their boundaries go away); the
        # layers' eager step work (step begin, slide merge / quantize) only
        # touches picokv state and is enqueued before it. Same kernels, same
        # order of dependent work: bitwise.
        self._compose = (os.environ.get("PICOKV_COMPOSE", "1") == "1"
                         and _bcg_enable_compose is not None and _bcg_enable_compose())
        logger.info(f"[PicoKVAttnBackend] composed decode graph: {'on' if self._compose else 'off'}")
        # PICOKV_COMPOSE_STATS=N: print the replay outcome counts every N steps
        self._cstats = int(os.environ.get("PICOKV_COMPOSE_STATS", "0")) if self._compose else 0
        self._cstats_n = 0
        # PICOKV_FILL_OFFPATH (composed replay only): compose each per-layer
        # graph as a variant whose FA3 semaphore reset runs beside the tail
        # instead of before it (_fa3_fill_offpath)
        self._fill_offpath = self._compose and os.environ.get("PICOKV_FILL_OFFPATH", "1") == "1"
        self._n_offpath = 0  # graphs captured with an off-path variant
        self._fi_pre = None
        self._prefill = os.environ.get("PICOKV_PREFILL", "fi").strip().lower()
        # C1 exact chunked prefill (_ChunkStage): on when SGLang's chunked
        # prefill can split a prompt (0 < chunked_prefill_size < context_len)
        # and the ragged prefill is on. PICOKV_CHUNK_STAGE=off: continuation
        # chunks attend only themselves. PICOKV_STAGE_BLOCK: pool block
        # (tokens); PICOKV_STAGE_PREALLOC=0: pin host blocks on first use.
        self._stg = None
        self._stg_chunk = self._chunk_size()
        ctx = int(getattr(cfg, "context_len", 0) or 0)
        # PICOKV_RADIX=host: the stage is also the radix tree's host K/V
        # store (attached at the first extend: the tree is built after the
        # backend), whatever the chunk size.
        self._radix = picokv_radix_mode() and self._prefill == "fi"
        if self._radix:
            try:  # registers the "picokv" radix backend in this process
                import sglang.srt.mem_cache.picokv_radix_cache  # noqa: F401
            except Exception as e:
                logger.warning(f"[PicoKVAttnBackend] picokv radix cache unavailable ({e!r})")
                self._radix = False
        self._own = {}  # radix mode: req row -> request id of its cache
        if (_stage_mode() == "host" and self._prefill == "fi"
                and (self._radix or (self._stg_chunk > 0
                                     and (ctx == 0 or self._stg_chunk < ctx)))):
            dt = getattr(cfg, "dtype", None)
            self._stg = _ChunkStage(
                self._n_layers, self.num_kv_heads, self.head_dim,
                dt if isinstance(dt, torch.dtype) else torch.bfloat16, self.device,
                ctx or 131072, self._stg_chunk if self._stg_chunk > 0 else (ctx or 131072),
                int(os.environ.get("PICOKV_STAGE_BLOCK", "8192")),
                os.environ.get("PICOKV_STAGE_PREALLOC", "1") == "1")
            if self._radix and os.environ.get("PICOKV_STAGE_PREALLOC", "1") == "1":
                # radix: pin the whole host store now (~0.45 s per 1 GiB page),
                # not page by page under the first requests' prefills
                self._stg._grow(self._stg.cap_pages(_radix_host_gb(tp)))
            logger.info(f"[PicoKVAttnBackend] exact chunked prefill: host stage ON (chunk "
                        f"{self._stg_chunk}, {len(self._stg.pool)} x {self._stg.BLK}-token pinned "
                        f"blocks preallocated{'; radix host K/V' if self._radix else ''})")
        if self._stg is not None:
            self._stg.sync = self._tp_sync(model_runner)
        self._tp = tp
        self._stg_warned = False
        # CUDA-graph decode (item 1): PICOKV_GRAPHS=0 disables; the first
        # PICOKV_GRAPH_WARMUP steps of each batch run eager (cublas/JIT warmup)
        # before per-layer graphs are captured.
        self._graphs_on = os.environ.get("PICOKV_GRAPHS", "1") != "0"
        self._graph_rearm = None  # monotonic time graphs come back after failures
        self._graph_warmup = max(0, int(os.environ.get("PICOKV_GRAPH_WARMUP", "3")))
        # Eager decode reads req ids (a device->host sync) once per step, at
        # the step's first layer, instead of at every layer call (-1.4 to
        # -2.4 ms/step measured). PICOKV_RIDS_PER_STEP=0 restores per-layer
        # reads. The breakable engine graph reads them in replay-prep instead.
        self._rids_per_step = os.environ.get("PICOKV_RIDS_PER_STEP", "1") == "1"
        # Host req ids come from the scheduler's CPU mirror
        # (forward_batch.req_pool_indices_cpu) when present: no device->host
        # sync, so the overlap scheduler keeps the GPU queue full across
        # steps. Spot-checked against the device copy on every batch change
        # and every 256 reads; a mismatch disables the mirror for good.
        # PICOKV_RIDS_CPU=0 always reads the device copy.
        self._rids_cpu = os.environ.get("PICOKV_RIDS_CPU", "1") == "1"
        self._rids_n = 0
        # PICOKV_EXT_FAST=1: forward_extend reads the batch's req ids / extend
        # lens once per forward (layer 0) from the scheduler's CPU mirrors
        # instead of two device->host syncs per layer, and an all-short extend
        # (every prompt <= sink+local) stores its dense K/V as views into ONE
        # per-layer copy instead of two copies per request. 0 = old path (A/B).
        self._ext_fast = os.environ.get("PICOKV_EXT_FAST", "1") == "1"
        self._ext_m = None  # (id(forward_batch), rids, eseq, plens) of this forward
        self._ext_n = 0
        # PICOKV_FAST_PREPARE=1 (offline + eas=0): a compressed prompt layer
        # calls the two fused blockk launches and the gather directly, with the
        # memoized offline projections (_slide_projs / _gl0_projections, the
        # tensors prepare_kv_cache itself resolves to), a cached kept-index map
        # and one shared key_projection_t, instead of prepare_kv_cache's generic
        # path (validation, ~16 .contiguous(), per-call arange, stream context
        # managers, a fresh key_projection_t copy per request layer). Same
        # kernels on the same inputs: bitwise the same cache. K and V are also
        # copied to [1,KVH,L,D] in one launch when they are adjacent views of
        # the qkv projection. 0 = old path (A/B).
        self._fprep = None
        sc = self._scfg
        if (os.environ.get("PICOKV_FAST_PREPARE", "1") == "1" and self._offline is not None
                and float(sc.eas) == 0.0 and bool(sc.use_fused_quant)
                and os.environ.get("PICOKV_DISABLE_FUSED_QUANT", "0") != "1"
                and blockk_fixed_cuda is not None and _prepare_fixed_block32_cached is not None):
            try:
                from picokv.picokv_prefill_cuda_extension.picokv_prefill_cuda import (
                    load_picokv_prefill_extension,
                )
                from picokv.picokv_standalone.prepare import (
                    _side_stream, _validate_code_bits, _validate_exp_bits,
                )
                _validate_code_bits("key_bits", sc.key_bits)
                _validate_code_bits("value_bits", sc.value_bits)
                _validate_exp_bits(sc.exp_bits)
                ext = load_picokv_prefill_extension(False)
                self._fprep = dict(
                    blockk=ext.picokv_prefill_blockk_fixed_fn,
                    gather=ext.picokv_prefill_gather_pack_from_packed_fn,
                    # same, dense window into caller-provided tensors (older
                    # builds: None -> small groups stay batched)
                    gather_out=getattr(
                        ext, "picokv_prefill_gather_pack_from_packed_out_fn", None),
                    # PICOKV_PREP_FUSED: a prompt below the overlap gate (its V
                    # and K launches serialized anyway) prepares in ONE call of
                    # picokv_prefill_prepare_fast_fn (K/V split + both launches +
                    # value regroup; bitwise, t73/test_fused73.py); older
                    # builds: None -> the separate calls
                    fused=(getattr(ext, "picokv_prefill_prepare_fast_fn", None)
                           if os.environ.get("PICOKV_PREP_FUSED", "1") == "1" else None),
                    side_of=_side_stream, side=None, ev=None,
                    # V on the shared prepare side stream from this length up;
                    # shorter prompts prefill host-bound, where the event pair
                    # + stream switch (~18 us/layer) costs more than the
                    # overlap saves (1 x 384-768: -0.3..-1.5 ms TTFT serial,
                    # 1 x 896-1536: +0.4..+0.7 ms)
                    ovl=int(os.environ.get("PICOKV_PREP_OVERLAP_MIN", "832")),
                    # same for a batched quantize group (its prompts' tokens):
                    # the group's extra host work moves the crossover up
                    # (2 x 512 / 3 x 384: -2.5 / -2.7 ms serial; >= 1280: +0.6)
                    ovl_g=int(os.environ.get("PICOKV_PREP_OVERLAP_MIN_GROUP", "1200")),
                    kept={}, eas={},
                    # PICOKV_PREP_BATCH: an extend with >= 2 compressed prompts
                    # of global length <= PICOKV_PREP_BATCH_GL quantizes them in
                    # ONE blockk launch per side (groups of <= ..._TOKENS global
                    # tokens) instead of one small latency-bound launch each.
                    batch=(os.environ.get("PICOKV_PREP_BATCH", "1") == "1"
                           and _prep_vgroup_kernel is not None),
                    bgl=int(os.environ.get("PICOKV_PREP_BATCH_GL", "4096")),
                    btok=int(os.environ.get("PICOKV_PREP_BATCH_TOKENS", "65536")),
                    # a group of fewer than bmin prompts and at most bmin_tok
                    # prompt tokens packs each prompt on the per-request path:
                    # ~3 ms less host per forward, ~1-2 ms more GPU (TTFT
                    # 2 x 384: -2.9 ms with 50 ms idle before each; 2 x 352 / 384
                    # back to back: -2.5; 2 x 448 / 512: -1.1..-1.9 idle but
                    # +1.0..+1.2 back to back, GPU-bound: hence 768 tokens)
                    bmin=max(2, int(os.environ.get("PICOKV_PREP_BATCH_MIN", "3"))),
                    bmin_tok=int(os.environ.get("PICOKV_PREP_BATCH_MIN_TOKENS", "768")),
                    # PICOKV_FILL_TAB: _fill_fresh's launch table in one call
                    # (older builds: None -> the Python loop)
                    ftab=getattr(ext, "picokv_fill_table_fn", None),
                    plan=None, splan=None)  # splan: staged rows' plan (from G)
            except Exception as e:  # pragma: no cover - old path stays
                logger.info(f"[PicoKVAttnBackend] fast prepare off ({type(e).__name__}: {e})")
                self._fprep = None
        self._kv_adj = None  # K/V adjacency of the qkv views (checked per call)
        self._kvm = None  # (k_, v_, _kv_tok) of the layer in forward_extend (radix)
        self._dplan = None  # (segments, device index) of _dense_staged
        # PICOKV_FILL_KERNEL: a build / slot change writes its fresh rows (packed
        # codes + dense window, every layer) with _fill_rows_kernel, one launch
        # instead of four copies per row and layer. 0 = the row-wise copies.
        self._fill_k = (os.environ.get("PICOKV_FILL_KERNEL", "1") == "1"
                        and _fill_rows_kernel is not None)
        # PICOKV_FILL_TAB=1 (default): that launch's table (17 int64 per job)
        # built by one extension call (picokv_fill_table_fn: the same checks
        # and entries in C++, pinned + non_blocking H2D as here) instead of
        # ~8 us of Python per job (32 jobs per fresh row); a job it does not
        # take returns None and the Python loop decides as before. Same
        # table, same launch. 0 (or a build without it) = the Python loop.
        self._fill_tab = (self._fprep.get("ftab") if self._fill_k and self._fprep is not None
                          and os.environ.get("PICOKV_FILL_TAB", "1") == "1" else None)
        # per-row storage: fresh-row bytes of one slot-change fill launch
        self._fill_group = int(float(os.environ.get("PICOKV_FILL_GROUP_MB", "512")) * (1 << 20))
        # PICOKV_SLOT_MEMO=1 (default): a slot change's host bookkeeping
        # without per-layer recomputation — fill-launch layout checks kept with
        # the batch (_fill_fresh memo); the row write positions / lens / live
        # mask are rows of one device buffer, rewritten in one H2D. 0 = as
        # before (recomputed every change, three H2D copies).
        self._slot_memo = os.environ.get("PICOKV_SLOT_MEMO", "1") == "1"
        # PICOKV_SLOT_FAST=1 (default): a slot change of a batch whose layers
        # all keep packed rectangles (bucket 1, or PICOKV_ROWS=0) and whose
        # fresh rows need no catch-up builds the per-row lists every layer
        # shares (eas=0: one gl / window len / capacity per row at every
        # layer) once and copies them per layer (_slot_change_fast): the same
        # checks, copies, fill launch and table writes. 0 = layer by layer.
        self._slot_fast = os.environ.get("PICOKV_SLOT_FAST", "1") == "1"
        # PICOKV_SLOT_COMPOSE=1 (opt-in): a slot change stays inside the
        # composed BCG replay (no flush): it reads and writes only PiCoKV state
        # (row stores, request caches, lens / schedule tables), never this
        # step's q/k/v or another segment's buffers, so running its work before
        # the step's composed launch instead of after segment 0 changes no
        # value. A rebuild, capture or eager layer still flushes. Off by
        # default: the first step after a slot change then sees a new graph set
        # and instantiates a composed exec (~3 ms on that step, +14 MiB), so the
        # first request per bucket gets slower TTFT. 0 (default) = flush first.
        self._slot_compose = os.environ.get("PICOKV_SLOT_COMPOSE", "0") == "1"
        # PICOKV_PREALLOC=1 (default; dedup + fast prepare): a compressed
        # prompt layer's packed codes live in ONE buffer at the request's
        # decode-row capacity (whole prompt + headroom) from its first chunk
        # on: later chunks append in place instead of regrowing a copy per
        # chunk and layer, and the decode batch adopts the buffer as the row's
        # storage (no copy, no free). Without it the caching allocator splits
        # into GBs of holes no request fits (default allocator: 120k x 24 OOM).
        # Only in extends of at most PICOKV_PREALLOC_MAX_TOKENS tokens (chunked
        # prefill): in a larger one the buffers' headroom is held while its
        # activations peak (one 3 x 8000-token extend: +0.6 GB reserved), and
        # the previous path builds the row capacity after the extend.
        self._prealloc = (os.environ.get("PICOKV_PREALLOC", "1") == "1" and self._dedup
                          and self._fprep is not None)
        self._prealloc_max = int(os.environ.get("PICOKV_PREALLOC_MAX_TOKENS", "16384"))
        self._fin = {}  # this extend (layer 0): req row -> full prompt length
        # PICOKV_KV_POOL: the full-capacity buffers come from their own caching
        # allocator pool. Allocated mid-forward, they would otherwise carve the
        # blocks the extend's activations just freed, and the remainders no
        # later activation or buffer fits pile up (~25% of every long prompt's
        # buffers at 120k). use_on_oom: the pool's cached free blocks serve
        # other allocations as a last resort.
        self._kvpool = None
        if self._prealloc and os.environ.get("PICOKV_KV_POOL", "1") == "1":
            try:
                self._kvpool = torch.cuda.MemPool(use_on_oom=True)
            except Exception:  # no MemPool (older torch): the default pool
                self._kvpool = None
        self._fused_append = os.environ.get("PICOKV_FUSED_APPEND", "1") == "1"
        # sglang breakable decode graph state (see init_forward_metadata_out_graph)
        self._eng_capture = False  # BCG warmup/capture on dummy requests
        self._eng_rids = None      # real req ids of the step being replayed
        self._eng_bs = 0           # padded bucket size of that step
        # BCG replay: the per-layer graph writes the attention output straight
        # into the break's captured output buffer (recorded per (layer, bucket)
        # at capture), so the break's output copy is a self-copy no-op.
        # PICOKV_ENG_DIRECT_OUT=0 restores the copy.
        self._eng_direct_out = os.environ.get("PICOKV_ENG_DIRECT_OUT", "1") == "1"
        self._eng_outs = {}
        # PICOKV_VIEW_MEMO=1 (default): BCG replay calls every layer's break
        # with the same captured q / k / v tensor objects (and the same output
        # buffer) each step, so their row slices / head views and the graph's
        # address key are kept per (layer, bucket) and reused while those
        # objects are the ones passed in (strong refs: an id is never reused
        # while kept) instead of being rebuilt per layer per step (~12 us of
        # host time each). Same views, same addresses: nothing on the GPU
        # changes. 0 = rebuild every call, as before.
        self._view_memo = os.environ.get("PICOKV_VIEW_MEMO", "1") == "1"
        self._eng_views = {}  # (layer, bucket) -> (q, k, v, n, (q_, k_, v_))
        self._eng_lgv = {}    # (layer, B) -> (q_, k_, v_, out, B, contiguous, unit stride, ov, ptrs)
        # Per-layer graph: the dense-window branch (K/V append + window or fi
        # attention) runs on a forked stream, overlapped with the query
        # projection/quantize/pack; the global kernel joins both.
        # PICOKV_GRAPH_FORK=auto (default) forks off the FA3 dense path, and
        # with FA3 only on tail grids above the fused query quantizer's gate
        # (decode fork_large, PICOKV_TAIL_QUANT_BLOCKS) at B <= 8 or >= 48
        # (PICOKV_FORK_B): elsewhere the serial order is faster (t41/lprof.py,
        # GPU ms per 32-layer step incl. a weight-GEMM filler: B1 6.58 vs
        # 6.69, B8 6.43 vs 6.69, B24 7.19 vs 7.51; t68 longgen B8 +0.8%
        # forked, bench_itl 8192x16 / x32 +0.5%) — the two-stream graph's
        # launch + join latency exceeds the ~5 us of qproj it hides. Where it
        # forks, the projection + prep op (B160: 13 us per layer) hide under
        # append + FA3 (t68 r160 step p50 10.77 -> 10.61 ms at B 129-160,
        # 122880x1 8.19 -> 8.10). 1 forces the fork, 0 the serial order.
        _gf = os.environ.get("PICOKV_GRAPH_FORK", "auto").strip().lower()
        self._graph_fork = _gf == "1" or (_gf not in ("0", "1") and self._fa3 is None)
        self._fork_large = _gf not in ("0", "1") and self._fa3 is not None
        # BCG replay reads q in place as a strided view of the qkv projection
        # when every kernel that reads it takes strides (FA3 dense window +
        # shared-projection Triton qproj) instead of staging a dense copy per
        # layer in the graph. PICOKV_Q_DIRECT=0 restores the staging copy.
        self._q_direct = os.environ.get("PICOKV_Q_DIRECT", "1") == "1"
        # A layer whose batch has no compressed token (gl = 0 in every row:
        # short prompts, nothing slid yet) captures only append + dense-window
        # attention: the tail would merge in exactly zero, so the output is
        # bitwise the same without qproj / pack / tail / vproj.
        # PICOKV_DENSE_ONLY=0 always runs the full sequence.
        self._dense_only = os.environ.get("PICOKV_DENSE_ONLY", "1") == "1"
        logger.info(f"[PicoKVAttnBackend] per-layer graph: "
                    f"{'forked' if self._graph_fork else 'serial'} dense branch"
                    f"{' (forked on large tail grids)' if self._fork_large else ''}, "
                    f"q direct {'on' if self._q_direct else 'off'}")
        self._fork_stream = None
        self._cap_stream = None    # per-layer graph capture stream
        self._graph_anchor = None  # keeps the shared graph pool alive
        self._graph_pool = None
        self._n_extend = 0
        self._n_decode = 0
        # PICOKV_STEPPROF=1: host wall time of every engine step (layer-0 call
        # to the next layer-0 call), bucketed by what the step did — prefill /
        # decode batch build / eager warmup / graph capture / replay / eager —
        # plus host time inside the batch-build and capture code; printed
        # every PICOKV_STEPPROF_EVERY (2048) steps. Attributes the batch-churn cost.
        self._sp = ({"t0": None, "kind": None, "f": set(), "acc": {}, "h": {}, "n": 0,
                     "every": max(1, int(os.environ.get("PICOKV_STEPPROF_EVERY", "2048")))}
                    if os.environ.get("PICOKV_STEPPROF", "0") == "1" else None)
        # per-layer decode timing (debug; eager decode only: it synchronizes)
        self._decprof = os.environ.get("PICOKV_DECPROF", "0") not in ("", "0")
        self._decprof_acc = {"prep": 0.0, "kern": 0.0, "n": 0}
        # PICOKV_WARM_INIT (default on; fast build 2): a CUDA kernel's first
        # launch loads its module (lazy loading) — index_select's <= 16-row
        # variant ~49 ms, the larger one ~5 ms, inside the first bucket
        # change that gathers that many carried windows. Launched here once
        # on tiny tensors of the window buffers' layout; nothing kept.
        if self._fast_build2 and os.environ.get("PICOKV_WARM_INIT", "1") == "1":
            self._warm_init(getattr(model_runner, "dtype", None) or torch.bfloat16)
        logger.info(f"[PicoKVAttnBackend] ACTIVE (multi-request) qh={self.num_q_heads} "
                    f"kvh={self.num_kv_heads} hd={self.head_dim} "
                    f"mode={'offline (' + path + ')' if self._offline else 'online'} "
                    f"prefill={self._prefill} graphs={'on' if self._graphs_on else 'off'}")

    def _warm_init(self, dtype):
        """PICOKV_WARM_INIT: the bucket-change window gather's kernels
        (_fixed_layer_fast2: index_select along rows of [B, cap, H, D]
        buffers, int64 rows; <= 16 and > 16 rows are separate kernels)."""
        H, D = self.num_kv_heads, self.head_dim
        src = torch.zeros(40, 2, H, D, device=self.device, dtype=dtype)
        out = torch.empty_like(src)
        for n in (8, 32):
            idx = torch.arange(n, device=self.device, dtype=torch.int64)
            torch.index_select(src, 0, idx, out=out[:n])
        torch.cuda.synchronize()

    @staticmethod
    def _attn_tp(model_runner):
        """(attention TP size, rank). sglang keeps them in the parallel context
        (ModelRunner has no tp_size); unit-test runners set tp_size/tp_rank."""
        if hasattr(model_runner, "tp_size"):
            return int(model_runner.tp_size), int(getattr(model_runner, "tp_rank", 0))
        from sglang.srt.runtime_context import get_parallel

        ps = get_parallel()
        return int(ps.attn_tp_size), int(ps.attn_tp_rank)

    def _check_model(self, cfg, model_runner):
        """Reject shapes the PiCoKV kernels are not built for (instead of
        failing deep in a kernel or decoding garbage)."""
        if self.num_q_heads != 4 * self.num_kv_heads:
            raise ValueError(f"PiCoKV needs GQA=4 per rank; got QH={self.num_q_heads}, "
                             f"KVH={self.num_kv_heads}")
        vd = getattr(cfg, "v_head_dim", self.head_dim)
        if self.head_dim != 128 or vd != 128:
            raise ValueError(f"PiCoKV needs head_dim == v_head_dim == 128; got "
                             f"{self.head_dim}/{vd}")
        dt = getattr(cfg, "dtype", torch.bfloat16)
        if dt != torch.bfloat16:
            raise ValueError(f"PiCoKV serving is built and validated for bfloat16; got {dt}")
        # Per-layer attention features the kernels do not implement (they use
        # 1/sqrt(head_dim), no window / softcap, causal decoder self-attention):
        # reject instead of silently decoding different attention.
        from sglang.srt.layers.radix_attention import AttentionType, RadixAttention

        model = getattr(model_runner, "model", None)
        bad = {}
        for m in (model.modules() if model is not None else ()):
            if not isinstance(m, RadixAttention):
                continue
            why = []
            if abs(float(m.scaling) - self.scaling) > 1e-6 * self.scaling:
                why.append(f"scaling={m.scaling:g} (kernels use head_dim**-0.5)")
            if (m.sliding_window_size or -1) > 0:
                why.append(f"sliding_window_size={m.sliding_window_size}")
            if m.logit_cap and m.logit_cap > 0:
                why.append(f"logit_cap={m.logit_cap}")
            if m.is_cross_attention:
                why.append("cross-attention")
            if m.attn_type != AttentionType.DECODER:
                why.append(f"attn_type={m.attn_type.value}")
            if getattr(m, "use_irope", False):
                why.append("irope")
            if getattr(m, "xai_temperature_len", -1) > 0:
                why.append("attention temperature")
            for w in why:
                bad.setdefault(w, []).append(m.layer_id)
        if bad:
            raise ValueError("PiCoKV does not support this model's attention: " + "; ".join(
                f"{w} (layers {v[:4]}{'...' if len(v) > 4 else ''})" for w, v in bad.items()))

    @staticmethod
    def _shard_heads(off, total, heads, tp, rank):
        """Keep this rank's KV heads of the [layers, total_kv_heads, bits, dim]
        projection stacks, matching QKVParallelLinear's split: contiguous
        blocks of total/tp heads, or (tp > total) rank // (tp/total) replicated."""
        if total is None:
            if off["key_encoder"].shape[1] != heads:
                raise ValueError(f"projection has {off['key_encoder'].shape[1]} KV heads; "
                                 f"this rank serves {heads} (model total unknown)")
            return off
        if off["key_encoder"].shape[1] != total:
            raise ValueError(f"projection has {off['key_encoder'].shape[1]} KV heads; "
                             f"model has {total}")
        if total == heads:
            return off
        start = (rank // max(1, tp // total)) * heads
        out = dict(off)
        for k in ("key_encoder", "value_encoder", "key_decoder", "value_decoder"):
            out[k] = off[k][:, start:start + heads].contiguous()
        return out

    @staticmethod
    def _default_offline_proj(cfg, scfg) -> Optional[str]:
        """Resolve the shipped offline projection .pt for the served model.

        Conventional location: ``<PiCoKV repo>/offline_projection/projections/
        {model_dir_name}_k{key_bits}_v{value_bits}.pt`` (produced by
        ``offline_projection/create_offline_projections.py``). Raises when the
        model has no shipped projection: the online path is a different
        (slower, slide-off) serving mode, opted into with PICOKV_ONLINE=1.
        """
        from pathlib import Path

        import picokv

        model_path = getattr(cfg, "model_path", None)
        proj_dir = (Path(picokv.__file__).resolve().parent.parent
                    / "offline_projection" / "projections")
        tag = Path(str(model_path or "").rstrip("/")).name
        cand = proj_dir / f"{tag}_k{scfg.key_bits}_v{scfg.value_bits}.pt"
        if model_path and cand.is_file():
            return str(cand)
        raise ValueError(
            f"picokv: no offline projection for {model_path!r} (looked for {cand}); "
            "set PICOKV_OFFLINE_PROJ=<file.pt> (offline_projection/"
            "create_offline_projections.py builds one) or PICOKV_ONLINE=1 for "
            "per-request online projections")

    def init_forward_metadata(self, forward_batch):
        # eager forward (no engine graph): forward_decode reads req ids itself
        self._eng_capture = False
        self._eng_rids = None

    # ------------------------------------------------------------------
    # engine-level CUDA graph hooks. Only the BREAKABLE decode graph is
    # supported (cuda_graph_backend_decode="breakable"): forward_decode is an
    # eager break (eager_on_graph), so replay runs the graphed embedding /
    # proj / MLP segments and calls this backend's Python once per layer —
    # the per-request cache state and the per-layer graphs keep working. A
    # whole-step ("full") graph would skip that Python entirely (capture runs
    # before any prefill, so replay would never run PiCoKV attention) and is
    # rejected at capture.
    # ------------------------------------------------------------------

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        pass  # no engine-graph buffers: the break owns its inputs/outputs

    def init_forward_metadata_out_graph(self, forward_batch, in_capture: bool = False):
        if in_capture:
            if not is_in_breakable_cuda_graph():
                raise NotImplementedError(
                    "picokv backend supports only the breakable decode CUDA "
                    "graph (cuda_graph_backend_decode='breakable'); a full "
                    "graph replay would skip the host-side per-request cache "
                    "state. Use breakable, or disable_cuda_graph=True.")
            # warmup + capture run on dummy requests: forward_decode returns
            # zeros (the break output only needs its shape/address recorded)
            self._eng_capture = True
            self._eng_rids = None
            return
        # replay-prep (outside the graph): the step's only read of the real
        # (unpadded) req ids; every layer's break reuses it.
        if torch.cuda.is_current_stream_capturing():
            # the metadata glue graph (SGLANG_ENABLE_METADATA_GLUE_GRAPH)
            # would freeze this host state at capture: refuse so it falls back
            raise RuntimeError("picokv replay-prep sets host state; not capturable")
        self._eng_capture = False
        raw = forward_batch.batch_size - forward_batch.num_padding
        self._eng_rids = self._host_rids(forward_batch, raw)
        self._eng_bs = forward_batch.batch_size

    def init_forward_metadata_in_graph(self, forward_batch):
        pass

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def _eng_pad(self, o):
        # BCG replay: the break output buffer has the padded bucket's rows
        pad = self._eng_bs - o.shape[0]
        return o if pad == 0 else torch.nn.functional.pad(o, (0, 0, 0, pad))

    def _warn_continuation(self, appended: bool) -> None:
        """One-time note when a chunked-prefill continuation is detected."""
        if self._cont_warned:
            return
        self._cont_warned = True
        if appended:
            logger.info(
                "[PicoKVAttnBackend] chunked-prefill continuation: cache merged "
                "incrementally (all chunks retained); "
                + ("prefill attention reads the staged prefix ++ chunk (exact)."
                   if self._stg is not None else
                   "prefill OUTPUT of chunks after the first attends only the "
                   "current chunk — set chunked_prefill_size >= prompt length "
                   "for exact hidden states (or PICOKV_CHUNK_STAGE=host)."),
            )
        else:
            logger.warning(
                "[PicoKVAttnBackend] WARNING: chunked-prefill continuation "
                "re-prepared from the current chunk only — earlier chunks' "
                "compressed KV LOST (incremental append needs eas=0 default + "
                "offline projections). Set chunked_prefill_size >= prompt length.",
            )

    def _count_lost(self, plen: int, rid=None) -> None:
        """A row extended a prefix of plen tokens it holds no cache / staged
        K/V of (its new tokens attend only themselves). Counted; logged at
        the 1st, 2nd, 4th, 8th, ... occurrence; with the chunk stage,
        PICOKV_LOST=abort notes the request id for picokv_lost_abort."""
        self.n_lost += 1
        self.n_lost_tokens += plen
        if rid is not None and self._lost_abort and self._stg is not None:
            # no stage (PICOKV_CHUNK_STAGE=off / sdpa + PICOKV_ALLOW_CHUNKED_PREFILL=1)
            # is an explicit opt-in to chunks without their prefix: counted only
            self._lost_rids[rid] = max(plen, self._lost_rids.get(rid, 0))
            if len(self._lost_rids) > 4096:  # ids of requests that ended otherwise
                self._lost_rids.pop(next(iter(self._lost_rids)))
        if self.n_lost & (self.n_lost - 1) == 0:
            logger.warning(f"[PicoKVAttnBackend] WARNING: {self.n_lost} extends so far attended "
                           f"without their prefix ({self.n_lost_tokens} prefix tokens unseen)")

    def picokv_lost_abort(self, rid) -> Optional[str]:
        """Scheduler hook (process_batch_result_prefill, at a request's last
        prefill chunk): the abort message for request ``rid`` when it
        extended a prefix this backend could not attend, else None. Every
        attention-TP rank answers alike (the staging verdict is all-reduced)."""
        n = self._lost_rids.pop(rid, None) if self._lost_rids else None
        if n is None:
            return None
        return (f"picokv: {n} prompt-prefix tokens could not be attended (exact chunked "
                f"prefill / radix host K/V staging failed for lack of device or pinned "
                f"host memory); retry the request")

    @staticmethod
    def _tp_sync(model_runner):
        """ok -> every attention-TP rank ok (a MIN all-reduce on the attention
        TP CPU group); None at TP 1. Called at the same points on every rank
        (the stage's control flow is deterministic up to its allocations)."""
        tp, _ = PicoKVAttnBackend._attn_tp(model_runner)
        if tp <= 1:
            return None
        import torch.distributed as dist
        from sglang.srt.distributed.parallel_state import get_attn_tp_group

        grp = get_attn_tp_group().cpu_group

        def sync(ok):
            t = torch.tensor([1 if ok else 0], dtype=torch.int32)
            dist.all_reduce(t, op=dist.ReduceOp.MIN, group=grp)
            return bool(t.item())

        return sync

    @staticmethod
    def _ints(x):
        if hasattr(x, "tolist"):
            return [int(v) for v in x.tolist()]
        return [int(v) for v in x]

    @staticmethod
    def _chunk_size() -> int:
        """SGLang's resolved chunked_prefill_size (<= 0: off). Read, not
        guessed: a wrong 0 would size no chunk stage and lose every
        continuation chunk's prefix."""
        from sglang.srt import runtime_context as rc

        if not rc._CONTEXT.is_config_namespace_published("schedule"):
            return 0  # no engine (direct-drive unit tests): no chunked prefill
        return int(rc.get_schedule().chunked_prefill_size or 0)

    def _stage_begin(self, fb, rids, eseq, plens):
        """Layer 0 of an extend with the chunk stage on: which rows continue
        a staged prefix / stage their chunk. Reads orig_seq_lens (the full
        prompt length; a device->host sync), only for an extend that can hold
        a chunked request (a continuation, or >= half a chunk of tokens: a
        truncated chunk fills the chunk budget)."""
        stg = self._stg
        stg.rows, stg.wplan, stg.rplan = [], None, None
        if self._radix:
            return self._radix_begin(fb, rids, eseq, plens)
        if max(plens) == 0 and 2 * sum(eseq) < self._stg_chunk:
            for rid in (rids if stg.ent else ()):
                if rid in stg.ent:  # a new request on a row whose staged prefix died
                    stg.drop(rid)   # (abort / retraction between chunks)
            return
        o = getattr(fb, "orig_seq_lens", None)
        if o is None:
            raise RuntimeError("PiCoKV chunked prefill needs ForwardBatch.orig_seq_lens "
                               "(the full prompt length) to stage a continuation")
        err = None
        ol = self._ints(o[:len(rids)])
        if self._prealloc:
            self._fin = dict(zip(rids, ol))
        try:
            lost = stg.begin(rids, eseq, plens, ol)
        except Exception as e:  # pinned host / device memory
            err = e
        # one verdict for all attention-TP ranks: their schedulers abort the
        # same requests (picokv_lost_abort)
        ok = err is None
        if stg.sync is not None and not stg.sync(ok) and ok:
            err = _StageFailed("on another attention-TP rank")
        if err is not None:
            logger.warning(f"[PicoKVAttnBackend] chunk stage failed ({err!r}): this extend's "
                           f"continuations lose their prefix")
            for rid in rids:
                stg.drop(rid)
            stg.rows, stg.wplan, lost = [], None, [r for r, p in zip(rids, plens) if p > 0]
        if lost and not self._stg_warned:
            self._stg_warned = True
            logger.warning(f"[PicoKVAttnBackend] WARNING: chunked-prefill continuation of req rows "
                           f"{lost} without a staged prefix: those chunks attend only themselves")

    def _radix_begin(self, fb, rids, eseq, plens):
        """_stage_begin under PICOKV_RADIX=host: attach the stage to the
        radix tree at the first extend (the tree is built after the backend),
        give a req row a fresh cache when it holds a new request (a radix hit
        has a prefix but no cache of it), stage every row (_ChunkStage.rbegin)."""
        stg, rstr = self._stg, getattr(fb, "rids", None)
        if stg.tree is None:
            from sglang.srt.mem_cache.picokv_radix_cache import live_tree

            tree = live_tree()
            if tree is None or rstr is None:
                logger.warning("[PicoKVAttnBackend] PICOKV_RADIX=host without a picokv radix tree "
                               "(radix cache disabled?): radix host K/V off")
                self._radix = False
                return self._stage_begin(fb, rids, eseq, plens)
            # (8 GiB per attention rank: a mirror page is 1.2 GB, unbounded
            # mirrors ran a 106k NIAH cell out of memory at 21 pages)
            hgb = float(os.environ.get("PICOKV_RADIX_HOT_GB", "8")) / max(1, int(self._tp))
            # PICOKV_RADIX_CODES (default on with the hot tier and the fast
            # prepare): mirrors keep their tokens' codes too (per token, +16%
            # mirror bytes), a hit recompresses only its new tokens
            codes, sc = None, self._scfg
            if (self._fprep is not None and _code_asm_kernel is not None
                    and os.environ.get("PICOKV_RADIX_CODES", "1") != "0"):
                codes = (int(sc.sink_size), int(sc.local_size), int(sc.key_bits) // 64,
                         int(sc.value_bits) // 64,
                         int(os.environ.get("PICOKV_RADIX_CODES_MIN", "1024")))
            stg.attach(tree, _radix_host_gb(self._tp),
                       picokv_cell_size_bytes(self.num_kv_heads, self.head_dim, self._n_layers),
                       self._kvpool, hgb, codes)
            logger.info(f"[PicoKVAttnBackend] radix host K/V: tree attached ({len(stg.pool)} x "
                        f"{stg.BLK}-token pages, up to {stg.rcap}; GPU hot tier "
                        f"{'on' if stg.hot is not None else 'off'}"
                        f"{', codes' if stg.code is not None else ''})")
        for row, rid in zip(rids, rstr):
            if self._own.get(row) != rid:
                self._own[row] = rid
                self._reqs.pop(row, None)
        try:
            lost = stg.rbegin(rids, eseq, plens, rstr)
        except Exception as e:  # pinned host / device memory
            logger.warning(f"[PicoKVAttnBackend] radix host K/V stage failed ({e!r}): this extend's "
                           f"prefixes are not attended")
            stg.rows, stg.rplan, stg.wplan = [], None, None
            stg.cplan, stg.cmarks, stg.cwr, stg.rsl = {}, [], {}, {}
            lost = [r for r, p in zip(rids, plens) if p > 0]
        if lost and not self._stg_warned:
            self._stg_warned = True
            logger.warning(f"[PicoKVAttnBackend] WARNING: req rows {lost} extend a prefix without "
                           f"host K/V: they attend only their new tokens")

    def _ext_meta(self, fb, layer_id):
        """PICOKV_EXT_FAST: (rids, extend lens, prefix lens) of this extend as
        host ints, read at layer 0 from the CPU mirrors (device copies only
        when a mirror is missing) and reused by the other layers. The req-id
        mirror is checked against the device copy on the first extend and
        every 64th; a mismatch disables the mirror for good (_rids_cpu)."""
        m = self._ext_m
        if layer_id != 0 and m is not None and m[0] == id(fb):
            return m[1], m[2], m[3]
        es = getattr(fb, "extend_seq_lens_cpu", None)
        eseq = self._ints(es) if es is not None else self._ints(fb.extend_seq_lens)
        n = len(eseq)
        mir = getattr(fb, "req_pool_indices_cpu", None) if self._rids_cpu else None
        if mir is None or mir.shape[0] < n:
            rids = self._ints(fb.req_pool_indices[:n])
        else:
            rids = self._ints(mir[:n])
            self._ext_n += 1
            if self._ext_n % 64 == 1:
                d = self._ints(fb.req_pool_indices[:n])
                if d != rids:
                    logger.warning(f"[PicoKVAttnBackend] extend req_pool_indices_cpu mirror {rids} != "
                                   f"device {d}; reading the device copy from now on")
                    self._rids_cpu = False
                    rids = d
        pc = getattr(fb, "extend_prefix_lens_cpu", None)
        pe = getattr(fb, "extend_prefix_lens", None)
        plens = (self._ints(pc) if pc is not None
                 else (self._ints(pe) if pe is not None else [0] * n))
        self._ext_m = (id(fb), rids, eseq, plens)
        return rids, eseq, plens

    def _host_rids(self, fb, n):
        """The batch's first n req pool indices as host ints: the CPU mirror
        when it is present and unpadded, else a device read (D2H sync)."""
        m = getattr(fb, "req_pool_indices_cpu", None) if self._rids_cpu else None
        if m is None or m.shape[0] != n:
            return self._ints(fb.req_pool_indices[:n])
        r = self._ints(m)
        self._rids_n += 1
        if self._rids_n == 1:
            logger.info("[PicoKVAttnBackend] decode req ids from the CPU mirror "
                        "(no D2H sync)")
        if (self._batch is None or self._batch.get("rids") != r
                or self._rids_n % 256 == 0):
            d = self._ints(fb.req_pool_indices[:n])
            if d != r:
                logger.warning(f"[PicoKVAttnBackend] req_pool_indices_cpu mirror {r} != "
                               f"device {d}; reading the device copy from now on")
                self._rids_cpu = False
                return d
        return r

    def _log_l0(self, path, msg):
        # step-first line: once per path change within a batch, then every
        # 256 steps (a per-step print+flush is host time on the critical path)
        b = self._batch
        b["nl0"] = b.get("nl0", 0) + 1
        if b.get("last_path") != path or b["nl0"] % 256 == 0:
            b["last_path"] = path
            logger.info(msg)

    def _sp_tick(self, kind):
        """PICOKV_STEPPROF: close the previous step (bucket its wall time by
        the flags it raised) and open a ``kind`` step ("X" prefill, "D")."""
        import time as _time
        sp = self._sp
        now = _time.perf_counter()
        if sp["t0"] is not None:
            f = sp["f"]
            if sp["kind"] == "X":
                cat = "prefill"
            elif "build" in f:
                cat = "build"
            elif "slot" in f:
                cat = "slot"
            elif "cap" in f:
                cat = "capture"
            elif "eager" in f:
                cat = "warm" if "warm" in f else "eager"
            else:
                cat = "replay"
            a = sp["acc"].setdefault(cat, [0, 0.0])
            a[0] += 1
            a[1] += now - sp["t0"]
        sp["t0"], sp["kind"], sp["f"] = now, kind, set()
        sp["n"] += 1
        if sp["n"] % sp["every"] == 0:
            tot = sum(v[1] for v in sp["acc"].values())
            parts = " ".join(f"{c}={v[0]}/{v[1]:.2f}s/{1e3 * v[1] / max(v[0], 1):.2f}ms"
                             for c, v in sorted(sp["acc"].items()))
            hs = " ".join(f"{c}={v:.2f}s" for c, v in sorted(sp["h"].items()))
            logger.info(f"[PICOKV_STEPPROF] steps={sp['n']} tot={tot:.2f}s {parts} | host {hs}")

    def _sp_host(self, name, t0):
        import time as _time
        h = self._sp["h"]
        h[name] = h.get(name, 0.0) + _time.perf_counter() - t0

    def _bhsd(self, chunk, heads):
        # [n, heads, D] -> [1, heads, n, D]
        return chunk.reshape(-1, heads, self.head_dim).permute(1, 0, 2).unsqueeze(0).contiguous()

    def _kv_tok(self, k_, v_):
        """k_ / v_ ([T,KVH,D]) as ONE [T,2,KVH,D] view when they are adjacent
        column views of the same qkv buffer (sglang's split + in-place rope),
        else None."""
        m = self._kvm
        if m is not None and m[0] is k_ and m[1] is v_:
            return m[2]  # this layer's, from the radix stage (forward_extend)
        H, D = self.num_kv_heads, self.head_dim
        if (k_.stride() == v_.stride() and k_.stride()[1:] == (D, 1)
                and v_.data_ptr() - k_.data_ptr() == H * D * k_.element_size()
                and k_.untyped_storage().data_ptr() == v_.untyped_storage().data_ptr()):
            if self._kv_adj is None:
                self._kv_adj = True
                logger.info("[PicoKVAttnBackend] fast prepare: one K+V copy (qkv views)")
            return k_.as_strided((k_.shape[0], 2, H, D), (k_.stride(0), H * D, D, 1))
        return None

    def _bhsd_kv(self, k_, v_, off, ln):
        """(_bhsd(k_[off:off+ln]), _bhsd(v_[off:off+ln])), in ONE copy when
        k_ / v_ are adjacent views (_kv_tok): both come out of one
        [2,KVH,ln,D] tensor."""
        kv = self._kv_tok(k_, v_)
        if kv is not None:
            kvb = kv[off:off + ln].permute(1, 2, 0, 3).contiguous()
            return kvb[0:1], kvb[1:2]
        H = self.num_kv_heads
        return self._bhsd(k_[off:off + ln], H), self._bhsd(v_[off:off + ln], H)

    def _blockk_pair(self, kb, vb, layer_id, start, glen, L, gate="ovl"):
        """Both sides' fused fixed-cascade launch over tokens [start,
        start+glen) of kb / vb ([1,KVH,*,D]): V on prepare's side stream,
        overlapped with K (as prepare_kv_cache), from fp[gate] prompt tokens
        L up (PICOKV_PREP_OVERLAP_MIN / ..._GROUP). Returns (packed key
        [1,H,KW,glen], value words [1,H,glen,VW])."""
        fp = self._fprep
        _, H, _, D = kb.shape
        dev = kb.device
        ek, dk, ev, dv = self._slide_projs(layer_id, H, D, dev, kb.dtype)
        blockk, zr = fp["blockk"], int(self._scfg.zerorope)
        if L < fp[gate]:
            pva, _ = blockk(vb, ev, dv, start, glen, 0, False, False)
            pka, _ = blockk(kb, ek, dk, start, glen, zr, True, True)
            return pka, pva
        if fp["side"] is None:
            fp["side"] = fp["side_of"](dev)
            fp["ev"] = (torch.cuda.Event(), torch.cuda.Event())
        side, (e0, e1) = fp["side"], fp["ev"]
        cur = torch.cuda.current_stream(dev)
        e0.record(cur)
        side.wait_event(e0)
        torch.cuda.set_stream(side)
        try:
            pva, _ = blockk(vb, ev, dv, start, glen, 0, False, False)
        finally:
            torch.cuda.set_stream(cur)
        pka, _ = blockk(kb, ek, dk, start, glen, zr, True, True)
        e1.record(side)
        cur.wait_event(e1)
        return pka, pva

    def _kept(self, glen, H, dev):
        """eas=0: every global token stays compressed (identity map), as in
        _select_eas_and_kept_indices; reused by all layers of a prompt."""
        fp = self._fprep
        kept = fp["kept"].get((glen, H))
        if kept is None:
            if len(fp["kept"]) >= 8:
                fp["kept"].pop(next(iter(fp["kept"])))
            kept = torch.arange(glen, device=dev, dtype=torch.int32).expand(
                1, H, glen).contiguous()
            fp["kept"][(glen, H)] = kept
        return kept

    def _prepare_kv(self, cache, layer_id, k_, v_, off, ln, dense=None, codes=None):
        """_prepare_fast(cache, layer_id, *_bhsd_kv(k_, v_, off, ln), dense,
        codes): below the overlap gate with k_ / v_ adjacent qkv views, in ONE
        extension call (PICOKV_PREP_FUSED) reading the token-major rows."""
        fp = self._fprep
        kv = (self._kv_tok(k_, v_) if fp["fused"] is not None and ln < fp["ovl"] and codes is None
              else None)
        if kv is None:
            self._prepare_fast(cache, layer_id, *self._bhsd_kv(k_, v_, off, ln), dense=dense,
                               codes=codes)
            return
        sc = self._scfg
        H, D = self.num_kv_heads, self.head_dim
        sink, local = int(sc.sink_size), int(sc.local_size)
        dev, dt = kv.device, kv.dtype
        ek, dk, ev, dv = self._slide_projs(layer_id, H, D, dev, dt)
        kp, vp, kpt = self._gl0_projections(layer_id, H, D, dev, dt)
        dkey, dval, pk, pv = fp["fused"](
            kv, off, ln, ek, dk, ev, dv, self._kept(ln - sink - local, H, dev), sink, local,
            int(sc.zerorope), *(dense if dense is not None else (None, None)))
        cache.layers[layer_id] = PreparedKVCache(
            dense_key=dkey, dense_value=dval, key_projection=kp, value_projection=vp,
            key_projection_t=kpt, packed_key=pk, packed_value=pv, exp_bits=int(sc.exp_bits))
        cache.seq_lengths[layer_id] = ln

    def _prepare_fast(self, cache, layer_id, kb, vb, dense=None, codes=None):
        """== cache.prepare(layer_id, kb, vb) for a compressed layer (L >
        sink+local) under PICOKV_FAST_PREPARE: the kernels prepare_kv_cache
        runs (offline, eas=0), called directly. kb/vb: contiguous bf16
        [1,KVH,L,D]; dense: optional contiguous [1,KVH,sink+local,D] pair
        the dense window is gathered into; codes: a radix code plan
        (_ChunkStage.code_plan) whose writes take the codes."""
        fp, sc = self._fprep, self._scfg
        _, H, L, D = kb.shape
        sink, local = int(sc.sink_size), int(sc.local_size)
        glen = L - sink - local
        dev, dt = kb.device, kb.dtype
        kp, vp, kpt = self._gl0_projections(layer_id, H, D, dev, dt)
        kept = self._kept(glen, H, dev)
        eas = fp["eas"].get(H)
        if eas is None:
            eas = fp["eas"][H] = torch.empty(1, H, 0, device=dev, dtype=torch.int32)
        pka, pva = self._blockk_pair(kb, vb, layer_id, sink, glen, L)
        if codes is not None:
            self._code_write(layer_id, pka, pva, codes, 0)
        if dense is None:
            dkey, dval, pk, pv = fp["gather"](kb, vb, pka, pva, eas, kept, sink, local)
        else:
            dkey, dval, pk, pv = fp["gather_out"](kb, vb, pka, pva, eas, kept, sink,
                                                  local, *dense)
        cache.layers[layer_id] = PreparedKVCache(
            dense_key=dkey, dense_value=dval, key_projection=kp, value_projection=vp,
            key_projection_t=kpt, packed_key=pk, packed_value=pv, exp_bits=int(sc.exp_bits))
        cache.seq_lengths[layer_id] = L

    @staticmethod
    def _code_write(layer_id, pka, pva, plan, j0):
        """A radix code plan's writes (_code_put_kernel, one launch): global
        tokens j >= plan["c"] at columns j0 + j of pka [1,H,KW,Tk] / pva
        [1,H,Tk,VW] to the code slots of their mirrors."""
        if not plan["writes"]:
            return
        _, H, KW, Tk = pka.shape
        g, c = plan["g"], plan["c"]
        _code_put_kernel[(triton.cdiv(g - c, 256), KW + pva.shape[3], H)](
            plan["tab"], plan["np"], pka, pva, Tk, j0, c, g, layer_id, plan["blk"],
            H=H, KW=KW, VW=pva.shape[3], BLOCK=256)

    def _prepare_coded(self, cache, layer_id, g, s, n, plan, fin=None):
        """_prepare_fast over a staged prompt (g[s:s+n]; g: the stage's layer
        buffer [T,2,KVH,D]) reusing codes (PICOKV_RADIX_CODES): the leading
        plan["c"] global tokens' codes come from the hot tier's mirrors,
        blockk runs on the rest (whose codes go to the mirrors), and ONE
        _code_asm_kernel launch assembles the packed key, the grouped value
        codes (as _prepare_batched) and the dense window out of g; ``fin``
        (PICOKV_PREALLOC, the request's prompt length): straight into the
        full-capacity buffers _prealloc_layer would copy them to. Codes are
        per token: the same cache bit for bit (PICOKV_RADIX_CODES_CHECK=1
        compares)."""
        sc = self._scfg
        H, D = self.num_kv_heads, self.head_dim
        sink, local = int(sc.sink_size), int(sc.local_size)
        KW, VW, CV = int(sc.key_bits) // 64, int(sc.value_bits) // 64, int(sc.value_bits)
        glen, c = n - sink - local, plan["c"]
        m = glen - c
        dev, dt = g.device, g.dtype
        kp, vp, kpt = self._gl0_projections(layer_id, H, D, dev, dt)
        G = (glen + 63) // 64
        cap = self._pre_cap(glen, fin) if fin is not None else None
        full = cap is not None
        if not full:
            cap = 64 * G
            pka = torch.empty((1, H, KW, glen), dtype=torch.int64, device=dev)
            pv = torch.empty((1, H, G, CV), dtype=torch.int64, device=dev)
            KS, zq = glen, 0
        else:
            pool = self._kvpool if not torch.cuda.is_current_stream_capturing() else None
            with torch.cuda.use_mem_pool(pool) if pool is not None else contextlib.nullcontext():
                pka = torch.empty((1, H, KW, cap), dtype=torch.int64, device=dev)
                pv = torch.empty((1, H, cap // 64, CV), dtype=torch.int64, device=dev)
            pka._pk_full = pv._pk_full = True
            KS, zq = cap, triton.cdiv(cap - glen, 64)
        dw = torch.empty((2, H, sink + local, D), dtype=dt, device=dev)
        if m:
            x = g[s + sink + c:s + n - local].permute(1, 2, 0, 3).contiguous()  # [2,H,m,D]
            tk, tv = self._blockk_pair(x[0:1], x[1:2], layer_id, 0, m, m)
        else:
            tk = tv = pka
        nwq = triton.cdiv(sink + local, 64)
        _code_asm_kernel[(G + nwq + zq, max(KW + VW, 2), H)](
            plan["tab"], plan["np"], tk, tv, m, c, glen, layer_id, pka, pv, KS, (cap // 64) * CV,
            g, g.stride(0), g.stride(1), g.stride(2), s, n, dw, plan["blk"], G,
            H=H, KW=KW, VW=VW, D=D, SINK=sink, LOCAL=local, BT=64, NWQ=nwq,
            num_warps=2)  # 4: 17x slower
        if full:  # prefix views (_pk_full): _prealloc_layer skips the row
            pka, pv = pka[..., :glen], pv[:, :, :G]
        cache.layers[layer_id] = PreparedKVCache(
            dense_key=dw[0:1], dense_value=dw[1:2], key_projection=kp, value_projection=vp,
            key_projection_t=kpt, packed_key=pka, packed_value=pv, exp_bits=int(sc.exp_bits))
        cache.seq_lengths[layer_id] = n
        if _RADIX_CODES_CHECK:
            self._codes_check(cache, layer_id, g, s, n, c)

    def _codes_check(self, cache, layer_id, g, s, n, c):
        """PICOKV_RADIX_CODES_CHECK: _prepare_coded's layer vs _prepare_fast
        over the same staged tokens."""
        ref = PicoKVRequestCache(self._scfg, self._offline)
        self._prepare_fast(ref, layer_id, *self._bhsd_kv(g[:, 0], g[:, 1], s, n))
        a, b = ref.layers[layer_id], cache.layers[layer_id]
        bad = [f for f in ("dense_key", "dense_value", "packed_key", "packed_value")
               if not torch.equal(getattr(a, f), getattr(b, f))]
        kf, vf = _pk_full(b.packed_key), _pk_full(b.packed_value)
        if kf is not None and bool(kf[..., b.packed_key.shape[-1]:].any()):
            bad.append("packed_key tail")  # full-capacity buffers: zero past the content
        if vf is not None and bool(vf[:, :, b.packed_value.shape[2]:].any()):
            bad.append("packed_value tail")
        st = self.__dict__.setdefault("_ccheck", [0, 0])
        st[0] += 1
        st[1] += bool(bad)
        if bad:
            print(f"[picokv radix] CODES CHECK MISMATCH layer {layer_id} n {n} read {c}: {bad}",
                  flush=True)
        if layer_id == self._n_layers - 1:
            print(f"[picokv radix] codes check: {st[0]} prepares, {st[1]} mismatched", flush=True)

    def _bplan(self, sel, dev, which="plan"):
        """PICOKV_PREP_BATCH groups of one extend's compressed prompts ([(cache,
        off, ln)], in order), each <= PICOKV_PREP_BATCH_TOKENS global tokens,
        with their device tables. Identical for every layer of the extend, so
        built once (cached on the (off, ln) list; ``which``: one cache per
        source, so an extend's fresh and staged rows don't evict each other's
        plan every layer). A 1-prompt group gets None, a group of fewer than
        PICOKV_PREP_BATCH_MIN prompts and at most PICOKV_PREP_BATCH_MIN_TOKENS
        prompt tokens {solo: True} (per-request fast path)."""
        fp, sc = self._fprep, self._scfg
        key = tuple((o, ln) for _, o, ln in sel)
        if fp[which] is not None and fp[which][0] == key:
            return fp[which][1]
        H = self.num_kv_heads
        sink, W = int(sc.sink_size), self._dense_max
        KW, CV = int(sc.key_bits) // 64, int(sc.value_bits)

        def h2d(t):
            return t.pin_memory().to(dev, non_blocking=True)

        groups, cur, tg = [], [], 0
        for m, (_, _, ln) in enumerate(sel):
            if cur and tg + ln - W > fp["btok"]:
                groups.append(cur)
                cur, tg = [], 0
            cur.append(m)
            tg += ln - W
        groups.append(cur)
        plan = []
        for mem in groups:
            if len(mem) == 1:
                plan.append((mem, None))
                continue
            if (fp["gather_out"] is not None and len(mem) < fp["bmin"]
                    and sum(sel[m][2] for m in mem) <= fp["bmin_tok"]):
                plan.append((mem, dict(solo=True)))
                continue
            offs = torch.tensor([sel[m][1] for m in mem])
            lens = torch.tensor([sel[m][2] for m in mem])
            # dense window: [off, off+sink) + [off+ln-local, off+ln)
            A = torch.arange(W)
            widx = (offs[:, None] + A[None, :] + (A[None, :] >= sink) * (lens[:, None] - W)).reshape(-1)
            gl = lens - W
            gls = gl.tolist()
            c0 = (torch.cumsum(gl, 0) - gl).tolist()   # column start in the group
            Tg = sum(gls)
            # global tokens: request i's [off+sink, off+ln-local) -> columns [c0, c0+gl)
            gidx = torch.arange(Tg) + torch.repeat_interleave(
                offs + sink - torch.tensor(c0), gl)
            Gs = [(g + 63) // 64 for g in gls]
            kof, vof, ko, vo = [], [], 0, 0
            for g, G in zip(gls, Gs):
                kof.append(ko)
                vof.append(vo)
                ko += H * KW * g
                vo += H * G * CV
            ktab = [(c, g, o) for c, g, o in zip(c0, gls, kof)]
            vtab = [(c + 64 * q, min(64, g - 64 * q), o + q * CV, G * CV)
                    for c, g, o, G in zip(c0, gls, vof, Gs) for q in range(G)]
            plan.append((mem, dict(
                Tg=Tg, gls=gls, c0=c0, Gs=Gs, kof=kof, vof=vof, ksum=ko, vsum=vo, glmax=max(gls),
                NG=len(vtab), gidx=h2d(gidx), widx=h2d(widx),
                ktab=h2d(torch.tensor(ktab, dtype=torch.int64)),
                vtab=h2d(torch.tensor(vtab, dtype=torch.int64)))))
        fp[which] = (key, plan)
        return plan

    def _prepare_batched(self, layer_id, k_, v_, sel, which="plan", codes=None):
        """== _prepare_fast for each compressed prompt of ``sel`` ([(cache,
        off, ln)] of one extend; k_ / v_ token-major [T,KVH,D]), but each
        _bplan group shares ONE blockk launch per side over its concatenated
        global tokens (one small 16-CTA launch per prompt and side is
        latency-bound), one window gather and two Triton splits (key columns,
        value 64-token regroup). Request tensors are views of per-group
        buffers. Bitwise the same cache (t43/test_fastprep.py). ``codes``:
        {sel index: radix code plan} (their codes go to the mirrors)."""
        sc = self._scfg
        codes = codes or {}
        H, D = self.num_kv_heads, self.head_dim
        dev, dt = k_.device, k_.dtype
        W = self._dense_max
        KW, VW, CV = int(sc.key_bits) // 64, int(sc.value_bits) // 64, int(sc.value_bits)
        eb = int(sc.exp_bits)
        kp, vp, kpt = self._gl0_projections(layer_id, H, D, dev, dt)
        kv = None
        for mem, t in self._bplan(sel, dev, which):
            if t is None:
                for m in mem:
                    cache, off, ln = sel[m]
                    self._prepare_kv(cache, layer_id, k_, v_, off, ln, codes=codes.get(m))
                continue
            n = len(mem)
            if t.get("solo"):
                # per-request fast path, dense windows into ONE group buffer: a
                # prompt's own ~640 KiB pair would come from the allocator's
                # small-block pool, which no other size reuses (r160 peak +78 MiB);
                # [2n,H,W,D] = keys then values, split in one call
                dw = torch.empty(2 * n, H, W, D, device=dev, dtype=dt).split(1)
                for i, m in enumerate(mem):
                    cache, off, ln = sel[m]
                    self._prepare_kv(cache, layer_id, k_, v_, off, ln, dense=(dw[i], dw[n + i]),
                                     codes=codes.get(m))
                continue
            if kv is None:
                kv = self._kv_tok(k_, v_)
                if kv is None:
                    kv = torch.stack((k_, v_), 1)  # [T,2,H,D]
            Tg = t["Tg"]
            xg = kv.permute(1, 2, 0, 3).index_select(2, t["gidx"]).contiguous()  # [2,H,Tg,D]
            # overlap gate on the group's prompt tokens (Tg global + n windows)
            pka, pva = self._blockk_pair(xg[0:1], xg[1:2], layer_id, 0, Tg, Tg + n * W,
                                         "ovl_g")
            for i, m in enumerate(mem):
                if codes.get(m) is not None:
                    self._code_write(layer_id, pka, pva, codes[m], t["c0"][i])
            dw = kv.index_select(0, t["widx"]).view(n, W, 2, H, D).permute(
                2, 0, 3, 1, 4).contiguous()  # [2,n,H,W,D]
            pk = torch.empty(t["ksum"], device=dev, dtype=torch.int64)
            _prep_kcopy_kernel[(n, H * KW, triton.cdiv(t["glmax"], 1024))](
                pka, pk, t["ktab"], Tg, BLOCK=1024)
            pv = torch.empty(t["vsum"], device=dev, dtype=torch.int64)
            _prep_vgroup_kernel[(t["NG"], VW, H)](pva, pv, t["vtab"], Tg, VW=VW)
            for i, m in enumerate(mem):
                cache, _, ln = sel[m]
                g, G, a, b = t["gls"][i], t["Gs"][i], t["kof"][i], t["vof"][i]
                cache.layers[layer_id] = PreparedKVCache(
                    dense_key=dw[0, i:i + 1], dense_value=dw[1, i:i + 1],
                    key_projection=kp, value_projection=vp, key_projection_t=kpt,
                    packed_key=pk[a:a + H * KW * g].view(1, H, KW, g),
                    packed_value=pv[b:b + H * G * CV].view(1, H, G, CV), exp_bits=eb)
                cache.seq_lengths[layer_id] = ln
    def _pre_cap(self, g, fin):
        """PICOKV_PREALLOC: the full-capacity buffer width (tokens) of a
        compressed prompt with g global tokens of a fin-token request, or
        None: rows up to PICOKV_ROWS_SHARE_CAP share decode storage."""
        dm = self._dense_max
        cw = max(0, dm - self._slide_W)  # window tokens a build catches up
        cap = max(512, _round_up(max(g, fin - dm) + cw + self._slide_head, 512))
        return None if cap <= _ROWS_SHARE_CAP else cap

    def _prealloc_layer(self, layer_id, pre):
        """PICOKV_PREALLOC: this layer's freshly prepared compressed prompts
        ([(cache, the request's full prompt length)]) move their packed codes
        into full-capacity buffers: the decode-row capacity of the whole
        prompt (_fixed_layer_fast's rowcap), zero past the content, all in
        one fill launch. The cache holds prefix views of them (_pk_full);
        later chunks append in place, the decode batch adopts them. Rows up
        to PICOKV_ROWS_SHARE_CAP stay (short rows share decode storage)."""
        alloc = torch.empty if self._fill_k else torch.zeros  # the fill zeroes past n
        pairs = []
        pool = self._kvpool if not torch.cuda.is_current_stream_capturing() else None
        for rc, fin in pre:
            c = rc.layers.get(layer_id)
            if c is None or c.packed_key is None:
                continue
            pk, pv = c.packed_key, c.packed_value
            g = int(pk.shape[-1])
            cap = self._pre_cap(g, fin)
            if (cap is None or _pk_full(pk) is not None
                    or not (pk.is_contiguous() and pv.is_contiguous())):
                continue
            _, H, KW, _ = pk.shape
            if pool is not None:
                with torch.cuda.use_mem_pool(pool):
                    kf = alloc(1, H, KW, cap, dtype=pk.dtype, device=pk.device)
                    vf = alloc(1, H, cap // 64, int(pv.shape[-1]), dtype=pv.dtype, device=pv.device)
            else:
                kf = alloc(1, H, KW, cap, dtype=pk.dtype, device=pk.device)
                vf = alloc(1, H, cap // 64, int(pv.shape[-1]), dtype=pv.dtype, device=pv.device)
            kf._pk_full = vf._pk_full = True
            pairs.append(((pk, pv), (kf, vf)))
            c.packed_key, c.packed_value = kf[..., :g], vf[:, :, :int(pv.shape[2])]
        if pairs:
            self._row_copy(pairs)

    def _dense_staged(self, layer_id, g, sel):
        """== cache.prepare's dense branch for each staged row of ``sel``
        ([(cache, G off, n)], n <= sink+local; g: this layer's stage buffer
        [cap,2,KVH,D]), from ONE copy of their G regions ([2,N,KVH,D]) instead
        of two per row: rows hold [1,KVH,n,D] views of it (as the kc path)."""
        segs = tuple((s, n) for _, s, n in sel)
        a = segs[0][0]
        if all(segs[j + 1][0] == segs[j][0] + segs[j][1] for j in range(len(segs) - 1)):
            kv = g[a:segs[-1][0] + segs[-1][1]].permute(1, 0, 2, 3).contiguous()
        else:
            dp = self._dplan
            if dp is None or dp[0] != segs:  # same for every layer of the extend
                idx = np.concatenate([np.arange(s, s + n, dtype=np.int64) for s, n in segs])
                dp = self._dplan = (segs, torch.from_numpy(idx).pin_memory().to(
                    g.device, non_blocking=True))
            kv = g.permute(1, 0, 2, 3).index_select(1, dp[1]).contiguous()
        kv, o = kv.permute(0, 2, 1, 3), 0  # [2,KVH,N,D]: one slice per row and side
        for cache, _, n in sel:
            cache._dense[layer_id] = (kv[0:1, :, o:o + n], kv[1:2, :, o:o + n])
            cache.seq_lengths[layer_id] = n
            o += n

    # ------------------------------------------------------------------
    # prefill
    # ------------------------------------------------------------------

    def forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kw):
        import torch as _t
        if self._sp is not None and layer.layer_id == 0:
            self._sp_tick("X")
        if self._ext_fast:
            rids, eseq, plens = self._ext_meta(forward_batch, layer.layer_id)
        else:
            rids = self._ints(forward_batch.req_pool_indices)
            eseq = self._ints(forward_batch.extend_seq_lens)
        q_ = q.view(-1, self.num_q_heads, self.head_dim)
        k_ = k.view(-1, self.num_kv_heads, self.head_dim)
        v_ = v.view(-1, self.num_kv_heads, self.head_dim)

        # A prefill invalidates any live decode batch (membership may change):
        # stash its pools so continuing requests carry their appended tokens.
        # Slot mode keeps a stepped slot batch alive instead (the next decode
        # admits the new rows in place, _slot_change); a slot whose req id
        # re-prefills holds that id's previous request: retired, never
        # synced back (``stale``).
        b = self._batch
        if b is not None and not (b.get("S") and b.get("shared") is not None):
            self._stash_pools()
            self._batch = b = None
        if layer.layer_id == 0:
            if b is not None:
                sl = set(rids).intersection(b["rids"])
                if sl:
                    b.setdefault("stale", set()).update(sl)
            self._prune_finished(rids)
            self._release_dead(rids)
        # extend_prefix_lens > 0 marks a chunked-prefill continuation: the
        # request already holds its prefix's compressed cache.
        if not self._ext_fast:
            _epl = getattr(forward_batch, "extend_prefix_lens", None)
            _epl_cpu = getattr(forward_batch, "extend_prefix_lens_cpu", None)
            plens = (
                self._ints(_epl_cpu)
                if _epl_cpu is not None
                else (self._ints(_epl) if _epl is not None else [0] * len(rids))
            )
        stg = self._stg
        if layer.layer_id == 0:
            self._fin = {}
        if stg is not None and layer.layer_id == 0:
            self._stage_begin(forward_batch, rids, eseq, plens)
        # chunk stage: this layer's buffer (staged prefixes + this extend's
        # staged chunks) and the continuation rows by batch index
        # (radix mode: every staged extend, rplan; g only with prefix rows)
        son = stg is not None and (stg.rows or stg.rplan is not None)
        g = None
        if son:  # (this layer's prepares reuse the K+V view: _kvm, dropped below)
            self._kvm = (k_, v_, self._kv_tok(k_, v_))
            g = stg.layer(layer.layer_id, k_, v_, self._kvm[2])
        srow = {r[0]: r for r in stg.rows if r[3] > 0} if g is not None else {}
        rstr = getattr(forward_batch, "rids", None) if layer.layer_id == 0 else None
        # An extend whose every prompt fits the dense window (no compression,
        # nothing to append to) stores views into one copy of this layer's K/V.
        kc = vc = None
        if (self._ext_fast and max(eseq) <= self._dense_max and max(plens) == 0
                and q_.shape[0] == sum(eseq)):
            kc = k_ if k_.is_contiguous() else k_.contiguous()
            vc = v_ if v_.is_contiguous() else v_.contiguous()
        off = 0
        bsel = ([] if self._fprep is not None and self._fprep["batch"] and len(rids) > 1
                else None)
        # staged rows (radix hits / chunk continuations) re-preparing over
        # their staged prompt: dense ones share one copy of their G regions,
        # compressed ones the PICOKV_PREP_BATCH launches (from G)
        dsel = [] if len(srow) > 1 else None
        ssel = [] if bsel is not None and len(srow) > 1 else None
        # PICOKV_PREALLOC: this layer's fresh compressed prepares [(cache, the
        # request's full prompt length)] -> full-capacity buffers after the loop
        pre = [] if self._prealloc and sum(eseq) <= self._prealloc_max else None
        # PICOKV_RADIX_CODES: compressed prepares' code plans (made at layer 0
        # where each row's prepare path is decided; the rows' bsel codes)
        cod = (son and stg.tree is not None and getattr(stg, "code", None) is not None
               and self._fprep is not None)
        l0, bcod = layer.layer_id == 0, {}
        for i, (rid, ln, plen) in enumerate(zip(rids, eseq, plens)):
            cache = self._reqs.get(rid)
            if layer.layer_id == 0:
                # Layer 0 starts every extend: decide append vs fresh prepare
                # once (can_append validates prefix length + prerequisites).
                cont = plen > 0 and cache is not None and cache.can_append(plen)
                rs = rstr[i] if rstr is not None else None
                if plen == 0 and self._lost_rids:
                    self._lost_rids.pop(rs, None)  # a reused id starts over
                if not cont:
                    if plen > 0 and i not in srow:
                        self._warn_continuation(False)
                        self._count_lost(plen, rs)
                    cache = PicoKVRequestCache(self._scfg, self._offline)
                    self._reqs[rid] = cache
                elif self._stg is not None and i not in srow:
                    # a continuation the stage holds no prefix of (its
                    # previous chunk was taken as the last): the chunk
                    # attends only itself
                    self._count_lost(plen, rs)
                elif not self._cont_warned:
                    self._warn_continuation(True)
                cache._append_mode = cont
            elif cache is None:
                cache = PicoKVRequestCache(self._scfg, self._offline)
                self._reqs[rid] = cache
            r = srow.get(i)
            if r is not None and not cache._append_mode:
                # staged prefix but no incremental merge (eas > 0 / online
                # projections): the request's last chunk prepares over the
                # whole staged prompt (== the single-chunk prepare), earlier
                # chunks build nothing
                if not r[5]:
                    s, n = r[4], plen + ln
                    cp = ((stg.code_plan(i, n, True) if l0 else stg.cplan.get(i))
                          if cod and n > self._dense_max else None)
                    if cp is not None:  # (straight into its full-capacity buffers)
                        self._prepare_coded(cache, layer.layer_id, g, s, n, cp,
                                            n if pre is not None else None)
                    elif dsel is not None and n <= self._dense_max:
                        dsel.append((cache, s, n))  # one copy after the loop
                    elif (ssel is not None and n > self._dense_max
                            and n - self._dense_max <= self._fprep["bgl"]):
                        ssel.append((cache, s, n))  # batched after the loop
                    elif self._fprep is not None and n > self._dense_max:
                        self._prepare_kv(cache, layer.layer_id, g[:, 0], g[:, 1], s, n)
                    else:
                        cache.prepare(layer.layer_id, self._bhsd(g[s:s + n, 0], self.num_kv_heads),
                                      self._bhsd(g[s:s + n, 1], self.num_kv_heads))
                    if pre is not None and n > self._dense_max:
                        pre.append((cache, n))  # the whole prompt
                if self._pools is not None:
                    self._pools["idx"].pop(rid, None)
                off += ln
                continue
            if kc is not None and not cache._append_mode:
                # == cache.prepare's dense branch, as [1,KVH,ln,D] views
                cache._dense[layer.layer_id] = (kc[off:off + ln].permute(1, 0, 2).unsqueeze(0),
                                                vc[off:off + ln].permute(1, 0, 2).unsqueeze(0))
                cache.seq_lengths[layer.layer_id] = ln
                if self._pools is not None:
                    self._pools["idx"].pop(rid, None)
                off += ln
                continue
            if (self._fprep is not None and ln > self._dense_max
                    and not cache._append_mode):
                cp = (stg.code_plan(i, ln, False) if l0 else stg.cplan.get(i)) if cod else None
                if bsel is not None and ln - self._dense_max <= self._fprep["bgl"]:
                    if cp is not None:
                        bcod[len(bsel)] = cp
                    bsel.append((cache, off, ln))  # batched after the loop
                else:
                    self._prepare_kv(cache, layer.layer_id, k_, v_, off, ln, codes=cp)
                if pre is not None:
                    # a first chunk: its request's whole prompt (orig_seq_lens;
                    # a continuation that lost its prefix: this chunk alone)
                    pre.append((cache, ln if plen else max(ln, self._fin.get(rid, 0))))
                if self._pools is not None:
                    self._pools["idx"].pop(rid, None)
                off += ln
                continue
            kb = self._bhsd(k_[off:off + ln], self.num_kv_heads)
            vb = self._bhsd(v_[off:off + ln], self.num_kv_heads)
            if cache._append_mode and cache.append(layer.layer_id, kb, vb):
                pass  # merged incrementally (exact for the cache state)
            else:
                cache.prepare(layer.layer_id, kb, vb)
            if self._pools is not None:
                # this request re-prefilled: its stashed pool data is stale
                self._pools["idx"].pop(rid, None)
            off += ln
        if bsel:
            self._prepare_batched(layer.layer_id, k_, v_, bsel, codes=bcod)
        if dsel:
            self._dense_staged(layer.layer_id, g, dsel)
        if ssel:
            self._prepare_batched(layer.layer_id, g[:, 0], g[:, 1], ssel, "splan")
        if pre:
            self._prealloc_layer(layer.layer_id, pre)
        # Prefill attention: ONE batched ragged flashinfer call over the
        # concatenated token-major q/k/v (exact causal attention — replaces the
        # per-request SDPA loop). PICOKV_PREFILL=sdpa forces the old path.
        # Continuation rows of the chunk stage attend their staged prefix.
        self._kvm = None
        o = self._prefill_attn(q_, k_, v_, eseq, g) if self._prefill != "sdpa" else None
        if o is not None:
            if son:
                stg.after(layer.layer_id)
            self._n_extend += 1
            return o.reshape(-1, self.num_q_heads * self.head_dim)
        F = _t.nn.functional
        outs, off = [], 0
        for i, (rid, ln) in enumerate(zip(rids, eseq)):
            qb = self._bhsd(q_[off:off + ln], self.num_q_heads)
            r = srow.get(i)
            if r is None:
                kb = self._bhsd(k_[off:off + ln], self.num_kv_heads)
                vb = self._bhsd(v_[off:off + ln], self.num_kv_heads)
                o = F.scaled_dot_product_attention(
                    qb, kb, vb, is_causal=True, enable_gqa=True, scale=self.scaling)
            else:  # stage continuation: staged prefix ++ chunk, bottom-right causal
                s, p = r[4], r[3]
                kb = self._bhsd(g[s:s + p + ln, 0], self.num_kv_heads)
                vb = self._bhsd(g[s:s + p + ln, 1], self.num_kv_heads)
                o = torch.empty_like(qb)
                for a in range(0, ln, 1024):  # mask [<=1024, p+b] per q block
                    b_ = min(ln, a + 1024)
                    m = torch.ones(b_ - a, p + b_, dtype=torch.bool,
                                   device=qb.device).tril(p + a)
                    o[:, :, a:b_] = F.scaled_dot_product_attention(
                        qb[:, :, a:b_], kb[:, :, :p + b_], vb[:, :, :p + b_],
                        attn_mask=m, enable_gqa=True, scale=self.scaling)
            outs.append(o.squeeze(0).permute(1, 0, 2).reshape(ln, self.num_q_heads * self.head_dim))
            off += ln
        if son:
            stg.after(layer.layer_id)
        self._n_extend += 1
        return outs[0] if len(outs) == 1 else _t.cat(outs, dim=0)

    def _prefill_attn(self, q_, k_, v_, eseq, g=None):
        """One batched exact causal prefill attention via flashinfer's ragged
        prefill kernel over the concatenated token-major q/k/v. Returns
        [ΣL,QH,D] or None when unavailable (caller falls back to per-request
        SDPA — same math). With ``g`` (the chunk stage's layer buffer), the
        stage's continuation rows instead attend their prefix ++ chunk in g
        through a second plan (one call over those rows; the batch head
        when the engine put them first, as SGLang does)."""
        if self._fi_pre is False:
            return None
        try:
            if self._fi_pre is None:
                import flashinfer

                ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=self.device)
                W = flashinfer.BatchPrefillWithRaggedKVCacheWrapper
                # plan cache: (qo lens, kv lens) of the last plan() (skip re-plan)
                self._fi_pre = {"wrapper": W(ws, kv_layout="NHD"), "planned": None,
                                "cont": W(ws, kv_layout="NHD"), "cplanned": None}
            cr = [r for r in self._stg.rows if r[3] > 0] if g is not None else []
            if not cr:
                return self._fi_run("", q_, k_, v_, eseq, eseq)
            nc = len(cr)
            o = torch.empty(q_.shape[0], self.num_q_heads, self.head_dim,
                            dtype=q_.dtype, device=q_.device)
            if [r[0] for r in cr] == list(range(nc)):
                t0 = cr[-1][1] + cr[-1][2]
                if nc < len(eseq):
                    self._fi_run("", q_[t0:], k_[t0:], v_[t0:], eseq[nc:], eseq[nc:], out=o[t0:])
                qc, oc = q_[:t0], o[:t0]
            else:
                self._fi_run("", q_, k_, v_, eseq, eseq, out=o)
                qc, oc = torch.cat([q_[r[1]:r[1] + r[2]] for r in cr]), None
            kvl = [r[3] + r[2] for r in cr]
            n = sum(kvl)
            res = self._fi_run("c", qc, g[:n, 0], g[:n, 1], [r[2] for r in cr], kvl, out=oc)
            if oc is None:
                s = 0
                for r in cr:
                    o[r[1]:r[1] + r[2]] = res[s:s + r[2]]
                    s += r[2]
            return o
        except Exception as e:
            if isinstance(e, torch.cuda.OutOfMemoryError):
                # transient: this extend uses SDPA, the next one retries
                logger.warning(f"[PicoKVAttnBackend] ragged prefill out of memory ({e!r}); "
                               f"this extend uses SDPA")
                return None
            if isinstance(self._fi_pre, dict):
                logger.warning(f"[PicoKVAttnBackend] ragged prefill DISABLED ({e!r}); "
                               f"using SDPA")
            self._fi_pre = False
            return None

    def _fi_run(self, which, q, k, v, qo, kv, out=None):
        """Run the ragged prefill wrapper ``which`` ("" main, "c" stage
        continuation) over q lens ``qo`` / kv lens ``kv`` (causal,
        bottom-right aligned), re-planning only when the lens change; the
        indptrs are built on the host (plan needs host copies: no sync).
        plan() writes the wrapper's ONE page-locked buffer on the host, then
        copies it to the device asynchronously: with host indptrs nothing
        syncs, so a re-plan while the GPU still lags behind the previous
        plan's copy would hand that forward this plan (garbage prefill of
        a request whose lens were planned once). The re-plan waits for the
        previous plan's copy (an event right after it; normally long done)."""
        import numpy as np

        fi = self._fi_pre
        w, pk = ("cont", "cplanned") if which else ("wrapper", "planned")
        key = (tuple(qo), tuple(kv))
        if fi[pk] != key:
            ip = np.zeros((2, len(qo) + 1), dtype=np.int32)
            ip[0, 1:] = np.cumsum(qo)
            ip[1, 1:] = np.cumsum(kv)
            ip = torch.from_numpy(ip)
            ev = fi.get(w + "_ev")
            if ev is not None:
                ev.synchronize()  # the previous plan's H2D read the buffer
            fi[w].plan(
                ip[0], ip[1],
                self.num_q_heads, self.num_kv_heads, self.head_dim,
                causal=True, sm_scale=self.scaling,
                q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
            )
            if ev is None:
                ev = fi[w + "_ev"] = torch.cuda.Event()
            ev.record()
            fi[pk] = key
        return fi[w].run(q, k, v, out=out)

    # ------------------------------------------------------------------
    # decode: batch/pool construction
    # ------------------------------------------------------------------

    def _prune_finished(self, live):
        """Drop the caches of finished requests: a req-pool row back in the
        pool's free list belongs to no live request. ``live`` (this forward's
        rids) is never pruned — under the overlap scheduler a request's row
        can be freed while its last step is still in flight. Besides freeing
        memory early, this releases the batched storage that dedup views
        (``_fixed_layer``) of finished requests would otherwise pin. Chunk
        stage entries of aborted chunked requests go the same way."""
        stg = self._stg.ent if self._stg is not None else None
        if not self._reqs and not stg:
            return
        rp = getattr(self._mr, "req_to_token_pool", None)
        fs = getattr(rp, "free_slots", None)
        if not isinstance(fs, list):
            if rp is not None and not getattr(self, "_prune_warned", False):
                self._prune_warned = True  # caches of finished requests would pile up
                logger.warning(f"[PicoKVAttnBackend] WARNING: req_to_token_pool.free_slots is "
                               f"{type(fs).__name__}, not a list: finished-request caches are "
                               f"not pruned")
            return
        free, live = set(fs), set(live)
        for rid in [r for r in (stg or ()) if r in free and r not in live]:
            self._stg.drop(rid)
        for rid in [r for r in self._reqs if r in free and r not in live]:
            del self._reqs[rid]

    def _release_dead(self, rids):
        """Prefill: a kept slot batch / stash none of whose rows is still a
        live request (all finished, or re-prefilled ids) pins their compressed
        codes until the next decode while this prefill allocates — SGLang's
        pool already counts those tokens as free (16 x ~100k finished rows ->
        ~30 GB; NIAH 128k OOM). Nothing is left to carry: drop it now. A dead
        slot batch within the slot waste guard is kept: the next decode
        refills it in place instead of rebuilding + re-capturing every layer
        (sequential single requests: ~6 ms TTFT)."""
        b = self._batch
        if b is not None:
            stale = b.get("stale") or ()
            if not any(r in self._reqs and r not in stale for r in b["rids"]) and (
                    self._packed_nbytes(b) > self._slot_waste):
                self._batch = None
        st = self._pools
        if st is not None:
            cur = set(rids)
            if not any(r in self._reqs and r not in cur for r in st["idx"]):
                self._pools = None

    def _stash_pools(self):
        """Snapshot the live decode batch's pools + per-request window lengths
        so the NEXT batch can carry decode-appended tokens across the rebuild:
        a request's cache ``dense_key`` stays the prefill-time window; only the
        pool holds the tokens appended since."""
        b = self._batch
        if b is None:
            return
        sh = b.get("shared")
        if self._fast:
            self._stash_lazy(b, sh)
            return
        if sh is None:
            return  # never stepped: the previous stash is still every row's truth
        for lid, fx in b["fixed"].items():
            if fx.get("slide"):
                self._slide_sync(fx, lid, b["rids"])
        old = self._pools
        stale = b.get("stale") or ()
        self._pools = {
            "idx": {rid: i for i, rid in enumerate(b["rids"]) if rid not in stale},
            "lens": list(sh["dlens_py"]),  # per-request window len incl. last append
            "pools": {lid: (fx["kbuf"], fx["vbuf"]) for lid, fx in b["fixed"].items()},
        }
        if old is not None:
            # Live requests carried by the previous stash but absent from this
            # batch would lose their decode tokens with it: write their window
            # back into their own cache (rare — e.g. a skipped decode step).
            self._write_back(old, skip=set(b["rids"]))

    def _write_back(self, pools, skip=()):
        """Copy each carried request's pooled dense window (prefill window +
        decode-appended tokens) back into its per-request cache, making that
        cache the source of truth again."""
        for rid, j in pools["idx"].items():
            rc = self._reqs.get(rid)
            if rid in skip or rc is None:
                continue
            ln = int(pools["lens"][j])
            for lid, (kp, vp) in pools["pools"].items():
                k = kp[j, :ln].transpose(0, 1).unsqueeze(0).contiguous()  # [1,KVH,ln,D]
                v = vp[j, :ln].transpose(0, 1).unsqueeze(0).contiguous()
                c = rc.layers.get(lid)
                if c is not None:
                    c.dense_key, c.dense_value = k, v
                elif lid in rc._dense:
                    rc._dense[lid] = (k, v)

    # ------------------------------------------------------------------
    # fast incremental batch build (slide)
    # ------------------------------------------------------------------

    @staticmethod
    def _h2d(vals, dev, dtype=torch.int32):
        """Small host list -> device WITHOUT a stream sync (a pageable
        torch.tensor(..., device=cuda) waits for the queue to drain): pinned
        staging + async copy; the caching host allocator recycles the pinned
        block only after the copy ran."""
        return torch.tensor(vals, dtype=dtype, pin_memory=True).to(dev, non_blocking=True)

    def _stash_lazy(self, b, sh):
        """Fast build: the live batch's per-layer tensors become the stash as
        they are (no copy, no sync). The next batch gathers its carried rows
        from them and hands the rows it drops back to their request caches,
        one layer at a time (_fixed_layer_fast)."""
        old = self._pools
        if old is not None and old.get("fixed"):
            self._flush_stash(old)  # layers this batch never built
        fixed = {lid: fx for lid, fx in b["fixed"].items() if fx is not None}
        if not fixed:
            return
        stale = b.get("stale") or ()  # slot mode: re-prefilled req ids
        self._pools = {
            "idx": {rid: i for i, rid in enumerate(b["rids"]) if rid not in stale},
            # per-request window len incl. the last append (never stepped:
            # the build's post-catch-up lens)
            "lens": list(sh["dlens_py"]) if sh is not None
            else list(next(iter(fixed.values()))["dlens0"]),
            "fixed": fixed,
            # epoch clock of the stashed rows: the next batch's step 0 is step + 1
            "step": sh["step"] if sh is not None else b.get("step0", 0) - 1,
            # their batch's slide window: a build at another W catches carried rows up
            "W": b.get("W", self._slide_W),
        }

    def _flush_stash(self, st):
        """Hand every row of the stash's remaining layers back to its request
        cache (robustness: a batch that never built some stashed layer)."""
        rows = list(st["idx"].items())
        for lid, fx in st["fixed"].items():
            self._sync_rows(fx, lid, rows, st["lens"])
        st["fixed"].clear()

    def _fill_fresh(self, jobs, memo=None):
        """PICOKV_FILL_KERNEL: fresh rows into batch tensors in ONE launch.
        jobs: [(pk, pv, kbuf, vbuf, row, cache or None, (dense k, dense v), g,
        d)], any mix of layers (each row: packed key / value zeroed past g,
        window [:d]) —
        what the row-wise copies + index_fill do. A one-row pk / pv is that
        row's own storage (_RowStore). False (nothing launched)
        if a tensor is not in the kernel's layout: the caller copies.
        ``memo`` (PICOKV_SLOT_MEMO, a dict kept with the slot batch): the
        layout checks + bases of its window buffers and packed rectangles
        carry over to the next slot change instead of being redone per layer
        (weak references: an entry holds only while its tensors live, and no
        tensor here changes its layout in place)."""
        if not self._fill_k or not jobs:
            return False
        H, D = self.num_kv_heads, self.head_dim
        ft = getattr(self, "_fill_tab", None)
        if ft is not None:  # PICOKV_FILL_TAB
            tt = ft(jobs, H, D)
            if tt is not None:
                HKW = jobs[0][0].shape[1] * jobs[0][0].shape[2]
                _fill_rows_kernel[(len(jobs), HKW + 3 * H)](
                    tt, HKW=HKW, H=H, D=D, BLOCK=1024, BT=16)
                return True
        bf, i64, wr = torch.bfloat16, torch.int64, weakref.ref
        tab, HKW = [], jobs[0][0].shape[1] * jobs[0][0].shape[2]
        mw, mp = ({}, {}) if memo is None else (memo.setdefault("w", {}), memo.setdefault("p", {}))
        ap = tab.extend
        for pk, pv, kb, vb, i, c, s, g, d in jobs:
            k, v = s
            e = mw.get((id(kb), id(vb)))
            if e is None or (e[0] is not None and (e[0]() is not kb or e[1]() is not vb)):
                # per batch-layer window buffers: layout check + row bases
                if not (kb.dtype == vb.dtype == bf and kb.shape[2] == H and kb.shape[3] == D
                        and kb.shape == vb.shape and kb.is_contiguous() and vb.is_contiguous()):
                    return False
                keep = memo is not None
                e = mw[(id(kb), id(vb))] = (wr(kb) if keep else None, wr(vb) if keep else None,
                                            kb.data_ptr(), vb.data_ptr(), kb.shape[1] * H * D * 2,
                                            kb.shape[1], kb.shape[0])
            kbp, vbp, kr, cap, S = e[2:]
            lg = mp.get((id(pk), id(pv)))
            if lg is None or (lg[0] is not None and (lg[0]() is not pk or lg[1]() is not pv)):
                # packed rectangle (or one _RowStore row): layout check + bases
                if not (pk.dtype == pv.dtype == i64 and pk.shape[1] == pv.shape[1] == H
                        and pk.shape[1] * pk.shape[2] == HKW and pk.is_contiguous()
                        and pv.is_contiguous()):
                    return False
                L, w = pk.shape[-1], pv.shape[2] * pv.shape[3]
                keep = memo is not None and pk.shape[0] == S  # a batch rectangle, not a row
                lg = (wr(pk) if keep else None, wr(pv) if keep else None,
                      pk.data_ptr(), pv.data_ptr(), L, w, HKW * L * 8, H * w * 8,
                      pv.shape[2], pv.shape[3], pk.shape[0] > 1)
                if keep or memo is None:
                    mp[(id(pk), id(pv))] = lg
            pkp, pvp, L, w, rk, rv, G, CV, multi = lg[2:]
            pi = i if multi else 0  # one-row pk / pv: the row's storage
            n, Lr, wr_ = g, L, w  # packed key: source / destination widths
            if c is not None:
                cpk, cpv = c.packed_key, c.packed_value
                _, h1, h2, ck = cpk.shape
                _, _, cg, cc = cpv.shape
                if not (ck == g <= L and h1 * h2 == HKW and cc == CV and cg <= G):
                    return False
                ps, vs, nv = cpk.data_ptr(), cpv.data_ptr(), cg * cc
                if not (cpk.is_contiguous() and cpv.is_contiguous()):
                    # PICOKV_PREALLOC prefix views: their whole buffers
                    # (zero past the content) at the buffers' widths
                    kf, vf = _pk_full(cpk), _pk_full(cpv)
                    if kf is None or vf is None or kf.data_ptr() != ps or vf.data_ptr() != vs:
                        return False
                    n, nv = int(kf.shape[-1]), int(vf.shape[2]) * cc
                if ps == pkp + pi * rk and vs == pvp + pi * rv:
                    n = nv = Lr = wr_ = 0  # adopted buffer: the codes are in place
            else:
                ps = vs = nv = n = 0
            _, kh, kd, _ = k.shape
            _, vh, vd, _ = v.shape
            _, k1, k2, k3 = k.stride()
            _, v1, v2, v3 = v.stride()
            if not (d <= cap and kd >= d and vd >= d and kh == vh == H and k3 == v3 == 1
                    and k.dtype is bf and v.dtype is bf):
                return False
            ap((ps, vs, k.data_ptr(), v.data_ptr(), pkp + pi * rk, pvp + pi * rv,
                kbp + i * kr, vbp + i * kr, n, nv, d, Lr, wr_, k2, k1, v2, v1))
        if memo is not None and len(mp) + len(mw) > 512:  # drop dead entries
            for m in (mp, mw):
                for x in [x for x, e in m.items() if e[0] is not None and (
                        e[0]() is None or e[1]() is None)]:
                    del m[x]
        # array -> pinned: half the host time of torch.tensor(list) at 1k+ ints
        tt = torch.frombuffer(array.array("q", tab), dtype=i64).pin_memory().to(
            jobs[0][2].device, non_blocking=True)
        _fill_rows_kernel[(len(jobs), HKW + 3 * H)](tt, HKW=HKW, H=H, D=D, BLOCK=1024, BT=16)
        return True

    def _adoptable(self, c, need, dtype, H, KW, CV):
        """PICOKV_PREALLOC (dedup): the full-capacity buffers (k, v) of a
        fresh row's request cache ``c`` if they can be the row's decode
        storage as they are (capacity >= ``need``, the batch's layout), else
        None. The dedup that follows drops the cache's views of them."""
        if not self._prealloc or c is None:
            return None
        kf, vf = _pk_full(c.packed_key), _pk_full(c.packed_value)
        if (kf is None or vf is None or kf.shape[-1] < need
                or kf.data_ptr() != c.packed_key.data_ptr()
                or vf.data_ptr() != c.packed_value.data_ptr()
                or kf.dtype != dtype or vf.dtype != dtype
                or tuple(kf.shape[1:3]) != (H, KW) or tuple(vf.shape[1::2]) != (H, CV)
                or vf.shape[2] * 64 != kf.shape[-1]):
            return None
        return kf, vf

    @staticmethod
    def _row_src(fx, j):
        """Row j's packed key / value in a batch layer: [1, H, KW, w] / [1, H,
        w/64, CV] (w: its storage width; None: a vacant _RowStore row)."""
        st = fx.get("store")
        if st is not None:
            return st.k[j], st.v[j]
        b = fx["batched"]
        return b.packed_key[j:j + 1], b.packed_value[j:j + 1]

    def _row_copy(self, pairs):
        """Packed rows into other storage: pairs [((src k, src v), (dst k, dst
        v))] of [1, H, KW, w] / [1, H, w/64, CV] rows, any widths; the rows are
        zero past gl <= both widths, so the prefix of the narrower one is
        copied (dst zeroed). One fill launch (PICOKV_FILL_KERNEL), else
        per-row copies."""
        pairs = [(s, d) for s, d in pairs if s[0] is not None]
        if not pairs:
            return
        if self._fill_k and all(t.is_contiguous() for s, d in pairs for t in s + d):
            tab = []
            for (sk, sv), (dk, dv) in pairs:
                tab.extend((sk.data_ptr(), sv.data_ptr(), 0, 0, dk.data_ptr(), dv.data_ptr(),
                            0, 0, sk.shape[-1], sv.shape[2] * sv.shape[3], 0,
                            dk.shape[-1], dv.shape[2] * dv.shape[3], 0, 0, 0, 0))
            k0 = pairs[0][1][0]
            H, HKW = k0.shape[1], k0.shape[1] * k0.shape[2]
            tt = torch.frombuffer(array.array("q", tab), dtype=torch.int64).pin_memory().to(
                k0.device, non_blocking=True)
            _fill_rows_kernel[(len(pairs), HKW + 3 * H)](tt, HKW=HKW, H=H, D=self.head_dim,
                                                          BLOCK=1024, BT=16)
            return
        for (sk, sv), (dk, dv) in pairs:
            n, g = min(sk.shape[-1], dk.shape[-1]), min(sv.shape[2], dv.shape[2])
            dk[..., :n] = sk[..., :n]
            dv[:, :, :g] = sv[:, :, :g]

    def _rows_stub(self, H, KW, CV, dtype, dev):
        """Shape-only packed_key / packed_value of a _RowStore batch (the tail
        op reads the rows through the tables); one shared pair."""
        key = (H, KW, CV, dtype, str(dev))
        s = self._row_stub.get(key)
        if s is None:
            s = self._row_stub[key] = (torch.zeros(1, H, KW, 512, dtype=dtype, device=dev),
                                       torch.zeros(1, H, 8, CV, dtype=dtype, device=dev))
        return s

    def _sync_rows(self, fx, layer_id, rows, lens):
        """Fast build: rows [(rid, j)] leave the batch whose layer tensors are
        ``fx``: copy each one's global segment + dense window back into its
        request cache (compact copies — nothing pins the batch tensors). A
        dense-only row whose window slid is promoted to a compressed layer."""
        for rid, j in rows:
            rc = self._reqs.get(rid)
            if rc is None:
                continue  # finished (pruned)
            g, ln = fx["gl_py"][j], int(lens[j])
            k = fx["kbuf"][j, :ln].transpose(0, 1).unsqueeze(0).contiguous()  # [1,KVH,ln,D]
            v = fx["vbuf"][j, :ln].transpose(0, 1).unsqueeze(0).contiguous()
            c = rc.layers.get(layer_id)
            if c is None:
                if g == 0:
                    rc._dense[layer_id] = (k, v)
                    continue
                rc._dense.pop(layer_id, None)
                kp, vp, kpt = self._gl0_projections(
                    layer_id, int(k.shape[1]), int(k.shape[-1]), k.device, k.dtype)
                c = PreparedKVCache(dense_key=k, dense_value=v, key_projection=kp,
                                    value_projection=vp, key_projection_t=kpt,
                                    packed_key=None, packed_value=None, exp_bits=fx["exp_bits"])
                rc.layers[layer_id] = c
            c.dense_key, c.dense_value = k, v
            pk, pv = self._row_src(fx, j)
            c.packed_key = pk[..., :g].contiguous()
            c.packed_value = pv[:, :, :(g + 63) // 64].contiguous()

    def _fixed_layer_fast(self, rids, layer_id):
        """Incremental _fixed_layer (slide): the same per-layer batch state,
        built on device from the stashed previous batch. Carried rows are
        copied run-wise from its tensors (already zero past gl, windows on the
        inherited epoch clock -> no catch-up); only fresh rows read their
        request cache (and catch up past sink+local); stashed rows this batch
        drops are synced back first. No per-row host work for carried rows,
        no stream sync. Slot mode (``self._batch["S"]``): S rows, the rows
        past ``rids`` vacant (gl=0, empty window)."""
        import torch as _t
        if self._fast_build2:
            fx = self._fixed_layer_fast2(rids, layer_id)
            if fx is not None:
                return fx
        st = self._pools
        ofx = st["fixed"].pop(layer_id, None) if st is not None else None
        carry = st["idx"] if ofx is not None else {}
        if ofx is not None:
            live = set(rids)
            gone = [(r, j) for r, j in carry.items() if r not in live]
            if gone:
                self._sync_rows(ofx, layer_id, gone, st["lens"])
        S = self._batch.get("S") if self._batch is not None else None
        B, W = S or len(rids), self._batch["W"]
        heads, dim = self.num_kv_heads, self.head_dim
        gls, dlens, ci, cj, fresh = [], [], [], [], []
        for i, rid in enumerate(rids):
            j = carry.get(rid)
            if j is not None:
                gls.append(ofx["gl_py"][j])
                dlens.append(int(st["lens"][j]))
                ci.append(i)
                cj.append(j)
                continue
            rc = self._reqs.get(rid)
            if rc is None:
                raise KeyError(f"decode for a request with no PiCoKV cache: {rid}")
            c = rc.layers.get(layer_id)
            if c is not None:
                src = (c.dense_key, c.dense_value)
            elif layer_id in rc._dense:
                src = rc._dense[layer_id]
            else:
                raise KeyError(f"request {rid} has no PiCoKV cache for layer {layer_id}")
            if src is None or src[0] is None or (c is not None and c.packed_key is None):
                raise RuntimeError(f"PiCoKV request {rid} layer {layer_id}: cache held by a "
                                   "decode batch that was never synced back")
            gls.append(int(c.packed_key.shape[-1]) if c is not None else 0)
            dlens.append(int(src[0].shape[-2]))
            fresh.append((i, c, src))
        nvac = B - len(rids)
        gls += [0] * nvac
        dlens += [0] * nvac
        comp = [c for _, c, _ in fresh if c is not None]
        if ofx is not None:
            opk, opv = ofx["batched"].packed_key, ofx["batched"].packed_value
            dev, dt = ofx["kbuf"].device, ofx["kbuf"].dtype
            KW, CV, pdt, exp_bits = int(opk.shape[2]), int(opv.shape[-1]), opk.dtype, ofx["exp_bits"]
        else:
            dev, dt = fresh[0][2][0].device, fresh[0][2][0].dtype
            if comp:
                c0 = comp[0]
                KW, CV = int(c0.packed_key.shape[2]), int(c0.packed_value.shape[-1])
                pdt, exp_bits = c0.packed_key.dtype, c0.exp_bits
            else:
                KW, CV = int(self._scfg.key_bits) // 64, int(self._scfg.value_bits)
                pdt, exp_bits = _t.int64, int(self._scfg.exp_bits)
        # geometry exactly as _fixed_layer (headroom for the next merge)
        need = [g + max(0, d - W) for g, d in zip(gls, dlens)]
        L = _round_up(max(need) + self._slide_head, 512) or 512
        rows = self._rows and (S or len(rids)) > 1
        if S and nvac and not rows and not self._batch["fixed"] and (
                nvac * L * self._n_layers * self._tok_bytes(heads, KW, CV, pdt) > self._slot_waste):
            # vacant rows too costly at this width (long context): unslotted
            # (per-row storage: vacant rows hold nothing)
            self._batch["S"] = S = None
            B, nvac = len(rids), 0
            gls, dlens = gls[:B], dlens[:B]
            need = need[:B]
        # per-row storage: each row at its own capacity, no rectangle to fit
        rowcap = [max(512, _round_up(nd + self._slide_head, 512)) if rows else
                  min(L, max(512, _round_up(nd + self._slide_head, 512)))
                  for nd in need[:len(rids)]] + [0] * nvac
        gmax = (L + 63) // 64
        cap = max(W + _SLIDE_S, max(dlens))
        store = None
        if rows:
            tabs = self._batch.get("rtab")
            if tabs is None:  # the batch's layers' tables: one block (_RowTabs)
                tabs = self._batch["rtab"] = _RowTabs(self._n_layers, B, dev)
            store = _RowStore(B, heads, KW, CV, pdt, dev, tabs=tabs)
            pk, pv = self._rows_stub(heads, KW, CV, pdt, dev)
            ost = ofx.get("store") if ofx is not None else None
            kept = set()
            if ost is not None:  # carried rows keep their storage: nothing copied
                for i, j in zip(ci, cj):
                    if ost.k[j] is not None and ost.k[j].shape[-1] >= need[i]:
                        store.set(i, ost.k[j], ost.v[j])
                        rowcap[i] = int(ost.k[j].shape[-1])
                        kept.add(i)
            for i, c, _ in fresh:  # PICOKV_PREALLOC: the request's buffer is the row
                a = self._adoptable(c, need[i], pdt, heads, KW, CV)
                if a is not None:
                    store.set(i, *a)
                    rowcap[i] = int(a[0].shape[-1])
                    kept.add(i)
            new = [i for i in range(len(rids)) if i not in kept]
            store.alloc(new, [rowcap[i] for i in new])
            store.sync()
            self._row_copy([(self._row_src(ofx, j), (store.k[i], store.v[i]))
                            for i, j in zip(ci, cj) if i not in kept])
        else:
            a = (self._adoptable(fresh[0][1], need[0], pdt, heads, KW, CV)
                 if B == 1 and fresh else None)
            if a is not None and a[0].shape[-1] == L:  # PICOKV_PREALLOC: one row
                pk, pv = a
            else:
                pk = _t.zeros(B, heads, KW, L, dtype=pdt, device=dev)
                pv = _t.zeros(B, heads, gmax, CV, dtype=pdt, device=dev)
        kbuf = _t.zeros(B, cap, heads, dim, device=dev, dtype=dt)
        vbuf = _t.zeros_like(kbuf)
        if ci and not rows and ofx.get("store") is not None:
            # per-row storage -> rectangle (e.g. down to one row): rows copied
            self._row_copy([(self._row_src(ofx, j), (pk[i:i + 1], pv[i:i + 1]))
                            for i, j in zip(ci, cj)])
            rows_done = True
        else:
            rows_done = rows
        if ci:
            okb, ovb = ofx["kbuf"], ofx["vbuf"]
            Lm, Gm, cm = min(L, opk.shape[-1]), min(gmax, opv.shape[2]), min(cap, okb.shape[1])
            runs = []  # maximal [i0, j0, n] runs consecutive in both batches
            for i, j in zip(ci, cj):
                if runs and runs[-1][0] + runs[-1][2] == i and runs[-1][1] + runs[-1][2] == j:
                    runs[-1][2] += 1
                else:
                    runs.append([i, j, 1])
            if len(runs) <= 4:
                for i0, j0, n in runs:
                    if not rows_done:
                        pk[i0:i0 + n, ..., :Lm] = opk[j0:j0 + n, ..., :Lm]
                        pv[i0:i0 + n, :, :Gm] = opv[j0:j0 + n, :, :Gm]
                    kbuf[i0:i0 + n, :cm] = okb[j0:j0 + n, :cm]
                    vbuf[i0:i0 + n, :cm] = ovb[j0:j0 + n, :cm]
            else:
                it, jt = self._h2d(ci, dev, _t.int64), self._h2d(cj, dev, _t.int64)
                if not rows_done:
                    pk[it, ..., :Lm] = opk[jt, ..., :Lm]
                    pv[it, :, :Gm] = opv[jt, :, :Gm]
                kbuf[it, :cm] = okb[jt, :cm]
                vbuf[it, :cm] = ovb[jt, :cm]
        if not self._fill_fresh([((store.k[i], store.v[i]) if rows else (pk, pv))
                                 + (kbuf, vbuf, i, c, src, gls[i], dlens[i])
                                 for i, c, src in fresh]):
            for i, c, src in fresh:
                if c is not None:
                    rk, rv = (store.k[i], store.v[i]) if rows else (pk[i:i + 1], pv[i:i + 1])
                    if rk.data_ptr() != c.packed_key.data_ptr():  # else adopted: in place
                        rk[..., :gls[i]] = c.packed_key
                        rv[:, :, :c.packed_value.shape[-2]] = c.packed_value
                kbuf[i, :dlens[i]] = src[0][0].transpose(0, 1)  # [len,KVH,D]
                vbuf[i, :dlens[i]] = src[1][0].transpose(0, 1)
        late = {i for i, _, _ in fresh if dlens[i] > W}
        if ci and st.get("W", W) != W:
            # carried from a batch with another window: past this W they would
            # break the boundary invariant (a sliding row enters at exactly W + n)
            late.update(i for i in ci if dlens[i] > W)
        if late:  # written-back rows past sink+local: compress down to <= W
            self._slide_catchup(layer_id, kbuf, vbuf, pk, pv, gls, dlens, rows=late, store=store)
        if kbuf.shape[1] > W + _SLIDE_S:
            # caught up: every window is <= W + one epoch's appends; a longer
            # carried window (e.g. from a <= PICOKV_DENSE_LOCAL_MAX_BS batch)
            # would pin B x its length for the batch's life
            # (picokv_fixed_reserve_bytes assumes B x (W + 64))
            kbuf = kbuf[:, :W + _SLIDE_S].contiguous()
            vbuf = vbuf[:, :W + _SLIDE_S].contiguous()
        ref = self._gl0_projections(layer_id, heads, dim, dev, dt)
        if comp:
            self._check_proj(layer_id, comp[0], ref)
        rep = self._proj_rep.get(layer_id)
        if rep is None or rep[0].shape[0] < B:
            nb = _round_up(B, 8)
            rep = tuple(r.expand(nb, *r.shape[1:]) if self._shared_proj
                        else r.expand(nb, *r.shape[1:]).contiguous() for r in ref)
            self._proj_rep[layer_id] = rep
        if self._dedup:
            # The batch is now these rows' only copy: drop the request-cache
            # originals (_sync_rows restores them when the row leaves a batch).
            for i, c, _ in fresh:
                if c is not None:
                    c.packed_key = c.packed_value = c.dense_key = c.dense_value = None
                    c.key_projection, c.value_projection, c.key_projection_t = ref
                else:
                    self._reqs[rids[i]]._dense[layer_id] = None
        # Device tables: identical for every layer of the batch (eas=0 -> same
        # gl per row), so built once; global_lens grows in place per layer.
        # Slot mode: one fixed-length schedule rewritten in place (_slot_sched).
        if S:
            sr_t, st_t = self._slot_sched(rowcap, max(L, max(rowcap)))
            gl_t = self._h2d(gls, dev)
        else:
            tkey = (tuple(gls), tuple(rowcap))
            tab = self._batch.get("tabs") if self._batch is not None else None
            if tab is None or tab[0] != tkey:
                sr, stl = self._sched_lists(rowcap)
                tab = (tkey, self._h2d(gls, dev), self._h2d(sr, dev), self._h2d(stl, dev))
                if self._batch is not None:
                    self._batch["tabs"] = tab
            sr_t, st_t, gl_t = tab[2], tab[3], tab[1].clone()
        batched = PreparedKVCache(
            dense_key=kbuf.permute(0, 2, 1, 3), dense_value=vbuf.permute(0, 2, 1, 3),
            key_projection=rep[0][:B], value_projection=rep[1][:B],
            key_projection_t=rep[2][:B], packed_key=pk, packed_value=pv, exp_bits=exp_bits,
            row_tables=store.tables if rows else None)
        return dict(
            batched=batched, kbuf=kbuf, vbuf=vbuf, exp_bits=exp_bits,
            global_lens=gl_t, gl_max=max(gls),
            sched_req=sr_t, sched_tile=st_t, dlens0=dlens,
            equal=False, dlens_equal=False, slide=True,
            gl_py=gls, rowcap=rowcap, kst=None, vst=None, bnd_done=-1,
            slot_k=(2 * layer_id) % _SLIDE_S, slot_v=(2 * layer_id + 1) % _SLIDE_S,
            store=store)

    def _carry_plan(self, rids, st):
        """PICOKV_FAST_BUILD2: the build's row mapping from the stash ``st``,
        the same for every layer (the stash's row index and window lens are
        per batch): new row i <- stash row j (ci / cj), fresh rows, rows
        leaving, the copy runs; device index tensors and the geometry / gl
        memos filled by the first layer that needs them."""
        b = self._batch
        cp = b.get("cplan")
        if cp is not None and cp["st"] is st and (cp["rids"] is rids or cp["rids"] == rids):
            return cp
        carry, lens = st["idx"], st["lens"]
        ci, cj, frow = [], [], []
        for i, rid in enumerate(rids):
            j = carry.get(rid)
            if j is None:
                frow.append(i)
            else:
                ci.append(i)
                cj.append(j)
        live = set(rids)
        runs = []  # maximal [i0, j0, n] runs consecutive in both batches
        for i, j in zip(ci, cj):
            if runs and runs[-1][0] + runs[-1][2] == i and runs[-1][1] + runs[-1][2] == j:
                runs[-1][2] += 1
            else:
                runs.append([i, j, 1])
        cp = b["cplan"] = dict(
            st=st, rids=rids, ci=ci, cj=cj, frow=frow, runs=runs,
            gone=[(r, j) for r, j in carry.items() if r not in live],
            dl=[int(lens[j]) for j in cj],
            ig=operator.itemgetter(*cj) if len(cj) > 1 else None,
            # the carried rows are new rows 0..len(ci)-1 (ci is increasing)
            prefix=bool(ci) and ci[-1] == len(ci) - 1,
            ci_np=np.asarray(ci, dtype=np.int64), cj_np=np.asarray(cj, dtype=np.int64),
            dev={}, geo=None, gl=None)
        return cp

    def _fixed_layer_fast2(self, rids, layer_id):
        """PICOKV_FAST_BUILD2: _fixed_layer_fast of a layer the stash holds
        (None for any other: the caller builds it) — the same tensors and
        host state with the per-build work done once (_carry_plan; geometry
        and gl table memoized across layers: eas=0 -> every layer has the
        same per-row gl and lens), the kept rows' storage handed over as
        table rows in numpy, and the window buffers written once: carried
        rows gathered straight into them (index_select, no gathered temp),
        only the other rows zeroed."""
        import torch as _t
        st = self._pools
        if st is None or self._batch is None or layer_id not in st["fixed"]:
            return None
        cp = self._carry_plan(rids, st)
        ofx = st["fixed"].pop(layer_id)
        if cp["gone"]:
            self._sync_rows(ofx, layer_id, cp["gone"], st["lens"])
        S = self._batch.get("S")
        B, W = S or len(rids), self._batch["W"]
        heads, dim = self.num_kv_heads, self.head_dim
        ci, cj, gl_o, fresh = cp["ci"], cp["cj"], ofx["gl_py"], []
        if not cp["frow"]:
            gls = list(cp["ig"](gl_o)) if cp["ig"] is not None else [gl_o[cj[0]]]
            dlens = list(cp["dl"])
        else:
            gls, dlens = [0] * len(rids), [0] * len(rids)
            for i, j, d in zip(ci, cj, cp["dl"]):
                gls[i], dlens[i] = gl_o[j], d
            for i in cp["frow"]:
                rid = rids[i]
                rc = self._reqs.get(rid)
                if rc is None:
                    raise KeyError(f"decode for a request with no PiCoKV cache: {rid}")
                c = rc.layers.get(layer_id)
                if c is not None:
                    src = (c.dense_key, c.dense_value)
                elif layer_id in rc._dense:
                    src = rc._dense[layer_id]
                else:
                    raise KeyError(f"request {rid} has no PiCoKV cache for layer {layer_id}")
                if src is None or src[0] is None or (c is not None and c.packed_key is None):
                    raise RuntimeError(f"PiCoKV request {rid} layer {layer_id}: cache held by a "
                                       "decode batch that was never synced back")
                gls[i] = int(c.packed_key.shape[-1]) if c is not None else 0
                dlens[i] = int(src[0].shape[-2])
                fresh.append((i, c, src))
        nvac = B - len(rids)
        if nvac:
            gls += [0] * nvac
            dlens += [0] * nvac
        comp = [c for _, c, _ in fresh if c is not None]
        opk, opv = ofx["batched"].packed_key, ofx["batched"].packed_value
        dev, dt = ofx["kbuf"].device, ofx["kbuf"].dtype
        KW, CV, pdt, exp_bits = int(opk.shape[2]), int(opv.shape[-1]), opk.dtype, ofx["exp_bits"]
        rows = self._rows and (S or len(rids)) > 1
        geo = cp["geo"]
        if geo is not None and geo[0] == W and geo[1] == S and geo[2] == gls and geo[3] == dlens:
            need, need_np, L, rc0, rc0_np = geo[4:]
        else:  # geometry exactly as _fixed_layer_fast
            need = [g + max(0, d - W) for g, d in zip(gls, dlens)]
            L = _round_up(max(need) + self._slide_head, 512) or 512
            if S and nvac and not rows and not self._batch["fixed"] and (
                    nvac * L * self._n_layers * self._tok_bytes(heads, KW, CV, pdt) > self._slot_waste):
                self._batch["S"] = S = None  # vacant rows too costly: unslotted
                B, nvac = len(rids), 0
                gls, dlens = gls[:B], dlens[:B]
                need = need[:B]
            rc0 = [max(512, _round_up(nd + self._slide_head, 512)) if rows else
                   min(L, max(512, _round_up(nd + self._slide_head, 512)))
                   for nd in need[:len(rids)]] + [0] * nvac
            need_np, rc0_np = np.asarray(need, dtype=np.int64), np.asarray(rc0, dtype=np.int64)
            cp["geo"] = (W, S, list(gls), list(dlens), need, need_np, L, rc0, rc0_np)
        gmax = (L + 63) // 64
        cap = max(W + _SLIDE_S, max(dlens))
        nr, nc = len(rids), len(ci)
        store = None
        if rows:
            tabs = self._batch.get("rtab")
            if tabs is None:  # the batch's layers' tables: one block (_RowTabs)
                tabs = self._batch["rtab"] = _RowTabs(self._n_layers, B, dev)
            store = _RowStore(B, heads, KW, CV, pdt, dev, tabs=tabs)
            pk, pv = self._rows_stub(heads, KW, CV, pdt, dev)
            ost = ofx.get("store")
            keep = np.zeros(B, dtype=bool)
            rcn = rc0_np.copy()
            if ost is not None and nc:
                # carried rows keep their storage (cap >= need; 0: vacant):
                # == store.set(i, ost.k[j], ost.v[j]) — the stash's mirrors
                # hold the rows' addresses and caps
                hc = ost._hc[cp["cj_np"]]
                ok = (hc > 0) & (hc >= need_np[cp["ci_np"]])
                ki, kj = cp["ci_np"][ok], cp["cj_np"][ok]
                if len(ki):
                    sk, sv, tk, tv = store.k, store.v, ost.k, ost.v
                    for i, j in zip(ki.tolist(), kj.tolist()):
                        sk[i], sv[i] = tk[j], tv[j]
                    store._h[:2, ki] = ost._h[:2, kj]
                    store._hc[ki] = hc[ok]
                    rcn[ki] = hc[ok]
                    keep[ki] = True
            for i, c, _ in fresh:  # PICOKV_PREALLOC: the request's buffer is the row
                a = self._adoptable(c, need[i], pdt, heads, KW, CV)
                if a is not None:
                    store.set(i, *a)
                    rcn[i] = int(a[0].shape[-1])
                    keep[i] = True
            rowcap = rcn.tolist()
            new = np.flatnonzero(~keep[:nr]).tolist()
            store.alloc(new, [rowcap[i] for i in new])
            store.sync()
            if nc and not keep[cp["ci_np"]].all():
                self._row_copy([(self._row_src(ofx, j), (store.k[i], store.v[i]))
                                for i, j in zip(ci, cj) if not keep[i]])
        else:
            a = (self._adoptable(fresh[0][1], need[0], pdt, heads, KW, CV)
                 if B == 1 and fresh else None)
            if a is not None and a[0].shape[-1] == L:  # PICOKV_PREALLOC: one row
                pk, pv = a
            else:
                pk = _t.zeros(B, heads, KW, L, dtype=pdt, device=dev)
                pv = _t.zeros(B, heads, gmax, CV, dtype=pdt, device=dev)
            rowcap = list(rc0)
        okb, ovb = ofx["kbuf"], ofx["vbuf"]
        runs = cp["runs"]
        # the carried rows are rows [0, nc) and fill them whole: only the rest
        # need zeros (a fresh row's fill writes its window, nothing past it)
        gather = nc and cp["prefix"] and okb.shape[1] == cap and okb.dtype == dt
        if gather:
            kbuf = _t.empty(B, cap, heads, dim, device=dev, dtype=dt)
            vbuf = _t.empty_like(kbuf)
            if B > nc:
                kbuf[nc:].zero_()
                vbuf[nc:].zero_()
        else:
            kbuf = _t.zeros(B, cap, heads, dim, device=dev, dtype=dt)
            vbuf = _t.zeros_like(kbuf)
        if nc and not rows and ofx.get("store") is not None:
            # per-row storage -> rectangle (e.g. down to one row): rows copied
            self._row_copy([(self._row_src(ofx, j), (pk[i:i + 1], pv[i:i + 1]))
                            for i, j in zip(ci, cj)])
            rows_done = True
        else:
            rows_done = rows
        if nc:
            Lm, Gm, cm = min(L, opk.shape[-1]), min(gmax, opv.shape[2]), min(cap, okb.shape[1])
            if len(runs) <= 4:
                for i0, j0, n in runs:
                    if not rows_done:
                        pk[i0:i0 + n, ..., :Lm] = opk[j0:j0 + n, ..., :Lm]
                        pv[i0:i0 + n, :, :Gm] = opv[j0:j0 + n, :, :Gm]
                    kbuf[i0:i0 + n, :cm] = okb[j0:j0 + n, :cm]
                    vbuf[i0:i0 + n, :cm] = ovb[j0:j0 + n, :cm]
            else:
                dv = cp["dev"]
                if "jt" not in dv:  # once per build
                    dv["it"], dv["jt"] = self._h2d(ci, dev, _t.int64), self._h2d(cj, dev, _t.int64)
                it, jt = dv["it"], dv["jt"]
                if not rows_done:
                    pk[it, ..., :Lm] = opk[jt, ..., :Lm]
                    pv[it, :, :Gm] = opv[jt, :, :Gm]
                if gather:  # rows [0, nc) <- stash rows cj, whole
                    _t.index_select(okb, 0, jt, out=kbuf[:nc])
                    _t.index_select(ovb, 0, jt, out=vbuf[:nc])
                else:
                    kbuf[it, :cm] = okb[jt, :cm]
                    vbuf[it, :cm] = ovb[jt, :cm]
        if not self._fill_fresh([((store.k[i], store.v[i]) if rows else (pk, pv))
                                 + (kbuf, vbuf, i, c, src, gls[i], dlens[i])
                                 for i, c, src in fresh]):
            for i, c, src in fresh:
                if c is not None:
                    rk, rv = (store.k[i], store.v[i]) if rows else (pk[i:i + 1], pv[i:i + 1])
                    if rk.data_ptr() != c.packed_key.data_ptr():  # else adopted: in place
                        rk[..., :gls[i]] = c.packed_key
                        rv[:, :, :c.packed_value.shape[-2]] = c.packed_value
                kbuf[i, :dlens[i]] = src[0][0].transpose(0, 1)  # [len,KVH,D]
                vbuf[i, :dlens[i]] = src[1][0].transpose(0, 1)
        late = {i for i, _, _ in fresh if dlens[i] > W}
        if nc and st.get("W", W) != W:
            late.update(i for i in ci if dlens[i] > W)  # carried from another window
        if late:  # written-back rows past sink+local: compress down to <= W
            self._slide_catchup(layer_id, kbuf, vbuf, pk, pv, gls, dlens, rows=late, store=store)
        if kbuf.shape[1] > W + _SLIDE_S:  # caught up: windows <= W + one epoch
            kbuf = kbuf[:, :W + _SLIDE_S].contiguous()
            vbuf = vbuf[:, :W + _SLIDE_S].contiguous()
        ref = self._gl0_projections(layer_id, heads, dim, dev, dt)
        if comp:
            self._check_proj(layer_id, comp[0], ref)
        rep = self._proj_rep.get(layer_id)
        if rep is None or rep[0].shape[0] < B:
            nb = _round_up(B, 8)
            rep = tuple(r.expand(nb, *r.shape[1:]) if self._shared_proj
                        else r.expand(nb, *r.shape[1:]).contiguous() for r in ref)
            self._proj_rep[layer_id] = rep
        if self._dedup:  # the batch is now these rows' only copy (_sync_rows restores)
            for i, c, _ in fresh:
                if c is not None:
                    c.packed_key = c.packed_value = c.dense_key = c.dense_value = None
                    c.key_projection, c.value_projection, c.key_projection_t = ref
                else:
                    self._reqs[rids[i]]._dense[layer_id] = None
        if S:
            sr_t, st_t = self._slot_sched(rowcap, max(L, max(rowcap)))
            g = cp["gl"]  # gl table: one H2D per build, a copy per layer (grows in place)
            if g is None or g[0] != gls:
                g = cp["gl"] = (list(gls), self._h2d(gls, dev))
            gl_t = g[1].clone()
        else:
            tkey = (tuple(gls), tuple(rowcap))
            tab = self._batch.get("tabs")
            if tab is None or tab[0] != tkey:
                sr, stl = self._sched_lists(rowcap)
                tab = (tkey, self._h2d(gls, dev), self._h2d(sr, dev), self._h2d(stl, dev))
                self._batch["tabs"] = tab
            sr_t, st_t, gl_t = tab[2], tab[3], tab[1].clone()
        batched = PreparedKVCache(
            dense_key=kbuf.permute(0, 2, 1, 3), dense_value=vbuf.permute(0, 2, 1, 3),
            key_projection=rep[0][:B], value_projection=rep[1][:B],
            key_projection_t=rep[2][:B], packed_key=pk, packed_value=pv, exp_bits=exp_bits,
            row_tables=store.tables if rows else None)
        return dict(
            batched=batched, kbuf=kbuf, vbuf=vbuf, exp_bits=exp_bits,
            global_lens=gl_t, gl_max=max(gls),
            sched_req=sr_t, sched_tile=st_t, dlens0=dlens,
            equal=False, dlens_equal=False, slide=True,
            gl_py=gls, rowcap=rowcap, kst=None, vst=None, bnd_done=-1,
            slot_k=(2 * layer_id) % _SLIDE_S, slot_v=(2 * layer_id + 1) % _SLIDE_S,
            store=store)
    @staticmethod
    def _packed_nbytes(b):
        """Bytes of a batch's batched packed codes over all layers."""
        n, held = 0, {}  # held: per-row buffers, each once (shared across layers)
        for fx in (b.get("fixed") or {}).values():
            if fx is None:
                continue
            bt = fx.get("batched")
            if fx.get("store") is not None:  # per-row storage (bt: shape stubs)
                fx["store"].nbytes(held)
                continue
            for t in ((bt.packed_key, bt.packed_value) if bt is not None else ()):
                if t is not None:
                    n += t.numel() * t.element_size()
        return n + sum(held.values())

    @staticmethod
    def _tok_bytes(heads, KW, CV, pdt):
        """Packed bytes per token-row per layer (keys KW words/token, values
        CV words per 64-token group)."""
        return heads * (KW + CV / 64.0) * pdt.itemsize

    def _check_proj(self, layer_id, c0, ref):
        """Fast build: a request's projections must BE the shared offline ones
        (checked once per layer)."""
        if layer_id in self._proj_ok:
            return
        for f, x, r in zip(("key_projection", "value_projection", "key_projection_t"),
                           (c0.key_projection, c0.value_projection, c0.key_projection_t), ref):
            if tuple(x.shape) != tuple(r.shape) or (
                    x.data_ptr() != r.data_ptr() and not torch.equal(x, r)):
                raise RuntimeError(f"PiCoKV fast build: request {f} differs from the "
                                   f"shared offline one at layer {layer_id}")
        self._proj_ok.add(layer_id)

    @staticmethod
    def _sched_lists(rowcap):
        """Tail-kernel block schedule: row b's tiles [0, rowcap[b]/512)."""
        sr, stl = [], []
        for b, c in enumerate(rowcap):
            sr.extend([b] * (c // 512))
            stl.extend(range(c // 512))
        return sr, stl

    # tile index of a padding schedule entry: tile*512 >= any gl -> the tail
    # kernels' varlen early exit (the block reads gl[0] and returns)
    _SCHED_PAD = 1 << 20

    def _slot_sched(self, rowcap, L, reset=False):
        """Slot mode: the batch's ONE tail schedule (every layer's sched_req /
        sched_tile). Fixed length (grid.x of the captured tail launch), so it
        is rewritten in place under the graphs: the rows' tiles, then padding
        entries (_SCHED_PAD -> early exit). Holds the per-row max capacity
        over layers — a superset is always valid (tiles past a row's gl
        exit) — unless ``reset`` (membership change: rows re-assigned and
        every layer's rowcap rewritten with it). Reallocated, dropping every
        captured graph, only when the tiles outgrow its length. Returns
        (sched_req, sched_tile)."""
        b = self._batch
        ss = b.get("ssched")
        if ss is not None and not reset and self._fast_build2 and ss["rowcap"] == rowcap:
            return ss["sr"], ss["st"]  # == the merge below (max(a, a) = a)
        if ss is not None and not reset:
            rowcap = [max(a, c) for a, c in zip(ss["rowcap"], rowcap)]
        if ss is not None and ss["rowcap"] == rowcap:
            return ss["sr"], ss["st"]
        nt = np.asarray(rowcap, dtype=np.int64) // 512
        n = int(nt.sum())
        if ss is not None and n <= ss["sr"].numel():
            # in place: the same entries as _sched_lists + padding, built in
            # numpy (one row-tile list of ~S x L/512 ints per change)
            m = ss["sr"].numel()
            sr = np.zeros(m, dtype=np.int32)
            stl = np.full(m, self._SCHED_PAD, dtype=np.int32)
            sr[:n] = np.repeat(np.arange(len(rowcap), dtype=np.int32), nt)
            stl[:n] = np.arange(n, dtype=np.int32) - np.repeat(
                (np.cumsum(nt) - nt).astype(np.int32), nt)
            for dst, x in ((ss["sr"], sr), (ss["st"], stl)):
                dst.copy_(torch.from_numpy(x).pin_memory(), non_blocking=True)
            ss["rowcap"] = rowcap
            return ss["sr"], ss["st"]
        sr, stl = self._sched_lists(rowcap)
        if ss is None or n > ss["sr"].numel():
            # room for every row at the full packed width, capped at 4 spare
            # tiles per row (a mixed long-context batch would otherwise launch
            # ~S x L/512 early-exit blocks per layer)
            S = len(rowcap)
            ncap = max(n, min(S * (L // 512), n + 4 * (S + 2)), 1)
            pad = ncap - n
            dev = self.device if ss is None else ss["sr"].device
            srt = self._h2d(sr + [0] * pad, dev)
            stt = self._h2d(stl + [self._SCHED_PAD] * pad, dev)
            if ss is not None:
                for fx in b["fixed"].values():
                    if fx is not None:
                        fx["sched_req"], fx["sched_tile"] = srt, stt
                self._graph_clear(b["g"])  # grid.x changed
            b["ssched"] = {"rowcap": rowcap, "sr": srt, "st": stt}
            return srt, stt

    @staticmethod
    def _h2d_into(dst, vals):
        """_h2d into an existing device tensor (stable address)."""
        dst.copy_(torch.tensor(vals, dtype=dst.dtype, pin_memory=True), non_blocking=True)

    def _slot_change(self, rids, S):
        """Slot mode (BCG): a membership change within the bucket keeps the
        live batch: its S-row tensors, tables, epoch clock and captured
        graphs. Rows leaving are synced back to their request caches;
        carried rows are compacted in place into the engine order (live rows
        first; sglang's filter fills each vacated row with one from past the
        new size (PICOKV_STABLE_ORDER, schedule_batch.py) and appends new
        requests, so a finish moves one row; the order-preserving filter
        shifts every row after it); fresh rows are filled from their request caches
        into the slots after them (packed rows zeroed first, since the merges
        need zero bits past gl; catch-up past sink+local as in the build);
        the per-row lens, write positions and tail schedule are rewritten in
        place. Staged quantize-ahead codes are dropped (the boundary
        re-quantizes). Returns False (the caller rebuilds) unless this is a
        stepped slot batch of bucket S on the window path whose packed width
        and pool fit every fresh row."""
        import torch as _t
        b = self._batch
        sh = b.get("shared") if b is not None else None
        if (b is None or b.get("S") != S or sh is None or not sh["win"] or len(rids) > S
                or not b["fixed"] or any(fx is None for fx in b["fixed"].values())):
            return False
        if self._slot_fast:
            r = self._slot_change_fast(b, sh, rids, S)
            if r is not None:
                return r
        old, W, nl = b["rids"], b["W"], len(rids)
        stale = b.get("stale") or ()  # slots of re-prefilled req ids: fresh
        pos = {r: j for j, r in enumerate(old) if r not in stale}
        cj = [pos.get(r) for r in rids]  # each new row's current slot (None: fresh)
        fresh = [i for i, j in enumerate(cj) if j is None]
        rcf = [self._reqs.get(rids[i]) for i in fresh]  # fresh rows' request caches
        if any(rc is None for rc in rcf):
            return False
        src = {}  # (row, layer) -> (cache, (dense k, dense v), gl, window len)
        for lid, fx in b["fixed"].items():
            L, cap = fx["batched"].packed_key.shape[-1], fx["kbuf"].shape[1]
            for i, rc in zip(fresh, rcf):
                c = rc.layers.get(lid)
                s = (c.dense_key, c.dense_value) if c is not None else rc._dense.get(lid)
                if s is None or s[0] is None or (c is not None and c.packed_key is None):
                    return False  # the rebuild raises the precise error
                g, d = (int(c.packed_key.shape[-1]) if c is not None else 0), int(s[0].shape[-2])
                if (fx.get("store") is None and g + max(0, d - W) > L) or d > cap:
                    return False
                src[(i, lid)] = (c, s, g, d)
            if fresh and lid not in self._proj_ok:  # raises before any mutation
                kb_ = fx["kbuf"]
                ref = self._gl0_projections(lid, kb_.shape[2], kb_.shape[3], kb_.device, kb_.dtype)
                for i in fresh:
                    if src[(i, lid)][0] is not None:
                        self._check_proj(lid, src[(i, lid)][0], ref)
        for i in fresh:  # one window len per row (carried rows share dl_old)
            if len({src[(i, lid)][3] for lid in b["fixed"]}) > 1:
                raise RuntimeError(f"PiCoKV slot change: request {rids[i]} window lens "
                                   "differ across layers")
        waste = 0  # packed bytes held past what a rebuild would allocate
        for lid, fx in b["fixed"].items():
            if fx.get("store") is not None:
                continue  # per-row storage: vacant / leaving slots hold nothing
            pk, pv = fx["batched"].packed_key, fx["batched"].packed_value
            L = pk.shape[-1]
            need = [fx["gl_py"][j] if j is not None else
                    src[(i, lid)][2] + max(0, src[(i, lid)][3] - W) for i, j in enumerate(cj)]
            Ln = _round_up(max(need) + self._slide_head, 512) or 512
            waste += (S * L - nl * Ln) * self._tok_bytes(
                pk.shape[1], pk.shape[2], pv.shape[-1], pk.dtype)
        if waste > self._slot_waste:
            return False
        # PICOKV_PREALLOC: fresh rows whose request buffers become their
        # per-row storage as they are ((row, layer) -> (k, v))
        ad = {}
        for lid, fx in (b["fixed"].items() if self._prealloc and fresh else ()):
            st = fx.get("store")
            for i in (fresh if st is not None else ()):
                c, _, g, d = src[(i, lid)]
                a = self._adoptable(c, g + max(0, d - W), st.dtype, st.H, st.KW, st.CV)
                if a is not None:
                    ad[(i, lid)] = a
        self._prune_finished(rids)
        live = set(rids)
        gone = [(r, j) for j, r in enumerate(old) if r not in live and r not in stale]
        moves = [(i, j) for i, j in enumerate(cj) if j is not None and i != j]
        dev = sh["wpos"].device
        # carried rows move as contiguous runs (a finished row shifts every
        # row after it by one): one memcpy-speed slice copy per run and
        # buffer instead of an element-wise index_select + index_copy_ pair
        # (~2x the GPU time; 160 rows: ~17 ms per drop of the first row)
        runs = []
        for i, j in moves:
            if runs and runs[-1][0] + runs[-1][2] == i and runs[-1][1] + runs[-1][2] == j:
                runs[-1][2] += 1
            else:
                runs.append([i, j, 1])
        if len(runs) > 8:
            runs = None
            it = self._h2d([i for i, _ in moves], dev, _t.int64)
            jt = self._h2d([j for _, j in moves], dev, _t.int64)
        # no row both written and read by the moves (hole fill: holes < nl <=
        # movers): every run copy at once, no gather-first clones
        disj = {i for i, _ in moves}.isdisjoint(j for _, j in moves)
        ft = None  # fresh row indices on device: only the copy fallback reads them
        vac = [0] * (S - nl)
        dl_old = sh["dlens_py"]
        dl0 = [dl_old[j] if j is not None else 0 for j in cj] + [1] * (S - nl)
        dl, rcs, Lmax, glt, gld, gls_ = None, [], 0, {}, [], []
        gsel = [S if j is None else j for j in cj]  # old slot; S: fresh (-> 0)
        pplan = (_RowStore.plan(cj, S)
                 if any(fx.get("store") is not None for fx in b["fixed"].values()) else None)

        def _move(lids):
            # gather first: source and destination rows overlap. Per-row
            # storage: carried rows' storage follows them to their new slots
            # (nothing copied), leaving / vacant slots drop theirs, fresh rows
            # get zeroed storage at their own capacity; tables in place (one
            # pinned staging block for the layers ``lids``).
            dsts, srcs, sts = [], [], []
            for lid in lids:
                fx = b["fixed"][lid]
                st, bt = fx.get("store"), fx["batched"]
                if moves:
                    for x in ((fx["kbuf"], fx["vbuf"]) if st is not None else
                              (bt.packed_key, bt.packed_value, fx["kbuf"], fx["vbuf"])):
                        if runs is None:
                            x.index_copy_(0, it, x.index_select(0, jt))
                        elif disj:
                            for i0, j0, n in runs:
                                dsts.append(x[i0:i0 + n])
                                srcs.append(x[j0:j0 + n])
                        else:  # source first: a run's rows overlap their targets
                            for i0, j0, n in runs:
                                x[i0:i0 + n].copy_(x[j0:j0 + n].clone())
                if st is not None:
                    sts.append((st, lid))
            if dsts:
                _t._foreach_copy_(dsts, srcs)
            if sts:  # every layer's mirrors at once (_RowTabs span), one H2D
                stl = [st for st, _ in sts]
                _RowStore.permute_all(stl, cj, S, pplan)
                for st, lid in sts:
                    for i in fresh:
                        a = ad.get((i, lid))
                        if a is not None:
                            st.set(i, *a)
                _RowStore.alloc_all(stl, fresh, [[None if (i, lid) in ad else max(
                    512, _round_up(src[(i, lid)][2] + max(0, src[(i, lid)][3] - W)
                                   + self._slide_head, 512)) for i in fresh]
                    for _, lid in sts])
                _RowStore.sync_all(stl)

        # PICOKV_FILL_KERNEL: per layer group, the leaving rows synced +
        # carried rows moved first, then ALL fresh rows of the group's layers
        # in one launch. Per-row storage: a layer's fresh rows get new storage
        # while their request caches still hold the codes (released by the
        # dedup below), so a group holds at most PICOKV_FILL_GROUP_MB of it
        # (one group = every layer: 16 fresh 120k rows needed +38 GB)
        pre = bool(fresh) and self._fill_k
        lids = list(b["fixed"])
        groups = [lids]
        if pre and any(fx.get("store") is not None for fx in b["fixed"].values()):
            st0 = next(fx["store"] for fx in b["fixed"].values() if fx.get("store") is not None)
            tb = self._tok_bytes(st0.H, st0.KW, st0.CV, st0.dtype)
            groups, acc = [[]], 0
            for lid in lids:
                n = tb * sum(max(512, _round_up(src[(i, lid)][2] + max(
                    0, src[(i, lid)][3] - W) + self._slide_head, 512)) for i in fresh
                    if (i, lid) not in ad)  # adopted rows allocate nothing
                if groups[-1] and acc + n > self._fill_group:
                    groups.append([])
                    acc = 0
                groups[-1].append(lid)
                acc += n
        for grp in groups:
            filled = False
            if gone:
                for lid in grp:
                    self._sync_rows(b["fixed"][lid], lid, gone, dl_old)
            _move(grp)
            if pre:
                jobs = []
                for lid in grp:
                    fx = b["fixed"][lid]
                    st, kv = fx.get("store"), (fx["kbuf"], fx["vbuf"])
                    if st is None:
                        hd = (fx["batched"].packed_key, fx["batched"].packed_value) + kv
                        jobs.extend(hd + (i,) + src[(i, lid)] for i in fresh)
                    else:
                        jobs.extend((st.k[i], st.v[i]) + kv + (i,) + src[(i, lid)] for i in fresh)
                filled = self._fill_fresh(jobs, b.setdefault("fgeo", {}) if self._slot_memo else None)
            for lid in grp:
                fx = b["fixed"][lid]
                bt, store = fx["batched"], fx.get("store")
                pk, pv, kbuf, vbuf = bt.packed_key, bt.packed_value, fx["kbuf"], fx["vbuf"]
                gls = list(map([*fx["gl_py"], 0].__getitem__, gsel)) + vac
                rowcap = list(map([*fx["rowcap"], 0].__getitem__, gsel)) + vac
                dls = list(dl0)
                if fresh:
                    L = pk.shape[-1]
                    if not filled and store is None:
                        if ft is None:
                            ft = self._h2d(fresh, dev, _t.int64)
                        pk.index_fill_(0, ft, 0)
                        pv.index_fill_(0, ft, 0)
                    late = set()
                    for i in fresh:
                        c, s, g, d = src[(i, lid)]
                        if not filled:
                            if c is not None and (i, lid) not in ad:  # adopted: in place
                                rk, rv = (store.k[i], store.v[i]) if store is not None else (
                                    pk[i:i + 1], pv[i:i + 1])
                                rk[..., :g] = c.packed_key
                                rv[:, :, :c.packed_value.shape[-2]] = c.packed_value
                            kbuf[i, :d] = s[0][0].transpose(0, 1)  # [len,KVH,D]
                            vbuf[i, :d] = s[1][0].transpose(0, 1)
                        gls[i], dls[i] = g, d
                        rowcap[i] = int(store.k[i].shape[-1]) if store is not None else min(
                            L, max(512, _round_up(g + max(0, d - W) + self._slide_head, 512)))
                        if d > W:
                            late.add(i)
                    if late:  # written-back rows past sink+local
                        self._slide_catchup(lid, kbuf, vbuf, pk, pv, gls, dls, rows=late,
                                            store=store)
                    ref = self._gl0_projections(lid, kbuf.shape[2], kbuf.shape[3], dev, kbuf.dtype)
                    for i in fresh:  # projections checked by the precondition
                        c = src[(i, lid)][0]
                        if self._dedup:  # the batch is now the row's only copy
                            if c is not None:
                                c.packed_key = c.packed_value = c.dense_key = c.dense_value = None
                                c.key_projection, c.value_projection, c.key_projection_t = ref
                            else:
                                self._reqs[rids[i]]._dense[lid] = None
                if dl is not None and dls != dl:
                    raise RuntimeError(f"PiCoKV slot change: layer {lid} window lens differ")
                dl = dls
                key = tuple(gls)
                if key not in glt:
                    glt[key] = self._h2d(gls, dev)
                gld.append(fx["global_lens"])
                gls_.append(glt[key])
                fx["gl_py"], fx["rowcap"], fx["gl_max"] = gls, rowcap, max(gls)
                fx["kst"] = fx["vst"] = None
                rcs.append(rowcap)
                Lmax = max(Lmax, max(rowcap) if store is not None else int(pk.shape[-1]))
        # per-row max capacity over layers
        if not rcs:
            rcm = [0] * S
        elif all(r == rcs[0] for r in rcs):  # the usual case: one capacity per row
            rcm = list(rcs[0])
        else:
            rcm = [max(t) for t in zip(*rcs)]
        return self._slot_finish(b, sh, rids, gld, gls_, dl, vac, rcm, Lmax)

    def _slot_finish(self, b, sh, rids, gld, gls_, dl, vac, rcm, Lmax):
        """_slot_change's device lens + row tables + tail schedule, once the
        layers' packed rows and windows are in place."""
        nl = len(rids)
        if gld:  # every layer's lens in one launch
            torch._foreach_copy_(gld, gls_)
        # previous-step convention (_step_begin advances live rows first):
        # wpos = window len incl. the last append - 1; vacant rows 0 / len 1
        lv = [1] * nl + vac
        w3 = sh.get("w3")
        if w3 is not None and sh["wpos"].shape[0] == len(dl):  # rows wpos / lpl / live
            z = [0] * (w3.shape[1] - len(dl))
            self._h2d_into(w3.view(-1), [d - 1 for d in dl] + z + dl + z + lv + z)
        else:
            self._h2d_into(sh["wpos"], [d - 1 for d in dl])
            self._h2d_into(sh["lpl"], dl)
            self._h2d_into(sh["live"], lv)
        sh["live_py"], sh["dlens_py"], sh["cur"] = lv, dl, max(dl)
        self._slot_sched(rcm, Lmax, reset=True)
        b["rids"] = rids
        b.pop("stale", None)
        g = b["g"]
        g["ncap"], g["free"], g["off"] = 0, 0, False  # capture budget per membership (as a rebuild)
        return True

    def _slot_change_fast(self, b, sh, rids, S):
        """PICOKV_SLOT_FAST: _slot_change of a batch whose layers all keep
        packed rectangles (no per-row storage) and whose fresh rows have the
        same (gl, window len) at every layer, within sink+local (no
        catch-up): the same checks, syncs, run copies, fill launch and table
        writes in the same order; a layer's per-row lists are copied from the
        last layer with the same old lists and packed width instead of
        rebuilt. None: not that case (nothing touched; the general path)."""
        fixed, nl = b["fixed"], len(rids)
        if not nl or any(fx.get("store") is not None for fx in fixed.values()):
            return None
        old, W, head = b["rids"], b["W"], self._slide_head
        stale = b.get("stale") or ()  # slots of re-prefilled req ids: fresh
        pos = {r: j for j, r in enumerate(old) if r not in stale}
        cj = [pos.get(r) for r in rids]  # each new row's current slot (None: fresh)
        fresh = [i for i, j in enumerate(cj) if j is None]
        rcf = [self._reqs.get(rids[i]) for i in fresh]
        if any(rc is None for rc in rcf):
            return False
        lay, gd0 = [], None  # per layer (id, fx, pk, pv, kbuf, vbuf, L, fresh rows)
        for lid, fx in fixed.items():
            bt, kb = fx["batched"], fx["kbuf"]
            pk = bt.packed_key
            L, cap = pk.shape[-1], kb.shape[1]
            rows, gd = [], []
            for i, rc in zip(fresh, rcf):
                c = rc.layers.get(lid)
                s = (c.dense_key, c.dense_value) if c is not None else rc._dense.get(lid)
                if s is None or s[0] is None or (c is not None and c.packed_key is None):
                    return False  # the rebuild raises the precise error
                g, d = (int(c.packed_key.shape[-1]) if c is not None else 0), int(s[0].shape[-2])
                if g + max(0, d - W) > L or d > cap:
                    return False
                rows.append((i, c, s, g, d))
                gd.append((g, d))
            if gd0 is None:
                if any(d > W for _, d in gd):
                    return None  # catch-up rows
                gd0 = gd
            elif gd != gd0:
                return None  # rows' lens differ across layers
            lay.append((lid, fx, pk, bt.packed_value, kb, fx["vbuf"], L, rows))
            if fresh and lid not in self._proj_ok:  # raises before any mutation
                ref = self._gl0_projections(lid, kb.shape[2], kb.shape[3], kb.device, kb.dtype)
                for r in rows:
                    if r[1] is not None:
                        self._check_proj(lid, r[1], ref)
        # packed bytes held past what a rebuild would allocate (max over the
        # carried rows' gl reused while the next layer's gl list is equal)
        fneed = max((g + max(0, d - W) for g, d in gd0), default=0)
        car = [j for j in cj if j is not None]
        waste, gpl, mc = 0, None, 0
        for lid, fx, pk, pv, kb, vb, L, rows in lay:
            gp = fx["gl_py"]
            if gpl is None or gp != gpl:
                gpl, mc = gp, max(map(gp.__getitem__, car), default=0)
            Ln = _round_up(max(mc, fneed) + head, 512) or 512
            waste += (S * L - nl * Ln) * self._tok_bytes(
                pk.shape[1], pk.shape[2], pv.shape[-1], pk.dtype)
        if waste > self._slot_waste:
            return False
        self._prune_finished(rids)
        live = set(rids)
        gone = [(r, j) for j, r in enumerate(old) if r not in live and r not in stale]
        moves = [(i, j) for i, j in enumerate(cj) if j is not None and i != j]
        dev = sh["wpos"].device
        runs = []  # carried rows as contiguous runs (_slot_change)
        for i, j in moves:
            if runs and runs[-1][0] + runs[-1][2] == i and runs[-1][1] + runs[-1][2] == j:
                runs[-1][2] += 1
            else:
                runs.append([i, j, 1])
        if len(runs) > 8:
            runs = None
            it = self._h2d([i for i, _ in moves], dev, torch.int64)
            jt = self._h2d([j for _, j in moves], dev, torch.int64)
        disj = {i for i, _ in moves}.isdisjoint(j for _, j in moves)
        dl_old = sh["dlens_py"]
        if gone:
            for lid, fx, *_ in lay:
                self._sync_rows(fx, lid, gone, dl_old)
        if moves:
            dsts, srcs = [], []
            for _, fx, pk, pv, kb, vb, *_ in lay:
                for x in (pk, pv, kb, vb):
                    if runs is None:
                        x.index_copy_(0, it, x.index_select(0, jt))
                    elif disj:
                        for i0, j0, n in runs:
                            dsts.append(x[i0:i0 + n])
                            srcs.append(x[j0:j0 + n])
                    else:  # source first: a run's rows overlap their targets
                        for i0, j0, n in runs:
                            x[i0:i0 + n].copy_(x[j0:j0 + n].clone())
            if dsts:
                torch._foreach_copy_(dsts, srcs)
        filled = False
        if fresh and self._fill_k:  # every fresh row of every layer: one launch
            jobs = []
            for _, fx, pk, pv, kb, vb, L, rows in lay:
                hd = (pk, pv, kb, vb)
                jobs.extend(hd + r for r in rows)
            filled = self._fill_fresh(jobs, b.setdefault("fgeo", {}) if self._slot_memo else None)
        vac = [0] * (S - nl)
        gsel = [S if j is None else j for j in cj]  # old slot; S: fresh (-> 0)
        dl = [dl_old[j] if j is not None else 0 for j in cj] + [1] * (S - nl)
        for i, (g, d) in zip(fresh, gd0):
            dl[i] = d
        ft, glt, gld, gls_, rcs, Lmax, sig, req = None, {}, [], [], [], 0, None, True
        for lid, fx, pk, pv, kbuf, vbuf, L, rows in lay:
            gp, rp = fx["gl_py"], fx["rowcap"]
            if sig is None or L != sig[0] or gp != sig[1] or rp != sig[2]:
                gls = list(map([*gp, 0].__getitem__, gsel)) + vac
                rowcap = list(map([*rp, 0].__getitem__, gsel)) + vac
                for i, (g, d) in zip(fresh, gd0):
                    gls[i] = g
                    rowcap[i] = min(L, max(512, _round_up(g + max(0, d - W) + head, 512)))
                key = tuple(gls)
                if key not in glt:
                    glt[key] = self._h2d(gls, dev)
                sig = (L, gp, rp, gls, rowcap, glt[key], max(gls))
                if rcs and rowcap != rcs[0]:
                    req = False
            else:  # the same lists as the last layer's (each layer its own: slides grow them)
                gls, rowcap = list(sig[3]), list(sig[4])
            if fresh:
                if not filled:
                    if ft is None:
                        ft = self._h2d(fresh, dev, torch.int64)
                    pk.index_fill_(0, ft, 0)
                    pv.index_fill_(0, ft, 0)
                    for i, c, s, g, d in rows:
                        if c is not None:
                            pk[i:i + 1][..., :g] = c.packed_key
                            pv[i:i + 1][:, :, :c.packed_value.shape[-2]] = c.packed_value
                        kbuf[i, :d] = s[0][0].transpose(0, 1)  # [len,KVH,D]
                        vbuf[i, :d] = s[1][0].transpose(0, 1)
                ref = self._gl0_projections(lid, kbuf.shape[2], kbuf.shape[3], dev, kbuf.dtype)
                if self._dedup:  # the batch is now the row's only copy
                    for i, c, *_ in rows:
                        if c is not None:
                            c.packed_key = c.packed_value = c.dense_key = c.dense_value = None
                            c.key_projection, c.value_projection, c.key_projection_t = ref
                        else:
                            self._reqs[rids[i]]._dense[lid] = None
            gld.append(fx["global_lens"])
            gls_.append(sig[5])
            fx["gl_py"], fx["rowcap"], fx["gl_max"] = gls, rowcap, sig[6]
            fx["kst"] = fx["vst"] = None
            rcs.append(rowcap)
            Lmax = max(Lmax, int(L))
        rcm = list(rcs[0]) if req else [max(t) for t in zip(*rcs)]
        return self._slot_finish(b, sh, rids, gld, gls_, dl, vac, rcm, Lmax)

    def _fixed_layer(self, rids, layer_id):
        """Stack the length-invariant + compressed-global tensors for one layer
        across the batch, padding packed_key/value to the batch's max global
        length (done ONCE per batch; these never change during decode), and
        build that layer's persistent token-major dense pool ``[B,cap,KVH,D]``
        (cap leaves headroom for generated tokens; doubled on overflow).
        Dense-only rows (prompt <= sink+local: no compressed global segment)
        join as gl=0 rows: the tail kernel runs no global block for them and
        their zero accumulators project to zero, so the output is exactly the
        dense-window attention. Every layer of every batch takes this path."""
        import torch as _t
        caches, srcs = [], []
        for rid in rids:
            rc = self._reqs.get(rid)
            if rc is None:
                raise KeyError(f"decode for a request with no PiCoKV cache: {rid}")
            c = rc.layers.get(layer_id)
            if c is not None:
                srcs.append((c.dense_key, c.dense_value))
            elif layer_id in rc._dense:
                srcs.append(rc._dense[layer_id])
            else:
                raise KeyError(f"request {rid} has no PiCoKV cache for layer {layer_id}")
            caches.append(c)  # None => dense-only row (gl = 0)
        comp = [c for c in caches if c is not None]
        dev, dt = srcs[0][0].device, srcs[0][0].dtype
        heads, dim = int(srcs[0][0].shape[1]), int(srcs[0][0].shape[-1])
        if comp:
            ref = (comp[0].key_projection, comp[0].value_projection, comp[0].key_projection_t)
        else:
            ref = self._gl0_projections(layer_id, heads, dim, dev, dt)
        for f, r in zip(("key_projection", "value_projection", "key_projection_t"), ref):
            if any(tuple(getattr(c, f).shape) != tuple(r.shape) for c in comp):
                raise RuntimeError(f"PiCoKV decode batch mixes {f} shapes at layer {layer_id}")
        gls = [int(c.packed_key.shape[-1]) if c is not None else 0 for c in caches]
        gvs = [int(c.packed_value.shape[-2]) if c is not None else 0 for c in caches]
        max_gl = max(gls)
        B = len(caches)
        # Seed lengths of each request's dense window. Requests carried over
        # from the last decode batch keep their appended tokens (their cache's
        # dense_key is the stale prefill window); freshly prefilled requests
        # seed from it.
        pools = self._pools
        carry = pools["idx"] if pools is not None else {}
        dlens = []
        for i, rid in enumerate(rids):
            j = carry.get(rid)
            dlens.append(int(pools["lens"][j]) if j is not None
                         else int(srcs[i][0].shape[-2]))
        slide = self._slide
        if slide:
            # Packed width with headroom for the tokens the window will drop:
            # this build's catch-up (dense tokens past sink+local) plus
            # PICOKV_SLIDE_HEADROOM per row. rowcap = each row's scheduled
            # tail blocks (x512 tokens); overflow regrows (_slide_regrow).
            W = self._batch["W"]
            need = [g + max(0, d - W) for g, d in zip(gls, dlens)]
            L = _round_up(max(need) + self._slide_head, 512) or 512
            rowcap = [min(L, max(512, _round_up(nd + self._slide_head, 512))) for nd in need]
        else:
            # Packed width = the batch's max global length; an all-dense batch
            # keeps one zero group so the tail launch stays well-formed
            # (global_lens=0 makes every block exit before reading it).
            L = max_gl if max_gl > 0 else 64
        gmax = (L + 63) // 64
        if comp:
            KW, CV = int(comp[0].packed_key.shape[2]), int(comp[0].packed_value.shape[-1])
            pdt = comp[0].packed_key.dtype
        else:
            KW, CV = int(self._scfg.key_bits) // 64, int(self._scfg.value_bits)
            pdt = _t.int64
        # Zero-padded batched copies, filled per request (no padded temporaries:
        # the build's transient peak stays at one layer's batched copy).
        pk = _t.zeros(B, heads, KW, L, dtype=pdt, device=dev)
        pv = _t.zeros(B, heads, gmax, CV, dtype=pdt, device=dev)
        for i, c in enumerate(caches):
            if c is not None:
                pk[i:i + 1, ..., :gls[i]] = c.packed_key
                pv[i:i + 1, ..., :gvs[i], :] = c.packed_value
        # Seed each request's dense window (carried rows from the stashed pool).
        if slide:
            cap = max(W + _SLIDE_S, max(dlens))  # window <= sink+local+64 once caught up
        else:
            cap = max(dlens) + 512
        kbuf = _t.zeros(B, cap, self.num_kv_heads, self.head_dim, device=dev, dtype=dt)
        vbuf = _t.zeros_like(kbuf)
        opool = pools["pools"].get(layer_id) if pools is not None else None
        for i in range(B):
            ln = dlens[i]
            j = carry.get(rids[i])
            if j is not None and opool is not None:
                kbuf[i, :ln] = opool[0][j, :ln]
                vbuf[i, :ln] = opool[1][j, :ln]
            else:
                kbuf[i, :ln] = srcs[i][0][0].transpose(0, 1)  # [len,KVH,D]
                vbuf[i, :ln] = srcs[i][1][0].transpose(0, 1)
        if slide:
            # rows past sink+local (carried mid-epoch / written back) compress
            # now, so every row starts the batch's first epoch at <= W
            self._slide_catchup(layer_id, kbuf, vbuf, pk, pv, gls, dlens)
            max_gl = max(gls)
        # Block schedule for the tail kernel's paged grid: map each grid.x block
        # -> (request, local tile) so ONLY real blocks launch. Block covers
        # BlockTiles(2)*kTileSize(256) = 512 global tokens. Slide: each row's
        # blocks cover its capacity (blocks past gl exit at once).
        BLK = 512
        sr, stl = [], []
        for b, g in enumerate(gls):
            nb = rowcap[b] // BLK if slide else (g + BLK - 1) // BLK  # gl=0: none
            sr.extend([b] * nb)
            stl.extend(range(nb))
        exp_bits = comp[0].exp_bits if comp else int(self._scfg.exp_bits)
        # Persistent batched cache: with the dense attention precomputed
        # (dense_attn) the dense_key/value views are never read, so bind the
        # FULL pool view once and keep the tail workspaces alive across steps
        # (the no-dense-attn fallback builds a temporary sliced cache instead).
        batched = PreparedKVCache(
            dense_key=kbuf.permute(0, 2, 1, 3),
            dense_value=vbuf.permute(0, 2, 1, 3),
            key_projection=_t.cat(
                [c.key_projection if c is not None else ref[0] for c in caches], 0).contiguous(),
            value_projection=_t.cat(
                [c.value_projection if c is not None else ref[1] for c in caches], 0).contiguous(),
            key_projection_t=_t.cat(
                [c.key_projection_t if c is not None else ref[2] for c in caches], 0).contiguous(),
            packed_key=pk,
            packed_value=pv,
            exp_bits=exp_bits,
        )
        if self._dedup:
            # Re-point each request at its slice of the batched copy so the
            # per-request originals are freed. Ragged packed slices are
            # non-contiguous; PicoKVRequestCache.decode re-materializes them
            # if the per-request path ever touches this layer again.
            for i, c in enumerate(caches):
                if c is None:
                    continue
                c.packed_key = pk[i:i + 1, ..., :gls[i]]
                ng = (gls[i] + 63) // 64 if slide else gvs[i]  # slide: post catch-up
                c.packed_value = pv[i:i + 1, ..., :ng, :]
                c.key_projection = batched.key_projection[i:i + 1]
                c.value_projection = batched.value_projection[i:i + 1]
                c.key_projection_t = batched.key_projection_t[i:i + 1]
        fx = dict(
            batched=batched,
            kbuf=kbuf,
            vbuf=vbuf,
            exp_bits=exp_bits,
            global_lens=_t.tensor(gls, dtype=_t.int32, device=dev),
            gl_max=max_gl,
            sched_req=_t.tensor(sr, dtype=_t.int32, device=dev),
            sched_tile=_t.tensor(stl, dtype=_t.int32, device=dev),
            dlens0=dlens,
            # uniform fast path (global_lens=None reads every row L wide);
            # slide: global lens grow per row -> always varlen
            equal=(not slide and len(set(gls)) == 1 and gls[0] == L and len(set(dlens)) == 1),
            dlens_equal=(not slide and len(set(dlens)) == 1),
            slide=slide,
        )
        if slide:
            fx.update(gl_py=gls, rowcap=rowcap, kst=None, vst=None, bnd_done=-1,
                      # quantize-ahead slots of this layer's key / value side
                      slot_k=(2 * layer_id) % _SLIDE_S, slot_v=(2 * layer_id + 1) % _SLIDE_S)
        return fx

    def _gl0_projections(self, layer_id, heads, dim, dev, dt):
        """Projection triple for a batch whose rows are ALL dense-only at this
        layer (no compressed row to borrow from). A gl=0 row never reads its
        projections into its output, so any well-formed tensors serve: the
        offline ones when loaded (the memoized tensors prepare itself uses),
        else zeros (online mode)."""
        key = (layer_id, heads, dim, dev, dt)
        hit = self._gl0_proj.get(key)
        if hit is None:
            kb, vb = int(self._scfg.key_bits), int(self._scfg.value_bits)
            off = self._offline
            if off is not None and _prepare_fixed_block32_cached is not None:
                kp = _prepare_fixed_block32_cached(
                    _slice_layer(off["key_decoder"], layer_id, heads), 1, heads, kb, dim, dev, dt
                ).view(1, heads, kb, dim)
                vp = _prepare_fixed_block32_cached(
                    _slice_layer(off["value_decoder"], layer_id, heads), 1, heads, vb, dim, dev, dt
                ).view(1, heads, vb, dim)
            else:
                kp = torch.zeros(1, heads, kb, dim, device=dev, dtype=dt)
                vp = torch.zeros(1, heads, vb, dim, device=dev, dtype=dt)
            hit = (kp, vp, kp.transpose(-1, -2).contiguous())
            self._gl0_proj[key] = hit
        return hit

    # ------------------------------------------------------------------
    # decode slide (P1-4): dense window -> compressed global every 64 steps
    # ------------------------------------------------------------------

    def _slide_projs(self, layer_id, heads, dim, dev, dt):
        """(enc_k, dec_k, enc_v, dec_v) of one layer: the memoized fixed
        projections prepare itself uses (same cascade -> same codes)."""
        hit = self._slide_proj.get(layer_id)
        if hit is None:
            off = self._offline
            kb, vb = int(self._scfg.key_bits), int(self._scfg.value_bits)

            def f(name, bits):
                return _prepare_fixed_block32_cached(
                    _slice_layer(off[name], layer_id, heads), 1, heads, bits, dim, dev, dt
                ).view(1, heads, bits, dim)

            hit = (f("key_encoder", kb), f("key_decoder", kb),
                   f("value_encoder", vb), f("value_decoder", vb))
            self._slide_proj[layer_id] = hit
        return hit

    def _slide_quant(self, layer_id, buf, key, P=1):
        """Quantize every row's staging slots [sink, sink+64) of one pool side
        in ONE fixed-cascade launch (codes are per-token, slice-independent):
        key -> [H, KW, B*64] packed columns, value -> [H, B*64, VW] words.
        P > 1 (catch-up): slots [sink, sink+64P) as P pass-major blocks of
        B*64 columns (pass p = tokens [sink+64p, sink+64p+64) of each row)."""
        B, _, H, D = buf.shape
        sink = int(self._scfg.sink_size)
        x = buf[:, sink:sink + P * _SLIDE_S].view(B, P, _SLIDE_S, H, D).permute(
            3, 1, 0, 2, 4).reshape(1, H, P * B * _SLIDE_S, D).contiguous()
        ek, dk, ev, dv = self._slide_projs(layer_id, H, D, buf.device, buf.dtype)
        if key:
            return blockk_fixed_cuda(
                x, ek, dk, 0, P * B * _SLIDE_S, zerorope=int(self._scfg.zerorope),
                apply_zerorope=True, pack_key_layout=True)[0][0]
        return blockk_fixed_cuda(
            x, ev, dv, 0, P * B * _SLIDE_S, zerorope=0,
            apply_zerorope=False, pack_key_layout=False)[0][0]

    def _batch_W(self, n):
        """Slide window (sink + local) of a new decode batch of n rows (slot
        mode: the bucket): PICOKV_SLIDE_LOCAL, or with the opt-in
        PICOKV_DENSE_LOCAL that local up to PICOKV_DENSE_LOCAL_MAX_BS rows.
        Fixed for the batch's life (an
        in-place slot change keeps it); a build at another W catches carried
        rows up (_fixed_layer_fast)."""
        loc = self._slide_local
        if self._dense_local and 0 < n <= self._dense_bs:
            loc = self._dense_local
        return int(self._scfg.sink_size) + loc

    def _slide_merge(self, kst, vst, pk, pv, gl_t, n_t, bs=None, store=None):
        """Append each row's first n staged tokens at its global offset gl.
        ``bs``: column count of the staging tensors when kst / vst are views
        at a pass offset of a larger catch-up quantization. ``store``: the
        rows' per-row storage (_RowStore; pk / pv are then shape stubs)."""
        VW = vst.shape[-1]
        if store is not None:
            kp, vp, ct = store.tables
            B, H = len(store.k), store.H
            _slide_kmerge_rows_kernel[(B * H,)](kst, kp, ct, gl_t, n_t, H, bs or kst.shape[-1],
                                                KB=store.KW, S=_SLIDE_S)
            _slide_vmerge_rows_kernel[(B * H, VW)](vst, vp, ct, gl_t, n_t, H, bs or vst.shape[1],
                                                   CV=store.CV, VB=VW, S=_SLIDE_S)
            return
        B, H, KW, L = pk.shape
        _slide_kmerge_kernel[(B * H,)](kst, pk, gl_t, n_t, H, bs or kst.shape[-1], L,
                                       KB=KW, S=_SLIDE_S)
        _slide_vmerge_kernel[(B * H, VW)](vst, pv, gl_t, n_t, H, bs or vst.shape[1], pv.shape[2],
                                          CV=pv.shape[3], VB=VW, S=_SLIDE_S)

    def _slide_shift(self, buf, n_t):
        """Drop each row's n outgoing tokens in place:
        [sink+n, sink+n+local) -> [sink, sink+local)."""
        row = buf.shape[2] * buf.shape[3]
        _slide_shift_kernel[(buf.shape[0], row // 128)](
            buf, n_t, buf.shape[1], ROWW=row, SINK=int(self._scfg.sink_size),
            LOCAL=self._batch["W"] - int(self._scfg.sink_size), T=32, DC=128)

    def _slide_catchup(self, layer_id, kbuf, vbuf, pk, pv, gls, dlens, rows=None, store=None):
        """Batch build: compress every row's dense tokens past sink+local
        (rows carried mid-epoch, written back, or from a batch with a longer
        window) with the epoch-boundary kernels, nothing moved in between:
        ONE quantization per side of the slots [sink, sink+64P) (P passes of
        <= 64 tokens per row), P merges, then ONE shift drops each row's
        whole excess E = d - W ([sink+E, d) -> [sink, W): exact, the shift
        takes any n per row). Updates gls / dlens in place. ``rows``: only
        these row indices (fast build: carried rows keep their epoch clock)."""
        W = self._batch["W"]
        dev = kbuf.device
        E = [max(0, d - W) if rows is None or i in rows else 0 for i, d in enumerate(dlens)]
        if not any(E):
            return
        P = (max(E) + _SLIDE_S - 1) // _SLIDE_S  # sink + 64P <= max(d): in bounds
        ns, gs = [], []
        for p in range(P):
            n = [min(_SLIDE_S, max(0, e - p * _SLIDE_S)) for e in E]
            ns.append(n)
            gs.append(list(gls))
            for i, m in enumerate(n):
                gls[i] += m
        n_t, gl_t = self._h2d(ns, dev), self._h2d(gs, dev)
        kq = self._slide_quant(layer_id, kbuf, True, P)   # [H, KW, P*B*64]
        vq = self._slide_quant(layer_id, vbuf, False, P)  # [H, P*B*64, VW]
        N, c = kq.shape[-1], kbuf.shape[0] * _SLIDE_S
        for p in range(P):
            self._slide_merge(kq[..., p * c:], vq[:, p * c:], pk, pv, gl_t[p], n_t[p], bs=N,
                              store=store)
        e_t = self._h2d(E, dev)
        self._slide_shift(kbuf, e_t)
        self._slide_shift(vbuf, e_t)
        for i, e in enumerate(E):
            dlens[i] -= e

    def _slide_layer(self, fx, sh, layer_id):
        """Per-layer slide work of this step, before its attention: at an
        epoch boundary merge the staged codes of the outgoing tokens into the
        packed global copy and shift the dense window (wpos/lpl were already
        rewound in _step_begin); then, on this layer side's slot, quantize the
        staging slots for the NEXT boundary (one fixed-cascade launch)."""
        step = sh["step"]
        if sh.get("bnd_step") == step and fx["bnd_done"] != step:
            fx["bnd_done"] = step
            n, n_t = sh["bnd_n"], sh["bnd_t"]
            gls = fx["gl_py"]
            if any(g + m > c for g, m, c in zip(gls, n, fx["rowcap"])):
                self._slide_regrow(fx, layer_id, n)
            kst = fx["kst"] if fx["kst"] is not None else self._slide_quant(layer_id, fx["kbuf"], True)
            vst = fx["vst"] if fx["vst"] is not None else self._slide_quant(layer_id, fx["vbuf"], False)
            b = fx["batched"]
            self._slide_merge(kst, vst, b.packed_key, b.packed_value, fx["global_lens"], n_t,
                              store=fx.get("store"))
            fx["global_lens"].add_(n_t)
            self._slide_shift(fx["kbuf"], n_t)
            self._slide_shift(fx["vbuf"], n_t)
            for i, m in enumerate(n):
                gls[i] += m
            fx["gl_max"] = max(gls)
            fx["kst"] = fx["vst"] = None
        ph = step % _SLIDE_S
        if ph == fx["slot_k"]:
            fx["kst"] = self._slide_quant(layer_id, fx["kbuf"], True)
        if ph == fx["slot_v"]:
            fx["vst"] = self._slide_quant(layer_id, fx["vbuf"], False)

    def _slide_regrow(self, fx, layer_id, n):
        """A row's global segment is about to pass its scheduled capacity:
        widen this layer's packed copy (zero-padded, contents kept), extend
        the overflowing rows' tail schedule, drop the layer's graphs. Slot
        mode (a long-lived batch): widen only when a row passes the packed
        width, by 2x headroom, and rewrite the shared schedule in place —
        a schedule-only regrow keeps the graphs. Per-row storage: only the
        overflowing rows move, into new storage (2x headroom in slot mode),
        tables rewritten in place — no widening of the others, no graph drop
        in slot mode."""
        import torch as _t
        b = fx["batched"]
        pk, pv = b.packed_key, b.packed_value
        need = [g + m for g, m in zip(fx["gl_py"], n)]
        slot = self._batch is not None and self._batch.get("S")
        st = fx.get("store")
        if st is not None:
            caps = st.caps()
            over = [i for i, (m, c) in enumerate(zip(need, caps)) if m > c]
            if over:
                hd = (2 if slot else 1) * self._slide_head
                old = [(st.k[i], st.v[i]) for i in over]
                st.alloc(over, [_round_up(need[i] + hd, 512) for i in over])
                self._row_copy(list(zip(old, [(st.k[i], st.v[i]) for i in over])))
                st.sync()
                del old
            fx["rowcap"] = st.caps()
            if slot:
                fx["sched_req"], fx["sched_tile"] = self._slot_sched(
                    fx["rowcap"], max(fx["rowcap"]))
                return
            sr, stl = self._sched_lists(fx["rowcap"])
            fx["sched_req"] = self._h2d(sr, pk.device)
            fx["sched_tile"] = self._h2d(stl, pk.device)
            self._graph_drop_layer(layer_id)
            return
        B, H, KW, L = pk.shape
        if slot:
            nl = L if max(need) <= L else _round_up(max(need) + 2 * self._slide_head, 512)
        else:
            nl = max(L, _round_up(max(need) + self._slide_head, 512))
        if nl > L:
            npk = _t.zeros(B, H, KW, nl, dtype=pk.dtype, device=pk.device)
            npv = _t.zeros(B, H, nl // 64, pv.shape[-1], dtype=pv.dtype, device=pv.device)
            npk[..., :L] = pk
            npv[:, :, :pv.shape[2]] = pv
            b.packed_key, b.packed_value = npk, npv
            if self._dedup and not self._fast:  # re-point the request views (else they pin the old copy)
                for i, rid in enumerate(self._batch["rids"]):
                    c = self._reqs[rid].layers.get(layer_id)
                    if c is not None:
                        g = fx["gl_py"][i]
                        c.packed_key = npk[i:i + 1, ..., :g]
                        c.packed_value = npv[i:i + 1, :, :(g + 63) // 64]
        fx["rowcap"] = [c if m <= c else min(nl, _round_up(m + self._slide_head, 512))
                        for c, m in zip(fx["rowcap"], need)]
        if slot:
            fx["sched_req"], fx["sched_tile"] = self._slot_sched(fx["rowcap"], nl)
            if nl > L:
                self._graph_drop_layer(layer_id)
            return
        sr, stl = self._sched_lists(fx["rowcap"])
        dev = pk.device
        fx["sched_req"] = self._h2d(sr, dev)
        fx["sched_tile"] = self._h2d(stl, dev)
        self._graph_drop_layer(layer_id)

    def _graph_drop_layer(self, layer_id):
        g = self._batch["g"] if self._batch is not None else None
        if g is None:
            return
        for key in [k for k in g["by_layer"]
                    if k == layer_id or (isinstance(k, tuple) and k[0] == layer_id)]:
            del g["by_layer"][key]
            g["free"] = g.get("free", 0) + 1  # as _graph_clear

    @staticmethod
    def _graph_clear(g):
        """Drop every captured graph of the batch for an address / launch
        shape change (pool or schedule growth, FA3 metadata buffer): their
        recaptures do not count against the capture budget, which bounds
        signature churn (an equal-length slot batch reallocates its schedule
        at every regrow, ~2 x PICOKV_SLIDE_HEADROOM generated tokens: 32
        recaptures each, which used to exhaust the budget and leave a long
        generation eager)."""
        g["free"] = g.get("free", 0) + len(g["by_layer"])
        g["by_layer"].clear()

    def _slide_sync(self, fx, layer_id, rids):
        """Batch end: hand each row's grown global segment back to its request
        cache (a dense-only row whose window slid is promoted to a compressed
        layer); the dense window itself travels in the stashed pool."""
        pk, pv = fx["batched"].packed_key, fx["batched"].packed_value
        for i, rid in enumerate(rids):
            rc = self._reqs.get(rid)
            g = fx["gl_py"][i]
            if rc is None:
                continue
            c = rc.layers.get(layer_id)
            if c is None:
                if g == 0:
                    continue
                dk, dv = rc._dense.pop(layer_id)
                kp, vp, kpt = self._gl0_projections(
                    layer_id, int(dk.shape[1]), int(dk.shape[-1]), dk.device, dk.dtype)
                c = PreparedKVCache(dense_key=dk, dense_value=dv, key_projection=kp,
                                    value_projection=vp, key_projection_t=kpt,
                                    packed_key=pk[i:i + 1, ..., :0],
                                    packed_value=pv[i:i + 1, :, :0], exp_bits=fx["exp_bits"])
                rc.layers[layer_id] = c
            k = pk[i:i + 1, ..., :g]
            v = pv[i:i + 1, :, :(g + 63) // 64]
            if not self._dedup:
                k, v = k.contiguous(), v.contiguous()
            c.packed_key, c.packed_value = k, v

    # ------------------------------------------------------------------
    # decode: per-step bookkeeping
    # ------------------------------------------------------------------

    def _step_begin(self, fixed):
        """Per-STEP bookkeeping, once at the step's first layer call (all
        layers of a step share this state): advance the shared write positions
        (one device add — no per-layer tensor churn), grow every layer's pool
        on overflow, and re-plan the paged dense kernel (one host sync per
        step)."""
        import torch as _t
        sh = self._batch["shared"]
        if sh is None:
            dl = fixed["dlens0"]
            dev = fixed["kbuf"].device
            B = len(dl)
            cap = fixed["kbuf"].shape[1]
            # slot mode: only live rows advance (vacant rows stay at one
            # dense token, write position 0)
            lv = ([1] * len(self._batch["rids"]) + [0] * (B - len(self._batch["rids"]))
                  if self._batch.get("S") else None)
            w3 = None  # PICOKV_SLOT_MEMO slot mode: wpos / lpl / live = rows of one
            if lv is not None and self._slot_memo:  # buffer (a slot change: one H2D);
                z = [0] * (_round_up(B, 4) - B)  # rows 16-byte aligned, as own tensors
                w3 = self._h2d(list(dl) + z + [d + 1 for d in dl] + z + lv + z, dev).view(3, -1)
            sh = {
                # write position of this step (device-side: scatter + graphs
                # read VALUES, never python ints)
                "wpos": self._h2d(dl, dev) if w3 is None else w3[0, :B],
                # per-request dense kv lens INCLUDING this step's token
                "lpl": self._h2d([d + 1 for d in dl], dev) if w3 is None else w3[1, :B],
                "dlens_py": list(dl),  # pre-append here; +1 below
                "live_py": lv,
                "live": None if lv is None else (self._h2d(lv, dev) if w3 is None else w3[2, :B]),
                "w3": w3,
                "cur": max(dl),
                "ar": _t.arange(B, device=dev),
                "ar32": _t.arange(B, dtype=_t.int32, device=dev),
                "cap": cap,
                # Paged tables (page_size=1 — the only page size this
                # flashinfer build runs correctly): persistent buffers whose
                # CONTENTS are rebuilt on device each step (stable kernel-input
                # addresses, CUDA-graph-friendly, no per-step H2D).
                "indptr": _t.zeros(B + 1, dtype=_t.int32, device=dev),
                "indices": _t.empty(B * cap, dtype=_t.int32, device=dev),
                "ones": _t.ones(B, dtype=_t.int32, device=dev),
                "T": 0,
                # custom window kernel for this batch (else flashinfer)
                "win": bool(self._win) and B <= self._win_max_bs,
                # decode steps since the batch was built (slide epoch clock;
                # fast build: inherited from the carried rows' batch)
                "step": self._batch.get("step0", 0),
            }
            self._batch["shared"] = sh
            # an inherited clock may start the batch ON an epoch boundary
            bnd = sh["step"] > 0
            inc = None
        else:
            inc = sh["live_py"]
            if inc is None:
                sh["wpos"].add_(1)
                sh["lpl"].add_(1)
            else:
                sh["wpos"].add_(sh["live"])
                sh["lpl"].add_(sh["live"])
            sh["step"] += 1
            bnd = True
        if bnd and fixed.get("slide") and sh["step"] % _SLIDE_S == 0:
            # Epoch boundary: each row past sink+local drops its n oldest
            # local tokens into the global segment (every layer merges +
            # shifts at its own call, before its append). wpos == the
            # pre-append window length here, so n is computed on device.
            W = self._batch["W"]
            n = [min(_SLIDE_S, max(0, d - W)) for d in sh["dlens_py"]]
            if any(n):
                n_t = (sh["wpos"] - W).clamp_(0, _SLIDE_S)
                sh["wpos"].sub_(n_t)
                sh["lpl"].sub_(n_t)
                sh["dlens_py"] = [d - m for d, m in zip(sh["dlens_py"], n)]
                sh["cur"] = max(sh["dlens_py"])
                sh["bnd_step"], sh["bnd_n"], sh["bnd_t"] = sh["step"], n, n_t
        sh["dlens_py"] = ([d + 1 for d in sh["dlens_py"]] if inc is None
                          else [d + m for d, m in zip(sh["dlens_py"], inc)])
        sh["cur"] += 1
        if sh["cur"] > sh["cap"]:  # grow every layer's pool together (rare)
            newcap = sh["cap"] * 2
            for fx in self._batch["fixed"].values():
                if fx is None:
                    continue
                kbuf, vbuf = fx["kbuf"], fx["vbuf"]
                nk = _t.zeros(kbuf.shape[0], newcap, kbuf.shape[2], kbuf.shape[3],
                              device=kbuf.device, dtype=kbuf.dtype)
                nv = _t.zeros_like(nk)
                nk[:, : kbuf.shape[1], :] = kbuf
                nv[:, : vbuf.shape[1], :] = vbuf
                fx["kbuf"], fx["vbuf"] = nk, nv
                fx["batched"].dense_key = nk.permute(0, 2, 1, 3)
                fx["batched"].dense_value = nv.permute(0, 2, 1, 3)
            sh["cap"] = newcap
            sh["indices"] = _t.empty(
                kbuf.shape[0] * newcap, dtype=_t.int32, device=sh["wpos"].device)
            self._graph_clear(self._batch["g"])  # pool/table addresses changed
        if not sh["win"]:  # the window kernel reads lpl only, never the tables
            self._rebuild_tables(sh)
        elif self._fa3 is not None and self._fa3_meta_fn is not None:
            self._fa3_sched(sh, fixed["kbuf"].dtype)
        self._plan_paged(sh)
        g = self._batch["g"]
        g["n"] += 1
        if g["bufs"] is None and (self._graphs_on or sh["win"]):
            dev = sh["wpos"].device
            B = len(sh["dlens_py"])
            g["bufs"] = {
                "q": _t.empty(B, self.num_q_heads, self.head_dim, device=dev, dtype=torch.bfloat16),
                "k": _t.empty(B, self.num_kv_heads, self.head_dim, device=dev, dtype=torch.bfloat16),
                "v": _t.empty(B, self.num_kv_heads, self.head_dim, device=dev, dtype=torch.bfloat16),
                "fi_out": _t.empty(B, self.num_q_heads, self.head_dim, device=dev, dtype=torch.bfloat16),
                "fi_lse": _t.empty(B, self.num_q_heads, device=dev, dtype=torch.float32),
                # custom window kernel output (ln-space LSE, unlike fi_lse)
                "win_lse": _t.empty(B, self.num_q_heads, device=dev, dtype=torch.float32),
            }
        return sh

    def _append_kv(self, kbuf, vbuf, sh, k_, v_, pdl=False):
        """Write this step's K/V [B,KVH,D] at each request's wpos in the dense
        pools: one fused Triton launch when every row is dense, else two
        index_put kernels. ``pdl``: the fused launch is a programmatic
        dependent of the previous kernel, which must neither write k_ / v_ /
        wpos nor access the pools (the shared query projection)."""
        D = self.head_dim
        row = self.num_kv_heads * D
        if (self._fused_append and _append_kv_kernel is not None
                and row & (row - 1) == 0
                and k_.stride(2) == 1 and k_.stride(1) == D
                and v_.stride(2) == 1 and v_.stride(1) == D
                and kbuf.is_contiguous() and vbuf.is_contiguous()):
            pdl = pdl and _gdc_wait is not None
            if (not pdl and self._pdl and row not in self._pdl_warm
                    and not torch.cuda.is_current_stream_capturing()):
                # compile (no launch) the graphs' PDL variant on the eager step
                # before any capture: a cold compile inside one stalls a step
                self._pdl_warm.add(row)
                _append_kv_kernel.warmup(
                    kbuf, vbuf, k_, v_, sh["wpos"], kbuf.shape[1], k_.stride(0),
                    v_.stride(0), ROW=row, PDL=True, launch_pdl=True, grid=(k_.shape[0],))
            _append_kv_kernel[(k_.shape[0],)](
                kbuf, vbuf, k_, v_, sh["wpos"], kbuf.shape[1],
                k_.stride(0), v_.stride(0), ROW=row, PDL=pdl, launch_pdl=pdl)
            return
        kbuf[sh["ar"], sh["wpos"]] = k_
        vbuf[sh["ar"], sh["wpos"]] = v_

    def _rebuild_tables(self, sh):
        """Rebuild the page_size=1 paged tables in place on device (once per
        step at the step's first layer): ``indices`` = each request's live pool
        positions (request-major), ``indptr`` = cumulative live lengths."""
        import torch as _t
        l = sh["lpl"]
        T = int(sum(sh["dlens_py"]))
        cum = _t.cumsum(l, 0, dtype=_t.int32)
        sh["indptr"][1:].copy_(cum)
        pos = _t.arange(T, device=l.device) - _t.repeat_interleave(
            cum - l, l, output_size=T)
        req = _t.repeat_interleave(sh["ar32"], l, output_size=T)
        sh["indices"][:T] = req * sh["cap"] + pos
        sh["T"] = T

    def _plan_paged(self, sh):
        """Plan the paged dense-window kernel for this step (once per step at
        the step's first layer). Every step advances the window lengths, so
        plan runs every step; its launch config is recorded as the graph
        signature (a change re-captures the per-layer graphs instead of
        replaying a stale config)."""
        if sh["win"]:
            # Custom kernel path: no fi plan (no per-step host work). The
            # kernel reads lpl from device memory, so one captured launch
            # serves every step — the signature only changes with the batch
            # or a pool grow, never as the window grows.
            self._batch["g"]["sig"] = ("win", len(sh["dlens_py"]), sh["cap"])
            return
        if self._fi is False:
            return
        try:
            if self._fi is None:
                import flashinfer

                ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
                self._fi = {"wrapper": flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, kv_layout="NHD")}
            fi = self._fi
            args = (sh["indptr"], sh["indices"], sh["ones"],
                    self.num_q_heads, self.num_kv_heads, self.head_dim, 1)
            if fi.get("ev") is None:
                # first plan: resolves the decode module (D2H indptr read)
                fi["wrapper"].plan(*args, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
                if os.environ.get("PICOKV_FI_NOSYNC", "1") == "1":
                    from flashinfer.decode import fast_decode_plan
                    fi["fast"] = fast_decode_plan
                fi["ev"] = torch.cuda.Event()
            elif fi.get("fast") is not None:
                # Sync-free plan: indptr from the host lens mirror (== lpl,
                # rebuilt on device above) instead of a D2H read that drains
                # the stream every step. The event makes the previous plan's
                # pinned-buffer H2D consumed before the buffer is rewritten
                # (it completes at the previous step's first layer).
                fi["ev"].synchronize()
                ih = torch.tensor([0] + list(itertools.accumulate(sh["dlens_py"])),
                                  dtype=torch.int32)
                fi["fast"](fi["wrapper"], *args, q_data_type=torch.bfloat16,
                           kv_data_type=torch.bfloat16, global_override_indptr_cpu=ih)
            else:
                fi["wrapper"].plan(*args, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
            fi["ev"].record()
            pi = getattr(self._fi["wrapper"], "_plan_info", None)
            try:
                # flashinfer 0.6.18 returns a tvm_ffi Array (not list/tuple);
                # convert so it can key the CUDA-graph store.
                sig = tuple(pi) if pi is not None else None
            except Exception:
                sig = None
            self._batch["g"]["sig"] = sig
            if sig is None and self._graphs_on:
                logger.warning("[PicoKVAttnBackend] plan sig unavailable; CUDA graphs "
                               "cannot engage (running eager)")
        except Exception as e:
            logger.warning(f"[PicoKVAttnBackend] paged dense path DISABLED ({e!r}); "
                           f"using fallback")
            self._fi = False

    _LN2 = 0.6931471805599453  # flashinfer returns log2 LSE; tail kernel wants ln

    def _fa3_sched(self, sh, dtype):
        """FA3 tile-scheduler metadata of this step's window attention (lens
        lpl, max_seqlen_k = the pool capacity: the host heuristics every
        layer's call would rerun), computed once per step into a persistent
        buffer the per-layer graphs read; layers whose pool shape differs
        fall back to the per-call path (_fa3_meta)."""
        B = int(sh["lpl"].shape[0])
        try:
            m = self._fa3_meta_fn(B, 1, sh["cap"], self.num_q_heads, self.num_kv_heads,
                                  self.head_dim, sh["lpl"], qkv_dtype=dtype)
        except Exception as e:
            logger.warning(f"[PicoKVAttnBackend] FA3 scheduler metadata unavailable ({e!r})")
            self._fa3_meta_fn = None
            sh["fa3_meta"] = None
            self._graph_clear(self._batch["g"])  # graphs captured with the buffer
            return
        buf = sh.get("fa3_meta")
        if buf is not None and buf.shape == m.shape:
            buf.copy_(m)
        else:
            if buf is not None:
                self._graph_clear(self._batch["g"])  # graphs read the old buffer
            sh["fa3_meta"] = m
        sh["fa3_key"] = (B, sh["cap"])

    def _fa3_meta(self, sh, kbuf):
        """This step's FA3 metadata if it was computed for ``kbuf``'s shape."""
        m = sh.get("fa3_meta")
        if m is None or sh.get("fa3_key") != (kbuf.shape[0], kbuf.shape[1]):
            return None
        return m

    def _window_attn(self, kbuf, vbuf, lpl, q_, out, lse, meta=None, pdl=False,
                     split=False):
        """Plan-free dense-window attention over the token-major pools with
        device lens ``lpl``: the Triton kernels (_tri_win; ``pdl``: launched
        as programmatic dependents; ``split``: the LSE now, and a third
        return value that launches the output from it), FA3 (ln-LSE
        returned, ``lse`` unused; ``meta``: its scheduler metadata for these
        lens, _fa3_sched) or the custom kernel (writes ``out``/``lse`` in
        place). Returns (out [B,QH,1,D], ln-LSE [B,QH,1][, out_fn])."""
        B = kbuf.shape[0]
        o4 = out.view(B, self.num_q_heads, 1, self.head_dim)
        w = self._tri_win
        if (w is not None and B <= self._tri_win_bs and kbuf.shape[1] <= 2048
                and q_.dim() == 3 and q_.stride(-1) == 1 and out.is_contiguous()
                and lse.is_contiguous() and kbuf.is_contiguous() and vbuf.is_contiguous()):
            # compile every variant on the eager steps (a no-op after the
            # first call and inside a capture, which reaches the split pair)
            w.warm(q_, kbuf, vbuf, lpl, lse, out, self.scaling)
        else:
            w = None
        if w is not None and (split or self._tri_serial):
            sc = self.scaling
            l3 = lse.view(B, self.num_q_heads, 1)
            if split and B <= self._tri_split_bs:
                w.window_lse(q_, kbuf, lpl, lse, sc, pdl=pdl)
                return o4, l3, lambda: w.window_out_from_lse(q_, kbuf, vbuf, lpl, lse, out, sc)
            w.window_attn(q_, kbuf, vbuf, lpl, lse, out, sc, pdl=pdl)
            return o4, l3
        if self._fa3 is not None:
            _, l, *_ = self._fa3(q_.unsqueeze(1), kbuf, vbuf, cache_seqlens=lpl,
                                 softmax_scale=self.scaling, causal=False,
                                 return_softmax_lse=True, scheduler_metadata=meta,
                                 out=out.view(B, 1, self.num_q_heads, self.head_dim))
            return o4, l.view(B, self.num_q_heads, 1)
        self._win(kbuf, vbuf, lpl, q_, out, lse, self.scaling)
        return o4, lse.unsqueeze(-1)

    def _paged_dense_attn(self, sh, fixed, q_):
        """Exact dense-window attention via flashinfer's paged decode kernel
        over the token-major pool (one fused kernel per layer; the plan ran
        once at layer 0). All branches (equal / dlens_equal / ragged) route
        through here. Returns (out [B,QH,1,D], ln-LSE [B,QH,1]) or None when
        flashinfer is unavailable (caller falls back)."""
        if sh["win"]:
            # Custom kernel path: same contract, one fused kernel + merge,
            # ln-LSE directly (no log2 conversion).
            kbuf, vbuf = fixed["kbuf"], fixed["vbuf"]
            B = kbuf.shape[0]
            g = self._batch["g"]
            bufs = g["bufs"] if g is not None else None
            if bufs is not None:
                out, lse = bufs["fi_out"], bufs["win_lse"]
            else:
                out = torch.empty(B, self.num_q_heads, self.head_dim,
                                  device=kbuf.device, dtype=torch.bfloat16)
                lse = torch.empty(B, self.num_q_heads,
                                  device=kbuf.device, dtype=torch.float32)
            # q_ is a view of the qkv projection and non-contiguous for B>1.
            if not q_.is_contiguous():
                q_ = q_.contiguous()
            return self._window_attn(kbuf, vbuf, sh["lpl"], q_, out, lse,
                                     meta=self._fa3_meta(sh, kbuf))
        if self._fi is False:
            return None
        try:
            kbuf, vbuf = fixed["kbuf"], fixed["vbuf"]
            B, cap = kbuf.shape[0], kbuf.shape[1]
            kt = kbuf.view(B * cap, self.num_kv_heads, self.head_dim)
            vt = vbuf.view(B * cap, self.num_kv_heads, self.head_dim)
            out, lse2 = self._fi["wrapper"].run(q_, (kt, vt), out=None, lse=None, return_lse=True)
            return (
                out.view(B, self.num_q_heads, 1, self.head_dim),
                (lse2 * self._LN2).unsqueeze(-1),
            )
        except Exception as e:
            logger.warning(f"[PicoKVAttnBackend] paged dense path DISABLED ({e!r}); "
                           f"using fallback")
            self._fi = False
            return None

    # ------------------------------------------------------------------
    # decode: CUDA graph (item 1)
    # ------------------------------------------------------------------

    def _layer_graph(self, fixed, sh, layer_id, q_, k_, v_, clone=True, direct=False,
                     eng_out=None):
        """CUDA-graph the per-layer decode op sequence: scatter-append this
        step's K/V, paged dense attention, query-projection bmm, fused tail —
        ~10 kernel launches whose configs depend only on shapes that are stable
        within a batch. Inputs are copied into fixed per-batch buffers; every
        kernel-side input (wpos, lpl, paged tables, pools) is a persistent
        device tensor whose VALUES advance per step, so one captured graph
        serves every step. The flashinfer launch config is pinned by keying on
        the plan signature; a signature change re-captures. ``direct`` (BCG
        replay, where q/k/v sit at the segment's fixed output addresses every
        step): the graph reads them in place, keyed on those addresses, and
        stages a non-contiguous q (the kernels need dense q) inside the graph.
        ``eng_out`` (BCG replay): the break's captured output buffer
        [bucket, QH*D]; the tail writes rows [:B] in place and that buffer is
        returned. Otherwise returns the output view [B, QH*D]. None when
        graphs are off / capture failed (caller runs the eager path)."""
        import torch as _t
        g = self._batch["g"]
        if g is None or g["bufs"] is None or g["sig"] is None:
            return None
        bufs = g["bufs"]
        B = fixed["kbuf"].shape[0]
        ov = None
        q_stage = False
        if direct:
            lv = self._eng_lgv.get((layer_id, B)) if self._view_memo else None
            if (lv is None or lv[0] is not q_ or lv[1] is not k_ or lv[2] is not v_
                    or lv[3] is not eng_out or lv[4] != B):
                # PICOKV_VIEW_MEMO: the same objects every replay step
                if (eng_out is not None and eng_out.is_contiguous()
                        and eng_out.dtype == _t.bfloat16 and eng_out.shape[0] >= B
                        and eng_out.shape[1] == self.num_q_heads * self.head_dim):
                    ov = eng_out[:B].view(B, self.num_q_heads, 1, self.head_dim)
                lv = (q_, k_, v_, eng_out, B, q_.is_contiguous(), q_.stride(-1) == 1, ov,
                      (q_.data_ptr(), tuple(q_.stride()), k_.data_ptr(), v_.data_ptr(),
                       None if ov is None else ov.data_ptr()))
                if self._view_memo:
                    self._eng_lgv[(layer_id, B)] = lv
            q_stage = not lv[5] and not (
                # FA3 window + Triton qproj read q through its strides
                self._q_direct and sh["win"] and self._fa3 is not None
                and lv[6] and self._shared_proj
                and fixed["batched"].key_projection_t.stride(0) == 0)
            qs = bufs["q"] if q_stage else q_
            ks, vs = k_, v_
            ov, ptrs = lv[7], lv[8]
        else:
            _bcg_flush()  # the staging copies read q/k/v
            bufs["q"].copy_(q_)
            bufs["k"].copy_(k_)
            bufs["v"].copy_(v_)
            qs, ks, vs = bufs["q"], bufs["k"], bufs["v"]
            ptrs = None
        # slide: the windows oscillate (sink+local .. +64), so the fi plan
        # signature cycles; keep one graph per (layer, signature)
        gkey = (layer_id, g["sig"]) if fixed["slide"] else layer_id
        # gl_max 0 -> > 0 (the first slide into the global segment, a row
        # with a compressed prompt joining) re-captures the full sequence
        dense_only = self._dense_only and fixed["gl_max"] == 0
        ent = g["by_layer"].get(gkey)
        if (ent is not None and ent["sig"] == g["sig"] and ent["cap"] == sh["cap"]
                and ent["ptrs"] == ptrs and ent["dense"] == dense_only):
            try:
                if ov is not None and _bcg_defer(ent["graph"]):
                    return eng_out  # launched by the engine's composed replay
                _bcg_flush()
                ent["graph"].replay()
                if ov is not None:
                    return eng_out
                # clone: the captured output buffer is persistent (reused next
                # step); callers may hold the returned tensor across steps.
                o = ent["out"].view(ent["out"].shape[0], -1)
                return o.clone() if clone else o
            except Exception as e:
                logger.warning(f"[PicoKVAttnBackend] graph replay failed ({e!r}); this batch runs eager")
                self._graphs_off(g)
                return None
        # circuit breaker: sig churn would recapture forever. Slide: a bounded
        # cycle, 7-11 signatures per layer measured at B=10..32 (transient
        # epochs; 1 per epoch in steady state; t36/scripts/sigcount.py).
        # Per batch: the next batch captures again (a global off would leave a
        # long-lived server eager for good after one churny batch).
        _bcg_flush()  # a capture executes this step's layer work right away
        budget = max(160, 16 * len(self._batch["fixed"])) if fixed["slide"] else 160
        if g["ncap"] >= budget + g.get("free", 0):
            if not g.get("off"):
                g["off"] = True
                g["by_layer"].clear()
                logger.warning(f"[PicoKVAttnBackend] graph capture budget ({budget}) exhausted; "
                               f"this batch runs eager")
            return None
        try:
            if self._sp is not None:
                import time as _time
                _ts = _time.perf_counter()
                self._sp["f"].add("cap")
            kbuf, vbuf = fixed["kbuf"], fixed["vbuf"]
            cap = kbuf.shape[1]
            kt = kbuf.view(B * cap, self.num_kv_heads, self.head_dim)
            vt = vbuf.view(B * cap, self.num_kv_heads, self.head_dim)
            gl = None if fixed["equal"] else fixed["global_lens"]
            sr = None if fixed["equal"] else fixed["sched_req"]
            st = None if fixed["equal"] else fixed["sched_tile"]
            batched = fixed["batched"]

            def dense_fn(out=None, pdl=False, split=False):
                # append this step's K/V, then exact dense-window attention
                # (window path: into ``out`` [B,QH,1,D] if given; the Triton
                # window a programmatic dependent of the append under PDL;
                # ``split``: see decode_attention's dense_split)
                self._append_kv(kbuf, vbuf, sh, ks, vs, pdl=pdl)
                if sh["win"]:
                    return self._window_attn(
                        kbuf, vbuf, sh["lpl"], qs,
                        bufs["fi_out"] if out is None else out.view(bufs["fi_out"].shape),
                        bufs["win_lse"], meta=self._fa3_meta(sh, kbuf),
                        pdl=pdl and self._tri_pdl, split=split)
                fi = self._fi["wrapper"]
                out, lse2 = fi.run(qs, (kt, vt),
                                   out=bufs["fi_out"], lse=bufs["fi_lse"], return_lse=True)
                return (out.view(B, self.num_q_heads, 1, self.head_dim),
                        (lse2 * self._LN2).unsqueeze(-1))

            # forced (PICOKV_GRAPH_FORK=1 / non-FA3) or, with FA3, decided per
            # capture by decode_attention from the tail grid (fork_large)
            forced = self._graph_fork and not self._scfg.use_pruning
            fork = forced or (self._fork_large and not self._scfg.use_pruning)
            if fork and self._fork_stream is None:
                self._fork_stream = _t.cuda.Stream(device=kbuf.device)
            # composed replay: keep the raw graph (the engine clones it into
            # its composed graph)
            cg = _t.cuda.CUDAGraph(keep_graph=True) if self._compose else _t.cuda.CUDAGraph()
            # Capture on a side stream WITHOUT torch.cuda.graph's entry work
            # (device synchronize + empty_cache + host empty_cache): per layer
            # per batch that work cost ~0.6 ms of host time plus slow
            # re-allocations afterwards (every cached block was released).
            if self._cap_stream is None:
                self._cap_stream = _t.cuda.Stream(device=kbuf.device)
            if self._graph_pool is None:
                # One private pool for every batch's graphs, pinned by an
                # anchor graph that lives as long as the backend: when a batch
                # change drops all graphs, the pool's use count would hit 0,
                # and without empty_cache in between re-capturing into it
                # trips the allocator's use_count assert. Freed graph memory
                # stays in the pool for the next batch's captures.
                self._graph_pool = _t.cuda.graph_pool_handle()
                anchor = _t.cuda.CUDAGraph()
                with _t.cuda.stream(self._cap_stream):
                    anchor.capture_begin(self._graph_pool)
                    try:
                        _t.zeros(1, device=kbuf.device)
                    finally:
                        anchor.capture_end()
                self._graph_anchor = anchor
            with _t.cuda.stream(self._cap_stream):
                cg.capture_begin(self._graph_pool)
                try:
                    if q_stage:
                        bufs["q"].copy_(q_)
                    if dense_only:
                        # the layer output is the dense-window attention:
                        # straight into the break output, else a buffer of
                        # this graph's own (fi_out is shared by all layers)
                        o = ov if ov is not None else _t.empty(
                            B, self.num_q_heads, 1, self.head_dim,
                            dtype=_t.bfloat16, device=kbuf.device)
                        od = dense_fn(o)[0]
                        if od.data_ptr() != o.data_ptr():
                            o.copy_(od.view(o.shape))
                    else:
                        # PICOKV_PDL: the query side first, the append a
                        # programmatic dependent of its projection
                        qf = not forced and self._pdl
                        o = decode_attention(
                            qs.unsqueeze(2), batched, update_cache=False,
                            global_lens=gl, dense_lens=None,
                            sched_req=sr, sched_tile=st,
                            dense_attn=None if fork or qf else dense_fn(),
                            dense_fn=dense_fn if fork or qf else None,
                            dense_stream=self._fork_stream if fork else None,
                            query_first=qf,
                            fork_large=not forced,
                            dense_split=self._tri_win is not None,
                            out=ov,
                            **self._scfg.decode_kwargs(),
                        )
                finally:
                    cg.capture_end()
            if self._compose:
                meta = self._fa3_meta(sh, kbuf) if (
                    self._fill_offpath and sh["win"] and self._fa3 is not None) else None
                if meta is not None:
                    try:
                        from cuda.bindings import runtime as _rt

                        c = _fa3_fill_offpath(
                            _rt.cudaGraph_t(init_value=cg.raw_cuda_graph()),
                            meta.data_ptr() + (meta.numel() - 1) * meta.element_size())
                    except Exception as e:
                        logger.warning(f"[PicoKVAttnBackend] FA3 fill off-path disabled ({e!r})")
                        self._fill_offpath = False
                        c = None
                    if c is not None:
                        import weakref

                        cg.compose_raw = c
                        self._n_offpath += 1
                        weakref.finalize(cg, _graph_destroy, c)
                cg.instantiate()
            cg.replay()  # capture records only — execute this step's work now
            g["by_layer"][gkey] = {"graph": cg, "out": o, "sig": g["sig"],
                                   "cap": sh["cap"], "ptrs": ptrs, "dense": dense_only}
            g["ncap"] += 1
            self._n_gfail = 0  # "3 failures": consecutive, not over the server's life
            if self._sp is not None:
                self._sp_host("capture", _ts)
            if ov is not None:
                return eng_out
            o = o.view(o.shape[0], -1)
            return o.clone() if clone else o
        except Exception as e:
            logger.warning(f"[PicoKVAttnBackend] graph capture failed ({e!r}); this batch runs eager")
            self._graphs_off(g)
            return None

    def _maybe_rearm_graphs(self):
        import time as _time

        if _time.monotonic() >= self._graph_rearm:
            self._graphs_on, self._graph_rearm = True, None
            logger.info("[PicoKVAttnBackend] CUDA graphs re-armed")

    def _graphs_off(self, g):
        # This batch runs eager: keep its graph-state dict (its bufs back the
        # window path and _step_begin still counts steps on it), drop only the
        # graphs; the next batch / membership captures again (a capture OOM is
        # transient). A third failure in a row (a capture resets the count)
        # turns graphs off for the server.
        g["by_layer"].clear()
        g["off"], g["ncap"] = True, 1 << 30
        self._n_gfail = getattr(self, "_n_gfail", 0) + 1
        if self._n_gfail >= 3:
            # graphs off for 1, 2, 4, ... 32 minutes, then re-armed at the next
            # batch change, rebuild or slot change (a long-lived server is not
            # eager for good)
            import time as _time

            self._n_goff = getattr(self, "_n_goff", 0) + 1
            wait = 60.0 * min(32, 1 << (self._n_goff - 1))
            self._graphs_on, self._n_gfail = False, 0
            self._graph_rearm = _time.monotonic() + wait
            logger.warning(f"[PicoKVAttnBackend] 3 graph failures: graphs off for {wait:.0f} s")

    # ------------------------------------------------------------------
    # decode
    # ------------------------------------------------------------------

    # Breakable decode graph: each call is an eager break between captured
    # segments; replay re-calls this with the capture-time args (q/k/v at the
    # same addresses, refilled by the preceding segment) and copies the
    # return value into the captured output buffer. Plain eager: a no-op.
    @_decode_break
    def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kw):
        import torch as _t
        import os as _os
        import time as _time
        from picokv.picokv_standalone import PreparedKVCache
        if self._eng_capture:
            z = q.new_zeros(q.shape[0], self.num_q_heads * self.head_dim)
            if self._eng_direct_out:
                # the last capture-time return per (layer, bucket) is the
                # break's captured output buffer for that bucket's graph
                self._eng_outs[(layer.layer_id, q.shape[0])] = z
            return z
        _prof = self._decprof
        if self._sp is not None and layer.layer_id == 0:
            self._sp_tick("D")
        b = self._batch
        # BCG replay: rows past the real batch are bucket padding
        eng = self._eng_rids is not None and is_in_breakable_cuda_graph()
        eo = None
        S = None  # slot mode: the bucket size (rows of every batch tensor)
        if eng:
            rids = self._eng_rids
            nq = q.shape[0]
            if self._eng_direct_out:
                eo = self._eng_outs.get((layer.layer_id, nq))
            if self._slots and self._win and nq <= self._win_max_bs:
                S = nq
            n = S or len(rids)
            vm = self._eng_views.get((layer.layer_id, nq)) if self._view_memo else None
            if vm is not None and vm[0] is q and vm[1] is k and vm[2] is v and vm[3] == n:
                q_, k_, v_ = vm[4]
            else:
                q0, k0, v0 = q, k, v
                q, k, v = q[:n], k[:n], v[:n]
                q_ = q.view(-1, self.num_q_heads, self.head_dim)
                k_ = k.view(-1, self.num_kv_heads, self.head_dim)
                v_ = v.view(-1, self.num_kv_heads, self.head_dim)
                if self._view_memo:
                    self._eng_views[(layer.layer_id, nq)] = (q0, k0, v0, n, (q_, k_, v_))
        else:
            if (self._rids_per_step and b is not None and b["min_layer"] is not None
                    and layer.layer_id > b["min_layer"]):
                # same step as the min-layer call that read them
                rids = b["rids"]
            else:
                rids = self._host_rids(forward_batch, forward_batch.req_pool_indices.shape[0])
            q_ = q.view(-1, self.num_q_heads, self.head_dim)
            k_ = k.view(-1, self.num_kv_heads, self.head_dim)
            v_ = v.view(-1, self.num_kv_heads, self.head_dim)

        sp = self._sp
        # "S_req": the bucket the batch was built for ("S": None if the build
        # fell back to unslotted rows, _fixed_layer_fast's memory guard)
        if (self._batch is None or self._batch.get("rids") != rids
                or self._batch.get("S_req") != S or self._batch.get("stale")):
            if not (self._slot_compose and S is not None):
                _bcg_flush()  # composed replay: a (re)build runs in order
            if sp is not None:
                _ts = _time.perf_counter()
            # before the slot path: a slot-mode server never rebuilds, and
            # graphs turned off by failures must come back at a batch change
            if self._graph_rearm is not None:
                self._maybe_rearm_graphs()
            if S is not None and self._slot_change(rids, S):
                if sp is not None:
                    sp["f"].add("slot")
                    self._sp_host("slot", _ts)
            else:
                _bcg_flush()  # no slot change: the rebuild runs in order
                if sp is not None:
                    sp["f"].add("build")
                if self._batch is not None:
                    self._stash_pools()
                self._prune_finished(rids)
                st = self._pools
                self._batch = {
                    "rids": rids, "fixed": {}, "shared": None, "min_layer": None,
                    "S": S, "S_req": S, "W": self._batch_W(S or len(rids)),
                    "g": {"n": 0, "bufs": None, "by_layer": {}, "ncap": 0, "sig": None},
                    # fast build: rows carried from the stash keep its epoch clock
                    "step0": (st["step"] + 1 if self._fast and st is not None and st.get("fixed")
                              and any(r in st["idx"] for r in rids) else 0),
                }
                if sp is not None:
                    self._sp_host("stash", _ts)
        if layer.layer_id not in self._batch["fixed"]:
            _bcg_flush()
            if sp is not None:
                _ts = _time.perf_counter()
            self._batch["fixed"][layer.layer_id] = (
                self._fixed_layer_fast if self._fast else self._fixed_layer)(rids, layer.layer_id)
            if sp is not None:
                self._sp_host("fixed", _ts)
        fixed = self._batch["fixed"][layer.layer_id]
        if S is not None and not self._batch["S"]:  # built unslotted
            S = None
            q_, k_, v_ = q_[:len(rids)], k_[:len(rids)], v_[:len(rids)]
        # A decode step's bookkeeping runs once, at the step's FIRST layer call
        # (the lowest layer id the batch ever calls — layer 0 in the engine; a
        # unit test may drive a single arbitrary layer, where every call is a
        # step).
        if self._batch["min_layer"] is None or layer.layer_id < self._batch["min_layer"]:
            self._batch["min_layer"] = layer.layer_id
        _step_first = layer.layer_id == self._batch["min_layer"]
        if self._cstats and _step_first and eng:
            self._cstats_n += 1
            if self._cstats_n % self._cstats == 0:
                print(f"[PICOKV_COMPOSE_STATS] steps={self._cstats_n} {_bcg_stats} "
                      f"offpath={self._n_offpath}", file=sys.stderr, flush=True)

        B = S or len(rids)
        if _prof:
            _t.cuda.synchronize(); _tp0 = _time.perf_counter()
        if sp is not None:
            _ts = _time.perf_counter()
        if _step_first:
            self._step_begin(fixed)
        sh = self._batch["shared"]
        if fixed["slide"]:
            self._slide_layer(fixed, sh, layer.layer_id)
        if sp is not None:
            self._sp_host("stepbeg+slide", _ts)
        # CUDA graph: after per-batch warmup steps, replay the captured
        # per-layer op sequence (one launch) instead of ~10 eager launches.
        g = self._batch["g"]
        if (self._graphs_on and g["n"] > self._graph_warmup
                and (self._fi not in (None, False) or sh["win"])):
            o = self._layer_graph(fixed, sh, layer.layer_id, q_, k_, v_,
                                  clone=not eng, direct=eng, eng_out=eo)
            if o is not None:
                self._n_decode += 1
                if _step_first:
                    self._log_l0(
                        "GRAPH",
                        f"[PicoKVAttnBackend] forward_decode L0 GRAPH{'/BCG' if eng else ''} B={B} "
                        f"gl_max={fixed['gl_max']} d_max={sh['cur']} "
                        f"(decode#{self._n_decode})")
                if o is eo:  # written in place into the break output
                    return o
                return self._eng_pad(o) if eng else o
        # Persistent paged pool: append this step's token in place at each
        # request's write position (device-side wpos — same op every step).
        _bcg_flush()  # composed replay: the eager path reads q/k/v
        if sp is not None:
            sp["f"].add("eager")
            if g["n"] <= self._graph_warmup:
                sp["f"].add("warm")
        kbuf, vbuf = fixed["kbuf"], fixed["vbuf"]
        self._append_kv(kbuf, vbuf, sh, k_, v_)
        # Tail-kernel varlen schedule: only the GLOBAL (compressed) lens
        # can be ragged; equal-global batches take the uniform fast path.
        if fixed["equal"]:
            gl = sr = st = None
        else:
            gl = fixed["global_lens"]
            sr, st = fixed["sched_req"], fixed["sched_tile"]
        # Dense-window attention: one paged flashinfer kernel for every
        # branch (equal batches included); fallback = internal paths.
        dense_attn = self._paged_dense_attn(sh, fixed, q_)
        if dense_attn is not None and self._dense_only and fixed["gl_max"] == 0:
            # nothing compressed in the batch: the output is the dense-window
            # attention (the tail would merge in exactly zero), as in the
            # dense-only graph — no qproj / pack / tail / vproj launches
            self._n_decode += 1
            if _step_first:
                self._log_l0("DENSE", f"[PicoKVAttnBackend] forward_decode L0 DENSE"
                             f"{'/BCG' if eng else ''} B={B} d_max={sh['cur']} "
                             f"(decode#{self._n_decode})")
            o = dense_attn[0].reshape(B, self.num_q_heads * self.head_dim)
            return self._eng_pad(o) if eng else o.clone()
        dl = None
        if dense_attn is None and not fixed["dlens_equal"]:
            dl = sh["wpos"] + 1  # ragged fallback: per-request dense lens
        batched = (fixed["batched"] if dense_attn is not None else PreparedKVCache(
            dense_key=kbuf[:, :sh["cur"]].permute(0, 2, 1, 3),
            dense_value=vbuf[:, :sh["cur"]].permute(0, 2, 1, 3),
            key_projection=fixed["batched"].key_projection,
            value_projection=fixed["batched"].value_projection,
            key_projection_t=fixed["batched"].key_projection_t,
            packed_key=fixed["batched"].packed_key,
            packed_value=fixed["batched"].packed_value,
            exp_bits=fixed["exp_bits"],
            row_tables=fixed["batched"].row_tables,
        ))
        qB = q_.unsqueeze(2).contiguous()  # [B,QH,1,D]
        if _prof:
            _t.cuda.synchronize(); _tp1 = _time.perf_counter()
        o = decode_attention(
            qB, batched, update_cache=False,
            global_lens=gl, dense_lens=dl,
            sched_req=sr, sched_tile=st,
            dense_attn=dense_attn,
            **self._scfg.decode_kwargs(),
        )
        self._n_decode += 1
        if _prof:
            _t.cuda.synchronize(); _tp2 = _time.perf_counter()
            d = self._decprof_acc
            d["prep"] += _tp1 - _tp0; d["kern"] += _tp2 - _tp1; d["n"] += 1
            if d["n"] % (32 * 8) == 0:
                tot = d["prep"] + d["kern"]
                logger.info(f"[PICOKV_DECPROF] per-layer-call n={d['n']} "
                            f"prep={1e3*d['prep']/d['n']:.4f}ms({100*d['prep']/tot:.1f}%) "
                            f"kernel={1e3*d['kern']/d['n']:.4f}ms({100*d['kern']/tot:.1f}%)")
        if _step_first:
            _path = "PAGED" if dense_attn is not None else ("EQUAL" if fixed["equal"] else "VARLEN")
            self._log_l0(
                _path,
                f"[PicoKVAttnBackend] forward_decode L0 {_path}{'/BCG' if eng else ''} "
                f"B={B} gl_max={fixed['gl_max']} d_max={sh['cur']} (decode#{self._n_decode})")
        # clone: the tail writes into the persistent cache workspace output
        # (reused next step); callers may hold the returned tensor. Under
        # BCG replay the break copies it into its own buffer right away.
        if eng:
            return self._eng_pad(o.reshape(B, self.num_q_heads * self.head_dim))
        return o.clone().reshape(B, self.num_q_heads * self.head_dim)

    def support_triton(self):
        return False
