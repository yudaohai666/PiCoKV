"""PiCoKV b1280 codec for MLA latent pools (GLM-4.7-Flash): K1 encode over all layers and a fill kernel that reads
latent rows for a slot list, copying dense rows and decoding compressed ones.

Arena layout per layer (``pool.py``): token slot s owns 288 B at byte ``288 s``: 40 int32 code words (word g, bit j =
code ``32 g + j``, set bit = +1) then the exact bf16 rope 64 at byte 160. A dense bf16 row (512 nope + 64 rope =
1152 B) occupies one allocator page = 4 slots, so dense row p is page p of the same arena.

K1 ``_pk_encode_kernel`` (generic in GROUPS, run at GROUPS = 40) is the reference's residual sign chain with its
bf16 rounding points (0 code mismatches vs ``quantize_block32_online``). The fill kernel decodes ``codes @ dec``
with an fp32 accumulate over 40 words and one bf16 rounding."""

from __future__ import annotations

from typing import Sequence

import torch
import triton
import triton.language as tl

NOPE = 512
ROPE = 64
BITS = 1280
WORDS = BITS // 32
UNIT = WORDS * 4 + ROPE * 2  # bytes per token slot per layer
ROPE_OFF = WORDS * 4  # byte offset of the rope inside a slot
UNITS_PER_ROW = (NOPE + ROPE) * 2 // UNIT  # a dense row spans one page of 4 slots
assert UNIT == 288 and UNITS_PER_ROW == 4


_ptr_tabs: dict = {}


def _ptr_table(tensors: Sequence[torch.Tensor], device: torch.device) -> torch.Tensor:
    """int64 device table of base addresses, cached by address (the pool buffers live for the whole run)."""
    key = (device, tuple(t.data_ptr() for t in tensors))
    tab = _ptr_tabs.get(key)
    if tab is None:
        tab = torch.tensor(key[1], dtype=torch.int64).to(device)
        _ptr_tabs[key] = tab
    return tab


@triton.jit
def _pk_encode_kernel(
    src_tab,
    src_rows,
    src_stride,
    enc_ptr,
    dec_ptr,
    codes_tab,
    codes_stride,
    rope_tab,
    rope_stride,
    dst_slots,
    n,
    D: tl.constexpr,
    R: tl.constexpr,
    GROUPS: tl.constexpr,
    HAS_ROPE: tl.constexpr,
    BM: tl.constexpr,
):
    pid = tl.program_id(0)
    layer = tl.program_id(1)
    src = tl.load(src_tab + layer).to(tl.pointer_type(tl.bfloat16))
    codes = tl.load(codes_tab + layer).to(tl.pointer_type(tl.int32))
    enc = enc_ptr + layer.to(tl.int64) * (GROUPS * 32 * D)
    dec = dec_ptr + layer.to(tl.int64) * (GROUPS * 32 * D)
    m = pid * BM + tl.arange(0, BM)
    live = m < n
    srow = tl.load(src_rows + m, mask=live, other=0).to(tl.int64)
    dst = tl.load(dst_slots + m, mask=live, other=0).to(tl.int64)
    d = tl.arange(0, D)
    j = tl.arange(0, 32)
    bit = tl.full([32], 1, tl.int32) << j  # bit 31 = INT_MIN, still the right pattern
    r = tl.load(src + srow[:, None] * src_stride + d[None, :], mask=live[:, None], other=0.0)
    for g in range(GROUPS):
        rows = g * 32 + j
        e_t = tl.load(enc + rows[None, :] * D + d[:, None])  # [D, 32] = enc_g^T
        # the reference's score matmul has a bf16 output; the sign is taken after that rounding
        pos = tl.dot(r, e_t).to(tl.bfloat16) >= 0
        c = tl.where(pos, 1.0, -1.0).to(tl.bfloat16)
        dg = tl.load(dec + rows[:, None] * D + d[None, :])  # [32, D]
        # baddbmm(r, c, dec_g, beta=1, alpha=-1) on bf16 rounds c @ dec_g to bf16 before the subtraction (measured:
        # 0 of 33.5M elements differ from x - bf16(c @ d), 23.5% differ from the single-rounded fp32 r - acc)
        r = (r.to(tl.float32) - tl.dot(c, dg).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
        word = tl.sum(tl.where(pos, bit, 0), axis=1)  # disjoint bits: the sum is the OR
        tl.store(codes + dst * codes_stride + g, word, mask=live)
    if HAS_ROPE:
        rope = tl.load(rope_tab + layer).to(tl.pointer_type(tl.bfloat16))
        k = tl.arange(0, R)
        v = tl.load(src + srow[:, None] * src_stride + D + k[None, :], mask=live[:, None], other=0.0)
        tl.store(rope + dst[:, None] * rope_stride + k[None, :], v, mask=live[:, None])



@triton.jit
def _pk_fill_kernel(
    idx_ptr,
    n,
    s2d_ptr,
    dense_ptr,
    dense_stride,
    codes_ptr,
    codes_stride,
    rope_ptr,
    rope_stride,
    dec_ptr,
    nope_out,
    nope_stride,
    rope_out,
    rope_out_stride,
    x_ptr,
    x_stride,
    D: tl.constexpr,
    R: tl.constexpr,
    WORDS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    HAS_X: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    e = pid_m * BM + tl.arange(0, BM)
    live = e < n
    slot = tl.load(idx_ptr + e, mask=live, other=0).to(tl.int64)
    dense = tl.load(s2d_ptr + slot, mask=live, other=0).to(tl.int64)
    if HAS_X:
        # s2d -2: a radix prefix row read exact (nope from the host mirror, rope from the slot)
        xr = live & (dense == -2)
        comp = live & (dense == -1)
        crope = comp | xr
    else:
        comp = live & (dense < 0)
        crope = comp
    hasd = live & (dense >= 0)
    nn = pid_n * BN + tl.arange(0, BN)
    out = tl.load(dense_ptr + dense[:, None] * dense_stride + nn[None, :], mask=hasd[:, None], other=0.0)
    if HAS_X:
        xv = tl.load(x_ptr + slot[:, None] * x_stride + nn[None, :], mask=xr[:, None], other=0.0)
        out = tl.where(xr[:, None], xv, out)
    if tl.max(comp.to(tl.int32), axis=0) > 0:
        j = tl.arange(0, 32)
        acc = tl.zeros([BM, BN], dtype=tl.float32)
        for w in range(WORDS):
            word = tl.load(codes_ptr + slot * codes_stride + w, mask=comp, other=0)
            c = tl.where(((word[:, None] >> j[None, :]) & 1) != 0, 1.0, -1.0).to(tl.bfloat16)
            dg = tl.load(dec_ptr + (w * 32 + j)[:, None] * D + nn[None, :])
            acc = tl.dot(c, dg, acc)
        out = tl.where(comp[:, None], acc.to(tl.bfloat16), out)
    e64 = e.to(tl.int64)
    tl.store(nope_out + e64[:, None] * nope_stride + nn[None, :], out, mask=live[:, None])
    if pid_n == 0:
        k = tl.arange(0, R)
        rd = tl.load(dense_ptr + dense[:, None] * dense_stride + D + k[None, :], mask=hasd[:, None], other=0.0)
        rc = tl.load(rope_ptr + slot[:, None] * rope_stride + k[None, :], mask=crope[:, None], other=0.0)
        tl.store(rope_out + e64[:, None] * rope_out_stride + k[None, :], tl.where(crope[:, None], rc, rd),
                 mask=live[:, None])


def pk_fill(idx, s2d, dense, codes, rope, dec, nope_out, nope_stride, rope_out, rope_out_stride, tile=None, x=None):
    """Latent rows of slots ``idx`` (int32/int64 [n]) of one layer into ``nope_out`` / ``rope_out`` (bf16, row
    strides in elements). dense: bf16 [rows, 576] view of the arena; codes: int32 view (stride 72); rope: bf16 view
    starting at the slot rope (stride 144); dec: bf16 [1280, 512]. x: the layer's host mirror (bf16 [slots, 512],
    pinned), read for slots marked -2 in s2d."""
    n = int(idx.numel())
    if n == 0:
        return
    bm, bn = tile or ((128, 128) if n < 16384 else (64, 256))
    _pk_fill_kernel[(triton.cdiv(n, bm), NOPE // bn)](
        idx, n, s2d, dense, dense.stride(0), codes, UNIT // 4, rope, UNIT // 2, dec,
        nope_out, nope_stride, rope_out, rope_out_stride, dense if x is None else x, NOPE,
        D=NOPE, R=ROPE, WORDS=WORDS, BM=bm, BN=bn, HAS_X=x is not None, num_warps=4,
    )


def pk_encode_layers(dense_views, rows, code_views, rope_views, enc, dec, slots):
    """K1 over every layer: dense rows ``rows`` -> codes / rope of ``slots``. dense_views[l]: bf16 [*, 576];
    code_views[l]: int32 (stride 72); rope_views[l]: bf16 slot-rope views (stride 144); enc/dec bf16 [L, 1280, 512]."""
    n, layers = int(rows.numel()), len(dense_views)
    if n == 0:
        return
    dev = rows.device
    _pk_encode_kernel[(triton.cdiv(n, 32), layers)](
        _ptr_table(dense_views, dev), rows, NOPE + ROPE, enc, dec, _ptr_table(code_views, dev), UNIT // 4,
        _ptr_table(rope_views, dev), UNIT // 2, slots, n,
        D=NOPE, R=ROPE, GROUPS=WORDS, HAS_ROPE=True, BM=32, num_warps=4,
    )


@triton.jit
def _pk_mirror_kernel(dense_tab, rows_ptr, slots_ptr, n, x_tab, ROW: tl.constexpr, D: tl.constexpr,
                      BM: tl.constexpr):
    pid = tl.program_id(0)
    layer = tl.program_id(1)
    dense = tl.load(dense_tab + layer).to(tl.pointer_type(tl.bfloat16))
    x = tl.load(x_tab + layer).to(tl.pointer_type(tl.bfloat16))
    e = pid * BM + tl.arange(0, BM)
    live = e < n
    row = tl.load(rows_ptr + e, mask=live, other=0).to(tl.int64)
    slot = tl.load(slots_ptr + e, mask=live, other=0).to(tl.int64)
    d = tl.arange(0, D)
    v = tl.load(dense + row[:, None] * ROW + d[None, :], mask=live[:, None], other=0.0)
    tl.store(x + slot[:, None] * D + d[None, :], v, mask=live[:, None])


def pk_mirror_layers(dense_views, rows, x_views, slots):
    """Exact nope of dense rows ``rows`` (every layer) into the host mirror at ``slots``. x_views[l]: pinned bf16
    [slots, 512] (written through UVA)."""
    n = int(rows.numel())
    if n == 0:
        return
    dev = rows.device
    _pk_mirror_kernel[(triton.cdiv(n, 32), len(dense_views))](
        _ptr_table(dense_views, dev), rows, slots, n, _ptr_table(x_views, dev), ROW=NOPE + ROPE, D=NOPE, BM=32,
        num_warps=4,
    )


@triton.jit
def _pk_priv_kernel(src_ptr, dst_ptr, n, s2d_ptr, dense_tab, rope_tab, rope_stride, x_tab, ROW: tl.constexpr,
                    D: tl.constexpr, R: tl.constexpr, BM: tl.constexpr):
    pid = tl.program_id(0)
    layer = tl.program_id(1)
    dense = tl.load(dense_tab + layer).to(tl.pointer_type(tl.bfloat16))
    rope = tl.load(rope_tab + layer).to(tl.pointer_type(tl.bfloat16))
    x = tl.load(x_tab + layer).to(tl.pointer_type(tl.bfloat16))
    e = pid * BM + tl.arange(0, BM)
    live = e < n
    src = tl.load(src_ptr + e, mask=live, other=0).to(tl.int64)
    dst = tl.load(dst_ptr + e, mask=live, other=0).to(tl.int64)
    srow = tl.load(s2d_ptr + src, mask=live, other=0).to(tl.int64)
    drow = tl.load(s2d_ptr + dst, mask=live, other=0).to(tl.int64)
    hs = live & (srow >= 0)
    cs = live & (srow < 0)
    d = tl.arange(0, D)
    v = tl.load(dense + srow[:, None] * ROW + d[None, :], mask=hs[:, None], other=0.0)
    vx = tl.load(x + src[:, None] * D + d[None, :], mask=cs[:, None], other=0.0)
    tl.store(dense + drow[:, None] * ROW + d[None, :], tl.where(cs[:, None], vx, v), mask=live[:, None])
    k = tl.arange(0, R)
    rd = tl.load(dense + srow[:, None] * ROW + D + k[None, :], mask=hs[:, None], other=0.0)
    rc = tl.load(rope + src[:, None] * rope_stride + k[None, :], mask=cs[:, None], other=0.0)
    tl.store(dense + drow[:, None] * ROW + D + k[None, :], tl.where(cs[:, None], rc, rd), mask=live[:, None])


def pk_priv_layers(src, dst, s2d, dense_views, rope_views, x_views):
    """Exact latent of slots ``src`` (dense row, or host mirror nope + slot rope once compressed) into the dense rows
    of slots ``dst``, every layer."""
    n = int(src.numel())
    if n == 0:
        return
    dev = src.device
    _pk_priv_kernel[(triton.cdiv(n, 32), len(dense_views))](
        src, dst, n, s2d, _ptr_table(dense_views, dev), _ptr_table(rope_views, dev), UNIT // 2,
        _ptr_table(x_views, dev), ROW=NOPE + ROPE, D=NOPE, R=ROPE, BM=32, num_warps=4,
    )
