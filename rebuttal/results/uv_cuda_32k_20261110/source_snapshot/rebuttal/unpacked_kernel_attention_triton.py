"""BF16 code attention with explicit codes or production output reuse.

The fused comparison preserves the original dense attention, query projection,
persistent workspace and CUDA output projection. Only global attention changes:
unpack BF16 tiles, compute floating scores/weights and aggregate value codes.
The explicit-code implementation remains available as a historical reference.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from picokv.picokv_standalone.decode import (
    _append_decode_tokens, _dense_attention, _get_workspace,
    _project_query_into_workspace, load_picokv_extension,
)
from .reconstruct_attention import unpack_codes


@triton.jit
def _project_query(Q, DK, QP, CK: tl.constexpr, BC: tl.constexpr):
    row = tl.program_id(0)
    m = tl.arange(0, 16)
    d = tl.arange(0, 128)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    q = tl.load(Q + row * 4 * 128 + m[:, None] * 128 + d[None, :], m[:, None] < 4, 0)
    decoder = tl.load(DK + row * 128 * CK + d[:, None] * CK + c[None, :], c[None, :] < CK, 0)
    projected = tl.dot(q, decoder)
    tl.store(QP + row * 4 * CK + m[:, None] * CK + c[None, :], projected, (m[:, None] < 4) & (c[None, :] < CK))


@triton.jit
def _scores(
    Q, QP, K, DENSE_K, SCORES, STATS,
    L: tl.constexpr, N, CK: tl.constexpr,
    SPLITS: tl.constexpr, SCALE: tl.constexpr,
    MATERIALIZED: tl.constexpr, BC: tl.constexpr, BT: tl.constexpr,
):
    row = tl.program_id(0)
    split = tl.program_id(1)
    m = tl.arange(0, 16)
    t = split * BT + tl.arange(0, BT)
    c = tl.arange(0, 64)
    result = tl.full((16, BT), 0, tl.float32)
    if split * BT < L:
        for code_block in range(triton.cdiv(CK, 64)):
            code = code_block * 64 + c
            qp = tl.load(QP + row * 4 * CK + m[:, None] * CK + code[None, :], (m[:, None] < 4) & (code[None, :] < CK), 0)
            valid = (code[:, None] < CK) & (t[None, :] < L)
            if MATERIALIZED:
                codes = tl.load(K + (row * L + t[None, :]) * CK + code[:, None], valid, 0)
            else:
                word = tl.load(K + (row * (CK // 64) + code_block) * L + t[None, :], t[None, :] < L, 0).to(tl.uint64)
                bit = ((word >> c[:, None].to(tl.uint64)) & 1).to(tl.float32)
                codes = tl.where(valid, 1.0 - 2.0 * bit, 0.0).to(tl.bfloat16)
            result = tl.dot(qp, codes, result)
    if (split + 1) * BT > L:
        d = tl.arange(0, 128)
        q = tl.load(Q + row * 4 * 128 + m[:, None] * 128 + d[None, :], m[:, None] < 4, 0)
        dense = tl.load(DENSE_K + (row * N + t[None, :] - L) * 128 + d[:, None], (t[None, :] >= L) & (t[None, :] < L + N), 0)
        result += tl.dot(q, dense)
    # Match the ordinary BF16 matmul output boundary before attention scaling.
    result = result.to(tl.bfloat16).to(tl.float32) * SCALE
    result = tl.where((t[None, :] < L + N) & (m[:, None] < 4), result, -float("inf"))
    maximum = tl.max(result, axis=1)
    probability = tl.where((t[None, :] < L + N) & (m[:, None] < 4), tl.exp(result - maximum[:, None]), 0.0)
    denominator = tl.sum(probability, axis=1)
    tl.store(SCORES + (row * 4 + m[:, None]) * (L + N) + t[None, :], result, (m[:, None] < 4) & (t[None, :] < L + N))
    stat = ((row * 4 + m) * SPLITS + split) * 2
    tl.store(STATS + stat, maximum, m < 4)
    tl.store(STATS + stat + 1, denominator, m < 4)


@triton.jit
def _normalizers(STATS, NORM, SPLITS: tl.constexpr, BS: tl.constexpr):
    head = tl.program_id(0)
    s = tl.arange(0, BS)
    maximums = tl.load(STATS + (head * SPLITS + s) * 2, s < SPLITS, -float("inf"))
    denominators = tl.load(STATS + (head * SPLITS + s) * 2 + 1, s < SPLITS, 0.0)
    maximum = tl.max(maximums, axis=0)
    denominator = tl.sum(denominators * tl.exp(maximums - maximum), axis=0)
    tl.store(NORM + head * 2, maximum)
    tl.store(NORM + head * 2 + 1, denominator)


@triton.jit
def _aggregate(
    SCORES, NORM, V, DENSE_V, PART_C, PART_D,
    L: tl.constexpr, N, CV: tl.constexpr,
    SPLITS: tl.constexpr, BC: tl.constexpr,
    MATERIALIZED: tl.constexpr, BT: tl.constexpr,
):
    row = tl.program_id(0)
    split = tl.program_id(1)
    m = tl.arange(0, 16)
    maximum = tl.load(NORM + (row * 4 + m) * 2, m < 4, 0.0)
    denominator = tl.load(NORM + (row * 4 + m) * 2 + 1, m < 4, 1.0)
    t = split * BT + tl.arange(0, BT)
    score = tl.load(SCORES + (row * 4 + m[:, None]) * (L + N) + t[None, :], (m[:, None] < 4) & (t[None, :] < L + N), -float("inf"))
    weights = (tl.exp(score - maximum[:, None]) / denominator[:, None]).to(tl.bfloat16)
    c = tl.program_id(2) * BC + tl.arange(0, BC)
    global_result = tl.full((16, BC), 0.0, tl.float32)
    if split * BT < L:
        valid = (t[:, None] < L) & (c[None, :] < CV)
        if MATERIALIZED:
            codes = tl.load(V + (row * L + t[:, None]) * CV + c[None, :], valid, 0)
        else:
            word = tl.load(V + (row * triton.cdiv(L, 64) + t[:, None] // 64) * CV + c[None, :], valid, 0).to(tl.uint64)
            bit = ((word >> (t[:, None] % 64).to(tl.uint64)) & 1).to(tl.float32)
            codes = tl.where(valid, 2.0 * bit - 1.0, 0.0).to(tl.bfloat16)
        global_result = tl.dot(weights, codes)
    tl.store(PART_C + ((row * SPLITS + split) * 4 + m[:, None]) * CV + c[None, :], global_result, (m[:, None] < 4) & (c[None, :] < CV))
    if tl.program_id(2) == 0:
        d = tl.arange(0, 128)
        dense_result = tl.full((16, 128), 0.0, tl.float32)
        if (split + 1) * BT > L:
            dense = tl.load(DENSE_V + (row * N + t[:, None] - L) * 128 + d[None, :], (t[:, None] >= L) & (t[:, None] < L + N), 0)
            dense_result = tl.dot(weights, dense)
        tl.store(PART_D + ((row * SPLITS + split) * 4 + m[:, None]) * 128 + d[None, :], dense_result, m[:, None] < 4)


@triton.jit
def _reduce_partials(PC, PD, GC, GD, CV: tl.constexpr, SPLITS: tl.constexpr, BS: tl.constexpr, BD: tl.constexpr):
    row = tl.program_id(0)
    q = tl.program_id(1)
    tile = tl.program_id(2)
    s = tl.arange(0, BS)
    c = tile * BD + tl.arange(0, BD)
    partial = tl.load(PC + ((row * SPLITS + s[:, None]) * 4 + q) * CV + c[None, :], (s[:, None] < SPLITS) & (c[None, :] < CV), 0)
    total = tl.sum(partial, axis=0)
    tl.store(GC + (row * 4 + q) * CV + c, total, c < CV)
    if tile * BD < 128:
        dense = tl.load(PD + ((row * SPLITS + s[:, None]) * 4 + q) * 128 + c[None, :], (s[:, None] < SPLITS) & (c[None, :] < 128), 0)
        dense_total = tl.sum(dense, axis=0)
        tl.store(GD + (row * 4 + q) * 128 + c, dense_total, c < 128)


@triton.jit
def _project_value(GC, GD, DV, OUT, CV: tl.constexpr, BC: tl.constexpr):
    row = tl.program_id(0)
    m = tl.arange(0, 16)
    c = tl.arange(0, BC)
    d = tl.program_id(1) * 32 + tl.arange(0, 32)
    code = tl.load(GC + (row * 4 + m[:, None]) * CV + c[None, :], (m[:, None] < 4) & (c[None, :] < CV), 0)
    decoder = tl.load(DV + (row * CV + c[:, None]) * 128 + d[None, :], c[:, None] < CV, 0)
    result = tl.dot(code, decoder).to(tl.bfloat16).to(tl.float32)
    dense = tl.load(GD + (row * 4 + m[:, None]) * 128 + d[None, :], m[:, None] < 4, 0).to(tl.float32)
    tl.store(OUT + (row * 4 + m[:, None]) * 128 + d[None, :], result + dense, m[:, None] < 4)


def _decode(query, cache, *, materialized, decode_key=None, decode_value=None,
            update_cache=False, use_pruning=False, prune_mode=3, log_eps=-10.0,
            attention_scale=None):
    if use_pruning:
        raise ValueError("the BF16 control requires pruning disabled")
    if not query.is_cuda or query.dtype != torch.bfloat16 or cache.dtype != query.dtype or query.device != cache.device:
        raise ValueError("query and cache must share a CUDA device and use BF16")
    if query.ndim != 4 or query.shape[2:] != (1, 128) or cache.head_dim != 128:
        raise ValueError("query must have shape [B,QH,1,128]")
    if query.shape[0] != cache.dense_key.shape[0] or query.shape[1] != 4 * cache.kv_heads:
        raise ValueError("batch must match and GQA must be 4")
    dk, dv = _append_decode_tokens(cache, decode_key, decode_value, update_cache)
    query = query.contiguous()
    rows = query.shape[0] * cache.kv_heads
    length, dense_len, ck, cv = cache.global_len, dk.shape[2], cache.key_bits, cache.value_bits
    block = 256
    splits = triton.cdiv(length + dense_len, block)
    if splits == 0:
        raise ValueError("attention requires at least one cached token")
    scale = 1.0 / math.sqrt(128) if attention_scale is None else float(attention_scale)
    bf16 = dict(device=query.device, dtype=torch.bfloat16)
    fp32 = dict(device=query.device, dtype=torch.float32)
    qp = torch.empty((rows, 4, ck), **bf16)
    scores = torch.empty((rows, 4, length + dense_len), **fp32)
    stats = torch.empty((rows, 4, splits, 2), **fp32)
    norm = torch.empty((rows, 4, 2), **fp32)
    pc = torch.empty((rows, splits, 4, cv), **fp32)
    pd = torch.empty((rows, splits, 4, 128), **fp32)
    gc = torch.empty((rows, 4, cv), **bf16)
    gd = torch.empty((rows, 4, 128), **bf16)
    output = torch.empty_like(query)
    with torch.cuda.device(query.device):
        _project_query[(rows, triton.cdiv(ck, 64))](query, cache.key_projection_t, qp, ck, 64, num_warps=4)
        key = unpack_codes(cache.packed_key, length=length, is_key=True) if materialized else cache.packed_key
        _scores[(rows, splits)](query, qp, key, dk, scores, stats, length, dense_len, ck, splits, scale, materialized, triton.next_power_of_2(ck), block, num_warps=8)
        del key
        _normalizers[(rows * 4,)](stats, norm, splits, triton.next_power_of_2(splits), num_warps=4)
        value = unpack_codes(cache.packed_value, length=length, is_key=False) if materialized else cache.packed_value
        _aggregate[(rows, splits, triton.cdiv(cv, 64))](scores, norm, value, dv, pc, pd, length, dense_len, cv, splits, 64, materialized, block, num_warps=8)
        del value
        _reduce_partials[(rows, 4, triton.cdiv(max(cv, 128), 16))](pc, pd, gc, gd, cv, splits, triton.next_power_of_2(splits), 16, num_warps=8)
        _project_value[(rows, 4)](gc, gd, cache.value_projection, output, cv, triton.next_power_of_2(cv), num_warps=4)
    return output


def decode_attention(query, cache, **kwargs):
    """Explicit full BF16 codes plus custom attention kernels."""
    return _decode(query, cache, materialized=True, **kwargs)



@triton.jit
def _global_attention(
    QP, K, V, DENSE_LSE, CODE, DEN,
    L: tl.constexpr, CK: tl.constexpr, CV: tl.constexpr,
    SCALE: tl.constexpr, BT: tl.constexpr,
):
    row = tl.program_id(0)
    token = tl.program_id(1) * BT + tl.arange(0, BT)
    head = tl.arange(0, 16)
    bit = tl.arange(0, 64)
    score = tl.full((16, BT), 0, tl.float32)
    for kb in range(CK // 64):
        c = kb * 64 + bit
        qp = tl.load(QP + row * 4 * CK + head[:, None] * CK + c[None, :], head[:, None] < 4, 0)
        word = tl.load(K + (row * (CK // 64) + kb) * L + token[None, :], token[None, :] < L, 0).to(tl.uint64)
        halfword = tl.where(bit[:, None] < 32, word.to(tl.uint32), (word >> 32).to(tl.uint32))
        negative = ((halfword >> (bit[:, None] % 32).to(tl.uint32)) & 1).to(tl.uint16)
        code = ((negative << 15) | 0x3F80).to(tl.uint16).to(tl.bfloat16, bitcast=True)
        code = tl.where(token[None, :] < L, code, 0.0).to(tl.bfloat16)
        score = tl.dot(qp, code, score)
    lse = tl.load(DENSE_LSE + row * 4 + head, head < 4, 0.0)
    # Keep the same BF16 score boundary as the prior BF16 GEMM control.
    score = score.to(tl.bfloat16).to(tl.float32) * SCALE
    weight = tl.where((head[:, None] < 4) & (token[None, :] < L), tl.exp(score - lse[:, None]), 0.0).to(tl.bfloat16)
    # The production output expects sum(w*C) and sum(w), not normalized codes.
    tl.atomic_add(DEN + row * 4 + head, tl.sum(weight.to(tl.float32), 1), head < 4, sem="relaxed")
    for vb in range(triton.cdiv(CV, 64)):
        c = vb * 64 + bit
        valid = (token[:, None] < L) & (c[None, :] < CV)
        word = tl.load(V + (row * triton.cdiv(L, 64) + token[:, None] // 64) * CV + c[None, :], valid, 0).to(tl.uint64)
        halfword = tl.where((token[:, None] % 64) < 32, word.to(tl.uint32), (word >> 32).to(tl.uint32))
        negative = (1 - ((halfword >> (token[:, None] % 32).to(tl.uint32)) & 1)).to(tl.uint16)
        code = ((negative << 15) | 0x3F80).to(tl.uint16).to(tl.bfloat16, bitcast=True)
        code = tl.where(valid, code, 0.0).to(tl.bfloat16)
        partial = tl.dot(weight, code)
        tl.atomic_add(CODE + (row * 4 + head[:, None]) * CV + c[None, :], partial, (head[:, None] < 4) & (c[None, :] < CV), sem="relaxed")


def decode_attention_fused(
    query, cache, *, decode_key=None, decode_value=None, update_cache=False,
    use_pruning=False, prune_mode=3, log_eps=-10.0, attention_scale=None,
):
    if use_pruning:
        raise ValueError("the BF16 comparison requires pruning disabled")
    if not query.is_cuda or query.dtype != torch.bfloat16 or cache.dtype != query.dtype or query.device != cache.device:
        raise ValueError("query and cache must share a CUDA device and use BF16")
    if query.ndim != 4 or query.shape[2:] != (1, 128) or cache.head_dim != 128:
        raise ValueError("query must have shape [B,QH,1,128]")
    if query.shape[0] != cache.dense_key.shape[0] or query.shape[1] != 4 * cache.kv_heads:
        raise ValueError("batch must match and GQA must be 4")
    dense_key, dense_value = _append_decode_tokens(cache, decode_key, decode_value, update_cache)
    projected_query, _, _, global_code, global_den, output = _get_workspace(cache, query)
    scale = 1.0 / math.sqrt(128) if attention_scale is None else float(attention_scale)
    extension = load_picokv_extension()
    # Identical production operations and order around the global body.
    dense_output, dense_lse = _dense_attention(query, dense_key, dense_value, scale)
    _project_query_into_workspace(query, cache, projected_query)
    global_code.zero_()
    global_den.zero_()
    if cache.global_len:
        with torch.cuda.device(query.device):
            _global_attention[(query.shape[0] * cache.kv_heads, triton.cdiv(cache.global_len, 64))](
                projected_query, cache.packed_key, cache.packed_value, dense_lse,
                global_code, global_den, cache.global_len, cache.key_bits,
                cache.value_bits, scale, 64, num_warps=4, num_stages=1,
            )
    extension.picokv_fused_output_projection_fn(
        dense_output, global_code, global_den, cache.value_projection, output,
    )
    return output
