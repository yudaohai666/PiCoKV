"""Reconstruct each BF16 KV tile and consume it within one attention kernel."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import triton
import triton.language as tl

from picokv.picokv_standalone.decode import (
    _append_decode_tokens, _dense_attention, _get_workspace, load_picokv_extension,
)

_EXTENSION = None


def _decoder_layouts(cache):
    version = None if torch.is_inference(cache.value_projection) else cache.value_projection._version
    key = ("rebuttal_reconstruct_layout", cache.value_projection.data_ptr(), version)
    layouts = cache._workspace.get(key)
    if layouts is None:
        layouts = (cache.value_projection.transpose(-1, -2).contiguous(),)
        cache._workspace[key] = layouts
    return cache.key_projection_t, layouts[0]


def load_reconstruct_extension():
    global _EXTENSION
    if _EXTENSION is None:
        from torch.utils.cpp_extension import load

        root = Path(__file__).resolve().parent / "cuda"
        _EXTENSION = load(
            name="picokv_rebuttal_reconstruct_cuda",
            sources=[str(root / "reconstruct.cpp"), str(root / "reconstruct.cu")],
            extra_cflags=["-O3"], extra_cuda_cflags=["-O3"],
        )
    return _EXTENSION


def reconstruct_full_kv(cache, dense_key=None, dense_value=None):
    """Rebuild BF16 K/V each step; only decoder layouts persist across calls.

    Compute D.T @ C.T to match the packed loads, then store ordinary contiguous
    token-major K/V. This remains a fused GPU unpack+GEMM, with no global BF16 C.
    """
    dense_key = cache.dense_key if dense_key is None else dense_key
    dense_value = cache.dense_value if dense_value is None else dense_value
    key_transposed, value_transposed = _decoder_layouts(cache)
    batch, heads = cache.dense_key.shape[:2]
    length, dense_len = cache.global_len, dense_key.shape[2]
    key = torch.empty((batch, heads, length + dense_len, 128),
                      device=cache.device, dtype=cache.dtype)
    value = torch.empty_like(key)
    if length + dense_len:
        with torch.cuda.device(cache.device):
            for is_key in (True, False):
                _reconstruct_transposed[(batch * heads, triton.cdiv(length + dense_len, 64), 2)](
                    cache.packed_key, cache.packed_value, key_transposed,
                    value_transposed, dense_key, dense_value, key, value, length,
                    dense_len, cache.key_bits, cache.value_bits, 64, 64, is_key,
                    num_warps=4, num_stages=1,
                )
    return key, value


@triton.jit
def _reconstruct_transposed(PK, PV, DKT, DVT, DENSE_K, DENSE_V, OUT_K, OUT_V,
                           L: tl.constexpr, N: tl.constexpr, CK: tl.constexpr, CV: tl.constexpr,
                           BM: tl.constexpr, BN: tl.constexpr, IS_KEY: tl.constexpr):
    row = tl.program_id(0)
    start = tl.program_id(1) * BM
    token = start + tl.arange(0,BM)
    dim = tl.program_id(2) * BN + tl.arange(0,BN)
    c = tl.arange(0,64)
    bits:tl.constexpr = CK if IS_KEY else CV
    dec = DKT if IS_KEY else DVT
    out = OUT_K if IS_KEY else OUT_V
    acc=tl.full((BN,BM),0.0,tl.float32)
    for kb in range(bits//64):
        code=kb*64+c
        if IS_KEY:
            word=tl.load(PK+(row*(CK//64)+kb)*L+token,token<L,0).to(tl.uint64)
            half=tl.where(c[:,None]<32,word[None,:].to(tl.uint32),(word[None,:]>>32).to(tl.uint32))
            neg=((half>>(c[:,None]%32).to(tl.uint32))&1).to(tl.uint16)
        else:
            g=start//64
            w0=tl.load(PV+(row*triton.cdiv(L,64)+g)*CV+code,g*64<L,0).to(tl.uint64)
            if BM>64:
                w1=tl.load(PV+(row*triton.cdiv(L,64)+g+1)*CV+code,((g+1)*64)<L,0).to(tl.uint64)
                word=tl.where(tl.arange(0,BM)[None,:]<64,w0[:,None],w1[:,None])
            else:
                word=w0[:,None]
            half=tl.where(token[None,:]%64<32,word.to(tl.uint32),(word>>32).to(tl.uint32))
            neg=(1-((half>>(token[None,:]%32).to(tl.uint32))&1)).to(tl.uint16)
        b=((neg<<15)|0x3f80).to(tl.uint16).to(tl.bfloat16,bitcast=True)
        b=tl.where(token[None,:]<L,b,0.0).to(tl.bfloat16)
        a=tl.load(dec+(row*128+dim[:,None])*bits+code[None,:],dim[:,None]<128,0)
        acc=tl.dot(a,b,acc)
    result=tl.trans(acc.to(tl.bfloat16))
    dense = DENSE_K if IS_KEY else DENSE_V
    valid_dense = (token[:,None]>=L) & (token[:,None]<L+N) & (dim[None,:]<128)
    exact = tl.load(dense+(row*N+token[:,None]-L)*128+dim[None,:],valid_dense,0)
    result = tl.where(token[:,None]<L,result,exact)
    tl.store(out+(row*(L+N)+token[:,None])*128+dim[None,:],result,(token[:,None]<L+N)&(dim[None,:]<128))


@triton.jit
def _reconstruct_attention_block(
    Q, PK, PV, DKT, DVT, DENSE_LSE, CODE_OUT, DEN_OUT,
    L: tl.constexpr, CK: tl.constexpr, CV: tl.constexpr,
    SCALE: tl.constexpr, BT: tl.constexpr,
):
    row = tl.program_id(0)
    start = tl.program_id(1) * BT
    token = start + tl.arange(0, BT)
    dim = tl.arange(0, 128)
    bit = tl.arange(0, 64)
    head = tl.arange(0, 16)

    # Reconstruct K.T entirely on chip; BF16 rounding matches explicit K=C@D.
    reconstructed_key = tl.full((128, BT), 0.0, tl.float32)
    for block in range(CK // 64):
        word = tl.load(PK + (row * (CK // 64) + block) * L + token,
                       token < L, 0).to(tl.uint64)
        halfword = tl.where(bit[:, None] < 32, word[None, :].to(tl.uint32),
                            (word[None, :] >> 32).to(tl.uint32))
        negative = ((halfword >> (bit[:, None] % 32).to(tl.uint32)) & 1).to(tl.uint16)
        codes = ((negative << 15) | 0x3F80).to(tl.uint16).to(tl.bfloat16, bitcast=True)
        codes = tl.where(token[None, :] < L, codes, 0.0).to(tl.bfloat16)
        c = block * 64 + bit
        decoder = tl.load(DKT + (row * 128 + dim[:, None]) * CK + c[None, :])
        reconstructed_key = tl.dot(decoder, codes, reconstructed_key)
    query = tl.load(Q + (row * 4 + head[:, None]) * 128 + dim[None, :],
                    head[:, None] < 4, 0)
    score = tl.dot(query, reconstructed_key.to(tl.bfloat16)) * SCALE
    lse = tl.load(DENSE_LSE + row * 4 + head, head < 4, 0.0)
    weight = tl.where((head[:, None] < 4) & (token[None, :] < L),
                      tl.exp(score - lse[:, None]), 0.0)

    # Reconstruct V.T in the same program, then immediately consume it in P@V.
    reconstructed_value = tl.full((128, BT), 0.0, tl.float32)
    for block in range(CV // 64):
        c = block * 64 + bit
        group = start // 64
        word0 = tl.load(PV + (row * triton.cdiv(L, 64) + group) * CV + c,
                        group * 64 < L, 0).to(tl.uint64)
        if BT > 64:
            word1 = tl.load(PV + (row * triton.cdiv(L, 64) + group + 1) * CV + c,
                            (group + 1) * 64 < L, 0).to(tl.uint64)
            word = tl.where(tl.arange(0, BT)[None, :] < 64,
                            word0[:, None], word1[:, None])
        else:
            word = word0[:, None]
        halfword = tl.where(token[None, :] % 64 < 32, word.to(tl.uint32),
                            (word >> 32).to(tl.uint32))
        negative = (1 - ((halfword >> (token[None, :] % 32).to(tl.uint32)) & 1)).to(tl.uint16)
        codes = ((negative << 15) | 0x3F80).to(tl.uint16).to(tl.bfloat16, bitcast=True)
        codes = tl.where(token[None, :] < L, codes, 0.0).to(tl.bfloat16)
        decoder = tl.load(DVT + (row * 128 + dim[:, None]) * CV + c[None, :])
        reconstructed_value = tl.dot(decoder, codes, reconstructed_value)
    partial = tl.dot(weight.to(tl.bfloat16), tl.trans(reconstructed_value.to(tl.bfloat16)))
    tl.atomic_add(CODE_OUT + (row * 4 + head[:, None]) * 128 + dim[None, :],
                  partial, head[:, None] < 4, sem="relaxed")
    tl.atomic_add(DEN_OUT + row * 4 + head, tl.sum(weight, 1), head < 4, sem="relaxed")


def decode_attention(
    query, cache, *, decode_key=None, decode_value=None, update_cache=False,
    use_pruning=False, prune_mode=3, log_eps=-10.0, attention_scale=None,
):
    if use_pruning:
        raise ValueError("the reconstruction control requires pruning disabled")
    if query.ndim != 4 or query.shape[-2] != 1:
        raise ValueError("query must have shape [B,QH,1,D]")
    if not query.is_cuda or query.device != cache.device:
        raise ValueError("query and cache must share a CUDA device")
    if query.dtype != torch.bfloat16 or cache.dtype != query.dtype:
        raise ValueError("query and cache must use BF16")
    if query.shape[0] != cache.dense_key.shape[0] or query.shape[-1] != 128 or cache.head_dim != 128:
        raise ValueError("batch must match and head_dim must be 128")
    if query.shape[1] != 4 * cache.kv_heads:
        raise ValueError("the comparison uses the production GQA=4 configuration")
    dense_key, dense_value = _append_decode_tokens(cache, decode_key, decode_value, update_cache)
    scale = 1 / math.sqrt(cache.head_dim) if attention_scale is None else float(attention_scale)
    # Preserve the production dense attention and CUDA fused-output operation.
    dense_output, dense_lse = _dense_attention(query, dense_key, dense_value, scale)
    _, _, _, code_storage, global_den, output = _get_workspace(cache, query)
    key = ("rebuttal_reconstruct_identity", query.shape[0], cache.kv_heads, query.device.index)
    identity = cache._workspace.get(key)
    if identity is None:
        identity = torch.eye(128, device=query.device, dtype=query.dtype).expand(
            query.shape[0], cache.kv_heads, 128, 128).contiguous()
        cache._workspace[key] = identity
    # Each tile already applied D_V before P@V. Reuse the original output
    # kernel with an identity decoder to avoid applying D_V a second time.
    code = code_storage.reshape(-1)[:query.shape[0] * query.shape[1] * 128].view_as(query)
    code.zero_()
    global_den.zero_()
    if cache.global_len:
        key_transposed, value_transposed = _decoder_layouts(cache)
        with torch.cuda.device(query.device):
            _reconstruct_attention_block[(query.shape[0] * cache.kv_heads,
                                          triton.cdiv(cache.global_len, 32))](
                query, cache.packed_key, cache.packed_value, key_transposed,
                value_transposed, dense_lse, code, global_den, cache.global_len,
                cache.key_bits, cache.value_bits, scale, 32,
                num_warps=4, num_stages=1,
            )
    load_picokv_extension().picokv_fused_output_projection_fn(
        dense_output, code, global_den, identity, output,
    )
    return output
