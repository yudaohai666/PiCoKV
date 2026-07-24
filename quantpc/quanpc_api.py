"""Public QuanPC API.

Only three operations are intended to be used directly:

* ``learn_projection``: learn one K or V binary reconstruction projection.
* ``prepare_kv_cache``: quantize/pack prefill K/V, select EAS tokens, and build
  the dense retained cache.
* ``decode_attention``: compute one decode attention step from a prepared cache.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional

import torch

from quantpc import quanpc
from quantpc.quanpc import QuantProjection, QuantizedTensor
from quantpc.quanpc_cuda_extension.quanpc_cuda import load_quanpc_extension, pack_quanpc_key_transposed_torch, pack_quanpc_value_1bit_transposed_torch

try:
    from quantpc.quanpc_prefill_cuda_extension.quanpc_prefill_cuda import (
        learn_projection_cuda,
        prepare_prefill_cuda,
        quantize_unpacked_cuda,
        select_eas_cuda,
        gather_pack_cuda,
    )
except Exception:  # pragma: no cover - CUDA extension may be unavailable at import time.
    learn_projection_cuda = None
    prepare_prefill_cuda = None
    quantize_unpacked_cuda = None
    select_eas_cuda = None
    gather_pack_cuda = None

SUPPORTED_CODE_BITS = (128, 192, 256, 320, 384, 448, 512)
SUPPORTED_EXP_BITS = (1, 2, 4, 8)
PROJECTED_QUERY_BITS = 8


@dataclass
class PreparedKVCache:
    """Packed QuanPC cache produced from a prefill K/V block."""

    dense_key: torch.Tensor
    dense_value: torch.Tensor
    packed_key: torch.Tensor
    packed_value: torch.Tensor
    key_projection: torch.Tensor
    value_projection: torch.Tensor
    key_err: Optional[torch.Tensor] = None
    value_err: Optional[torch.Tensor] = None
    eas_indices: Optional[torch.Tensor] = None
    kept_indices: Optional[torch.Tensor] = None
    sink_size: int = 64
    local_size: int = 1024
    eas: float = 0.01
    zerorope: int = 20
    exp_bits: int = 4
    projected_query_bits: int = PROJECTED_QUERY_BITS
    key_projection_t: Optional[torch.Tensor] = None
    value_projection_t: Optional[torch.Tensor] = None
    key_encoder_projection: Optional[torch.Tensor] = None
    value_encoder_projection: Optional[torch.Tensor] = None
    _workspace: dict[tuple, tuple[torch.Tensor, ...]] = field(default_factory=dict, repr=False)
    _dense_stream: Optional[torch.cuda.Stream] = field(default=None, repr=False)

    @property
    def device(self):
        return self.dense_key.device

    @property
    def dtype(self):
        return self.dense_key.dtype

    @property
    def kv_heads(self) -> int:
        return int(self.dense_key.shape[1])

    @property
    def head_dim(self) -> int:
        return int(self.dense_key.shape[-1])

    @property
    def key_bits(self) -> int:
        return int(self.key_projection.shape[-2])

    @property
    def value_bits(self) -> int:
        return int(self.value_projection.shape[-2])

    @property
    def global_len(self) -> int:
        return int(self.packed_key.shape[-1])

    @property
    def dense_len(self) -> int:
        return int(self.dense_key.shape[-2])


def _check_code_bits(name: str, bits: int) -> int:
    bits = int(bits)
    if bits not in SUPPORTED_CODE_BITS:
        raise ValueError(f"{name} must be one of {SUPPORTED_CODE_BITS}; got {bits}.")
    return bits


def _check_exp_bits(exp_bits: int) -> int:
    exp_bits = int(exp_bits)
    if exp_bits not in SUPPORTED_EXP_BITS:
        raise ValueError(f"exp_bits must be one of {SUPPORTED_EXP_BITS}; got {exp_bits}.")
    return exp_bits


def _global_span(full_states: torch.Tensor, sink_size: int, local_size: int) -> tuple[int, int]:
    total_len = int(full_states.shape[-2])
    glob_len = total_len - int(sink_size) - int(local_size)
    if sink_size < 0 or local_size < 0 or glob_len <= 0:
        raise ValueError(
            f"invalid sink/local sizes for total_len={total_len}: "
            f"sink_size={sink_size}, local_size={local_size}"
        )
    return int(sink_size), glob_len


def _apply_zerorope_(states: torch.Tensor, zerorope: int) -> torch.Tensor:
    if zerorope <= 0:
        return states
    half = states.shape[-1] // 2
    if zerorope > half:
        raise ValueError(f"zerorope ({zerorope}) must be <= half_head_dim ({half}).")
    states[..., :zerorope] = 0
    states[..., half:half + zerorope] = 0
    return states


def _torch_learn_projection(
    full_states: torch.Tensor,
    init_vectors: torch.Tensor,
    glob_start: int,
    glob_len: int,
    zerorope: int,
    apply_zerorope: bool,
) -> torch.Tensor:
    residual = full_states[:, :, glob_start:glob_start + glob_len, :].clone()
    if apply_zerorope:
        _apply_zerorope_(residual, zerorope)
    code_norm = math.sqrt(float(glob_len))
    projection = torch.empty_like(init_vectors)
    for bit_idx in range(init_vectors.shape[-2]):
        probe = init_vectors[:, :, bit_idx, :]
        ata = residual.transpose(-1, -2) @ residual
        direction = torch.einsum("bhde,bhe->bhd", ata, probe)
        scores = torch.einsum("bhld,bhd->bhl", residual, direction)
        sign = torch.where(scores >= 0, torch.ones_like(scores), -torch.ones_like(scores))
        basis = sign / code_norm
        direction = torch.einsum("bhld,bhl->bhd", residual, basis)
        residual = residual - torch.einsum("bhl,bhd->bhld", basis, direction)
        projection[:, :, bit_idx, :] = direction
    return projection.contiguous()


def learn_projection(
    full_states: torch.Tensor,
    bits: int,
    *,
    sink_size: int = 64,
    local_size: int = 1024,
    zerorope: int = 0,
    apply_zerorope: bool = False,
    init_vectors: Optional[torch.Tensor] = None,
    use_cuda: bool = True,
    verbose: bool = False,
) -> torch.Tensor:
    """Learn one QuanPC binary projection for a full prefill K or V tensor.

    ``full_states`` is ``[B, KVH, L, D]``. The learned projection has shape
    ``[B, KVH, bits, D]`` and is intentionally not folded by ``sqrt(global_len)``;
    folding happens during ``prepare_kv_cache`` quantization.
    """
    bits = _check_code_bits("bits", bits)
    glob_start, glob_len = _global_span(full_states, sink_size, local_size)
    if init_vectors is None:
        init_vectors = torch.randn(
            *full_states.shape[:2],
            bits,
            full_states.shape[-1],
            device=full_states.device,
            dtype=full_states.dtype,
        )
    else:
        expected = (*full_states.shape[:2], bits, full_states.shape[-1])
        if tuple(init_vectors.shape) != expected:
            raise ValueError(f"init_vectors must have shape {expected}; got {tuple(init_vectors.shape)}.")
    if use_cuda and full_states.is_cuda and learn_projection_cuda is not None:
        return learn_projection_cuda(
            full_states,
            init_vectors,
            glob_start,
            glob_len,
            zerorope,
            apply_zerorope,
            verbose=verbose,
        )
    return _torch_learn_projection(full_states, init_vectors, glob_start, glob_len, zerorope, apply_zerorope)


def _joint_eas_indices(k_err: torch.Tensor, v_err: torch.Tensor, eas_size: int) -> torch.Tensor:
    length = k_err.shape[-1]
    k_count = min(eas_size, length)
    k_indices = torch.topk(k_err, k=k_count, dim=-1).indices
    if k_count == length:
        return k_indices.sort(dim=-1).values
    v_count = min(eas_size, length - k_count)
    selected = torch.zeros_like(v_err, dtype=torch.bool).scatter_(-1, k_indices, True)
    v_indices = torch.topk(v_err.masked_fill(selected, -torch.inf), k=v_count, dim=-1).indices
    return torch.cat([k_indices, v_indices], dim=-1).sort(dim=-1).values


def _gather_tokens(states: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    dim = states.shape[-1]
    return torch.gather(states, dim=-2, index=indices.unsqueeze(-1).expand(-1, -1, -1, dim)).contiguous()


def _drop_quantized_tokens(block: QuantizedTensor, keep_indices: torch.Tensor) -> QuantizedTensor:
    length = block.codes.shape[-2]
    mask = torch.ones(block.codes.shape[:3], device=block.codes.device, dtype=torch.bool)
    mask.scatter_(-1, keep_indices, False)
    bsz, heads, _, code_dim = block.codes.shape
    codes = block.codes[mask].reshape(bsz, heads, length - keep_indices.shape[-1], code_dim).contiguous()
    err = block.err[mask].reshape(bsz, heads, length - keep_indices.shape[-1]).contiguous()
    return QuantizedTensor(codes, block.proj, err, proj_t=block.proj_t)


def _drop_token_states(states: torch.Tensor, keep_indices: torch.Tensor) -> torch.Tensor:
    length = states.shape[-2]
    mask = torch.ones(states.shape[:3], device=states.device, dtype=torch.bool)
    mask.scatter_(-1, keep_indices, False)
    bsz, heads, _, dim = states.shape
    return states[mask].reshape(bsz, heads, length - keep_indices.shape[-1], dim).contiguous()


def _prepare_kv_cache_torch(
    full_key: torch.Tensor,
    full_value: torch.Tensor,
    key_projection: torch.Tensor,
    value_projection: torch.Tensor,
    sink_size: int,
    local_size: int,
    eas: float,
    zerorope: int,
    exp_bits: int,
    refit_decoder: bool,
) -> PreparedKVCache:
    glob_start, glob_len = _global_span(full_key, sink_size, local_size)
    key_glob = full_key[:, :, glob_start:glob_start + glob_len, :].clone()
    _apply_zerorope_(key_glob, zerorope)
    value_glob = full_value[:, :, glob_start:glob_start + glob_len, :]

    k_cache = quanpc.KQuantCache(
        key_projection.shape[-2],
        sink_size=sink_size,
        local_size=local_size,
        zerorope=zerorope,
        refit_decoder=refit_decoder,
    )
    v_cache = quanpc.VQuantCache(
        value_projection.shape[-2],
        sink_size=sink_size,
        local_size=local_size,
        refit_decoder=refit_decoder,
    )
    k_block = k_cache.quantize_with_projection(key_glob, QuantProjection(key_projection))
    v_block = v_cache.quantize_with_projection(value_glob, QuantProjection(value_projection))

    eas_size = max(1, int(glob_len * float(eas)))
    eas_indices = _joint_eas_indices(k_block.err, v_block.err, eas_size)
    dense_key = torch.cat(
        [
            full_key[:, :, :sink_size, :],
            _gather_tokens(full_key[:, :, glob_start:glob_start + glob_len, :], eas_indices),
            full_key[:, :, -local_size:, :],
        ],
        dim=-2,
    ).contiguous()
    dense_value = torch.cat(
        [
            full_value[:, :, :sink_size, :],
            _gather_tokens(value_glob, eas_indices),
            full_value[:, :, -local_size:, :],
        ],
        dim=-2,
    ).contiguous()
    k_block = _drop_quantized_tokens(k_block, eas_indices)
    v_block = _drop_quantized_tokens(v_block, eas_indices)
    if refit_decoder:
        k_target = _drop_token_states(key_glob, eas_indices)
        v_target = _drop_token_states(value_glob, eas_indices)
        k_block = k_cache.refit_quantized_tensor(k_block, k_target)
        v_block = v_cache.refit_quantized_tensor(v_block, v_target)
    packed_key = pack_quanpc_key_transposed_torch(k_block.codes).contiguous()
    packed_value = pack_quanpc_value_1bit_transposed_torch(v_block.codes).contiguous()
    return PreparedKVCache(
        dense_key=dense_key,
        dense_value=dense_value,
        packed_key=packed_key,
        packed_value=packed_value,
        key_projection=k_block.proj,
        value_projection=v_block.proj,
        key_projection_t=k_block.proj_t,
        value_projection_t=v_block.proj_t,
        key_encoder_projection=key_projection,
        value_encoder_projection=value_projection,
        key_err=k_block.err,
        value_err=v_block.err,
        eas_indices=eas_indices,
        sink_size=sink_size,
        local_size=local_size,
        eas=eas,
        zerorope=zerorope,
        exp_bits=exp_bits,
    )


def _prepare_kv_cache_cuda_refit(
    full_key: torch.Tensor,
    full_value: torch.Tensor,
    key_projection: torch.Tensor,
    value_projection: torch.Tensor,
    sink_size: int,
    local_size: int,
    eas: float,
    zerorope: int,
    exp_bits: int,
    verbose: bool = False,
) -> PreparedKVCache:
    """CUDA prefill path for refit_decoder.

    Heavy steps (quantization, EAS selection, dense gather + packing) run on the
    CUDA prefill kernels; the two decode-matrix refits are batched least-squares
    solves that run on the GPU via ``torch.linalg``. Codes are unchanged by refit,
    so packing is done from the original CUDA codes plus the kept indices.
    """
    glob_start, glob_len = _global_span(full_key, sink_size, local_size)

    # Refit targets must match what the quantizer saw: K is zero-roped, V is not.
    key_glob = full_key[:, :, glob_start:glob_start + glob_len, :].clone()
    _apply_zerorope_(key_glob, zerorope)
    value_glob = full_value[:, :, glob_start:glob_start + glob_len, :].contiguous()

    # 1) Quantize on CUDA, keeping the raw int8 codes (unpacked).
    k_codes, _, _ = quantize_unpacked_cuda(
        full_key, key_projection, glob_start, glob_len,
        zerorope=zerorope, apply_zerorope=True, verbose=verbose,
    )
    v_codes, _, _ = quantize_unpacked_cuda(
        full_value, value_projection, glob_start, glob_len,
        zerorope=0, apply_zerorope=False, verbose=verbose,
    )

    k_cache = quanpc.KQuantCache(
        key_projection.shape[-2], sink_size=sink_size, local_size=local_size,
        zerorope=zerorope, refit_decoder=True,
    )
    v_cache = quanpc.VQuantCache(
        value_projection.shape[-2], sink_size=sink_size, local_size=local_size,
        refit_decoder=True,
    )

    # 2) Refit #1: decoder fit over the full global set, only to produce the
    #    reconstruction error that drives EAS token selection.
    k_dec1 = k_cache._refit_decoder_from_codes(k_codes, key_glob)
    v_dec1 = v_cache._refit_decoder_from_codes(v_codes, value_glob)
    k_err = k_cache._reconstruction_error(k_codes, key_glob, k_dec1)
    v_err = v_cache._reconstruction_error(v_codes, value_glob, v_dec1)

    # 3) Joint EAS selection on CUDA.
    eas_size = max(1, int(glob_len * float(eas)))
    eas_indices, _mask, kept_indices = select_eas_cuda(k_err, v_err, eas_size, verbose=verbose)

    # 4) Refit #2: re-solve the decoder over exactly the retained (quantized)
    #    tokens; this is the final key/value reconstruction matrix.
    Ck = key_projection.shape[-2]
    Cv = value_projection.shape[-2]
    D = full_key.shape[-1]
    keep = kept_indices.long()
    kept_k_codes = torch.gather(k_codes, 2, keep.unsqueeze(-1).expand(-1, -1, -1, Ck))
    kept_v_codes = torch.gather(v_codes, 2, keep.unsqueeze(-1).expand(-1, -1, -1, Cv))
    kept_k_target = torch.gather(key_glob, 2, keep.unsqueeze(-1).expand(-1, -1, -1, D))
    kept_v_target = torch.gather(value_glob, 2, keep.unsqueeze(-1).expand(-1, -1, -1, D))
    key_proj = k_cache._refit_decoder_from_codes(kept_k_codes, kept_k_target)
    value_proj = v_cache._refit_decoder_from_codes(kept_v_codes, kept_v_target)

    # 5) Dense gather + packing on CUDA (codes are unaffected by refit).
    dense_key, dense_value, packed_key, packed_value = gather_pack_cuda(
        full_key, full_value, k_codes, v_codes,
        eas_indices, kept_indices, sink_size, local_size, verbose=verbose,
    )

    return PreparedKVCache(
        dense_key=dense_key,
        dense_value=dense_value,
        packed_key=packed_key,
        packed_value=packed_value,
        key_projection=key_proj,
        value_projection=value_proj,
        key_projection_t=key_proj.transpose(-1, -2).contiguous(),
        value_projection_t=value_proj.transpose(-1, -2).contiguous(),
        key_encoder_projection=key_projection,
        value_encoder_projection=value_projection,
        key_err=k_err,
        value_err=v_err,
        eas_indices=eas_indices,
        kept_indices=kept_indices,
        sink_size=sink_size,
        local_size=local_size,
        eas=eas,
        zerorope=zerorope,
        exp_bits=exp_bits,
    )


def prepare_kv_cache(
    full_key: torch.Tensor,
    full_value: torch.Tensor,
    key_projection: torch.Tensor,
    value_projection: torch.Tensor,
    *,
    sink_size: int = 64,
    local_size: int = 1024,
    eas: float = 0.01,
    zerorope: int = 20,
    exp_bits: int = 4,
    refit_decoder: bool = False,
    use_cuda: bool = True,
    keep_intermediates: bool = False,
    verbose: bool = False,
) -> PreparedKVCache:
    """Prepare the decode cache from full prefill K/V and learned projections."""
    if tuple(full_key.shape) != tuple(full_value.shape):
        raise ValueError("full_key and full_value must have the same shape.")
    _check_code_bits("key bits", key_projection.shape[-2])
    _check_code_bits("value bits", value_projection.shape[-2])
    exp_bits = _check_exp_bits(exp_bits)
    if use_cuda and refit_decoder and full_key.is_cuda and quantize_unpacked_cuda is not None:
        return _prepare_kv_cache_cuda_refit(
            full_key,
            full_value,
            key_projection,
            value_projection,
            sink_size,
            local_size,
            eas,
            zerorope,
            exp_bits,
            verbose=verbose,
        )
    if use_cuda and not refit_decoder and full_key.is_cuda and prepare_prefill_cuda is not None:
        out = prepare_prefill_cuda(
            full_key,
            full_value,
            key_projection,
            value_projection,
            sink_size=sink_size,
            local_size=local_size,
            eas=eas,
            zerorope=zerorope,
            keep_intermediates=keep_intermediates,
            verbose=verbose,
        )
        return PreparedKVCache(
            dense_key=out.dense_key,
            dense_value=out.dense_value,
            packed_key=out.packed_key,
            packed_value=out.packed_value,
            key_projection=out.key_proj,
            value_projection=out.value_proj,
            key_projection_t=out.key_proj.transpose(-1, -2).contiguous(),
            value_projection_t=out.value_proj.transpose(-1, -2).contiguous(),
            key_encoder_projection=key_projection,
            value_encoder_projection=value_projection,
            key_err=out.key_err,
            value_err=out.value_err,
            eas_indices=out.eas_indices,
            kept_indices=out.kept_indices,
            sink_size=sink_size,
            local_size=local_size,
            eas=eas,
            zerorope=zerorope,
            exp_bits=exp_bits,
        )
    return _prepare_kv_cache_torch(
        full_key,
        full_value,
        key_projection,
        value_projection,
        sink_size,
        local_size,
        eas,
        zerorope,
        exp_bits,
        refit_decoder,
    )


def _get_workspace(cache: PreparedKVCache, query: torch.Tensor) -> tuple[torch.Tensor, ...]:
    bsz, query_heads, seq, head_dim = query.shape
    key = (
        query.device.type,
        query.device.index,
        query.dtype,
        bsz,
        query_heads,
        seq,
        head_dim,
        cache.key_bits,
        cache.value_bits,
    )
    workspace = cache._workspace.get(key)
    if workspace is not None:
        return workspace
    device = query.device
    workspace = (
        torch.empty(bsz, query_heads, seq, cache.key_bits, device=device, dtype=query.dtype),
        torch.empty(bsz, query_heads, seq, cache.key_bits // 64, PROJECTED_QUERY_BITS, device=device, dtype=torch.long),
        torch.empty(bsz, query_heads, seq, device=device, dtype=torch.float32),
        torch.empty(bsz, query_heads, seq, cache.value_bits, device=device, dtype=torch.float32),
        torch.empty(bsz, query_heads, seq, device=device, dtype=torch.float32),
        torch.empty(bsz, query_heads, seq, head_dim, device=device, dtype=query.dtype),
    )
    cache._workspace[key] = workspace
    return workspace


def _append_decode_token(cache: PreparedKVCache, decode_key: Optional[torch.Tensor], decode_value: Optional[torch.Tensor], update_cache: bool) -> tuple[torch.Tensor, torch.Tensor]:
    if decode_key is None and decode_value is None:
        return cache.dense_key, cache.dense_value
    if decode_key is None or decode_value is None:
        raise ValueError("decode_key and decode_value must be provided together.")
    dense_key = torch.cat([cache.dense_key, decode_key], dim=-2).contiguous()
    dense_value = torch.cat([cache.dense_value, decode_value], dim=-2).contiguous()
    if update_cache:
        cache.dense_key = dense_key
        cache.dense_value = dense_value
    return dense_key, dense_value


def decode_attention(
    query: torch.Tensor,
    cache: PreparedKVCache,
    *,
    decode_key: Optional[torch.Tensor] = None,
    decode_value: Optional[torch.Tensor] = None,
    update_cache: bool = False,
    exp_bits: Optional[int] = None,
    prune_mode: int = 3,
    log_eps: float = -10.0,
    use_pruning: bool = True,
    attention_scale: Optional[float] = None,
) -> torch.Tensor:
    """Compute one QuanPC decode attention step from a prepared cache."""
    if not query.is_cuda:
        raise ValueError("decode_attention currently requires a CUDA query tensor.")
    if query.shape[-2] != 1:
        raise ValueError(f"decode_attention expects single-token query S=1; got S={query.shape[-2]}.")
    exp_bits = cache.exp_bits if exp_bits is None else _check_exp_bits(exp_bits)
    if cache.projected_query_bits != PROJECTED_QUERY_BITS:
        raise ValueError(f"CUDA decode requires projected_query_bits={PROJECTED_QUERY_BITS}.")
    dense_key, dense_value = _append_decode_token(cache, decode_key, decode_value, update_cache)
    if attention_scale is None:
        attention_scale = 1.0 / math.sqrt(float(query.shape[-1]))

    projected_query, packed_query, query_scale, glob_code, glob_den, attn_output = _get_workspace(cache, query)
    ext = load_quanpc_extension()
    gqa = query.shape[1] // cache.kv_heads
    key_projection_t = cache.key_projection_t
    if key_projection_t is None:
        key_projection_t = cache.key_projection.transpose(-1, -2).contiguous()
        cache.key_projection_t = key_projection_t
    can_use_pruned_tail = (
        use_pruning
        and gqa == 4
        and hasattr(ext, "quanpc_decode_tail_v1_exp8_tiled_gqa4_atomic_pruned_fn")
    )

    if can_use_pruned_tail:
        current_stream = torch.cuda.current_stream(query.device)
        if cache._dense_stream is None:
            cache._dense_stream = torch.cuda.Stream(device=query.device)
        cache._dense_stream.wait_stream(current_stream)
        with torch.cuda.stream(cache._dense_stream):
            dense_output, dense_lse = quanpc._dense_attention_numerator_flash_gqa(
                query,
                dense_key,
                dense_value,
                attention_scale,
            )
        quanpc._project_query_gqa(query, cache.key_projection, key_projection_t, out=projected_query)
        current_stream.wait_stream(cache._dense_stream)
        ext.quanpc_decode_tail_v1_exp8_tiled_gqa4_atomic_pruned_fn(
            projected_query,
            packed_query,
            cache.packed_key,
            cache.packed_value,
            query_scale,
            dense_lse.contiguous(),
            dense_output.contiguous(),
            glob_code,
            glob_den,
            cache.value_projection.contiguous(),
            attn_output,
            int(exp_bits),
            256,
            1,
            float(attention_scale),
            int(prune_mode),
            float(log_eps),
        )
        return attn_output

    dense_output, dense_lse = quanpc._dense_attention_numerator_flash_gqa(
        query,
        dense_key,
        dense_value,
        attention_scale,
    )
    quanpc._project_query_gqa(query, cache.key_projection, key_projection_t, out=projected_query)

    if use_pruning and gqa == 4 and hasattr(ext, "quanpc_decode_tail_v1_exp8_tiled_gqa4_atomic_pruned_fn"):
        ext.quanpc_decode_tail_v1_exp8_tiled_gqa4_atomic_pruned_fn(
            projected_query,
            packed_query,
            cache.packed_key,
            cache.packed_value,
            query_scale,
            dense_lse.contiguous(),
            dense_output.contiguous(),
            glob_code,
            glob_den,
            cache.value_projection.contiguous(),
            attn_output,
            int(exp_bits),
            256,
            1,
            float(attention_scale),
            int(prune_mode),
            float(log_eps),
        )
    else:
        ext.quanpc_decode_tail_v1_exp8_tiled_gqa4_atomic_fn(
            projected_query,
            packed_query,
            cache.packed_key,
            cache.packed_value,
            query_scale,
            dense_lse.contiguous(),
            dense_output.contiguous(),
            glob_code,
            glob_den,
            cache.value_projection.contiguous(),
            attn_output,
            int(exp_bits),
            256,
            1,
            float(attention_scale),
        )
    return attn_output


__all__ = ["PreparedKVCache", "learn_projection", "prepare_kv_cache", "decode_attention"]
