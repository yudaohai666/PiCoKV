"""Torch-only QuanPC implementation for algorithm-accuracy experiments.

This module intentionally does not import or call the CUDA extension.  It keeps
binary key/value codes unpacked as torch tensors and uses torch operations to
simulate projected-query quantization plus per-group exp quantization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple

import torch
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP, apply_rotary_pos_emb
from transformers.processing_utils import Unpack


PROJECTED_QUERY_BITS = 8
EXP_BITS = 4
EXP_QUANT_GROUP_SIZE = 64


def _validate_bit_range(name: str, value: int, min_bits: int, max_bits: int) -> int:
    if not min_bits <= value <= max_bits:
        raise ValueError(f"{name} must be in [{min_bits}, {max_bits}]; got {value}")
    return int(value)


def _validate_attention_quant_bits(projected_query_bits: int, exp_bits: int) -> tuple[int, int]:
    return (
        _validate_bit_range("projected_query_bits", projected_query_bits, 2, 8),
        _validate_bit_range("exp_bits", exp_bits, 1, 16),
    )


def load_offline_projections(path):
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict) or "key" not in data or "value" not in data:
        raise ValueError("offline projection file must be a dict containing 'key' and 'value'")
    if "meta" not in data:
        data = dict(data)
        data["meta"] = {
            "format": "quanpc_offline_projection_legacy",
        }
    return data


@dataclass
class QuantProjection:
    """Binary reconstruction projection learned for one cache block."""

    proj: torch.Tensor  # [B, H, C, D]

    @property
    def bits(self) -> int:
        return self.proj.shape[-2]


@dataclass
class QuantizedTensor:
    """Unpacked binary quantized cache block.

    ``codes`` stores {-1, +1} values. ``proj`` reconstructs model-dimension KV
    vectors when attention pushes projection onto query/output.
    """

    codes: torch.Tensor  # [B, H, L, C]
    proj: torch.Tensor   # [B, H, C, D]
    err: torch.Tensor    # [B, H, L], float32
    packed_cuda: Optional[torch.Tensor] = None
    proj_t: Optional[torch.Tensor] = None

    @property
    def bits(self) -> int:
        return self.codes.shape[-1]


@dataclass
class MixedCacheTensor:
    """Dense retained KV plus unpacked quantized global KV."""

    dense: torch.Tensor
    glob: QuantizedTensor
    use_cuda_v1_exp8: bool = False
    projected_query_bits: int = PROJECTED_QUERY_BITS
    exp_bits: int = EXP_BITS

    @property
    def device(self):
        return self.dense.device

    @property
    def dtype(self):
        return self.dense.dtype

    @property
    def shape(self):
        B, H, _, D = self.dense.shape
        L = self.dense.shape[-2] + self.glob.codes.shape[-2]
        return (B, H, L, D)


class QuantCache(Cache):
    def __init__(self, bits: int, sink_size: int = 64, local_size: int = 1024, refit_decoder: bool = False, offline_projs = None):
        if bits <= 0:
            raise ValueError(f"bits must be positive; got {bits}")
        self.bits_per_token = bits
        self.sink_size = sink_size
        self.local_size = local_size
        self.refit_decoder = refit_decoder
        self.offline_projs = offline_projs
        self.reset()

    def reset(self):
        self.glob = {}
        self.projections = {}
        self.seq_length = {}

    @staticmethod
    def _binary_codes(scores: torch.Tensor) -> torch.Tensor:
        return torch.where(scores >= 0, torch.ones_like(scores), -torch.ones_like(scores))

    @staticmethod
    def _refit_decoder_from_codes(codes: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Fit decode-time reconstruction vectors for fixed binary codes."""
        decoder = torch.linalg.pinv(codes.float()) @ target.float()
        return decoder.to(dtype=target.dtype).contiguous()

    @staticmethod
    def _reconstruction_error(
        codes: torch.Tensor,
        target: torch.Tensor,
        decoder: torch.Tensor,
        chunk_size: int = 8192,
    ) -> torch.Tensor:
        B, H, L, _ = codes.shape
        err = torch.empty(B, H, L, device=target.device, dtype=torch.float32)
        decoder_fp32 = decoder.float()
        for start in range(0, L, chunk_size):
            end = min(start + chunk_size, L)
            recon = codes[:, :, start:end, :].float() @ decoder_fp32
            err[:, :, start:end] = (target[:, :, start:end, :].float() - recon).norm(dim=-1, p=2)
        return err.contiguous()

    def _learn_projection_chunk(self, A: torch.Tensor, code_dim: int) -> torch.Tensor:
        """Learn binary projection vectors from the current residual."""
        B, H, L, D = A.shape
        residual = A
        proj = torch.empty(B, H, code_dim, D, device=A.device, dtype=A.dtype)

        for bit_idx in range(code_dim):
            probe = torch.randn(B, H, D, device=A.device, dtype=A.dtype)
            AtA = residual.transpose(-1, -2) @ residual
            direction = torch.einsum("bhde,bhe->bhd", AtA, probe)
            direction = torch.einsum("bhde,bhe->bhd", AtA, direction)

            scores = torch.einsum("bhld,bhd->bhl", residual, direction)
            code = self._binary_codes(scores)
            direction = torch.einsum("bhld,bhl->bhd", residual, code) / float(L)
            residual = residual - torch.einsum("bhl,bhd->bhld", code, direction)
            proj[:, :, bit_idx, :] = direction

        return proj

    def find_projection(self, A: torch.Tensor, bits: Optional[int] = None) -> QuantProjection:
        bits = self.bits_per_token if bits is None else bits
        return QuantProjection(self._learn_projection_chunk(A, bits).contiguous())

    def quantize_with_projection(
        self,
        A: torch.Tensor,
        projection: QuantProjection,
        refit_decoder: Optional[bool] = None,
    ) -> QuantizedTensor:
        if refit_decoder is None:
            refit_decoder = self.refit_decoder
        B, H, L, _ = A.shape
        C = projection.proj.shape[-2]
        codes = torch.empty(B, H, L, C, device=A.device, dtype=A.dtype)
        block_proj = torch.empty_like(projection.proj)
        residual = A

        for bit_idx in range(C):
            direction = projection.proj[:, :, bit_idx, :]
            scores = torch.einsum("bhld,bhd->bhl", residual, direction)
            code = self._binary_codes(scores)
            residual = residual - torch.einsum("bhl,bhd->bhld", code, direction)
            codes[:, :, :, bit_idx] = code
            block_proj[:, :, bit_idx, :] = direction

        if refit_decoder:
            block_proj = self._refit_decoder_from_codes(codes, A)
            err = self._reconstruction_error(codes, A, block_proj)
        else:
            err = residual.norm(dim=-1, p=2).float()
        return QuantizedTensor(
            codes.contiguous(),
            block_proj.contiguous(),
            err.contiguous(),
            proj_t=block_proj.transpose(-1, -2).contiguous(),
        )

    def refit_quantized_tensor(self, block: QuantizedTensor, target: torch.Tensor) -> QuantizedTensor:
        decoder = self._refit_decoder_from_codes(block.codes, target)
        err = self._reconstruction_error(block.codes, target, decoder)
        return QuantizedTensor(
            block.codes.contiguous(),
            decoder,
            err,
            block.packed_cuda,
            decoder.transpose(-1, -2).contiguous(),
        )

    def qpc(self, A: torch.Tensor) -> tuple[QuantizedTensor, QuantProjection]:
        projection = self.find_projection(A)
        return self.quantize_with_projection(A, projection), projection

    def offline_projection(self, A: torch.Tensor, layer_idx: int) -> Optional[QuantProjection]:
        if self.offline_projs is None:
            return None

        B, H, _, D = A.shape
        try:
            per_layer = self.offline_projs[layer_idx]
        except (IndexError, KeyError, TypeError) as exc:
            raise ValueError(f"offline projection missing layer {layer_idx}") from exc

        if not torch.is_tensor(per_layer):
            try:
                per_layer = torch.stack([per_layer[head_idx] for head_idx in range(H)], dim=0)
            except (IndexError, KeyError, TypeError) as exc:
                raise ValueError(f"offline projection layer {layer_idx} cannot be indexed by head") from exc

        if per_layer.dim() == 2 and H == 1:
            per_layer = per_layer.unsqueeze(0)
        if per_layer.dim() != 3:
            raise ValueError(
                f"offline projection layer {layer_idx} must have shape [heads, bits, dim]; "
                f"got {tuple(per_layer.shape)}"
            )
        if per_layer.shape[0] != H or per_layer.shape[1] != self.bits_per_token or per_layer.shape[2] != D:
            raise ValueError(
                f"offline projection layer {layer_idx} shape mismatch: expected "
                f"[{H}, {self.bits_per_token}, {D}], got {tuple(per_layer.shape)}"
            )

        proj = per_layer.to(device=A.device, dtype=A.dtype)
        proj = proj.unsqueeze(0).expand(B, H, self.bits_per_token, D).contiguous()
        return QuantProjection(proj)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.seq_length[layer_idx] if layer_idx in self.seq_length else 0

    def compress(self, glob: torch.Tensor, layer_idx: int) -> QuantizedTensor:
        raise NotImplementedError

    def init_prefill(self, vecs: torch.Tensor, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert layer_idx not in self.glob
        self.seq_length[layer_idx] = vecs.shape[-2]
        sink_vecs = vecs[:, :, :self.sink_size, :].contiguous()
        local_vecs = vecs[:, :, -self.local_size:, :].contiguous()
        glob_vecs = vecs[:, :, self.sink_size:-self.local_size, :]
        self.glob[layer_idx] = self.compress(glob_vecs, layer_idx)
        return sink_vecs, glob_vecs, local_vecs

    def append_local(self, vecs: torch.Tensor, layer_idx: int):
        self.seq_length[layer_idx] = self.seq_length[layer_idx] + vecs.shape[-2]


class KQuantCache(QuantCache):
    def __init__(self, bits: int, sink_size: int = 64, local_size: int = 1024, zerorope: int = 30, refit_decoder: bool = False, offline_projs = None):
        if zerorope < 0:
            raise ValueError(f"zerorope must be non-negative; got {zerorope}")
        super().__init__(bits=bits, sink_size=sink_size, local_size=local_size, refit_decoder=refit_decoder, offline_projs=offline_projs)
        self.zerorope = zerorope

    def quantization_target(self, glob: torch.Tensor) -> torch.Tensor:
        glob = glob.clone()
        half_head_dim = glob.shape[-1] // 2
        if self.zerorope > half_head_dim:
            raise ValueError(f"zerorope ({self.zerorope}) must be <= half_head_dim ({half_head_dim})")
        glob[:, :, :, :self.zerorope] = 0
        glob[:, :, :, half_head_dim:half_head_dim + self.zerorope] = 0
        return glob

    def compress(self, glob: torch.Tensor, layer_idx: int) -> QuantizedTensor:
        glob = self.quantization_target(glob)
        projection = self.offline_projection(glob, layer_idx)
        if projection is None:
            quantized, projection = self.qpc(glob)
        else:
            quantized = self.quantize_with_projection(glob, projection)
        self.projections[layer_idx] = projection
        return quantized


class VQuantCache(QuantCache):
    def __init__(self, bits: int, sink_size: int = 64, local_size: int = 1024, refit_decoder: bool = False, offline_projs = None):
        super().__init__(bits=bits, sink_size=sink_size, local_size=local_size, refit_decoder=refit_decoder, offline_projs=offline_projs)

    def compress(self, glob: torch.Tensor, layer_idx: int) -> QuantizedTensor:
        projection = self.offline_projection(glob, layer_idx)
        if projection is None:
            quantized, projection = self.qpc(glob)
        else:
            quantized = self.quantize_with_projection(glob, projection)
        self.projections[layer_idx] = projection
        return quantized


class QuantKVCache(Cache):
    def __init__(
        self,
        sink_size: int = 64,
        local_size: int = 1024,
        k_bits: int = 128,
        v_bits: int = 256,
        projected_query_bits: int = PROJECTED_QUERY_BITS,
        exp_bits: int = EXP_BITS,
        eas: float = 0.01,
        zerorope: int = 30,
        use_cuda_v1_exp8: bool = False,
        refit_decoder: bool = False,
        offline_projs = None,
    ):
        projected_query_bits, exp_bits = _validate_attention_quant_bits(projected_query_bits, exp_bits)
        if offline_projs is None:
            key_offline_projs = None
            value_offline_projs = None
        else:
            key_offline_projs = offline_projs["key"]
            value_offline_projs = offline_projs["value"]
        self.kcache = KQuantCache(k_bits, sink_size, local_size, zerorope, refit_decoder=refit_decoder, offline_projs=key_offline_projs)
        self.vcache = VQuantCache(v_bits, sink_size, local_size, refit_decoder=refit_decoder, offline_projs=value_offline_projs)
        self.projected_query_bits = projected_query_bits
        self.exp_bits = exp_bits
        self.eas = eas
        self.use_cuda_v1_exp8 = False
        self.refit_decoder = refit_decoder
        self.k_dense = {}
        self.v_dense = {}

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.kcache.get_seq_length(layer_idx)

    def _joint_eas_indices(self, k_block: QuantizedTensor, v_block: QuantizedTensor) -> torch.Tensor:
        L = k_block.err.shape[-1]
        eas_size = max(1, int(L * self.eas))
        k_count = min(eas_size, L)
        k_indices = torch.topk(k_block.err, k=k_count, dim=-1).indices
        if k_count == L:
            return k_indices.sort(dim=-1).values

        v_count = min(eas_size, L - k_count)
        selected = torch.zeros_like(v_block.err, dtype=torch.bool).scatter_(-1, k_indices, True)
        v_indices = torch.topk(v_block.err.masked_fill(selected, -torch.inf), k=v_count, dim=-1).indices
        return torch.cat([k_indices, v_indices], dim=-1).sort(dim=-1).values

    @staticmethod
    def _gather_tokens(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        D = x.shape[-1]
        return torch.gather(x, dim=-2, index=indices.unsqueeze(-1).expand(-1, -1, -1, D)).contiguous()

    @staticmethod
    def _drop_tokens(block: QuantizedTensor, keep_indices: torch.Tensor) -> QuantizedTensor:
        L = block.codes.shape[-2]
        mask = torch.ones(block.codes.shape[:3], device=block.codes.device, dtype=torch.bool)
        mask.scatter_(-1, keep_indices, False)
        B, H, _, C = block.codes.shape
        codes = block.codes[mask].reshape(B, H, L - keep_indices.shape[-1], C).contiguous()
        err = block.err[mask].reshape(B, H, L - keep_indices.shape[-1]).contiguous()
        return QuantizedTensor(codes, block.proj, err, block.packed_cuda, block.proj_t)

    @staticmethod
    def _drop_token_states(states: torch.Tensor, keep_indices: torch.Tensor) -> torch.Tensor:
        L = states.shape[-2]
        mask = torch.ones(states.shape[:3], device=states.device, dtype=torch.bool)
        mask.scatter_(-1, keep_indices, False)
        B, H, _, D = states.shape
        return states[mask].reshape(B, H, L - keep_indices.shape[-1], D).contiguous()

    def _make_mixed(self, layer_idx: int):
        return (
            MixedCacheTensor(
                self.k_dense[layer_idx],
                self.kcache.glob[layer_idx],
                self.use_cuda_v1_exp8,
                self.projected_query_bits,
                self.exp_bits,
            ),
            MixedCacheTensor(
                self.v_dense[layer_idx],
                self.vcache.glob[layer_idx],
                self.use_cuda_v1_exp8,
                self.projected_query_bits,
                self.exp_bits,
            ),
        )

    def update(self, keys, vals, layer_idx, cache_kwargs):
        L = keys.shape[-2]
        if L > 1:
            key_sink, key_glob, key_local = self.kcache.init_prefill(keys, layer_idx)
            val_sink, val_glob, val_local = self.vcache.init_prefill(vals, layer_idx)
            eas_indices = self._joint_eas_indices(self.kcache.glob[layer_idx], self.vcache.glob[layer_idx])

            eas_keys = self._gather_tokens(key_glob, eas_indices)
            eas_vals = self._gather_tokens(val_glob, eas_indices)
            dense_keys = [key_sink, eas_keys, key_local]
            dense_vals = [val_sink, eas_vals, val_local]
            self.k_dense[layer_idx] = torch.cat(dense_keys, dim=-2).contiguous()
            self.v_dense[layer_idx] = torch.cat(dense_vals, dim=-2).contiguous()
            self.kcache.glob[layer_idx] = self._drop_tokens(self.kcache.glob[layer_idx], eas_indices)
            self.vcache.glob[layer_idx] = self._drop_tokens(self.vcache.glob[layer_idx], eas_indices)
            if self.refit_decoder:
                key_target = self._drop_token_states(self.kcache.quantization_target(key_glob), eas_indices)
                value_target = self._drop_token_states(val_glob, eas_indices)
                self.kcache.glob[layer_idx] = self.kcache.refit_quantized_tensor(self.kcache.glob[layer_idx], key_target)
                self.vcache.glob[layer_idx] = self.vcache.refit_quantized_tensor(self.vcache.glob[layer_idx], value_target)
            self.kcache.projections.pop(layer_idx, None)
            self.vcache.projections.pop(layer_idx, None)
            return keys, vals

        self.kcache.append_local(keys, layer_idx)
        self.vcache.append_local(vals, layer_idx)
        self.k_dense[layer_idx] = torch.cat([self.k_dense[layer_idx], keys], dim=-2).contiguous()
        self.v_dense[layer_idx] = torch.cat([self.v_dense[layer_idx], vals], dim=-2).contiguous()
        return self._make_mixed(layer_idx)


def _repeat_quantized_tensor(block: QuantizedTensor, repeat_factor: int) -> QuantizedTensor:
    return QuantizedTensor(
        block.codes.repeat_interleave(repeat_factor, dim=1),
        block.proj.repeat_interleave(repeat_factor, dim=1),
        block.err.repeat_interleave(repeat_factor, dim=1),
        block.packed_cuda.repeat_interleave(repeat_factor, dim=1) if block.packed_cuda is not None else None,
        block.proj_t.repeat_interleave(repeat_factor, dim=1) if block.proj_t is not None else None,
    )


def _repeat_mixed_cache(cache: MixedCacheTensor, target_heads: int) -> MixedCacheTensor:
    current_heads = cache.dense.shape[1]
    if current_heads == target_heads:
        return cache
    if target_heads % current_heads != 0:
        raise ValueError(f"query heads {target_heads} must be divisible by kv heads {current_heads}")
    repeat_factor = target_heads // current_heads
    return MixedCacheTensor(
        cache.dense.repeat_interleave(repeat_factor, dim=1),
        _repeat_quantized_tensor(cache.glob, repeat_factor),
        cache.use_cuda_v1_exp8,
        cache.projected_query_bits,
        cache.exp_bits,
    )


def _quantize_projected_query(projected_query: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    bits = _validate_bit_range("projected_query_bits", bits, 2, 8)
    projected_query = projected_query.float()
    max_code = (1 << (bits - 1)) - 1
    max_abs = projected_query.abs().amax(dim=-1).clamp_min(torch.finfo(torch.float32).eps)
    query_scale = (max_abs / float(max_code)).contiguous()
    quantized = torch.round(projected_query / query_scale[..., None]).clamp(-max_code, max_code)
    return quantized, query_scale.float()


def _dequantize_projected_query(projected_query: torch.Tensor, bits: int) -> torch.Tensor:
    quantized, query_scale = _quantize_projected_query(projected_query, bits)
    return (quantized * query_scale[..., None]).contiguous()


def _quantize_exp_weights(
    exp_weights: torch.Tensor,
    bits: int,
    group_size: int = EXP_QUANT_GROUP_SIZE,
) -> torch.Tensor:
    bits = _validate_bit_range("exp_bits", bits, 1, 16)
    exp_weights = exp_weights.float()
    L = exp_weights.shape[-1]
    groups = (L + group_size - 1) // group_size
    padded_L = groups * group_size
    if padded_L != L:
        exp_weights = torch.nn.functional.pad(exp_weights, (0, padded_L - L))

    grouped = exp_weights.reshape(*exp_weights.shape[:-1], groups, group_size)
    max_weight = grouped.amax(dim=-1, keepdim=True)
    max_code = float((1 << bits) - 1)
    inv_scale = torch.where(max_weight > 0, max_code / max_weight, torch.zeros_like(max_weight))
    quantized = torch.round(grouped * inv_scale).clamp(0, max_code)
    dequantized = torch.where(max_weight > 0, quantized * (max_weight / max_code), torch.zeros_like(quantized))
    return dequantized.reshape(*exp_weights.shape[:-1], padded_L)[..., :L].contiguous()


def _dense_attention_numerator(query_states, key_states, value_states, scale):
    dense_scores = torch.matmul(query_states, key_states.transpose(-1, -2)) * scale
    dense_lse = torch.logsumexp(dense_scores.float(), dim=-1, keepdim=True)
    dense_weights = torch.exp(dense_scores.float() - dense_lse).to(query_states.dtype)
    return torch.matmul(dense_weights, value_states), dense_lse


def _quantized_global_scores(query_states, key_block: QuantizedTensor, projected_query_bits: int):
    projected_query = torch.matmul(query_states, key_block.proj.transpose(-1, -2))
    projected_query = _dequantize_projected_query(projected_query, projected_query_bits)
    return torch.matmul(projected_query.to(key_block.codes.dtype), key_block.codes.transpose(-1, -2))


def _quantized_global_output(attn_probs, value_block: QuantizedTensor):
    output_codes = torch.matmul(attn_probs.to(value_block.codes.dtype), value_block.codes)
    return torch.matmul(output_codes, value_block.proj)


def _quantized_global_attention_numerator(
    query_states,
    key_block: QuantizedTensor,
    value_block: QuantizedTensor,
    dense_lse,
    scale,
    projected_query_bits: int,
    exp_bits: int,
):
    glob_scores = _quantized_global_scores(query_states, key_block, projected_query_bits) * scale
    glob_weights = torch.exp(glob_scores.float() - dense_lse)
    glob_weights = _quantize_exp_weights(glob_weights, exp_bits)
    glob_den = glob_weights.sum(dim=-1, keepdim=True)
    glob_output = _quantized_global_output(glob_weights, value_block)
    return glob_output, glob_den


def quantized_cache_attention_forward(query_states, key_states: MixedCacheTensor, value_states: MixedCacheTensor):
    target_heads = query_states.shape[1]
    head_dim = query_states.shape[-1]
    scale = 1 / math.sqrt(head_dim)

    key_states = _repeat_mixed_cache(key_states, target_heads)
    value_states = _repeat_mixed_cache(value_states, target_heads)

    dense_output, dense_lse = _dense_attention_numerator(
        query_states,
        key_states.dense,
        value_states.dense,
        scale,
    )
    glob_output, glob_den = _quantized_global_attention_numerator(
        query_states,
        key_states.glob,
        value_states.glob,
        dense_lse,
        scale,
        key_states.projected_query_bits,
        key_states.exp_bits,
    )
    return (dense_output + glob_output) / (1 + glob_den.to(dense_output.dtype))


def eager_attention_forward(query_states, key_states, value_states):
    Lq = query_states.shape[-2]
    if isinstance(key_states, MixedCacheTensor) or isinstance(value_states, MixedCacheTensor):
        assert isinstance(key_states, MixedCacheTensor)
        assert isinstance(value_states, MixedCacheTensor)
        return quantized_cache_attention_forward(query_states, key_states, value_states)

    return torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        is_causal=Lq != 1,
        enable_gqa=True,
    )


kproj_cache = {}
vproj_cache = {}


def find_orth_proj(X: torch.Tensor):
    # Normalize the Gram by the number of summed rows so the whitening matrix is
    # the (inverse) sqrt of the *covariance*, not of the unnormalized sum X^T X
    # (which grows ~L for keys and injected a sqrt(L) scale into the projection).
    # A and Ainv stay exact inverses so Q . K^T is preserved.
    n = X.shape[-2]
    U, S, V = torch.svd((X.transpose(-1, -2) @ X).float() / float(n))
    S = S.sqrt()
    U, S, V = U.to(torch.bfloat16), S.to(torch.bfloat16), V.to(torch.bfloat16)
    A = (V * S.unsqueeze(-2)) @ V.transpose(-1, -2)
    Ainv = (V / S.unsqueeze(-2)) @ V.transpose(-1, -2)
    return A, Ainv


def chunked_llama_mlp_forward(self, x, chunk_size: int = 8192):
    if x.shape[-2] <= chunk_size:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    outputs = []
    for chunk in x.split(chunk_size, dim=-2):
        gated = self.act_fn(self.gate_proj(chunk)) * self.up_proj(chunk)
        outputs.append(self.down_proj(gated))
    return torch.cat(outputs, dim=-2)


def modify(model):
    for module in model.modules():
        if isinstance(module, LlamaMLP):
            module.forward = partial(chunked_llama_mlp_forward, module)

    for module in model.modules():
        if not isinstance(module, LlamaAttention):
            continue

        def modified_forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: Tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor],
            past_key_value: Optional[Cache] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs: Unpack[FlashAttentionKwargs],
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)

            query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            B, H, L, D = key_states.shape
            query_states = query_states.reshape(B, H, -1, D)

            if L != 1:
                kproj_cache[self.layer_idx] = find_orth_proj(query_states)
                output_proj = self.o_proj.weight.reshape(
                    self.o_proj.weight.shape[0],
                    H,
                    -1,
                    self.head_dim,
                )
                output_proj = output_proj.transpose(0, 1).reshape(H, -1, self.head_dim)
                vproj_cache[self.layer_idx] = find_orth_proj(output_proj)

            query_states = query_states @ kproj_cache[self.layer_idx][1]
            key_states = key_states @ kproj_cache[self.layer_idx][0].transpose(-1, -2)
            value_states = value_states @ vproj_cache[self.layer_idx][0].transpose(-1, -2)
            query_states = query_states.reshape(B, -1, L, D)

            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(
                    key_states,
                    value_states,
                    self.layer_idx,
                    cache_kwargs,
                )

            attn_output = eager_attention_forward(query_states, key_states, value_states)
            attn_output = attn_output.reshape(B, H, -1, D)
            attn_output = attn_output @ vproj_cache[self.layer_idx][1].transpose(-1, -2)
            attn_output = attn_output.reshape(B, -1, L, D)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            return self.o_proj(attn_output), None

        module.forward = partial(modified_forward, module)
