"""QuanPC quantized KV-cache algorithm used by run_bin_v4 experiments.

QuanPC contains the model surgery, cache data structures, projection
learning, projection-based quantization, EAS token retention, and decode-time
attention over dense+quantized cache blocks.  It intentionally does not load a
model or run experiments at import time.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple
import math
import warnings

import torch
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP, apply_rotary_pos_emb
from transformers.processing_utils import Unpack

try:
    from quantpc.quanpc_cuda_extension.quanpc_cuda import (
        load_quanpc_extension,
        pack_projected_query_torch,
        pack_projected_query_cuda,
        quantize_pack_projected_query_cuda,
        quantize_pack_projected_query_cuda_out,
        project_quantize_pack_query_cuda,
        pack_quanpc_key_transposed_torch,
        pack_quanpc_value_1bit_transposed_torch,
        quanpc_dense_attention_numerator_cuda,
        quanpc_fused_output_projection_cuda,
        quanpc_fused_output_projection_cuda_out,
        quanpc_global_code_output_v1_exp8_tiled_gqa4_cuda,
        quanpc_global_output_v1_exp8_tiled_gqa4_fused_reduce_cuda,
        quanpc_global_code_output_v1_exp8_tiled_gqa4_atomic_cuda,
        quanpc_global_code_output_v1_exp8_tiled_gqa4_atomic_cuda_out,
    )
except Exception:
    load_quanpc_extension = None
    pack_projected_query_torch = None
    pack_projected_query_cuda = None
    quantize_pack_projected_query_cuda = None
    quantize_pack_projected_query_cuda_out = None
    project_quantize_pack_query_cuda = None
    pack_quanpc_key_transposed_torch = None
    pack_quanpc_value_1bit_transposed_torch = None
    quanpc_dense_attention_numerator_cuda = None
    quanpc_fused_output_projection_cuda = None
    quanpc_fused_output_projection_cuda_out = None
    quanpc_global_code_output_v1_exp8_tiled_gqa4_cuda = None
    quanpc_global_output_v1_exp8_tiled_gqa4_fused_reduce_cuda = None
    quanpc_global_code_output_v1_exp8_tiled_gqa4_atomic_cuda = None
    quanpc_global_code_output_v1_exp8_tiled_gqa4_atomic_cuda_out = None


CUDA_PROJECTED_QUERY_BITS = 8
CUDA_EXP_BITS_SUPPORTED = (1, 2, 4, 8)
EXP_QUANT_GROUP_SIZE = 64


def _validate_bit_range(name, value, min_bits, max_bits):
    if not min_bits <= value <= max_bits:
        raise ValueError(f"{name} must be in [{min_bits}, {max_bits}]; got {value}")
    return int(value)


def _validate_attention_quant_bits(projected_query_bits, exp_bits):
    return (
        _validate_bit_range("projected_query_bits", projected_query_bits, 2, 8),
        _validate_bit_range("exp_bits", exp_bits, 1, 16),
    )


@dataclass
class QuantProjection:
    """Projection learned offline/online and reused for binary quantization."""

    proj: torch.Tensor   # [B, H, C, D]

    @property
    def bits(self):
        return self.proj.shape[-2]


@dataclass
class QuantizedTensor:
    """Unpacked binary quantized representation of a cache block.

    ``codes`` stores ``{-1, +1}`` coordinates in unpacked form.  ``proj`` maps
    those coordinates back to the model head dimension, but attention pushes
    that projection onto query/output instead of materializing ``codes @ proj``.
    ``err`` is the per-token reconstruction error used by QuantKVCache to choose
    EAS tokens jointly for K and V.
    """

    codes: torch.Tensor  # [B, H, L, C]
    proj: torch.Tensor   # [B, H, C, D]
    err: torch.Tensor    # [B, H, L], float32
    packed_cuda: Optional[torch.Tensor] = None
    proj_t: Optional[torch.Tensor] = None  # [B, H, D, C], cached for decode-time query projection.

    @property
    def bits(self):
        return self.codes.shape[-1]


@dataclass
class MixedCacheTensor:
    """Cache view with one dense retained block and one quantized global block."""

    dense: torch.Tensor         # [B, H, sink + eas + local, D]
    glob: QuantizedTensor       # quantized non-retained global tokens
    use_cuda_v1_exp8: bool = False
    projected_query_bits: int = 8
    exp_bits: int = 4
    use_pruning: bool = True     # decode-time global block/token pruning (gqa4 CUDA only)
    prune_mode: int = 3
    log_eps: float = -10.0

    @property
    def device(self):
        return self.dense.device

    @property
    def dtype(self):
        return self.dense.dtype

    @property
    def shape(self):
        B, H, _, D = self.dense.shape
        glob_len = 0 if self.glob is None else self.glob.codes.shape[-2]
        L = self.dense.shape[-2] + glob_len
        return (B, H, L, D)

class QuantCache(Cache):
    def __init__(self, bits, sink_size = 64, local_size = 1024, refit_decoder = False):
        if bits <= 0:
            raise ValueError(f"bits must be positive; got {bits}")
        self.bits_per_token = bits
        self.sink_size = sink_size
        self.local_size = local_size
        self.refit_decoder = refit_decoder
        self.reset()

    def reset(self):
        self.sink = {}
        self.glob = {}
        self.locl = {}
        self.projections = {}
        self.seq_length = {}

    def _binary_codes(self, scores):
        return torch.where(
            scores >= 0,
            torch.ones_like(scores),
            -torch.ones_like(scores),
        )


    def _refit_decoder_from_codes(self, codes, target):
        """Fit decode-time reconstruction vectors for fixed binary codes."""
        decoder = torch.linalg.pinv(codes.float()) @ target.float()
        return decoder.to(dtype = target.dtype).contiguous()

    def _reconstruction_error(self, codes, target, decoder, chunk_size = 8192):
        B, H, L, _ = codes.shape
        err = torch.empty(B, H, L, device = target.device, dtype = torch.float32)
        decoder_fp32 = decoder.float()
        for start in range(0, L, chunk_size):
            end = min(start + chunk_size, L)
            recon = codes[:, :, start:end, :].float() @ decoder_fp32
            err[:, :, start:end] = (target[:, :, start:end, :].float() - recon).norm(dim = -1, p = 2)
        return err.contiguous()

    def _learn_projection_chunk(self, A, code_dim):
        """Learn binary projection vectors from the current residual."""
        B, H, L, D = A.shape
        residual = A
        proj = torch.empty(B, H, code_dim, D, device = A.device, dtype = A.dtype)
        for bit_idx in range(code_dim):
            Vh = torch.randn(B, H, D, device = A.device, dtype = A.dtype)
            AtA = residual.transpose(-1, -2) @ residual
            spectral_direction = torch.einsum("bhde,bhe->bhd", AtA, Vh)
            spectral_direction = torch.einsum("bhde,bhe->bhd", AtA, spectral_direction)
            scores = torch.einsum("bhld,bhd->bhl", residual, spectral_direction)
            code = self._binary_codes(scores)
            Vh = torch.einsum("bhld,bhl->bhd", residual, code) / float(L)
            residual = residual - torch.einsum("bhl,bhd->bhld", code, Vh)
            proj[:, :, bit_idx, :] = Vh
        return proj

    def find_projection(self, A, bits = None):
        """Learn a fixed-size binary projection."""
        bits = self.bits_per_token if bits is None else bits
        proj = self._learn_projection_chunk(A, bits)
        return QuantProjection(proj.contiguous())

    def quantize_with_projection(self, A, projection, refit_decoder = None):
        """Quantize A using precomputed binary projection directions."""
        if refit_decoder is None:
            refit_decoder = self.refit_decoder
        B, H, L, _ = A.shape
        C = projection.proj.shape[-2]
        codes = torch.empty(B, H, L, C, device = A.device, dtype = A.dtype)
        block_proj = torch.empty_like(projection.proj)
        residual = A
        for bit_idx in range(C):
            direction = projection.proj[:, :, bit_idx, :]
            scores = torch.einsum("bhld,bhd->bhl", residual, direction)
            Uq = self._binary_codes(scores)
            residual = residual - torch.einsum("bhl,bhd->bhld", Uq, direction)
            codes[:, :, :, bit_idx] = Uq
            block_proj[:, :, bit_idx, :] = direction
        if refit_decoder:
            block_proj = self._refit_decoder_from_codes(codes, A)
            token_err = self._reconstruction_error(codes, A, block_proj)
        else:
            token_err = residual.norm(dim = -1, p = 2).float()
        return QuantizedTensor(
            codes.contiguous(),
            block_proj.contiguous(),
            token_err.contiguous(),
            proj_t = block_proj.transpose(-1, -2).contiguous(),
        )

    def refit_quantized_tensor(self, block, target):
        """Refit decoder and token errors after EAS tokens are removed."""
        decoder = self._refit_decoder_from_codes(block.codes, target)
        err = self._reconstruction_error(block.codes, target, decoder)
        return QuantizedTensor(
            block.codes.contiguous(),
            decoder,
            err,
            block.packed_cuda,
            decoder.transpose(-1, -2).contiguous(),
        )

    def qpc(self, A):
        """Compatibility wrapper: find fixed projection, then quantize with it."""
        projection = self.find_projection(A)
        return self.quantize_with_projection(A, projection), projection

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.seq_length[layer_idx] if layer_idx in self.seq_length else 0

    def compress(self, glob, layer_idx):
        raise NotImplementedError
        
    def init_prefill(self, vecs, layer_idx):
        assert layer_idx not in self.sink
        assert layer_idx not in self.glob
        assert layer_idx not in self.locl
        total_len = vecs.shape[-2]
        self.seq_length[layer_idx] = total_len
        # Clamp sink/local so the three regions (sink | glob | local) form a
        # non-overlapping partition even when the sequence is short. Without
        # this, `vecs[:, :, -local_size:]` starts before `sink_size` and the
        # sink tokens get duplicated into the dense cache; the glob slice also
        # becomes empty, which later divides by zero in projection learning and
        # launches CUDA kernels with gridDim.x == 0.
        eff_sink = min(self.sink_size, total_len)
        eff_local = min(self.local_size, total_len - eff_sink)
        glob_end = total_len - eff_local
        self.sink[layer_idx] = vecs[:, :, :eff_sink, :].contiguous()
        self.locl[layer_idx] = vecs[:, :, glob_end:, :].contiguous()
        glob_vecs = vecs[:, :, eff_sink:glob_end, :]
        if glob_vecs.shape[-2] == 0:
            # No compressible global region: leave glob empty and skip
            # compression (projection learning would divide by glob_len == 0).
            self.glob[layer_idx] = None
        else:
            self.glob[layer_idx] = self.compress(glob_vecs, layer_idx)
        return glob_vecs

    def append_local(self, vecs, layer_idx):
        if self.locl.get(layer_idx) is not None:
            self.locl[layer_idx] = torch.cat([self.locl[layer_idx], vecs], dim = -2).contiguous()
        self.seq_length[layer_idx] = self.seq_length[layer_idx] + vecs.shape[-2]

class KQuantCache(QuantCache):
    def __init__(self, bits, sink_size = 64, local_size = 1024, zerorope = 30, refit_decoder = False):
        if zerorope < 0:
            raise ValueError(f"zerorope must be non-negative; got {zerorope}")
        super().__init__(bits = bits, sink_size = sink_size, local_size = local_size, refit_decoder = refit_decoder)
        self.zerorope = zerorope

    def quantization_target(self, glob):
        glob = glob.clone()
        half_head_dim = glob.shape[-1] // 2
        if self.zerorope > half_head_dim:
            raise ValueError(f"zerorope ({self.zerorope}) must be <= half_head_dim ({half_head_dim})")
        glob[:, :, :, :self.zerorope] = 0
        glob[:, :, :, half_head_dim:(half_head_dim + self.zerorope)] = 0
        return glob

    def compress(self, glob, layer_idx):
        glob = self.quantization_target(glob)
        quantized, projection = self.qpc(glob)
        self.projections[layer_idx] = projection
        return quantized
        
class VQuantCache(QuantCache):
    def __init__(self, bits, sink_size = 64, local_size = 1024, refit_decoder = False):
        super().__init__(bits = bits, sink_size = sink_size, local_size = local_size, refit_decoder = refit_decoder)

    def compress(self, glob, layer_idx):
        quantized, projection = self.qpc(glob)
        self.projections[layer_idx] = projection
        return quantized

class QuantKVCache(Cache):
    def __init__(
        self,
        sink_size = 64,
        local_size = 1024,
        k_bits = 128,
        v_bits = 256,
        projected_query_bits = 8,
        exp_bits = 4,
        eas = 0.01,
        zerorope = 30,
        use_cuda_v1_exp8 = False,
        refit_decoder = False,
        use_pruning = True,
        prune_mode = 3,
        log_eps = -10.0,
    ):
        projected_query_bits, exp_bits = _validate_attention_quant_bits(projected_query_bits, exp_bits)
        self.kcache = KQuantCache(k_bits, sink_size, local_size, zerorope, refit_decoder = refit_decoder)
        self.vcache = VQuantCache(v_bits, sink_size, local_size, refit_decoder = refit_decoder)
        self.projected_query_bits = projected_query_bits
        self.exp_bits = exp_bits
        self.eas = eas
        self.use_cuda_v1_exp8 = use_cuda_v1_exp8
        self.refit_decoder = refit_decoder
        self.use_pruning = use_pruning
        self.prune_mode = prune_mode
        self.log_eps = log_eps
        self.k_dense = {}
        self.v_dense = {}

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.kcache.get_seq_length(layer_idx)

    def _joint_eas_indices(self, k_block, v_block):
        """Select dense EAS tokens by K top-k, then V top-k from the remainder."""
        L = k_block.err.shape[-1]
        eas_size = max(1, int(L * self.eas))
        k_count = min(eas_size, L)
        k_indices = torch.topk(k_block.err, k = k_count, dim = -1).indices
        if k_count == L:
            return k_indices.sort(dim = -1).values

        v_count = min(eas_size, L - k_count)
        v_score = v_block.err.masked_fill(
            torch.zeros_like(v_block.err, dtype = torch.bool).scatter_(-1, k_indices, True),
            -torch.inf,
        )
        v_indices = torch.topk(v_score, k = v_count, dim = -1).indices
        return torch.cat([k_indices, v_indices], dim = -1).sort(dim = -1).values

    def _gather_tokens(self, x, indices):
        D = x.shape[-1]
        return torch.gather(x, dim = -2, index = indices.unsqueeze(-1).expand(-1, -1, -1, D)).contiguous()

    def _drop_tokens(self, block, keep_indices):
        L = block.codes.shape[-2]
        mask = torch.ones(block.codes.shape[:3], device = block.codes.device, dtype = torch.bool)
        mask.scatter_(-1, keep_indices, False)
        B, H, _, C = block.codes.shape
        codes = block.codes[mask].reshape(B, H, L - keep_indices.shape[-1], C).contiguous()
        err = block.err[mask].reshape(B, H, L - keep_indices.shape[-1]).contiguous()
        return QuantizedTensor(codes, block.proj, err, block.packed_cuda, block.proj_t)

    def _drop_token_states(self, states, keep_indices):
        L = states.shape[-2]
        mask = torch.ones(states.shape[:3], device = states.device, dtype = torch.bool)
        mask.scatter_(-1, keep_indices, False)
        B, H, _, D = states.shape
        return states[mask].reshape(B, H, L - keep_indices.shape[-1], D).contiguous()

    def _make_mixed(self, layer_idx):
        return (
            MixedCacheTensor(
                self.k_dense[layer_idx],
                self.kcache.glob[layer_idx],
                self.use_cuda_v1_exp8,
                self.projected_query_bits,
                self.exp_bits,
                self.use_pruning,
                self.prune_mode,
                self.log_eps,
            ),
            MixedCacheTensor(
                self.v_dense[layer_idx],
                self.vcache.glob[layer_idx],
                self.use_cuda_v1_exp8,
                self.projected_query_bits,
                self.exp_bits,
                self.use_pruning,
                self.prune_mode,
                self.log_eps,
            ),
        )

    def _prepare_cuda_v1_exp8(self, layer_idx):
        if not self.use_cuda_v1_exp8:
            return
        if (
            self.projected_query_bits != CUDA_PROJECTED_QUERY_BITS
            or self.exp_bits not in CUDA_EXP_BITS_SUPPORTED
        ):
            return
        if pack_quanpc_key_transposed_torch is None or pack_quanpc_value_1bit_transposed_torch is None:
            self.use_cuda_v1_exp8 = False
            return

        k_block = self.kcache.glob[layer_idx]
        v_block = self.vcache.glob[layer_idx]
        supported_dims = (128, 192, 256, 320, 384, 448, 512)
        supported = (
            k_block.codes.shape[-1] in supported_dims
            and v_block.codes.shape[-1] in supported_dims
            and k_block.codes.is_cuda
            and v_block.codes.is_cuda
        )
        if not supported:
            return

        k_shape = k_block.codes.shape
        v_shape = v_block.codes.shape
        k_block.packed_cuda = pack_quanpc_key_transposed_torch(k_block.codes).contiguous()
        k_block.codes = k_block.codes.new_empty(k_shape[0], k_shape[1], 0, k_shape[3])
        k_block.err = k_block.err.new_empty(k_shape[0], k_shape[1], 0)
        torch.cuda.empty_cache()

        v_block.packed_cuda = pack_quanpc_value_1bit_transposed_torch(v_block.codes).contiguous()
        v_block.codes = v_block.codes.new_empty(v_shape[0], v_shape[1], 0, v_shape[3])
        v_block.err = v_block.err.new_empty(v_shape[0], v_shape[1], 0)
        torch.cuda.empty_cache()
        
    def update(self, keys, vals, layer_idx, cache_kwargs):
        L = keys.shape[-2]
        if L > 1:
            key_glob = self.kcache.init_prefill(keys, layer_idx)
            val_glob = self.vcache.init_prefill(vals, layer_idx)
            if self.kcache.glob[layer_idx] is None:
                # Short sequence (total_len <= sink + local): there is no
                # compressible global region, so the whole sequence stays in the
                # full-precision dense cache. sink and local already partition
                # the sequence without overlap, so concatenating them reproduces
                # the original token order exactly once.
                self.k_dense[layer_idx] = torch.cat(
                    [self.kcache.sink[layer_idx], self.kcache.locl[layer_idx]], dim = -2
                ).contiguous()
                self.v_dense[layer_idx] = torch.cat(
                    [self.vcache.sink[layer_idx], self.vcache.locl[layer_idx]], dim = -2
                ).contiguous()
                self.kcache.sink[layer_idx] = None
                self.kcache.locl[layer_idx] = None
                self.vcache.sink[layer_idx] = None
                self.vcache.locl[layer_idx] = None
                self.kcache.projections.pop(layer_idx, None)
                self.vcache.projections.pop(layer_idx, None)
                return keys, vals
            eas_indices = self._joint_eas_indices(self.kcache.glob[layer_idx], self.vcache.glob[layer_idx])

            eas_keys = self._gather_tokens(key_glob, eas_indices)
            eas_vals = self._gather_tokens(val_glob, eas_indices)
            dense_keys = [self.kcache.sink[layer_idx], eas_keys, self.kcache.locl[layer_idx]]
            dense_vals = [self.vcache.sink[layer_idx], eas_vals, self.vcache.locl[layer_idx]]
            self.k_dense[layer_idx] = torch.cat(dense_keys, dim = -2).contiguous()
            self.v_dense[layer_idx] = torch.cat(dense_vals, dim = -2).contiguous()
            self.kcache.glob[layer_idx] = self._drop_tokens(self.kcache.glob[layer_idx], eas_indices)
            self.vcache.glob[layer_idx] = self._drop_tokens(self.vcache.glob[layer_idx], eas_indices)
            if self.refit_decoder:
                key_target = self._drop_token_states(self.kcache.quantization_target(key_glob), eas_indices)
                value_target = self._drop_token_states(val_glob, eas_indices)
                self.kcache.glob[layer_idx] = self.kcache.refit_quantized_tensor(self.kcache.glob[layer_idx], key_target)
                self.vcache.glob[layer_idx] = self.vcache.refit_quantized_tensor(self.vcache.glob[layer_idx], value_target)
            self.kcache.sink[layer_idx] = None
            self.kcache.locl[layer_idx] = None
            self.vcache.sink[layer_idx] = None
            self.vcache.locl[layer_idx] = None
            self.kcache.projections.pop(layer_idx, None)
            self.vcache.projections.pop(layer_idx, None)
            self._prepare_cuda_v1_exp8(layer_idx)

            return keys, vals

        self.kcache.append_local(keys, layer_idx)
        self.vcache.append_local(vals, layer_idx)
        self.k_dense[layer_idx] = torch.cat([self.k_dense[layer_idx], keys], dim = -2).contiguous()
        self.v_dense[layer_idx] = torch.cat([self.v_dense[layer_idx], vals], dim = -2).contiguous()
        return self._make_mixed(layer_idx)
        
def _repeat_quantized_tensor(block, repeat_factor):
    return QuantizedTensor(
        block.codes.repeat_interleave(repeat_factor, dim = 1),
        block.proj.repeat_interleave(repeat_factor, dim = 1),
        block.err.repeat_interleave(repeat_factor, dim = 1),
        block.packed_cuda.repeat_interleave(repeat_factor, dim = 1) if block.packed_cuda is not None else None,
        block.proj_t.repeat_interleave(repeat_factor, dim = 1) if block.proj_t is not None else None,
    )


def _repeat_mixed_cache(cache, target_heads):
    current_heads = cache.dense.shape[1]
    if current_heads == target_heads:
        return cache
    if target_heads % current_heads != 0:
        raise ValueError(f"query heads {target_heads} must be divisible by kv heads {current_heads}")
    repeat_factor = target_heads // current_heads
    return MixedCacheTensor(
        cache.dense.repeat_interleave(repeat_factor, dim = 1),
        _repeat_quantized_tensor(cache.glob, repeat_factor),
        cache.use_cuda_v1_exp8,
        cache.projected_query_bits,
        cache.exp_bits,
        cache.use_pruning,
        cache.prune_mode,
        cache.log_eps,
    )



def _can_use_cuda_v1_exp8(query_states, key_states, value_states):
    if not key_states.use_cuda_v1_exp8 or not value_states.use_cuda_v1_exp8:
        return False
    has_cuda_kernel = (
        quanpc_global_code_output_v1_exp8_tiled_gqa4_atomic_cuda is not None
        or quanpc_global_code_output_v1_exp8_tiled_gqa4_cuda is not None
    )
    has_query_packer = pack_projected_query_cuda is not None or pack_projected_query_torch is not None
    if not has_cuda_kernel or not has_query_packer:
        return False
    if not query_states.is_cuda:
        return False
    q_heads = query_states.shape[1]
    kv_heads = key_states.dense.shape[1]
    if q_heads % kv_heads != 0 or q_heads // kv_heads not in (1, 4, 8):
        return False
    if (
        key_states.projected_query_bits != CUDA_PROJECTED_QUERY_BITS
        or key_states.exp_bits not in CUDA_EXP_BITS_SUPPORTED
    ):
        return False
    supported_dims = (128, 192, 256, 320, 384, 448, 512)
    key_code_dim = key_states.glob.packed_cuda.shape[2] * 64 if key_states.glob.packed_cuda is not None else key_states.glob.codes.shape[-1]
    value_code_dim = value_states.glob.packed_cuda.shape[3] if value_states.glob.packed_cuda is not None else value_states.glob.codes.shape[-1]
    if key_code_dim not in supported_dims or value_code_dim not in supported_dims:
        return False
    if key_states.glob.packed_cuda is None or value_states.glob.packed_cuda is None:
        return False
    return True


def _quantize_projected_query(projected_query, bits):
    """Symmetric signed quantization used before the global key-code dot."""
    bits = _validate_bit_range("projected_query_bits", bits, 2, 8)
    projected_query = projected_query.float()
    max_code = (1 << (bits - 1)) - 1
    max_abs = projected_query.abs().amax(dim = -1).clamp_min(torch.finfo(torch.float32).eps)
    query_scale = (max_abs / float(max_code)).contiguous()
    quantized = torch.round(projected_query / query_scale[..., None]).clamp(-max_code, max_code)
    return quantized, query_scale.float()


def _dequantize_projected_query(projected_query, bits):
    quantized, query_scale = _quantize_projected_query(projected_query, bits)
    return (quantized * query_scale[..., None]).contiguous(), query_scale


def _quantize_projected_query_int8(projected_query):
    quantized, query_scale = _quantize_projected_query(projected_query, CUDA_PROJECTED_QUERY_BITS)
    return quantized.to(torch.int8).contiguous(), query_scale


def _quantize_exp_weights(exp_weights, bits, group_size = EXP_QUANT_GROUP_SIZE):
    """Quantize exp weights per contiguous token group, matching CUDA's per-64-token exp quantization scheme."""
    bits = _validate_bit_range("exp_bits", bits, 1, 16)
    exp_weights = exp_weights.float()
    L = exp_weights.shape[-1]
    groups = (L + group_size - 1) // group_size
    padded_L = groups * group_size
    if padded_L != L:
        exp_weights = torch.nn.functional.pad(exp_weights, (0, padded_L - L))
    grouped = exp_weights.reshape(*exp_weights.shape[:-1], groups, group_size)
    max_weight = grouped.amax(dim = -1, keepdim = True)
    max_code = float((1 << bits) - 1)
    inv_scale = torch.where(max_weight > 0, max_code / max_weight, torch.zeros_like(max_weight))
    quantized = torch.round(grouped * inv_scale).clamp(0, max_code)
    dequantized = torch.where(max_weight > 0, quantized * (max_weight / max_code), torch.zeros_like(quantized))
    return dequantized.reshape(*exp_weights.shape[:-1], padded_L)[..., :L].contiguous()


def _dense_attention_numerator_flash_gqa(query_states, key_states, value_states, scale):
    # The internal flash op accepts GQA layout directly ([B,QH,S,D] query and
    # [B,KVH,L,D] K/V) while also returning the row logsumexp needed by the
    # QuanPC global path. Avoiding repeat_interleave is the main speed win here.
    output, dense_lse, *_ = torch.ops.aten._scaled_dot_product_flash_attention(
        query_states,
        key_states,
        value_states,
        0.0,
        False,
        False,
        scale = float(scale),
    )
    return output, dense_lse.contiguous()



def _project_query_gqa(query_states, key_proj, key_proj_t = None, out = None):
    """Project query with GQA layout without materializing repeated projection heads."""
    B, QH, S, D = query_states.shape
    KVH = key_proj.shape[1]
    C = key_proj.shape[2]
    group_size = QH // KVH
    query_grouped = query_states.reshape(B, KVH, group_size, S, D).reshape(B * KVH, group_size * S, D)
    if key_proj_t is None:
        proj_grouped = key_proj.transpose(-1, -2).reshape(B * KVH, D, C)
    else:
        proj_grouped = key_proj_t.reshape(B * KVH, D, C)
    if out is None:
        projected = torch.bmm(query_grouped, proj_grouped)
        return projected.reshape(B, KVH, group_size, S, C).reshape(B, QH, S, C)
    projected_grouped = out.reshape(B, KVH, group_size, S, C).reshape(B * KVH, group_size * S, C)
    torch.bmm(query_grouped, proj_grouped, out = projected_grouped)
    return out


@dataclass
class _CudaAttentionWorkspace:
    projected_query: torch.Tensor
    packed_query: torch.Tensor
    query_scale: torch.Tensor
    glob_scratch: torch.Tensor
    glob_code_output: torch.Tensor
    glob_den: torch.Tensor
    attn_output: torch.Tensor


_cuda_attention_workspaces = {}
_cuda_dense_flash_streams = {}


def _workspace_device_key(tensor):
    device = tensor.device
    return (device.type, device.index if device.index is not None else torch.cuda.current_device())


def _get_cuda_dense_flash_stream(tensor):
    key = _workspace_device_key(tensor)
    stream = _cuda_dense_flash_streams.get(key)
    if stream is None:
        stream = torch.cuda.Stream(device = tensor.device)
        _cuda_dense_flash_streams[key] = stream
    return stream


def _get_cuda_attention_workspace(query_states, key_states, value_states):
    B, QH, S, D = query_states.shape
    Ck = key_states.glob.proj.shape[-2]
    Cv = value_states.glob.proj.shape[-2]
    key = (
        _workspace_device_key(query_states),
        query_states.dtype,
        B,
        QH,
        S,
        D,
        Ck,
        Cv,
    )
    workspace = _cuda_attention_workspaces.get(key)
    if workspace is not None:
        return workspace
    device = query_states.device
    glob_code_numel = B * QH * S * Cv
    glob_den_numel = B * QH * S
    glob_scratch = torch.empty(glob_code_numel + glob_den_numel, device = device, dtype = torch.float32)
    workspace = _CudaAttentionWorkspace(
        projected_query = torch.empty(B, QH, S, Ck, device = device, dtype = query_states.dtype),
        packed_query = torch.empty(B, QH, S, Ck // 64, CUDA_PROJECTED_QUERY_BITS, device = device, dtype = torch.long),
        query_scale = torch.empty(B, QH, S, device = device, dtype = torch.float32),
        glob_scratch = glob_scratch,
        glob_code_output = glob_scratch[:glob_code_numel].view(B, QH, S, Cv),
        glob_den = glob_scratch[glob_code_numel:].view(B, QH, S),
        attn_output = torch.empty(B, QH, S, D, device = device, dtype = query_states.dtype),
    )
    _cuda_attention_workspaces[key] = workspace
    return workspace


def _project_query_to_workspace(query_states, key_states, workspace):
    return _project_query_gqa(
        query_states,
        key_states.glob.proj,
        key_states.glob.proj_t,
        out = workspace.projected_query,
    )


def _dense_flash_and_project_workspace(query_states, key_states, value_states, scale, workspace):
    current_stream = torch.cuda.current_stream(query_states.device)
    dense_stream = _get_cuda_dense_flash_stream(query_states)
    dense_stream.wait_stream(current_stream)
    with torch.cuda.stream(dense_stream):
        dense_output, dense_lse_for_global = _dense_attention_numerator_flash_gqa(
            query_states,
            key_states.dense,
            value_states.dense,
            scale,
        )
    projected_query = _project_query_to_workspace(query_states, key_states, workspace)
    current_stream.wait_stream(dense_stream)
    return dense_output, dense_lse_for_global, projected_query


_PRUNING_WARNED_GQA: set = set()


def _warn_pruning_unavailable(gqa):
    if gqa in _PRUNING_WARNED_GQA:
        return
    _PRUNING_WARNED_GQA.add(gqa)
    warnings.warn(
        f"QuanPC decode pruning requested but unavailable for gqa={gqa}; the pruned "
        f"CUDA kernel only supports gqa==4. Falling back to the non-pruned decode path.",
        RuntimeWarning,
        stacklevel=2,
    )


def _run_cuda_decode_tail(
    ext,
    projected_query,
    workspace,
    key_states,
    value_states,
    dense_output,
    dense_lse_for_global,
    scale,
):
    if hasattr(ext, "quanpc_decode_tail_v1_exp8_tiled_gqa4_atomic_fn"):
        gqa = dense_output.shape[1] // key_states.dense.shape[1]
        use_pruning = getattr(key_states, "use_pruning", False)
        has_pruned = hasattr(ext, "quanpc_decode_tail_v1_exp8_tiled_gqa4_atomic_pruned_fn")
        if use_pruning and gqa == 4 and has_pruned:
            ext.quanpc_decode_tail_v1_exp8_tiled_gqa4_atomic_pruned_fn(
                projected_query,
                workspace.packed_query,
                key_states.glob.packed_cuda,
                value_states.glob.packed_cuda,
                workspace.query_scale,
                dense_lse_for_global,
                dense_output.contiguous(),
                workspace.glob_code_output,
                workspace.glob_den,
                value_states.glob.proj.contiguous(),
                workspace.attn_output,
                int(key_states.exp_bits),
                256,
                1,
                float(scale),
                int(getattr(key_states, "prune_mode", 3)),
                float(getattr(key_states, "log_eps", -10.0)),
            )
            return workspace.attn_output
        if use_pruning and gqa != 4:
            _warn_pruning_unavailable(gqa)
        ext.quanpc_decode_tail_v1_exp8_tiled_gqa4_atomic_fn(
            projected_query,
            workspace.packed_query,
            key_states.glob.packed_cuda,
            value_states.glob.packed_cuda,
            workspace.query_scale,
            dense_lse_for_global,
            dense_output.contiguous(),
            workspace.glob_code_output,
            workspace.glob_den,
            value_states.glob.proj.contiguous(),
            workspace.attn_output,
            int(key_states.exp_bits),
            256,
            1,
            float(scale),
        )
    else:
        ext.quanpc_quantize_pack_projected_query_fn(
            projected_query,
            workspace.packed_query,
            workspace.query_scale,
        )
        torch._foreach_zero_([workspace.glob_code_output, workspace.glob_den])
        ext.quanpc_global_code_output_v1_exp8_tiled_gqa4_atomic_fn(
            workspace.packed_query,
            key_states.glob.packed_cuda,
            value_states.glob.packed_cuda,
            workspace.query_scale,
            dense_lse_for_global,
            workspace.glob_code_output,
            workspace.glob_den,
            int(key_states.exp_bits),
            256,
            1,
            float(scale),
        )
        ext.quanpc_fused_output_projection_fn(
            dense_output.contiguous(),
            workspace.glob_code_output,
            workspace.glob_den,
            value_states.glob.proj.contiguous(),
            workspace.attn_output,
        )
    return workspace.attn_output


def _dense_attention_numerator(query_states, key_states, value_states, scale):
    dense_scores = torch.matmul(query_states, key_states.transpose(-1, -2)) * scale
    dense_lse = torch.logsumexp(dense_scores.float(), dim = -1, keepdim = True)
    dense_weights = torch.exp(dense_scores.float() - dense_lse).to(query_states.dtype)
    return torch.matmul(dense_weights, value_states), dense_lse


def _quantized_global_attention_numerator(
    query_states,
    key_block,
    value_block,
    dense_lse,
    scale,
    projected_query_bits,
    exp_bits,
):
    glob_scores = _quantized_global_scores(query_states, key_block, projected_query_bits) * scale
    glob_weights = torch.exp(glob_scores.float() - dense_lse)
    glob_weights = _quantize_exp_weights(glob_weights, exp_bits)
    glob_den = glob_weights.sum(dim = -1, keepdim = True)
    glob_output = _quantized_global_output(glob_weights, value_block)
    return glob_output, glob_den


def _quantized_cache_attention_forward_cuda_v1_exp8(query_states, key_states, value_states, scale):
    target_heads = query_states.shape[1]
    repeat_factor = target_heads // key_states.dense.shape[1]
    can_use_dense_flash = (
        query_states.shape[-2] == 1
        and query_states.dtype == torch.bfloat16
        and key_states.dense.dtype == torch.bfloat16
        and value_states.dense.dtype == torch.bfloat16
        and query_states.shape[1] % key_states.dense.shape[1] == 0
        and key_states.dense.shape[-2] >= 256
        and hasattr(torch.ops.aten, "_scaled_dot_product_flash_attention")
    )
    can_use_dense_cuda = (
        quanpc_dense_attention_numerator_cuda is not None
        and query_states.dtype in (torch.bfloat16, torch.float32)
        and key_states.dense.shape[-1] <= 256
        and key_states.dense.shape[-2] <= 12288
    )
    can_use_workspace_without_dense = (
        load_quanpc_extension is not None
        and quantize_pack_projected_query_cuda_out is not None
        and quanpc_global_code_output_v1_exp8_tiled_gqa4_atomic_cuda_out is not None
        and quanpc_fused_output_projection_cuda_out is not None
        and query_states.dtype in (torch.bfloat16, torch.float32)
        and value_states.glob.proj.shape[-2] <= 512
        and value_states.glob.proj.shape[-1] <= 256
    )
    if can_use_dense_flash and can_use_workspace_without_dense:
        workspace = _get_cuda_attention_workspace(query_states, key_states, value_states)
        ext = load_quanpc_extension()
        dense_output, dense_lse_for_global, projected_query = _dense_flash_and_project_workspace(
            query_states,
            key_states,
            value_states,
            scale,
            workspace,
        )
        return _run_cuda_decode_tail(
            ext,
            projected_query,
            workspace,
            key_states,
            value_states,
            dense_output,
            dense_lse_for_global,
            scale,
        )

    if can_use_dense_flash:
        dense_output, dense_lse_for_global = _dense_attention_numerator_flash_gqa(
            query_states,
            key_states.dense,
            value_states.dense,
            scale,
        )
    elif can_use_dense_cuda:
        dense_output, dense_lse_for_global = quanpc_dense_attention_numerator_cuda(
            query_states,
            key_states.dense,
            value_states.dense,
            scale,
        )
    else:
        dense_key = key_states.dense.repeat_interleave(repeat_factor, dim = 1)
        dense_value = value_states.dense.repeat_interleave(repeat_factor, dim = 1)
        dense_output, dense_lse = _dense_attention_numerator(query_states, dense_key, dense_value, scale)
        dense_lse_for_global = dense_lse.squeeze(-1).contiguous()

    can_use_workspace = (
        load_quanpc_extension is not None
        and quantize_pack_projected_query_cuda_out is not None
        and quanpc_global_code_output_v1_exp8_tiled_gqa4_atomic_cuda_out is not None
        and quanpc_fused_output_projection_cuda_out is not None
        and dense_output.dtype in (torch.bfloat16, torch.float32)
        and value_states.glob.proj.shape[-2] <= 512
        and value_states.glob.proj.shape[-1] <= 256
    )
    if can_use_workspace:
        workspace = _get_cuda_attention_workspace(query_states, key_states, value_states)
        ext = load_quanpc_extension()
        projected_query = _project_query_to_workspace(query_states, key_states, workspace)
        return _run_cuda_decode_tail(
            ext,
            projected_query,
            workspace,
            key_states,
            value_states,
            dense_output,
            dense_lse_for_global,
            scale,
        )

    projected_query = _project_query_gqa(query_states, key_states.glob.proj, key_states.glob.proj_t)
    if quantize_pack_projected_query_cuda is not None and projected_query.dtype in (torch.bfloat16, torch.float32):
        packed_query, query_scale = quantize_pack_projected_query_cuda(projected_query)
    else:
        quant_projected_query, query_scale = _quantize_projected_query_int8(projected_query)
        pack_query = pack_projected_query_cuda if pack_projected_query_cuda is not None else pack_projected_query_torch
        packed_query = pack_query(quant_projected_query)

    cuda_kernel = (
        quanpc_global_code_output_v1_exp8_tiled_gqa4_atomic_cuda
        or quanpc_global_code_output_v1_exp8_tiled_gqa4_cuda
    )
    glob_code_output, glob_den = cuda_kernel(
        packed_query,
        key_states.glob.packed_cuda,
        value_states.glob.packed_cuda,
        query_scale,
        dense_lse_for_global,
        scale,
        exp_bits = key_states.exp_bits,
        tile_size = 256,
        key_transposed = True,
    )
    can_use_fused_output = (
        quanpc_fused_output_projection_cuda is not None
        and dense_output.dtype in (torch.bfloat16, torch.float32)
        and value_states.glob.proj.shape[-2] <= 512
        and value_states.glob.proj.shape[-1] <= 256
    )
    if can_use_fused_output:
        return quanpc_fused_output_projection_cuda(
            dense_output.contiguous(),
            glob_code_output.contiguous(),
            glob_den.contiguous(),
            value_states.glob.proj.contiguous(),
        )

    value_proj = value_states.glob.proj.repeat_interleave(repeat_factor, dim = 1)
    glob_output = torch.matmul(glob_code_output.to(value_proj.dtype), value_proj)
    return (dense_output + glob_output) / (1 + glob_den[..., None].to(dense_output.dtype))


def _quantized_global_scores(query_states, key_block, projected_query_bits):
    # key ~= codes @ proj, so q @ key.T == quantize(q @ proj.T) @ codes.T.
    projected_query = torch.matmul(query_states, key_block.proj.transpose(-1, -2))
    projected_query, _ = _dequantize_projected_query(projected_query, projected_query_bits)
    return torch.matmul(projected_query.to(key_block.codes.dtype), key_block.codes.transpose(-1, -2))


def _quantized_global_output(attn_probs, value_block):
    # value ~= codes @ proj, so p @ value == (p @ codes) @ proj.
    output_codes = torch.matmul(attn_probs.to(value_block.codes.dtype), value_block.codes)
    return torch.matmul(output_codes, value_block.proj)


def quantized_cache_attention_forward(query_states, key_states, value_states):
    """Attention over dense retained KV and unpacked quantized global KV.

    Decode uses dense-cache logsumexp as the softmax offset, then accumulates
    dense and quantized-global numerators separately.  This avoids concatenating
    score/probability tensors while keeping one normalization over both cache
    regions.
    """
    target_heads = query_states.shape[1]
    head_dim = query_states.shape[-1]
    scale = 1 / math.sqrt(head_dim)
    if _can_use_cuda_v1_exp8(query_states, key_states, value_states):
        return _quantized_cache_attention_forward_cuda_v1_exp8(query_states, key_states, value_states, scale)

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
        if key_states.glob is not None:
            return quantized_cache_attention_forward(query_states, key_states, value_states)
        # Empty global cache (short sequence): only the dense region exists, so
        # fall through to the same dense attention path as the non-quantized case.
        key_states = key_states.dense
        value_states = value_states.dense

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        is_causal = Lq != 1,
        enable_gqa = True,
    )
    return attn_output

kproj_cache = {}
vproj_cache = {}

def find_orth_proj(X):
    U, S, V = torch.svd((X.transpose(-1, -2) @ X).float())
    S = S.sqrt()
    U, S, V = U.to(torch.bfloat16), S.to(torch.bfloat16), V.to(torch.bfloat16)
    A = (V * S.unsqueeze(-2)) @ V.transpose(-1, -2)
    Ainv = (V / S.unsqueeze(-2)) @ V.transpose(-1, -2)
    return A, Ainv
    

def chunked_llama_mlp_forward(self, x, chunk_size = 8192):
    if x.shape[-2] <= chunk_size:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    outputs = []
    for x_chunk in x.split(chunk_size, dim = -2):
        outputs.append(self.down_proj(self.act_fn(self.gate_proj(x_chunk)) * self.up_proj(x_chunk)))
    return torch.cat(outputs, dim = -2)

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
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
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
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
    
            attn_output = eager_attention_forward(query_states, key_states, value_states)
            attn_output = attn_output.reshape(B, H, -1, D)
            attn_output = attn_output @ vproj_cache[self.layer_idx][1].transpose(-1, -2)
            attn_output = attn_output.reshape(B, -1, L, D)

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output, None
        
        module.forward = partial(modified_forward, module)
