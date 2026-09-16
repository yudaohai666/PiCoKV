"""
BPC: binary-hash-based sparse attention.

- find_binary_projection: learn projection that compresses keys to 64-bit hashes (LSH-like).
- compute_hashscores: estimate Q-K similarity via hash codes, skipping full attention.
- LayerCache: prefill caches all KV + hashes and records high-error tokens; decode picks
  top-k tokens by hash score, force-selecting err_topk to bound precision loss.
"""

from typing import Optional, Tuple
import torch
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb
from transformers.models.mistral.modeling_mistral import MistralAttention
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
from transformers.cache_utils import Cache
from functools import partial
import os
import transformers

# transformers 4.53+ attention.forward returns 2 values (not 3).
def _parse_version(version_str):
    """Parse major.minor from version string."""
    import re
    match = re.match(r'^(\d+)\.(\d+)', version_str)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return (0, 0)

_TRANSFORMERS_VERSION = _parse_version(transformers.__version__)
_USE_NEW_CACHE_API = _TRANSFORMERS_VERSION >= (4, 53)

# Per-process flag for CUDA kernel load state.
_cuda_kernel_loaded = False

def _load_cuda_kernel(device_id=None):

    global _cuda_kernel_loaded

    if device_id is not None:
        torch.cuda.set_device(device_id)

    current_device = torch.cuda.current_device()

    # Try direct import first (already-compiled case).
    try:
        import binary_proj
        if not _cuda_kernel_loaded:
            print(f"[BPC] CUDA kernel imported successfully on GPU {current_device}")
            _cuda_kernel_loaded = True
        return True
    except ImportError:
        pass
    
    # Fall back to JIT compile.
    try:
        from torch.utils.cpp_extension import load
        import filelock

        curr_path = os.path.dirname(os.path.abspath(__file__))
        src_files = [os.path.join(curr_path, 'extension', file) for file in ['cuda_kernel.cu', 'torch_extension.cpp']]

        # File lock so concurrent processes don't race on compilation.
        lock_file = os.path.join(curr_path, 'extension', '.compile.lock')
        os.makedirs(os.path.dirname(lock_file), exist_ok=True)

        with filelock.FileLock(lock_file, timeout=300):
            # Recheck inside the lock — another process may have finished.
            try:
                import binary_proj
                if not _cuda_kernel_loaded:
                    print(f"[BPC] CUDA kernel imported successfully on GPU {current_device} (compiled by another process)")
                    _cuda_kernel_loaded = True
                return True
            except ImportError:
                pass
            
            print(f"[BPC] JIT compiling CUDA kernel on GPU {current_device}...")
            load(
                'binary_proj', 
                src_files, 
                extra_cflags=['-O3'],
                extra_ldflags=['-O3'],
                verbose=False
            )
        
        print(f"[BPC] CUDA kernel JIT compiled successfully on GPU {current_device}")
        _cuda_kernel_loaded = True
        return True
    except Exception as e:
        print(f"[BPC] Failed to load CUDA kernel on GPU {current_device}: {e}")
        return False

# PyTorch fallback implementation.
from .bpc_pytorch import (
    find_binary_projection as find_binary_projection_pytorch,
    binary_project_pytorch,
    compute_errors_pytorch,
    compute_hashscores_pytorch,
)

SUPPORTED_ATTENTION_CLASSES = (LlamaAttention, MistralAttention, Qwen2Attention)

DEFAULT_CONFIG = {
    "enable": True,
    "fix_layers": [],         # layers that skip sparse attention
    "mask_out": 0.98,         # token_budget = context_len * (1 - mask_out)
    "min_remain": 128,
    "token_budget": None,     # if set, overrides mask_out
    "err_ratio": 0.1,         # err_k = token_budget * err_ratio
    "use_offline_proj": False,
    "offline_proj_path": None,
    "use_cuda_kernel": False,
}


def compute_hashscores_cuda(probe, hashcode, T, hash_T):
    """Fused CUDA kernel for hash similarity (quantize+pack+hash in one launch)."""
    import binary_proj
    
    B, H, S, D = probe.shape
    B, H, L = hashcode.shape
    assert S == 4        # 4 queries per group
    assert L % 256 == 0  # length must be multiple of 256
    assert D == 64       # 64-bit hash
    
    device = probe.device
    
    if hashcode.device != device:
        hashcode = hashcode.to(device)
    
    with torch.cuda.device(device):
        if not hashcode.is_contiguous():
            hashcode = hashcode.contiguous()
        if not probe.is_contiguous():
            probe = probe.contiguous()
        
        # Fused kernel handles quantize+pack+hash and boundary masking internally.
        scores = torch.empty(B, H, L, dtype=torch.short, device=device)
        binary_proj.fused_quant_pack_probe_hash64_fn(probe, hashcode, scores, T, hash_T)

    return scores


def find_binary_projection(A, num_bits, num_iters=4):

    return find_binary_projection_pytorch(A, num_bits, num_iters)


def binary_project_cuda(A, quan_proj):

    import binary_proj
    
    B, H, L, D = A.shape

    # All tensors must live on the same device.
    device = A.device

    if quan_proj.device != device:
        quan_proj = quan_proj.to(device)

    # Offline proj is [H, num_bits, D]; expand to [B, H, num_bits, D].
    if quan_proj.dim() == 3:
        quan_proj = quan_proj.unsqueeze(0).expand(B, -1, -1, -1)
    elif quan_proj.shape[0] == 1 and B > 1:
        quan_proj = quan_proj.expand(B, -1, -1, -1)
    
    with torch.cuda.device(device):
        if not A.is_contiguous():
            A = A.contiguous()
        if not quan_proj.is_contiguous():
            quan_proj = quan_proj.contiguous()
        y = torch.empty(B, H, L, device=device, dtype=torch.long)
        binary_proj.binary_project_dim128_hash64_fn(A, quan_proj, y)
    return y


def compute_errors_cuda(keys, quant_keys, proj):
    """CUDA kernel: reconstruction error of binary quantization."""
    import binary_proj

    B, H, L = quant_keys.shape

    # Use keys.device as the reference device.
    device = keys.device

    if quant_keys.device != device:
        quant_keys = quant_keys.to(device)
    if proj.device != device:
        proj = proj.to(device)

    if proj.dim() == 3:
        proj = proj.unsqueeze(0).expand(B, -1, -1, -1)
    elif proj.shape[0] == 1 and B > 1:
        proj = proj.expand(B, -1, -1, -1)

    with torch.cuda.device(device):
        # Unpack hash to 64 bits.
        quant_keys_unpacked = torch.empty(B, H, L, 64, dtype=torch.int, device=device)
        binary_proj.format_bits_hash64_fn(quant_keys_unpacked, quant_keys, False)
        # Reconstruct approximate key and measure L2 distance.
        approx_keys = quant_keys_unpacked.to(proj.dtype) @ proj
        errs = torch.norm(keys - approx_keys, p=2, dim=-1)
    return errs


def binary_project(A, quan_proj, use_cuda_kernel=False):
    if use_cuda_kernel:
        return binary_project_cuda(A, quan_proj)
    else:
        return binary_project_pytorch(A, quan_proj)


def compute_errors(keys, quant_keys, proj, use_cuda_kernel=False):
    if use_cuda_kernel:
        return compute_errors_cuda(keys, quant_keys, proj)
    else:
        return compute_errors_pytorch(keys, quant_keys, proj)


def compute_hashscores(probe, hashcode, T, hash_T, use_cuda_kernel=False):
    if use_cuda_kernel:
        return compute_hashscores_cuda(probe, hashcode, T, hash_T)
    else:
        return compute_hashscores_pytorch(probe, hashcode, T, hash_T)

class LayerCache:

    def __init__(self, mask_out, min_remain, token_budget, err_ratio, use_cuda_kernel=False, proj=None):

        self.mask_out = mask_out
        self.min_remain = min_remain
        self.token_budget = token_budget
        self.err_ratio = err_ratio
        self.use_cuda_kernel = use_cuda_kernel
        self.offline_proj = proj  # offline projection (if provided)
        self.proj = proj
        self.reset()

    def reset(self):
        self.keys = None
        self.values = None
        self.hashcodes = None
        self.err_topk = None
        self.length = 0
        self.hash_length = 0
        self.k = None
        self.err_k = None
        self.prefill_complete = False
        self.proj_T = None  # precomputed proj transpose, reused across decode steps
        # Online mode: clear proj so next prefill relearns it for the new context length.
        if self.offline_proj is None:
            self.proj = None
        else:
            self.proj = self.offline_proj

    def _compute_budget(self, context_len):
        if self.token_budget is not None:
            k = self.token_budget
        else:
            k = max(int(context_len * (1 - self.mask_out)), self.min_remain)
        
        k = min(k, context_len)
        err_k = max(int(k * self.err_ratio), 1)
        err_k = min(err_k, context_len)
        
        return k, err_k

    def update(self, queries, keys, values, sparse=True):
        """Update KV cache. sparse=False disables hash-based selection."""
        B, Hq, L, D = queries.shape
        _, Hk, _, _ = keys.shape

        # ===== Prefill (single chunk) =====
        if L > 1:
            # Reserve 256 extra slots to reduce reallocations during decode.
            rounded_L = ((L - 1) // 256 + 1) * 256 + 256
            self.keys = torch.empty(B, Hk, rounded_L, D, dtype=keys.dtype, device=keys.device)
            self.values = torch.empty(B, Hk, rounded_L, D, dtype=keys.dtype, device=keys.device)
            self.hashcodes = torch.empty(B, Hk, rounded_L, dtype=torch.long, device=keys.device)
            
            self.keys[:, :, :L, :] = keys
            self.values[:, :, :L, :] = values
            self.length = L

            self.k, self.err_k = self._compute_budget(L)

            if sparse:
                # Learn projection if no offline one is set.
                if self.proj is None:
                    _, self.proj = find_binary_projection(keys, 64)
                else:
                    if self.proj.device != keys.device or self.proj.dtype != keys.dtype:
                        self.proj = self.proj.to(device=keys.device, dtype=keys.dtype)

                # Precompute proj_T (expand + transpose + contiguous) once.
                proj = self.proj
                if proj.dim() == 3:
                    proj = proj.unsqueeze(0).expand(B, -1, -1, -1)
                elif proj.shape[0] == 1 and B > 1:
                    proj = proj.expand(B, -1, -1, -1)
                self.proj_T = proj.transpose(-1, -2).contiguous()

                # Hash codes for all keys (packed long).
                if self.use_cuda_kernel:
                    hashcode = binary_project_cuda(keys, self.proj)
                else:
                    hashcode = binary_project_pytorch(keys, self.proj)
                self.hashcodes[:, :, :L] = hashcode
                self.hash_length = L

                # Tokens with the largest quantization error are force-selected during decode.
                if self.use_cuda_kernel:
                    errors = compute_errors_cuda(keys, hashcode, self.proj)
                else:
                    errors = compute_errors_pytorch(keys, hashcode, self.proj)
                self.err_topk = torch.topk(errors, k=self.err_k, dim=-1, sorted=False).indices
            
            self.prefill_complete = True
            return keys, values
        else:
            # ===== Decode =====
            rounded_L = (self.length // 256 + 1) * 256 + 256
            if rounded_L > self.keys.shape[-2]:
                originals = (self.keys, self.values, self.hashcodes)

                self.keys = torch.empty(B, Hk, rounded_L, D, dtype=keys.dtype, device=keys.device)
                self.values = torch.empty(B, Hk, rounded_L, D, dtype=keys.dtype, device=keys.device)
                self.hashcodes = torch.empty(B, Hk, rounded_L, dtype=torch.long, device=keys.device)
                
                self.keys[:, :, :self.length, :] = originals[0][:, :, :self.length, :]
                self.values[:, :, :self.length, :] = originals[1][:, :, :self.length, :]
                self.hashcodes[:, :, :self.length] = originals[2][:, :, :self.length]
            
            self.keys[:, :, self.length, :] = keys[:, :, 0, :]
            self.values[:, :, self.length, :] = values[:, :, 0, :]
            self.length = self.length + 1

            if not sparse:
                return self.keys[:, :, :self.length, :], self.values[:, :, :self.length, :]

            # actual_k = prefill budget + new (un-hashed) decode tokens.
            # The new tokens have no hash codes and are force-selected, so adding them here
            # prevents them from displacing the original-context budget.
            actual_k = min(self.k + (self.length - self.hash_length), self.length)

            # GQA reshape; reuse precomputed proj_T.
            queries = queries.reshape(B, Hk, Hq // Hk, D)
            probe = queries @ self.proj_T

            if self.use_cuda_kernel:
                hashscores = compute_hashscores_cuda(probe, self.hashcodes, self.length, self.hash_length)
            else:
                hashscores = compute_hashscores_pytorch(
                    probe,
                    self.hashcodes,
                    self.length,
                    self.hash_length
                )

            # Force-select high-error tokens by setting their score high.
            hashscores.scatter_(dim=2, index=self.err_topk, value=10000)

            token_idxes = torch.topk(hashscores, k=actual_k, dim=-1, sorted=False).indices

            # token_idxes: [B, Hk, k] -> [B, Hk, k, D]
            token_idxes_expanded = token_idxes.unsqueeze(-1).expand(-1, -1, -1, D)
            selected_keys = torch.gather(self.keys, dim=2, index=token_idxes_expanded)
            selected_values = torch.gather(self.values, dim=2, index=token_idxes_expanded)
            return selected_keys, selected_values


class AllCache(Cache):

    def __init__(self, mask_out, min_remain, token_budget, err_ratio, use_cuda_kernel=False, projs=None):
        super().__init__()
        self.cache = []
        self.mask_out = mask_out
        self.min_remain = min_remain
        self.token_budget = token_budget
        self.err_ratio = err_ratio
        self.use_cuda_kernel = use_cuda_kernel
        self.projs = projs
        self.use_offline = projs is not None and len(projs) > 0
        
        if self.use_offline:
            if isinstance(projs, dict):
                num_layers = max(projs.keys()) + 1
                for layer_idx in range(num_layers):
                    proj = projs.get(layer_idx, None)
                    self.cache.append(LayerCache(self.mask_out, self.min_remain, self.token_budget, self.err_ratio, self.use_cuda_kernel, proj))
            else:
                for proj in projs:
                    self.cache.append(LayerCache(self.mask_out, self.min_remain, self.token_budget, self.err_ratio, self.use_cuda_kernel, proj))

    def to(self, *args, **kwargs):
        return self

    def to_legacy_cache(self):
        return None
    
    def get_seq_length(self, layer_idx: int = 0) -> int:
        if len(self.cache) > layer_idx:
            return self.cache[layer_idx].length
        return 0
    
    def get_max_length(self) -> Optional[int]:
        return None

    def reset(self):
        for cache in self.cache:
            cache.reset()

    def update(self, query_states, key_states, value_states, layer_idx, sparse=True):
        if len(self.cache) == layer_idx:
            if self.use_offline:
                raise RuntimeError(
                    f"[BPC] ERROR: Layer {layer_idx} not found in offline projections! "
                    f"Offline projections only have {len(self.projs)} layers. "
                    f"Please regenerate the offline projection file with all layers."
                )
            self.cache.append(LayerCache(self.mask_out, self.min_remain, self.token_budget, self.err_ratio, self.use_cuda_kernel))
            return self.cache[layer_idx].update(query_states, key_states, value_states, sparse)
        elif len(self.cache) > layer_idx:
            return self.cache[layer_idx].update(query_states, key_states, value_states, sparse)
        else:
            raise Exception(f"Layer index {layer_idx} skipped, current cache size: {len(self.cache)}")


def reset_cache(model):
    if hasattr(model, '_bpc_cache') and model._bpc_cache is not None:
        model._bpc_cache.reset()


def load_offline_projections(proj_path):
    if proj_path is None:
        print("[BPC] WARNING: offline_proj_path is None")
        return None
    
    try:
        print(f"[BPC] Loading offline projections from: {proj_path}")
        projs = torch.load(proj_path, map_location='cpu')
        if not projs or len(projs) == 0:
            print("[BPC] WARNING: Loaded projections are empty")
            return None
        print(f"[BPC] Loaded projections for {len(projs)} layers")
        return projs
    except FileNotFoundError:
        print(f"[BPC] WARNING: File not found: {proj_path}")
        return None
    except Exception as e:
        print(f"[BPC] WARNING: Failed to load offline projections: {e}")
        return None


def get_bpc_cache(model):
    return getattr(model, '_bpc_cache', None)


def get_bpc_config(model):
    return getattr(model, '_bpc_config', None)


def is_using_cuda_kernel(model):
    config = get_bpc_config(model)
    if config is not None:
        return config.get('use_cuda_kernel', False)
    return False


def modify(model, config=None):

    # Per-model copy of config.
    model._bpc_config = DEFAULT_CONFIG.copy()
    if config is not None:
        model._bpc_config.update(config)

    cfg = model._bpc_config

    use_cuda = cfg.get('use_cuda_kernel', False)
    if use_cuda:
        # Bind kernel to current device for multi-GPU.
        current_device = torch.cuda.current_device() if torch.cuda.is_available() else None
        if _load_cuda_kernel(device_id=current_device):
            print("[BPC] Using CUDA kernel backend")
        else:
            # Fall back to PyTorch on load failure.
            cfg['use_cuda_kernel'] = False
            print("[BPC] CUDA kernel requested but failed, using PyTorch backend")
    else:
        print("[BPC] Using PyTorch backend")
    
    mask_out = cfg.get('mask_out', 0.98)
    min_remain = cfg.get('min_remain', 128)
    token_budget = cfg.get('token_budget', None)
    err_ratio = cfg.get('err_ratio', 0.1)
    fix_layers = cfg.get('fix_layers', [])
    use_offline = cfg.get('use_offline_proj', False)
    use_cuda_kernel = cfg.get('use_cuda_kernel', False)

    print("=" * 40 + " BPC Config " + "=" * 40)
    print(f"  enable: {cfg['enable']}")
    print(f"  backend: {'CUDA kernel' if use_cuda_kernel else 'PyTorch'}")
    print(f"  fix_layers: {fix_layers}")
    if token_budget is not None:
        print(f"  token_budget: {token_budget} (fixed)")
    else:
        print(f"  mask_out: {mask_out}")
        print(f"  min_remain: {min_remain}")
        print(f"  token_budget: dynamic (max(context_len * (1 - mask_out), min_remain))")
    print(f"  err_ratio: {err_ratio}")
    print(f"  use_offline_proj: {use_offline}")
    if use_offline:
        print(f"    offline_proj_path: {cfg.get('offline_proj_path', None)}")
    print("=" * 98)
    
    offline_projs = None
    if use_offline:
        offline_projs = load_offline_projections(cfg.get('offline_proj_path'))
        if offline_projs is not None:
            print(f"[BPC] Offline mode enabled, loaded {len(offline_projs)} layers")
        else:
            print("[BPC] Offline projection loading failed, will use online mode")
    
    # If disabled, skip patching.
    if not cfg.get('enable', True):
        print("[BPC] Disabled, using full attention")


        cache_obj = AllCache(mask_out, min_remain, token_budget, err_ratio, use_cuda_kernel, offline_projs)
        object.__setattr__(model, '_bpc_cache', cache_obj)
        object.__setattr__(model, '_bpc_config', cfg)
        return model
    
    patched_count = 0
    patched_types = set()
    

    bpc_cache_obj = AllCache(mask_out, min_remain, token_budget, err_ratio, use_cuda_kernel, offline_projs)


    for module in model.modules():
        if not isinstance(module, SUPPORTED_ATTENTION_CLASSES):
            continue
        
        patched_types.add(type(module).__name__)
        
        module.flash_forward = module.forward

        object.__setattr__(module, '_bpc_cache', bpc_cache_obj)
        object.__setattr__(module, '_bpc_config', cfg)
        
        def modified_forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            **kwargs,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

            bpc_cache = self._bpc_cache
            bpc_cfg = self._bpc_config
            fix_layers = bpc_cfg.get('fix_layers', [])
            use_sparse = self.layer_idx not in fix_layers
            
            seq_len = hidden_states.shape[1]
            
            # ===== Prefill: delegate attention to original forward to avoid recomputing Q/K/V =====
            if seq_len > 1:
                # Compute Q/K/V to refresh bpc_cache (hash metadata) before delegating.
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, self.head_dim)
                
                query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            
                if position_embeddings is not None:
                    cos, sin = position_embeddings
                else:
                    cos, sin = self.rotary_emb(value_states, position_ids)
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
                
                if bpc_cache is not None:
                    bpc_cache.update(
                        query_states, key_states, value_states,
                        self.layer_idx, sparse=use_sparse
                    )

                # Delegate attention compute to the original flash forward.
                if position_embeddings is not None:
                    flash_output = self.flash_forward(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=None,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        **kwargs,
                    )
                else:
                    flash_output = self.flash_forward(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=None,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        **kwargs,
                    )
                
                if len(flash_output) == 2:
                    attn_output, attn_weights = flash_output
                else:
                    attn_output, attn_weights, _ = flash_output
                
                if _USE_NEW_CACHE_API:
                    return attn_output, attn_weights
                return attn_output, attn_weights, bpc_cache
            
            # ===== Decode =====
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)
    
            query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        
            if position_embeddings is not None:
                cos, sin = position_embeddings
            else:
                cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            
            selected_keys, selected_values = bpc_cache.update(
                query_states, key_states, value_states, 
                self.layer_idx, sparse=use_sparse
            )
            
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                selected_keys,
                selected_values,
                enable_gqa=True,
            )
            
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(*input_shape, -1)
            attn_output = self.o_proj(attn_output)
            
            if _USE_NEW_CACHE_API:
                return attn_output, None
            return attn_output, None, bpc_cache
        
        module.forward = partial(modified_forward, module)
        patched_count += 1
    

    object.__setattr__(model, '_bpc_cache', bpc_cache_obj)
    object.__setattr__(model, '_bpc_config', cfg)
    
    print(f"[BPC] Patched {patched_count} attention layers")
    print(f"[BPC] Patched types: {patched_types}")
    print(f"[BPC] fix_layers (use flash attention): {fix_layers}")
    
    return model
