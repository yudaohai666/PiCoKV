"""
BPC: binary-hash sparse attention (CUDA kernel only).

Prefill uses HF DynamicCache and computes hash metadata; decode picks
top-k KV via hash scores for sparse attention.

Key components:
- BPCMetadata: per-layer hash metadata (proj matrix, hashcodes, err topk)
- find_binary_projection: learn projection matrix
- compute_hashscores: hash similarity scores
"""

from typing import Optional, Tuple, List
import torch
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb
from transformers.models.mistral.modeling_mistral import MistralAttention
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from functools import partial
import os
import transformers
import re
from flash_attn import flash_attn_with_kvcache, flash_attn_func

# transformers 4.53+ attention.forward returns only 2 values
def _parse_version(version_str):
    match = re.match(r'^(\d+)\.(\d+)', version_str)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return (0, 0)

_TRANSFORMERS_VERSION = _parse_version(transformers.__version__)
_USE_NEW_CACHE_API = _TRANSFORMERS_VERSION >= (4, 53)

_cuda_kernel_loaded = False

def _load_cuda_kernel(device_id=None):
    global _cuda_kernel_loaded

    if device_id is not None:
        torch.cuda.set_device(device_id)

    current_device = torch.cuda.current_device()

    # try direct import first
    try:
        import binary_proj
        if not _cuda_kernel_loaded:
            print(f"[BPC] CUDA kernel imported successfully on GPU {current_device}")
            _cuda_kernel_loaded = True
        return True
    except ImportError:
        pass

    # JIT compile fallback
    try:
        from torch.utils.cpp_extension import load
        import filelock
        
        curr_path = os.path.dirname(os.path.abspath(__file__))
        src_files = [os.path.join(curr_path, 'extension', file) for file in ['cuda_kernel.cu', 'torch_extension.cpp']]
        
        lock_file = os.path.join(curr_path, 'extension', '.compile.lock')
        os.makedirs(os.path.dirname(lock_file), exist_ok=True)
        
        with filelock.FileLock(lock_file, timeout=300):
            try:
                import binary_proj
                if not _cuda_kernel_loaded:
                    print(f"[BPC] CUDA kernel imported successfully on GPU {current_device}")
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


SUPPORTED_ATTENTION_CLASSES = (LlamaAttention, MistralAttention, Qwen2Attention)

DEFAULT_CONFIG = {
    "enable": True,
    "fix_layers": [0,1],
    "mask_out": 0.98,
    "min_remain": 128,
    "token_budget": None,
    "err_ratio": 0.1,
    "use_offline_proj": False,
    "offline_proj_path": None,
    "use_cuda_kernel": True,
    "use_fused_kernel": True,  # fused kernel: quant+pack+probe_hash
}


def find_binary_projection(A, num_bits, num_iters=4):
    """
    Learn binary projection matrix.

    Args:
        A: [B, H, L, D] input vectors
        num_bits: number of hash bits (typically 64)
        num_iters: power-iteration count

    Returns:
        quan_A: [B, H, L, num_bits] binarized result
        quan_proj: [B, H, num_bits, D] projection matrix
    """
    B, H, L, D = A.shape
    quan_A = torch.empty(B, H, L, num_bits, device=A.device, dtype=A.dtype)
    quan_proj = torch.empty(B, H, num_bits, D, device=A.device, dtype=A.dtype)
    
    A = A.clone()
    
    for bit_idx in range(num_bits):
        Vh = torch.randn(B, H, D, device=A.device, dtype=A.dtype)
        
        for _ in range(num_iters):
            Uq = torch.einsum("bhnd,bhd->bhn", A, Vh).sign()
            Vh = torch.einsum("bhnd,bhn->bhd", A, Uq)
        
        Vh = Vh / L
        A = A - torch.einsum("bhn,bhd->bhnd", Uq, Vh)
        quan_A[:, :, :, bit_idx] = Uq
        quan_proj[:, :, bit_idx, :] = Vh

    return quan_A, quan_proj


def binary_project_cuda(A, quan_proj):
    """Project vectors to hashcodes via CUDA kernel."""
    import binary_proj
    
    B, H, L, D = A.shape
    device = A.device
    
    if quan_proj.device != device:
        quan_proj = quan_proj.to(device)
    
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
    """Reconstruction error of binary quantization via CUDA kernel."""
    import binary_proj
    
    B, H, L = quant_keys.shape
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
        quant_keys_unpacked = torch.empty(B, H, L, 64, dtype=torch.int, device=device)
        binary_proj.format_bits_hash64_fn(quant_keys_unpacked, quant_keys, False)
        approx_keys = quant_keys_unpacked.to(proj.dtype) @ proj
        errs = torch.norm(keys - approx_keys, p=2, dim=-1)
    return errs


def compute_hashscores_cuda(probe, hashcode, T, hash_T):
    """Hash similarity scores via CUDA kernel."""
    import binary_proj

    B, H, S, D = probe.shape
    B, H, L = hashcode.shape
    assert S >= 1  # MHA (S=1) and GQA (S=4)
    assert L % 256 == 0
    assert D == 64
    
    device = probe.device
    
    if hashcode.device != device:
        hashcode = hashcode.to(device)
    
    with torch.cuda.device(device):
        if not hashcode.is_contiguous():
            hashcode = hashcode.contiguous()
        if not probe.is_contiguous():
            probe = probe.contiguous()
        
        quant_probe = torch.empty(B, H, S, D, dtype=torch.int, device=device)
        packed_probe = torch.empty(B, H, S, 8, dtype=torch.long, device=device)
        binary_proj.quant_probe_hash64_fn(probe, quant_probe)
        binary_proj.format_probe_hash64_fn(quant_probe, packed_probe, True)

        scores = torch.empty(B, H, L, dtype=torch.short, device=device)
        binary_proj.probe_hash64_group4_block256_fn(packed_probe, hashcode, scores)

    if T > hash_T:
        scores[:, :, hash_T:T] = 10000
    scores[:, :, T:] = -10000
    return scores


def compute_hashscores_cuda_fused(probe, hashcode, T, hash_T):
    """
    Fused kernel: quant_probe + pack_probe + probe_hash64 + boundary handling.
    Reduces kernel launch overhead and intermediate memory traffic.

    Args:
        probe: [B, H, S, 64] bf16 query hash representation (S=1 MHA, S=4 GQA)
        hashcode: [B, H, L] long KV hashcodes
        T: effective sequence length
        hash_T: hash-valid length

    Returns:
        scores: [B, H, L] short
            [0, hash_T): hash scores
            [hash_T, T): 10000 (force-select new tokens)
            [T, L): -10000 (excluded)
    """
    import binary_proj

    B, H, S, D = probe.shape
    _, _, L = hashcode.shape
    assert S >= 1  # MHA (S=1) and GQA (S=4)
    assert L % 256 == 0
    assert D == 64
    
    device = probe.device
    
    if hashcode.device != device:
        hashcode = hashcode.to(device)
    
    with torch.cuda.device(device):
        if not hashcode.is_contiguous():
            hashcode = hashcode.contiguous()
        if not probe.is_contiguous():
            probe = probe.contiguous()
        
        scores = torch.empty(B, H, L, dtype=torch.short, device=device)
        # boundary handling done inside kernel
        binary_proj.fused_quant_pack_probe_hash64_fn(probe, hashcode, scores, T, hash_T)

    return scores


class LayerMetadata:
    """Per-layer hash metadata."""
    
    def __init__(self, mask_out, min_remain, token_budget, err_ratio, offline_proj=None, use_fused_kernel=False):
        self.mask_out = mask_out
        self.min_remain = min_remain
        self.token_budget = token_budget
        self.err_ratio = err_ratio
        self.offline_proj = offline_proj
        self.use_fused_kernel = use_fused_kernel
        self.reset()
    
    def reset(self):
        self.proj = self.offline_proj  # projection matrix [B, H, 64, D]
        self.proj_T = None             # pre-transposed proj for decode
        self.hashcodes = None          # [B, H, L]
        self.err_topk = None           # indices of largest-error tokens [B, H, err_k]
        self.hash_length = 0           # length already hashed
        self.k = None                  # token budget
        self.err_k = None              # error token budget
        self.finalized = False
    
    def _compute_budget(self, context_len):
        if self.token_budget is not None:
            k = self.token_budget
        else:
            k = max(int(context_len * (1 - self.mask_out)), self.min_remain)
        
        k = min(k, context_len)
        err_k = max(int(k * self.err_ratio), 1)
        err_k = min(err_k, context_len)
        
        return k, err_k
    
    def finalize(self, keys, sparse=True):
        """
        Compute projection, hashcodes and error-topk from keys.

        Args:
            keys: [B, H, L, D] keys from DynamicCache
            sparse: whether sparse attention is used
        """
        if self.finalized:
            return

        B, H, L, D = keys.shape
        self.k, self.err_k = self._compute_budget(L)

        if not sparse:
            self.finalized = True
            return

        if self.proj is None:
            _, self.proj = find_binary_projection(keys, 64)
        else:
            if self.proj.device != keys.device or self.proj.dtype != keys.dtype:
                self.proj = self.proj.to(device=keys.device, dtype=keys.dtype)

        hashcode = binary_project_cuda(keys, self.proj)

        # Align hashcodes to 256 with extra 256 padding to avoid resize during decode
        rounded_L = ((L - 1) // 256 + 1) * 256 + 256
        # empty + selective init (padding scores will be set to -10000)
        self.hashcodes = torch.empty(B, H, rounded_L, dtype=torch.long, device=keys.device)
        self.hashcodes[:, :, :L] = hashcode
        if rounded_L > L:
            self.hashcodes[:, :, L:] = 0
        self.hash_length = L

        errors = compute_errors_cuda(keys, hashcode, self.proj)
        self.err_topk = torch.topk(errors, k=self.err_k, dim=-1).indices

        # Precompute proj_T (expand+transpose+contiguous) to avoid per-decode work
        proj = self.proj
        if proj.dim() == 3:
            proj = proj.unsqueeze(0).expand(B, -1, -1, -1)
        elif proj.shape[0] == 1 and B > 1:
            proj = proj.expand(B, -1, -1, -1)
        self.proj_T = proj.transpose(-1, -2).contiguous()  # [B, H, D, 64]

        self.finalized = True
    
    def select_kv(self, queries, keys, values, current_length):
        """
        Select top-k keys/values by hash score.

        Args:
            queries: [B, Hq, 1, D]
            keys: [B, Hk, L, D]
            values: [B, Hk, L, D]
            current_length: current sequence length

        Returns:
            selected_keys, selected_values: [B, Hk, k, D]
        """
        B, Hq, _, D = queries.shape
        _, Hk, L, _ = keys.shape

        # actual k accounts for tokens added since last hash compute
        actual_k = min(self.k + (current_length - self.hash_length), current_length)

        # Grow hashcodes buffer if needed
        if self.hashcodes.shape[2] < current_length:
            rounded_L = ((current_length - 1) // 256 + 1) * 256
            old_hashcodes = self.hashcodes
            old_len = old_hashcodes.shape[2]
            self.hashcodes = torch.empty(B, Hk, rounded_L, dtype=torch.long, device=keys.device)
            self.hashcodes[:, :, :old_len] = old_hashcodes
            # Init new padding (kernel reads these, even though scores get overwritten)
            if rounded_L > old_len:
                self.hashcodes[:, :, old_len:] = 0

        # Probe = query hash representation; reuse precomputed proj_T
        queries_reshaped = queries.reshape(B, Hk, Hq // Hk, D)
        probe = queries_reshaped @ self.proj_T  # [B, Hk, num_groups, 64]

        if self.use_fused_kernel:
            hashscores = compute_hashscores_cuda_fused(probe, self.hashcodes, current_length, self.hash_length)
        else:
            hashscores = compute_hashscores_cuda(probe, self.hashcodes, current_length, self.hash_length)

        # Force-select largest-error tokens
        hashscores.scatter_(dim=2, index=self.err_topk, value=10000)

        # sorted=False: attention does not need ordered indices
        token_idxes = torch.topk(hashscores, k=actual_k, dim=-1, sorted=False).indices  # [B, Hk, k]

        token_idxes_expanded = token_idxes.unsqueeze(-1).expand(-1, -1, -1, D)
        selected_keys = torch.gather(keys, dim=2, index=token_idxes_expanded)
        selected_values = torch.gather(values, dim=2, index=token_idxes_expanded)

        return selected_keys, selected_values


class BPCMetadata:
    """Manager for per-layer hash metadata."""
    
    def __init__(self, mask_out, min_remain, token_budget, err_ratio, offline_projs=None, use_fused_kernel=False):
        self.mask_out = mask_out
        self.min_remain = min_remain
        self.token_budget = token_budget
        self.err_ratio = err_ratio
        self.offline_projs = offline_projs
        self.use_fused_kernel = use_fused_kernel
        self.layers: List[LayerMetadata] = []
        self._cached_current_length = None  # cached seq length to avoid per-layer .item()

        # Pre-create layers when offline projections are provided
        if offline_projs is not None:
            if isinstance(offline_projs, dict):
                num_layers = max(offline_projs.keys()) + 1
                for i in range(num_layers):
                    proj = offline_projs.get(i, None)
                    self.layers.append(LayerMetadata(mask_out, min_remain, token_budget, err_ratio, proj, use_fused_kernel))
            else:
                for proj in offline_projs:
                    self.layers.append(LayerMetadata(mask_out, min_remain, token_budget, err_ratio, proj, use_fused_kernel))
    
    def reset(self):
        for layer in self.layers:
            layer.reset()
        self._cached_current_length = None

    def get_current_length(self, cache_position):
        """Cached current seq length to avoid repeated GPU sync."""
        if cache_position is None:
            return None
        # cache_position is current token position; +1 gives seq length.
        # Computed on first call per step; later layers reuse cached value.
        if self._cached_current_length is None:
            self._cached_current_length = cache_position[-1].item() + 1
        return self._cached_current_length

    def clear_length_cache(self):
        """Clear cached length at start of each forward step."""
        self._cached_current_length = None

    def get_layer(self, layer_idx):
        while len(self.layers) <= layer_idx:
            self.layers.append(LayerMetadata(
                self.mask_out, self.min_remain, self.token_budget, self.err_ratio,
                use_fused_kernel=self.use_fused_kernel
            ))
        return self.layers[layer_idx]
    
    def finalize_all(self, past_key_values, fix_layers=None):
        """
        Compute hash metadata for all layers from past_key_values.

        Args:
            past_key_values: DynamicCache, StaticCache, or tuple of (key, value)
            fix_layers: layer indices that skip sparse attention
        """
        fix_layers = fix_layers or []

        if hasattr(past_key_values, 'key_cache'):
            num_layers = len(past_key_values.key_cache)

            # StaticCache key_cache is [B, H, max_cache_len, D]; need real fill length
            if isinstance(past_key_values, StaticCache):
                seq_len = past_key_values.get_seq_length()
            else:
                seq_len = None  # DynamicCache needs no slicing

            for layer_idx in range(num_layers):
                keys = past_key_values.key_cache[layer_idx]  # [B, H, L, D]

                if seq_len is not None:
                    keys = keys[:, :, :seq_len, :]

                sparse = layer_idx not in fix_layers
                layer_meta = self.get_layer(layer_idx)
                layer_meta.finalize(keys, sparse=sparse)
        elif isinstance(past_key_values, tuple):
            num_layers = len(past_key_values)
            for layer_idx in range(num_layers):
                keys = past_key_values[layer_idx][0]
                sparse = layer_idx not in fix_layers
                layer_meta = self.get_layer(layer_idx)
                layer_meta.finalize(keys, sparse=sparse)
        else:
            raise TypeError(f"Unsupported past_key_values type: {type(past_key_values)}")


def reset_bpc(model):
    if hasattr(model, '_bpc_metadata') and model._bpc_metadata is not None:
        model._bpc_metadata.reset()


def get_bpc_metadata(model):
    return getattr(model, '_bpc_metadata', None)


def get_bpc_config(model):
    return getattr(model, '_bpc_config', None)


def finalize_bpc(model, past_key_values, fix_layers=None):
    """Finalize BPC prefill: compute hash metadata from past_key_values."""
    metadata = get_bpc_metadata(model)
    if metadata is not None:
        metadata.finalize_all(past_key_values, fix_layers=fix_layers)


def load_offline_projections(proj_path):
    """Load offline projection matrices."""
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


# Legacy API aliases
def reset_cache(model):
    reset_bpc(model)

def get_bpc_cache(model):
    return get_bpc_metadata(model)

def is_using_cuda_kernel(model):
    return True


def modify(model, config=None):
    """
    Patch model to use BPC sparse attention.

    Workflow:
    1. Prefill: use HF DynamicCache (or StaticCache).
    2. After prefill: call finalize_bpc(model, past_key_values, fix_layers).
    3. Decode: sparse attention is applied automatically.
    """
    model._bpc_config = DEFAULT_CONFIG.copy()
    if config is not None:
        model._bpc_config.update(config)
    
    cfg = model._bpc_config

    current_device = torch.cuda.current_device() if torch.cuda.is_available() else None
    if not _load_cuda_kernel(device_id=current_device):
        raise RuntimeError("[BPC] CUDA kernel is required but failed to load!")
    
    print("[BPC] Using CUDA kernel backend")
    
    mask_out = cfg.get('mask_out', 0.98)
    min_remain = cfg.get('min_remain', 128)
    token_budget = cfg.get('token_budget', None)
    err_ratio = cfg.get('err_ratio', 0.1)
    fix_layers = cfg.get('fix_layers', [])
    use_offline = cfg.get('use_offline_proj', False)
    use_fused_kernel = cfg.get('use_fused_kernel', False)

    print("=" * 40 + " BPC Config " + "=" * 40)
    print(f"  enable: {cfg['enable']}")
    print(f"  backend: CUDA kernel")
    print(f"  use_fused_kernel: {use_fused_kernel}")
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

    if not cfg.get('enable', True):
        print("[BPC] Disabled, using full attention")
        return model

    metadata = BPCMetadata(mask_out, min_remain, token_budget, err_ratio, offline_projs, use_fused_kernel)
    object.__setattr__(model, '_bpc_metadata', metadata)
    object.__setattr__(model, '_bpc_config', cfg)
    
    patched_count = 0
    patched_types = set()

    for module in model.modules():
        if not isinstance(module, SUPPORTED_ATTENTION_CLASSES):
            continue
        
        patched_types.add(type(module).__name__)

        # Keep original forward for prefill (uses HF native flash_attention_2)
        object.__setattr__(module, '_bpc_original_forward', module.forward)

        # Bind metadata and config
        object.__setattr__(module, '_bpc_metadata', metadata)
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

            metadata = self._bpc_metadata
            cfg = self._bpc_config
            fix_layers = cfg.get('fix_layers', [])
            use_sparse = self.layer_idx not in fix_layers
            
            seq_len = hidden_states.shape[1]

            # Prefill: delegate to original forward (HF flash_attention_2)
            if seq_len > 1:
                return self._bpc_original_forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
            
            # Decode: sparse attention path
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)
            B = hidden_states.shape[0]

            query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            if position_embeddings is not None:
                cos, sin = position_embeddings
            else:
                cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            # Clear length cache once per step (at first layer)
            if self.layer_idx == 0:
                metadata.clear_length_cache()

            if past_key_value is not None:
                if hasattr(past_key_value, 'update'):
                    cache_kwargs = {"cache_position": cache_position}
                    key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

                    # StaticCache: reuse cached current_length to avoid per-layer .item()
                    if isinstance(past_key_value, StaticCache):
                        current_length = metadata.get_current_length(cache_position)
                        if current_length is None:
                            current_length = past_key_value.get_seq_length()
                        key_states = key_states[:, :, :current_length, :]
                        value_states = value_states[:, :, :current_length, :]
                    else:
                        current_length = key_states.shape[2]
                elif isinstance(past_key_value, tuple):
                    past_key = past_key_value[self.layer_idx][0]
                    past_value = past_key_value[self.layer_idx][1]
                    key_states = torch.cat([past_key, key_states], dim=2)
                    value_states = torch.cat([past_value, value_states], dim=2)
                    current_length = key_states.shape[2]
            else:
                current_length = key_states.shape[2]
            
            layer_meta = metadata.get_layer(self.layer_idx)

            # Fall back to full attention if sparse disabled or metadata not finalized
            if not use_sparse or not layer_meta.finalized:
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query_states,
                    key_states,
                    value_states,
                    enable_gqa=True,
                )
            else:
                # Sparse path: pick top-k KV
                selected_keys, selected_values = layer_meta.select_kv(
                    query_states, key_states, value_states, current_length
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
            return attn_output, None, past_key_value
        
        module.forward = partial(modified_forward, module)
        patched_count += 1

    print(f"[BPC] Patched {patched_count} attention layers")
    print(f"[BPC] Patched types: {patched_types}")
    print(f"[BPC] fix_layers (use full attention): {fix_layers}")
    
    return model


def origin_modify(model, config=None):
    """
    Baseline: leave model unchanged; HF flash_attention_2 handles
    both prefill and decode. `config` is accepted for API compatibility.
    """
    print("=" * 40 + " Origin Config " + "=" * 40)
    print("  Using HuggingFace native flash_attention_2 for both prefill and decode")
    print("  No modification applied - this is the unmodified baseline")
    print("=" * 95)
    return model
