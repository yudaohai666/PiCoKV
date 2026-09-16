"""
Pure-PyTorch BPC implementation (no CUDA kernel required).
"""

import torch
from transformers.cache_utils import Cache
from typing import Optional


def find_binary_projection(A, num_bits, num_iters=4):

    B, H, L, D = A.shape
    quan_A = torch.empty(B, H, L, num_bits, device=A.device, dtype=A.dtype)
    quan_proj = torch.empty(B, H, num_bits, D, device=A.device, dtype=A.dtype)

    # clone to avoid mutating the caller's tensor
    A = A.clone()

    # Greedy per-bit projection (PCA-like principal-component extraction).
    for bit_idx in range(num_bits):
        Vh = torch.randn(B, H, D, device=A.device, dtype=A.dtype)

        # Alternating optimization: Uq = sign(A @ Vh); Vh = A^T @ Uq.
        for idx in range(num_iters):
            Uq = torch.einsum("bhnd,bhd->bhn", A, Vh).sign()
            Vh = torch.einsum("bhnd,bhn->bhd", A, Uq)

        Vh = Vh / L
        # Subtract residual captured by this bit.
        A = A - torch.einsum("bhn,bhd->bhnd", Uq, Vh)
        quan_A[:, :, :, bit_idx] = Uq
        quan_proj[:, :, bit_idx, :] = Vh

    return quan_A, quan_proj



def binary_project_pytorch(A, quan_proj):

    B, H, L, D = A.shape
    num_bits = quan_proj.shape[-2]
    device = A.device
    dtype = A.dtype
    
    # Offline proj is [H, num_bits, D]; expand to [B, H, num_bits, D].
    if quan_proj.dim() == 3:
        quan_proj = quan_proj.unsqueeze(0).expand(B, -1, -1, -1)
    elif quan_proj.shape[0] == 1 and B > 1:
        quan_proj = quan_proj.expand(B, -1, -1, -1)

    A = A.clone()

    # Unpacked +-1 storage; mirrors learning loop in find_binary_projection.
    hashcode_unpacked = torch.empty(B, H, L, num_bits, device=device, dtype=dtype)

    for bit_idx in range(num_bits):
        Uq = torch.einsum("bhld,bhd->bhl", A, quan_proj[:, :, bit_idx, :]).sign()
        A = A - torch.einsum("bhl,bhd->bhld", Uq, quan_proj[:, :, bit_idx, :])
        hashcode_unpacked[:, :, :, bit_idx] = Uq

    # Pack to long, matching CUDA kernel: val < 0 -> bit 1, val >= 0 -> bit 0.
    binary_bits = (hashcode_unpacked < 0).to(torch.int64)  # [B, H, L, 64]
    bit_weights = (2 ** torch.arange(num_bits, device=device, dtype=torch.int64)).view(1, 1, 1, num_bits)
    hashcode = (binary_bits * bit_weights).sum(dim=-1)  # [B, H, L]
    
    return hashcode


def compute_errors_pytorch(keys, hashcode, proj):

    B, H, L = hashcode.shape
    device = keys.device
    num_bits = proj.shape[-2]

    if proj.dim() == 3:
        proj = proj.unsqueeze(0).expand(B, -1, -1, -1)
    elif proj.shape[0] == 1 and B > 1:
        proj = proj.expand(B, -1, -1, -1)

    # Unpack packed long to per-bit values in +-1 form (CUDA: bit 0 -> +1, bit 1 -> -1).
    bit_indices = torch.arange(num_bits, device=device, dtype=torch.int64)
    unpacked = ((hashcode.unsqueeze(-1) >> bit_indices) & 1).to(proj.dtype)  # [B, H, L, 64], 0/1
    unpacked = 1 - unpacked * 2  # [B, H, L, 64], +-1

    # Reconstruct approximate key, then L2 distance from original.
    approx_keys = torch.einsum("bhln,bhnd->bhld", unpacked, proj)
    errs = torch.norm(keys - approx_keys, p=2, dim=-1)
    return errs


def compute_hashscores_pytorch(probe, hashcode, T, hash_T):

    B, H, S, D = probe.shape
    _, _, L = hashcode.shape
    device = probe.device
    num_bits = D  # D == 64 bits

    # Quantize probe to 7-bit range [-127, 127].
    probe_max = probe.abs().reshape(B, H, -1).max(dim=-1, keepdim=True).values.unsqueeze(-1)  # [B, H, 1, 1]
    probe_max = probe_max.clamp(min=1e-6)
    quant_probe = torch.round(probe * ((2 ** 7 - 1) / probe_max))

    # Unpack to +-1 (CUDA: bit 0 -> +1, bit 1 -> -1).
    bit_indices = torch.arange(num_bits, device=device, dtype=torch.int64)
    unpacked = ((hashcode.unsqueeze(-1) >> bit_indices) & 1).to(quant_probe.dtype)
    unpacked = 1 - unpacked * 2

    # Hash similarity: quant_probe @ unpacked.T -> [B, H, S, L], then reduce S.
    scores = torch.einsum("bhsd,bhld->bhsl", quant_probe, unpacked)
    scores = scores.max(dim=-2).values

    # Boundary handling: force-select [hash_T, T) (new tokens), mask out [T, L).
    if T > hash_T:
        scores[:, :, hash_T:T] = 10000
    scores[:, :, T:] = -10000
    
    return scores


class LayerCache:

    def __init__(self, mask_out, min_remain, token_budget, err_ratio, proj=None):

        self.mask_out = mask_out
        self.min_remain = min_remain
        self.token_budget = token_budget
        self.err_ratio = err_ratio
        self.offline_proj = proj  # offline projection (if provided)
        self.proj = proj
        self.reset()

    def reset(self):
        self.keys = None
        self.values = None
        self.hashcodes = None  # packed-long hash codes
        self.err_topk = None   # indices of high quantization-error tokens
        self.length = 0
        self.hash_length = 0   # number of tokens whose hashes are computed
        self.k = None
        self.err_k = None
        self.prefill_complete = False
        self.proj_T = None     # precomputed proj transpose, reused across decode steps
        # Online mode: clear proj so prefill relearns it; offline: keep it.
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

                hashcode = binary_project_pytorch(keys, self.proj)
                self.hashcodes[:, :, :L] = hashcode
                self.hash_length = L

                # Tokens with the largest quantization error are force-selected during decode.
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

            # fix_layers: return full KV (no sparsity).
            if not sparse:
                return self.keys[:, :, :self.length, :], self.values[:, :, :self.length, :]

            # actual_k = prefill budget + new (un-hashed) decode tokens.
            # The new tokens have no hash codes and are force-selected, so adding them here
            # prevents them from displacing the original-context budget.
            actual_k = min(self.k + (self.length - self.hash_length), self.length)

            # GQA reshape: [B, Hk, num_groups, D]; reuse precomputed proj_T.
            queries = queries.reshape(B, Hk, Hq // Hk, D)
            probe = queries @ self.proj_T

            hashscores = compute_hashscores_pytorch(
                probe,
                self.hashcodes,
                self.length,
                self.hash_length
            )

            # Force-select high-error tokens by setting their score high.
            hashscores.scatter_(dim=2, index=self.err_topk, value=10000)

            token_idxes = torch.topk(hashscores, k=actual_k, dim=-1, sorted=False).indices

            token_idxes_expanded = token_idxes.unsqueeze(-1).expand(-1, -1, -1, D)
            selected_keys = torch.gather(self.keys, dim=2, index=token_idxes_expanded)
            selected_values = torch.gather(self.values, dim=2, index=token_idxes_expanded)
            return selected_keys, selected_values


class AllCache(Cache):

    def __init__(self, mask_out, min_remain, token_budget, err_ratio, projs=None):
        super().__init__()
        self.cache = []
        self.mask_out = mask_out
        self.min_remain = min_remain
        self.token_budget = token_budget
        self.err_ratio = err_ratio
        self.projs = projs
        self.use_offline = projs is not None and len(projs) > 0
        
        if self.use_offline:
            if isinstance(projs, dict):
                num_layers = max(projs.keys()) + 1
                for layer_idx in range(num_layers):
                    proj = projs.get(layer_idx, None)
                    self.cache.append(LayerCache(self.mask_out, self.min_remain, self.token_budget, self.err_ratio, proj))
            else:
                for proj in projs:
                    self.cache.append(LayerCache(self.mask_out, self.min_remain, self.token_budget, self.err_ratio, proj))

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
                    f"[BPC-PyTorch] ERROR: Layer {layer_idx} not found in offline projections!"
                )
            self.cache.append(LayerCache(self.mask_out, self.min_remain, self.token_budget, self.err_ratio))
            return self.cache[layer_idx].update(query_states, key_states, value_states, sparse)
        elif len(self.cache) > layer_idx:
            return self.cache[layer_idx].update(query_states, key_states, value_states, sparse)
        else:
            raise Exception(f"Layer index {layer_idx} skipped, current cache size: {len(self.cache)}")
