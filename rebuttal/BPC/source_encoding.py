"""Verbatim BinaryPC functions; only unpacked-output adapter bypasses uint64 packing.

Source: src/BPC/bpc/bpc/bpc_pytorch.py.
"""

import torch



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


def binary_project_unpacked(A, quan_proj):

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

    return hashcode_unpacked
