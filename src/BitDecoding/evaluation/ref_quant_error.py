"""Pure-PyTorch reference for BitDecoding/KIVI min-max KV quantization.

Reproduces the exact setting of evaluation/test_sweep.py (B=1, H=32, d=128,
seqlen_k=1024, q~U[0,1), K/V~N(0,1), seed 42) in fp32/fp16 torch only, so the
measured kernel error can be compared against the error the algorithm *should*
produce at each (num_bits, group_size).

K is quantized per-channel  (groups of `group_size` tokens, per head_dim channel).
V is quantized per-token    (groups of `group_size` head_dim entries, per token).
"""

import math

import torch


def fake_quant(x, group_size, bits, axis):
    """Asymmetric min-max round-to-nearest fake-quant along `axis` in groups."""
    x = x.movedim(axis, -1)
    shape = x.shape
    x = x.reshape(*shape[:-1], shape[-1] // group_size, group_size)
    mn = x.min(dim=-1, keepdim=True).values
    mx = x.max(dim=-1, keepdim=True).values
    rng = mx - mn
    max_int = 2**bits - 1
    scale = rng / max_int
    scale = torch.where(rng > 0, scale, torch.ones_like(scale))
    q = ((x - mn) / scale).round().clamp_(0, max_int)
    deq = q * scale + mn
    return deq.reshape(*shape).movedim(-1, axis)


def attention(q, k, v):
    d = q.shape[-1]
    s = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    return torch.einsum("bhts,bshd->bthd", torch.softmax(s, dim=-1).to(v.dtype), v)


def main():
    device, dtype = "cuda", torch.float16
    B, H, d, S = 1, 32, 128, 1024

    torch.manual_seed(42)
    q = torch.rand(B, 1, H, d, device=device, dtype=dtype)
    k = torch.randn(B, S, H, d, device=device, dtype=dtype)
    v = torch.randn(B, S, H, d, device=device, dtype=dtype)

    ref = attention(q, k, v)

    print(f"{'bits':>5} {'group':>6} {'k-chan range':>13} {'out err':>10}")
    for bits in (4, 2):
        for group in (128, 32):
            # K per-channel: group along the token axis (dim=1).
            kq = fake_quant(k.float(), group, bits, axis=1).to(dtype)
            # V per-token: group along head_dim (dim=-1).
            vq = fake_quant(v.float(), group, bits, axis=-1).to(dtype)
            err = (attention(q, kq, vq) - ref).abs().mean().item()
            kr = (
                k.float()
                .movedim(1, -1)
                .reshape(B, H, d, S // group, group)
                .amax(-1)
                .sub(k.float().movedim(1, -1).reshape(B, H, d, S // group, group).amin(-1))
                .mean()
                .item()
            )
            print(f"{bits:>5} {group:>6} {kr:>13.3f} {err:>10.5f}")


if __name__ == "__main__":
    main()
