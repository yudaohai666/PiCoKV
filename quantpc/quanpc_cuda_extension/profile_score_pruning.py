"""Profile score-pruning opportunities for QuanPC global attention.

This is a statistics-only script.  It does not change the CUDA fast path.

The current packed CUDA score computes:

    score = dot(int8(projected_query), binary_key) * query_scale * attn_scale

This script compares the full score against two partial-score upper bounds:

1. Prefix code dimensions, e.g. first 64/128/192 of Ck=256.
2. High magnitude bits of the int8 projected query.

For a partial integer dot ``p`` and an absolute remaining bound ``r``, the
global score is guaranteed to be no larger than:

    (p + r) * query_scale * attn_scale

If that upper bound is below ``dense_lse + log_eps``, the token can be safely
skipped for that threshold.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import torch


@dataclass
class Counts:
    exact_small: int = 0
    safe_skip: int = 0
    heuristic_skip: int = 0
    heuristic_false_skip: int = 0
    total: int = 0

    def update(
        self,
        full_small: torch.Tensor,
        safe_skip: torch.Tensor,
        heuristic_skip: torch.Tensor,
    ) -> None:
        false_skip = heuristic_skip & ~full_small
        self.exact_small += int(full_small.sum().item())
        self.safe_skip += int(safe_skip.sum().item())
        self.heuristic_skip += int(heuristic_skip.sum().item())
        self.heuristic_false_skip += int(false_skip.sum().item())
        self.total += full_small.numel()

    def row(self, label: str) -> str:
        exact = self.exact_small / self.total if self.total else 0.0
        safe = self.safe_skip / self.total if self.total else 0.0
        heur = self.heuristic_skip / self.total if self.total else 0.0
        false = self.heuristic_false_skip / self.heuristic_skip if self.heuristic_skip else 0.0
        capture = self.safe_skip / self.exact_small if self.exact_small else 0.0
        return (
            f"{label:>12}  exact={exact:8.4%}  safe={safe:8.4%}  "
            f"capture={capture:8.4%}  heur={heur:8.4%}  heur_false={false:8.4%}"
        )


def parse_int_list(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


def parse_float_list(value: str) -> list[float]:
    return [float(part) for part in value.split(",") if part]


def project_query_gqa(query: torch.Tensor, key_proj: torch.Tensor) -> torch.Tensor:
    B, QH, S, D = query.shape
    KVH = key_proj.shape[1]
    C = key_proj.shape[-2]
    group = QH // KVH
    query_grouped = query.reshape(B, KVH, group, S, D).reshape(B * KVH, group * S, D)
    proj_t = key_proj.transpose(-1, -2).reshape(B * KVH, D, C)
    projected = torch.bmm(query_grouped, proj_t)
    return projected.reshape(B, KVH, group, S, C).reshape(B, QH, S, C).contiguous()


def quantize_projected_query(projected: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    projected_f = projected.float()
    max_abs = projected_f.abs().amax(dim=-1).clamp_min(torch.finfo(torch.float32).eps)
    query_scale = max_abs / 127.0
    q_int = torch.round(projected_f / query_scale[..., None]).clamp(-127, 127).to(torch.int16)
    return q_int.contiguous(), query_scale.contiguous()


def make_inputs(args, gqa: int):
    torch.manual_seed(args.seed + gqa * 1009)
    device = torch.device(args.device)
    dtype = torch.bfloat16
    B = args.batch_size
    KVH = args.kv_heads
    QH = KVH * gqa
    D = args.head_dim
    C = args.code_dim

    query = torch.randn(B, QH, 1, D, device=device, dtype=dtype).contiguous()
    dense_k = torch.randn(B, KVH, args.dense_len, D, device=device, dtype=dtype).contiguous()
    key_proj = (torch.randn(B, KVH, C, D, device=device, dtype=dtype) * args.proj_std).contiguous()
    projected = project_query_gqa(query, key_proj)
    q_int, query_scale = quantize_projected_query(projected)

    dense_k_exp = dense_k.repeat_interleave(gqa, dim=1)
    attn_scale = 1.0 / math.sqrt(D)
    dense_scores = torch.matmul(query, dense_k_exp.transpose(-1, -2)) * attn_scale
    dense_lse = torch.logsumexp(dense_scores.float(), dim=-1).contiguous()

    key_bits = torch.randint(0, 2, (B, KVH, args.global_len, C), device=device, dtype=torch.int8)
    key_codes = (key_bits * 2 - 1).contiguous()
    return q_int, query_scale, dense_lse, key_codes, attn_scale


def dot_chunk(q_int: torch.Tensor, key_chunk: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bhsc,bhlc->bhsl", q_int.float(), key_chunk.float())


def profile_gqa(args, gqa: int) -> None:
    q_int, query_scale, dense_lse, key_codes, attn_scale = make_inputs(args, gqa)
    B, QH, S, C = q_int.shape
    KVH = key_codes.shape[1]
    group = QH // KVH
    q_abs = q_int.abs()
    q_sign = torch.sign(q_int).to(torch.int16)
    threshold_dot = (dense_lse[:, :, :, None] + args.log_eps_tensor) / (
        query_scale[:, :, :, None].clamp_min(1.0e-30) * attn_scale
    )

    prefix_dims = [dim for dim in args.prefix_dims if 0 < dim < C]
    high_bits = [bits for bits in args.high_bits if 0 < bits <= 7]
    prefix_counts = {(dim, log_eps): Counts() for dim in prefix_dims for log_eps in args.log_eps}
    high_bit_counts = {(bits, log_eps): Counts() for bits in high_bits for log_eps in args.log_eps}

    prefix_remaining = {dim: q_abs[..., dim:].sum(dim=-1).float() for dim in prefix_dims}
    high_q = {}
    high_remaining = {}
    for bits in high_bits:
        low_bits = 7 - bits
        high_abs = ((q_abs >> low_bits) << low_bits).to(torch.int16)
        high_q[bits] = (q_sign * high_abs).contiguous()
        high_remaining[bits] = (q_abs - high_abs).sum(dim=-1).float()

    total_tokens = 0
    for start in range(0, key_codes.shape[2], args.chunk_size):
        end = min(start + args.chunk_size, key_codes.shape[2])
        key_chunk = key_codes[:, :, start:end, :].repeat_interleave(group, dim=1)
        full_dot = dot_chunk(q_int, key_chunk)
        total_tokens += full_dot.numel()

        for log_idx, log_eps in enumerate(args.log_eps):
            threshold = threshold_dot[..., log_idx].unsqueeze(-1)
            full_small = full_dot <= threshold

            for dim in prefix_dims:
                partial = dot_chunk(q_int[..., :dim], key_chunk[..., :dim])
                safe = partial + prefix_remaining[dim][..., None] <= threshold
                heuristic = partial <= threshold
                prefix_counts[(dim, log_eps)].update(full_small, safe, heuristic)

            for bits in high_bits:
                partial = dot_chunk(high_q[bits], key_chunk)
                safe = partial + high_remaining[bits][..., None] <= threshold
                heuristic = partial <= threshold
                high_bit_counts[(bits, log_eps)].update(full_small, safe, heuristic)

    print(
        f"\nGQA={gqa}  shape: B={B}, QH={QH}, KVH={KVH}, S={S}, "
        f"L={key_codes.shape[2]}, C={C}, tokens={total_tokens}"
    )
    for log_eps in args.log_eps:
        print(f"\n  log_eps={log_eps:g}")
        print("  prefix-dim:")
        for dim in prefix_dims:
            print("   " + prefix_counts[(dim, log_eps)].row(f"dim={dim}"))
        print("  high-query-bits:")
        for bits in high_bits:
            print("   " + high_bit_counts[(bits, log_eps)].row(f"bits={bits}"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--gqa", default="1,4,8")
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--code-dim", type=int, default=256)
    parser.add_argument("--dense-len", type=int, default=256)
    parser.add_argument("--global-len", type=int, default=8192)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--proj-std", type=float, default=0.02)
    parser.add_argument("--prefix-dims", default="64,128,192")
    parser.add_argument("--high-bits", default="2,3,4,5,6")
    parser.add_argument("--log-eps", default="-4,-6,-8,-10,-12")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the default device.")

    args.gqa = parse_int_list(args.gqa)
    args.prefix_dims = parse_int_list(args.prefix_dims)
    args.high_bits = parse_int_list(args.high_bits)
    args.log_eps = parse_float_list(args.log_eps)
    args.log_eps_tensor = torch.tensor(args.log_eps, device=args.device, dtype=torch.float32)

    if args.code_dim % 64 != 0:
        raise ValueError("--code-dim must be divisible by 64.")

    if args.device == "cuda":
        print(f"device: {torch.cuda.get_device_name()}")
    for gqa in args.gqa:
        profile_gqa(args, gqa)


if __name__ == "__main__":
    main()
