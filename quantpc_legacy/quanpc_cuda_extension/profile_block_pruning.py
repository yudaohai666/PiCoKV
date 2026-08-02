"""Profile 64-token block pruning for QuanPC global attention.

This script is statistics-only.  It does not change the CUDA fast path.

Current CUDA kernels quantize global exp weights per 64-token group.  If the
largest score in a group is already far below the dense logsumexp offset, the
whole group can be skipped:

    max_score_in_group <= dense_lse + log_eps

Equivalently:

    max(exp(score - dense_lse)) <= exp(log_eps)

The script reports how many 64-token groups would be skipped and how much
actual global denominator mass those skipped groups contain.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import torch

from profile_score_pruning import dot_chunk, make_inputs, parse_float_list, parse_int_list


@dataclass
class BlockStats:
    groups: int = 0
    skipped_groups: int = 0
    small_tokens: int = 0
    tokens: int = 0

    def group_ratio(self) -> float:
        return self.skipped_groups / self.groups if self.groups else 0.0

    def small_token_ratio(self) -> float:
        return self.small_tokens / self.tokens if self.tokens else 0.0


def _format_pct(value: float) -> str:
    return f"{value:8.4%}"


def profile_gqa(args, gqa: int) -> None:
    q_int, query_scale, dense_lse, key_codes, attn_scale = make_inputs(args, gqa)
    B, QH, S, C = q_int.shape
    KVH = key_codes.shape[1]
    head_group = QH // KVH
    L = key_codes.shape[2]
    block_size = args.block_size

    if L % block_size != 0:
        raise ValueError(f"global length must be divisible by block size; got L={L}, block={block_size}")
    if args.chunk_size % block_size != 0:
        raise ValueError(f"chunk size must be divisible by block size; got {args.chunk_size}, block={block_size}")

    eps_values = [math.exp(log_eps) for log_eps in args.log_eps]
    stats = {log_eps: BlockStats() for log_eps in args.log_eps}
    row_den = torch.zeros(B, QH, S, device=q_int.device, dtype=torch.float32)
    row_skip_den = {log_eps: torch.zeros_like(row_den) for log_eps in args.log_eps}
    row_skip_bound = {log_eps: torch.zeros_like(row_den) for log_eps in args.log_eps}

    for start in range(0, L, args.chunk_size):
        end = min(start + args.chunk_size, L)
        if (end - start) % block_size != 0:
            raise ValueError("chunk tail is not block aligned; choose a block-aligned chunk size")

        key_chunk = key_codes[:, :, start:end, :].repeat_interleave(head_group, dim=1)
        full_dot = dot_chunk(q_int, key_chunk)
        scores = full_dot * query_scale[..., None] * attn_scale
        weights = torch.exp(scores - dense_lse[..., None])
        row_den += weights.sum(dim=-1)

        group_count = (end - start) // block_size
        scores_grouped = scores.view(B, QH, S, group_count, block_size)
        weights_grouped = weights.view(B, QH, S, group_count, block_size)
        group_max_score = scores_grouped.amax(dim=-1)
        group_den = weights_grouped.sum(dim=-1)

        for log_eps, eps in zip(args.log_eps, eps_values):
            threshold = dense_lse[..., None] + log_eps
            prune = group_max_score <= threshold
            current = stats[log_eps]
            current.groups += prune.numel()
            current.skipped_groups += int(prune.sum().item())
            current.tokens += weights.numel()
            current.small_tokens += int((weights <= eps).sum().item())
            row_skip_den[log_eps] += torch.where(prune, group_den, torch.zeros_like(group_den)).sum(dim=-1)
            row_skip_bound[log_eps] += prune.sum(dim=-1).to(torch.float32) * (block_size * eps)

    total_global_den = row_den.sum().clamp_min(1.0e-30)
    total_final_den = (row_den + 1.0).sum()
    print(
        f"\nGQA={gqa}  shape: B={B}, QH={QH}, KVH={KVH}, S={S}, "
        f"L={L}, C={C}, block={block_size}, groups/row={L // block_size}"
    )
    print(
        "     log_eps  groups_skip  token_small  "
        "skip_global_total  skip_final_total  skip_final_mean  skip_final_max  "
        "bound_final_mean  bound_final_max"
    )
    for log_eps in args.log_eps:
        current = stats[log_eps]
        skipped_den = row_skip_den[log_eps]
        skipped_bound = row_skip_bound[log_eps]
        final_frac = skipped_den / (row_den + 1.0)
        bound_final_frac = skipped_bound / (row_den + 1.0)
        print(
            f"{log_eps:12g}  "
            f"{_format_pct(current.group_ratio())}  "
            f"{_format_pct(current.small_token_ratio())}  "
            f"{_format_pct(float(skipped_den.sum() / total_global_den))}  "
            f"{_format_pct(float(skipped_den.sum() / total_final_den))}  "
            f"{_format_pct(float(final_frac.mean()))}  "
            f"{_format_pct(float(final_frac.max()))}  "
            f"{_format_pct(float(bound_final_frac.mean()))}  "
            f"{_format_pct(float(bound_final_frac.max()))}"
        )


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
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--proj-std", type=float, default=0.02)
    parser.add_argument("--log-eps", default="-4,-5,-5.5,-6,-6.5,-7,-8,-10,-12")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the default device.")
    if args.code_dim % 64 != 0:
        raise ValueError("--code-dim must be divisible by 64.")

    args.gqa = parse_int_list(args.gqa)
    args.log_eps = parse_float_list(args.log_eps)

    if args.device == "cuda":
        print(f"device: {torch.cuda.get_device_name()}")
    for gqa in args.gqa:
        profile_gqa(args, gqa)


if __name__ == "__main__":
    main()
