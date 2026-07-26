"""Profile the DECODE stage of a KV-cache method with torch.profiler.

Mirrors the decode-only path in run_efficiency.py: prefill once (untimed) to
build the KV cache, then run single-token forwards through
``prepare_inputs_for_generation`` (cache_position + logits_to_keep=1, exactly
like ``generate``) while torch.profiler records CPU/CUDA activity. Only the
decode steps are inside the profiler, so the table reflects per-token decode
cost — useful for explaining why quantpc's decode speedup vs a dense baseline
is smaller than expected (dequant kernels, RoPE recompute, torch fallbacks,
python/launch overhead all show up here).

Examples:
    python profile_decode.py --method quantpc --k-bits 320 --v-bits 320 \
        --cuda-v1-exp8 --length 32768 --active 24 --trace results/efficiency/prof/qpc.json
    python profile_decode.py --method flash_attention_2 --length 32768 --active 24
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, schedule

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common import load_quantpc_model, seed_everything  # noqa: E402


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _decode_step(model, gen_ids, attn, pkv, pos, device):
    """One single-token decode forward; returns (pkv, gen_ids, attn)."""
    mi = model.prepare_inputs_for_generation(
        gen_ids, past_key_values=pkv, attention_mask=attn, use_cache=True,
        cache_position=torch.arange(pos, pos + 1, device=device), logits_to_keep=1)
    out = model(**mi)
    next_tok = out.logits[:, -1:, :].argmax(dim=-1)
    gen_ids = torch.cat([gen_ids, next_tok], dim=1)
    attn = torch.cat([attn, torch.ones_like(next_tok)], dim=1)
    return out.past_key_values, gen_ids, attn


def main() -> None:
    p = argparse.ArgumentParser(description="Profile the decode stage of a KV-cache method.")
    p.add_argument("--model", default="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct")
    p.add_argument("--method", default="quantpc", help="quantpc | kivi | flash_attention_2 | sdpa | eager")
    p.add_argument("--length", type=int, default=32768, help="prompt length to prefill before decoding")
    p.add_argument("--warmup", type=int, default=8, help="untimed decode steps before profiling")
    p.add_argument("--active", type=int, default=24, help="decode steps recorded by the profiler")
    p.add_argument("--rows", type=int, default=30, help="rows in the key_averages table")
    p.add_argument("--group-by-shape", action="store_true", help="group ops by input shape in the table")
    p.add_argument("--record-shapes", action="store_true", help="record op input shapes (adds overhead)")
    p.add_argument("--trace", default=None, help="export a chrome trace to this path")
    # KV-cache config (same semantics as run_efficiency.py).
    p.add_argument("--k-bits", type=int, default=128)
    p.add_argument("--v-bits", type=int, default=256)
    p.add_argument("--projected-query-bits", type=int, default=8)
    p.add_argument("--exp-bits", type=int, default=4)
    p.add_argument("--eas", type=float, default=0.01)
    p.add_argument("--zerorope", type=int, default=30)
    p.add_argument("--cuda-v1-exp8", action="store_true")
    p.add_argument("--refit-decoder", action="store_true")
    p.add_argument("--no-pruning", dest="use_pruning", action="store_false",
                   help="disable decode-time global pruning (gqa4 CUDA path); pruning is on by default")
    p.add_argument("--prune-mode", type=int, default=3)
    p.add_argument("--log-eps", type=float, default=-10.0)
    p.set_defaults(use_pruning=True)
    p.add_argument("--group-size", type=int, default=32)
    p.add_argument("--residual-length", type=int, default=32)
    args = p.parse_args()

    seed_everything(42)
    model, _tok, make_cache = load_quantpc_model(
        args.model, method=args.method, k_bits=args.k_bits, v_bits=args.v_bits,
        projected_query_bits=args.projected_query_bits, exp_bits=args.exp_bits,
        eas=args.eas, zerorope=args.zerorope, cuda_v1_exp8=args.cuda_v1_exp8,
        refit_decoder=args.refit_decoder, group_size=args.group_size,
        residual_length=args.residual_length,
        use_pruning=args.use_pruning, prune_mode=args.prune_mode, log_eps=args.log_eps,
    )
    device = model.device
    vocab = model.config.vocab_size
    input_ids = torch.randint(1, vocab, (1, args.length), device=device, dtype=torch.long)

    with torch.no_grad():
        pkv = make_cache() if make_cache is not None else None
        gen_ids = input_ids
        attn = torch.ones_like(input_ids)
        # Prefill (untimed): build the KV cache over the whole prompt.
        mi = model.prepare_inputs_for_generation(
            gen_ids, past_key_values=pkv, attention_mask=attn, use_cache=True,
            cache_position=torch.arange(args.length, device=device), logits_to_keep=1)
        out = model(**mi)
        pkv = out.past_key_values
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        gen_ids = torch.cat([gen_ids, next_tok], dim=1)
        attn = torch.cat([attn, torch.ones_like(next_tok)], dim=1)
        pos = args.length

        # Untimed warmup decode steps.
        for _ in range(args.warmup):
            pkv, gen_ids, attn = _decode_step(model, gen_ids, attn, pkv, pos, device)
            pos += 1
        _sync()

        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        sched = schedule(wait=0, warmup=1, active=args.active, repeat=1)
        with profile(activities=activities, schedule=sched,
                     record_shapes=args.record_shapes, with_stack=False) as prof:
            for _ in range(1 + args.active):  # +1 for the schedule warmup step
                pkv, gen_ids, attn = _decode_step(model, gen_ids, attn, pkv, pos, device)
                pos += 1
                prof.step()
        _sync()

    sort_key = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
    print(f"\n[{args.method}] decode profile  (length={args.length}, active={args.active} steps)\n")
    print(prof.key_averages(group_by_input_shape=args.group_by_shape).table(
        sort_by=sort_key, row_limit=args.rows))

    if args.trace:
        tp = Path(args.trace)
        tp.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(tp))
        print(f"[trace] {tp}")


if __name__ == "__main__":
    main()
