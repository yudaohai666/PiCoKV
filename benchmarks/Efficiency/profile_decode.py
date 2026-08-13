"""Profile the PREFILL and/or DECODE stage of a KV-cache method with torch.profiler.

``--stage`` selects what is recorded:
  * ``decode``  (default) — prefill once (untimed) to build the KV cache, then run
    single-token forwards through ``prepare_inputs_for_generation`` (cache_position
    + logits_to_keep=1, exactly like ``generate``) inside the profiler. The table
    reflects per-token decode cost (dequant kernels, RoPE recompute, torch
    fallbacks, python/launch overhead).
  * ``prefill`` — run ``--prefill-warmup`` untimed prefill passes to absorb
    first-call cuBLAS autotune / kernel init, then profile one full-prompt prefill.
    Useful for explaining why quantpc's prefill is slower/faster than a dense
    baseline (projection/quantize/pack kernels vs a single flash-attn prefill).
  * ``both`` — profile prefill, then reuse that cache to profile decode. Two tables
    are printed; if ``--trace`` is set, ``.prefill``/``.decode`` are appended.

Examples:
    python profile_decode.py --method quantpc --stage prefill --length 32768
    python profile_decode.py --method flash_attention_2 --stage prefill --length 32768
    python profile_decode.py --method quantpc --stage both --length 32768 --active 24 \
        --trace results/efficiency/prof/qpc.json
    python profile_decode.py --method flash_attention_2 --length 32768 --active 24
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common import load_model, resolve_offline_proj_path, seed_everything  # noqa: E402


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


def _prefill(model, input_ids, make_cache, device):
    """Run one full-prompt prefill forward; returns (pkv, gen_ids, attn, pos)."""
    pkv = make_cache() if make_cache is not None else None
    gen_ids = input_ids
    attn = torch.ones_like(input_ids)
    length = input_ids.shape[1]
    mi = model.prepare_inputs_for_generation(
        gen_ids, past_key_values=pkv, attention_mask=attn, use_cache=True,
        cache_position=torch.arange(length, device=device), logits_to_keep=1)
    out = model(**mi)
    pkv = out.past_key_values
    next_tok = out.logits[:, -1:, :].argmax(dim=-1)
    gen_ids = torch.cat([gen_ids, next_tok], dim=1)
    attn = torch.cat([attn, torch.ones_like(next_tok)], dim=1)
    return pkv, gen_ids, attn, length


def _report(prof, args, stage, extra):
    """Print the key_averages table for a finished profiler and export a trace."""
    sort_key = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
    if args.sort == "cpu":
        sort_key = "self_cpu_time_total"
    elif args.sort == "cuda":
        sort_key = "self_cuda_time_total"
    print(f"\n[{args.method}] {stage} profile  ({extra})\n")
    print(prof.key_averages(group_by_input_shape=args.group_by_shape).table(
        sort_by=sort_key, row_limit=args.rows))
    if args.trace:
        tp = Path(args.trace)
        if args.stage == "both":  # keep both stages' traces distinct
            tp = tp.with_name(f"{tp.stem}.{stage}{tp.suffix}")
        tp.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(tp))
        print(f"[trace] {tp}")


def main() -> None:
    p = argparse.ArgumentParser(description="Profile the decode stage of a KV-cache method.")
    p.add_argument("--model", default="/data/Llama-3.1-8B-Instruct")
    p.add_argument("--method", default="quantpc", help="quantpc | kivi | oscar | flash_attention_2 | sdpa | eager")
    p.add_argument("--stage", default="decode", choices=["decode", "prefill", "both"],
                   help="which stage to profile: decode (default) | prefill | both")
    p.add_argument("--length", type=int, default=32768, help="prompt length to prefill before decoding")
    p.add_argument("--batch", type=int, default=1, help="batch size for prefill/decode (default 1)")
    p.add_argument("--warmup", type=int, default=2, help="untimed decode steps before profiling (matches run_efficiency)")
    p.add_argument("--prefill-warmup", type=int, default=1,
                   help="untimed prefill passes before the profiled prefill (absorbs first-call cuBLAS autotune/kernel init)")
    p.add_argument("--active", type=int, default=64, help="decode steps recorded by the profiler (matches run_efficiency --decode-tokens)")
    p.add_argument("--rows", type=int, default=30, help="rows in the key_averages table")
    p.add_argument("--group-by-shape", action="store_true", help="group ops by input shape in the table")
    p.add_argument("--record-shapes", action="store_true", help="record op input shapes (adds overhead)")
    p.add_argument("--trace", default=None, help="export a chrome trace to this path")
    # ── quantpc KV-cache params (--method quantpc) ──
    p.add_argument("--k-bits", type=int, default=None, help="QuanPC K projection bits; KIVI defaults to 2")
    p.add_argument("--v-bits", type=int, default=None, help="QuanPC V projection bits; KIVI defaults to 2")
    p.add_argument("--sink-size", type=int, default=64, help="initial (sink) tokens kept in full precision")
    p.add_argument("--local-size", type=int, default=256, help="most-recent (local) tokens kept in full precision")
    p.add_argument("--exp-bits", type=int, default=4)
    p.add_argument("--eas", type=float, default=0.01)
    p.add_argument("--zerorope", type=int, default=20)
    p.add_argument("--no-pruning", dest="use_pruning", action="store_false",
                   help="disable decode-time global pruning (gqa4 CUDA path); pruning is on by default")
    p.add_argument("--prune-mode", type=int, default=3)
    p.add_argument("--log-eps", type=float, default=-10.0)
    p.add_argument("--offline", action="store_true",
                   help="quantpc: use offline encoders (skip online QR at prefill); path defaults to offline_projection/projections/{model}_k{K}_v{V}_nowhiten.pt")
    p.add_argument("--offline-proj-path", default=None,
                   help="quantpc: explicit offline encoder .pt (overrides the --offline default path)")
    p.set_defaults(use_pruning=True)
    # ── KIVI params (--method kivi; also uses --k-bits/--v-bits as 2/4-bit KV) ──
    p.add_argument("--group-size", type=int, default=32, help="KIVI per-group quant size; OScaR group size (INT2 必须=32)")
    p.add_argument("--residual-length", type=int, default=32, help="KIVI recent fp16 tokens kept unquantized")
    # ── OScaR params (--method oscar) ──
    p.add_argument("--num-bits", type=int, default=2, help="OScaR KV quant bits (INT2 默认)")
    p.add_argument("--quant-mode", default="k-channel", help="OScaR quant mode")
    p.add_argument("--kv-rotation", default="hadamard", help="OScaR kv rotation (INT2 默认 hadamard)")
    p.add_argument("--kv-norm", default="1", help="OScaR kv norm (INT2 默认 1)")
    p.add_argument("--residual-evict-size", type=int, default=256, help="OScaR residual evict size (INT2 默认 256)")
    p.add_argument("--offline-v-hadamard", action=argparse.BooleanOptionalAction, default=True, help="OScaR: offline per-head Hadamard into v_proj/o_proj (INT2 默认开启)")
    p.add_argument("--sort", default=None, choices=["cpu", "cuda"],
                   help="override sort key: cpu=self_cpu_time_total, cuda=self_cuda_time_total")
    # ── PolarQuant params (--method polar) ──
    p.add_argument("--polar-mode", default="mixed", choices=["mixed", "group", "snapkv"], help="PolarQuant attention variant: mixed (K polar + V KIVI-quant) | group (K-only) | snapkv")
    p.add_argument("--polar-rbits", type=int, default=3, help="PolarQuant K radius bits (rbits+tbits<=8)")
    p.add_argument("--polar-tbits", type=int, default=3, help="PolarQuant K angle bits (rbits+tbits<=8)")
    p.add_argument("--polar-vbits", type=int, default=2, help="PolarQuant V bits (mixed only; KIVI kernel supports 2 or 4)")
    args = p.parse_args()
    if args.k_bits is None:
        args.k_bits = 2 if args.method == "kivi" else 320
    if args.v_bits is None:
        args.v_bits = 2 if args.method == "kivi" else 320

    seed_everything(42)
    offline_proj_path = resolve_offline_proj_path(
        args.method, args.offline, args.offline_proj_path,
        model_name=args.model, k_bits=args.k_bits, v_bits=args.v_bits,
        whiten=False,
    )
    model, _tok, make_cache = load_model(
        args.model,
        method=args.method,
        # quantpc KV-cache params (--method quantpc)
        k_bits=args.k_bits,
        v_bits=args.v_bits,
        sink_size=args.sink_size,
        local_size=args.local_size,
        exp_bits=args.exp_bits,
        eas=args.eas,
        zerorope=args.zerorope,
        use_pruning=args.use_pruning,
        prune_mode=args.prune_mode,
        log_eps=args.log_eps,
        # KIVI params (--method kivi)
        group_size=args.group_size,
        residual_length=args.residual_length,
        # OScaR params (--method oscar)
        num_bits=args.num_bits,
        quant_mode=args.quant_mode,
        kv_rotation=args.kv_rotation,
        kv_norm=args.kv_norm,
        residual_evict_size=args.residual_evict_size,
        offline_v_hadamard=args.offline_v_hadamard,
        offline_proj_path=offline_proj_path,
        # PolarQuant params (--method polar)
        polar_mode=args.polar_mode,
        polar_rbits=args.polar_rbits,
        polar_tbits=args.polar_tbits,
        polar_vbits=args.polar_vbits,
    )
    device = model.device
    vocab = model.config.vocab_size
    input_ids = torch.randint(1, vocab, (args.batch, args.length), device=device, dtype=torch.long)

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    do_prefill = args.stage in ("prefill", "both")
    do_decode = args.stage in ("decode", "both")

    with torch.no_grad():
        pkv = gen_ids = attn = None
        pos = args.length

        # ── prefill stage ─────────────────────────────────────────────────
        if do_prefill:
            # Untimed warmup prefill(s): absorb first-call cuBLAS autotune /
            # kernel init / workspace allocation so the profiled pass is steady.
            for _ in range(args.prefill_warmup):
                _prefill(model, input_ids, make_cache, device)
            _sync()
            with profile(activities=activities, record_shapes=True,
                         profile_memory=True, with_stack=False) as prof:
                with record_function("PREFILL"):
                    pkv, gen_ids, attn, pos = _prefill(model, input_ids, make_cache, device)
            _sync()
            _report(prof, args, "prefill", f"batch={args.batch}, length={args.length}")

        # ── decode stage ──────────────────────────────────────────────────
        if do_decode:
            if pkv is None:
                # No profiled prefill ran (stage=decode): build the cache untimed.
                pkv, gen_ids, attn, pos = _prefill(model, input_ids, make_cache, device)
                _sync()
            # Untimed warmup decode steps (matches run_efficiency).
            for _ in range(args.warmup):
                pkv, gen_ids, attn = _decode_step(model, gen_ids, attn, pkv, pos, device)
                pos += 1
            _sync()
            with profile(activities=activities, record_shapes=True,
                         profile_memory=True, with_stack=False) as prof:
                with record_function("DECODE"):
                    for _ in range(args.active):
                        pkv, gen_ids, attn = _decode_step(model, gen_ids, attn, pkv, pos, device)
                        pos += 1
            _sync()
            _report(prof, args, "decode", f"batch={args.batch}, length={args.length}, active={args.active} steps")


if __name__ == "__main__":
    main()
