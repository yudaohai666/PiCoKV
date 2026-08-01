"""torch.profiler per-op breakdown for OScaR vs flash_attention_2 at long context.

Mirrors bench_e2e.py's driving path (inputs_embeds + past_key_values threading),
but instead of only reporting wall-clock ms/tok, it wraps the timed region in
torch.profiler and dumps:
  * a key_averages() table (top ops by self CUDA time),
  * a chrome trace (--out-dir), openable in chrome://tracing or perfetto,
  * total CUDA time for the profiled phase.

Phases:
  --phase decode   : profile `decode_tokens` single-token forwards (KV-quant regime).
  --phase prefill  : profile one full prompt_len prefill.
  --phase both     : profile prefill then decode, tagged via record_function.

Run ONE backend per process (pass --attn-backend) so memory is released between
runs; invoke twice (oscar, flash_attention_2) on the same GPU and compare tables.
"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.profiler import ProfilerActivity, profile, record_function, schedule

EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))
PROJECT_ROOT = EVAL_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bench_e2e import load_model, seed_everything  # noqa: E402


def rand_prompt(prompt_len, hidden, dtype, device):
    return torch.randn(1, prompt_len, hidden, dtype=dtype, device=device)


def rand_tok(hidden, dtype, device):
    return torch.randn(1, 1, hidden, dtype=dtype, device=device)


def prefill(model, prompt_len, hidden, dtype, device):
    return model(inputs_embeds=rand_prompt(prompt_len, hidden, dtype, device),
                 use_cache=True, logits_to_keep=1).past_key_values


def decode_step(model, past, hidden, dtype, device):
    return model(inputs_embeds=rand_tok(hidden, dtype, device), past_key_values=past,
                 use_cache=True, logits_to_keep=1).past_key_values


@torch.inference_mode()
def run(model, args, hidden, dtype, device):
    def sync():
        torch.cuda.synchronize(device)

    # ---- Warmup: kernel compile / autotune / allocator growth (untimed). ----
    for _ in range(args.warmup):
        past = prefill(model, args.length, hidden, dtype, device)
        for _ in range(args.decode_tokens):
            past = decode_step(model, past, hidden, dtype, device)
        del past
    torch.cuda.empty_cache()
    sync()

    # For decode-only, prefill must run OUTSIDE the profiler so the captured
    # region is purely the single-token decode steps (otherwise the 128k prefill
    # dominates the table). For `both`, prefill is intentionally inside + tagged.
    past = None
    if args.phase == "decode":
        past = prefill(model, args.length, hidden, dtype, device)
        sync()

    torch.cuda.reset_peak_memory_stats(device)
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    trace_path = Path(args.out_dir) / f"{args.tag}_{args.phase}.json"
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    with profile(activities=activities, record_shapes=True, profile_memory=True,
                 with_stack=False) as prof:
        if args.phase in ("prefill", "both"):
            with record_function("PREFILL"):
                past = prefill(model, args.length, hidden, dtype, device)
            sync()
        if args.phase != "prefill":
            with record_function("DECODE"):
                for _ in range(args.decode_tokens):
                    past = decode_step(model, past, hidden, dtype, device)
            sync()
    del past

    peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
    prof.export_chrome_trace(str(trace_path))

    print(f"\n==== [{args.tag}] phase={args.phase} L={args.length} "
          f"decode_tokens={args.decode_tokens} peak={peak_gb:.3f}GB ====")
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=args.row_limit))
    print(f"[trace] {trace_path}")


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct")
    p.add_argument("--attn-backend", default="oscar",
                   choices=["oscar", "flash_attention_2", "flash_decoding", "eager"])
    p.add_argument("--length", type=int, default=131072)
    p.add_argument("--phase", default="decode", choices=["decode", "prefill", "both"])
    p.add_argument("--decode-tokens", type=int, default=32)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--row-limit", type=int, default=30)
    p.add_argument("--out-dir", default=str(EVAL_DIR / "profiles"))
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-bits", type=int, default=4)
    p.add_argument("--quant-mode", default="k-channel")
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--kv-rotation", default="none")
    p.add_argument("--kv-norm", default="0")
    p.add_argument("--residual-evict-size", type=int, default=None)
    args = p.parse_args()
    tag = args.attn_backend
    if args.attn_backend == "oscar":
        tag += f"_int{args.num_bits}_{args.quant_mode}_g{args.group_size}"
    args.tag = tag
    return args


def main():
    args = build_args()
    seed_everything(42)
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    model = load_model(args, dtype)
    run(model, args, model.config.hidden_size, dtype, device)


if __name__ == "__main__":
    main()
