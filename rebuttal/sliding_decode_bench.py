"""Per-step decode latency of PiCoKV offline caches: static vs sliding.

Synthetic bf16 K/V with Llama-3.1-8B geometry (8 KV heads, GQA 4, D=128),
``--layers`` attention layers per step (no MLP, so less work exists to hide
the slide behind than in the real model).  Each mode prefills ``--contexts``
tokens and decodes ``--steps`` tokens through ``StandaloneLlamaCache.decode``.

* ``us_per_step``: wall time of the whole loop / steps (GPU-synchronized), so
  it includes the amortized slide cost and host launch overhead.
* ``slide_step_extra_us``: CUDA-event time of steps that trigger a slide minus
  the median of other steps, i.e. what a slide adds to the critical path.

Modes: ``static`` (slide_tokens=0), ``sliding`` (production: slide work on a
side stream), ``sliding_sync`` (same code with the side stream replaced by the
current stream, to show what the async overlap saves).

    CUDA_VISIBLE_DEVICES=3 python rebuttal/sliding_decode_bench.py --tag after
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from picokv.picokv_standalone import StandaloneConfig
from picokv.picokv_standalone.transformers import (
    StandaloneLlamaCache,
    load_offline_projections,
)

REPO = Path(__file__).resolve().parents[1]
MODES = {"static": (0, False), "sliding": (None, False), "sliding_sync": (None, True)}


def run(projs, context: int, steps: int, slide: int, sync: bool, layers: int) -> dict:
    torch.manual_seed(0)
    heads, dim = 8, 128
    key = torch.randn(1, heads, context, dim, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    new_key = torch.randn(1, heads, steps, dim, device="cuda", dtype=torch.bfloat16)
    new_value = torch.randn_like(new_key)
    query = torch.randn(1, 4 * heads, 1, dim, device="cuda", dtype=torch.bfloat16)
    cache = StandaloneLlamaCache(
        StandaloneConfig(slide_tokens=slide),
        key_encoders=projs["key_encoder"],
        value_encoders=projs["value_encoder"],
        key_decoders=projs["key_decoder"],
        value_decoders=projs["value_decoder"],
    )
    for layer in range(layers):
        cache.prepare(layer, key, value)
    del key, value
    if sync:
        cache._slide_stream = torch.cuda.current_stream()
    events = [torch.cuda.Event(enable_timing=True) for _ in range(steps + 1)]
    slide_steps = []
    torch.cuda.synchronize()
    start = time.perf_counter()
    events[0].record()
    for t in range(steps):
        before = cache.slide_count
        k, v = new_key[:, :, t : t + 1], new_value[:, :, t : t + 1]
        for layer in range(layers):
            cache.decode(layer, query, k, v)
        events[t + 1].record()
        if cache.slide_count != before:
            slide_steps.append(t)
    torch.cuda.synchronize()
    wall = (time.perf_counter() - start) / steps * 1e6
    step_us = [events[t].elapsed_time(events[t + 1]) * 1e3 for t in range(steps)]
    slide_set = set(slide_steps)
    # Skip the first 64 steps (allocator/launch warm-up inside the loop).
    plain = [x for t, x in enumerate(step_us) if t not in slide_set and t >= 64]
    base = statistics.median(plain)
    # A slide's side-stream work can also spill into the next step.
    extra = [step_us[t] + (step_us[t + 1] if t + 1 < steps else base) - 2 * base
             for t in slide_steps]
    layer0 = cache.layers[0]
    out = {
        "us_per_step": wall,
        "median_plain_step_us": base,
        "slide_step_extra_us": statistics.mean(extra) if extra else 0.0,
        "dense_len": layer0.dense_len,
        "global_len": layer0.global_len,
        "slides_per_layer": len(slide_steps),
    }
    del cache, layer0
    torch.cuda.empty_cache()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--proj",
        default=str(REPO / "offline_projection/projections/Llama-3.1-8B-Instruct_k320_v320.pt"),
    )
    parser.add_argument("--contexts", default="32768,131072")
    parser.add_argument("--steps", type=int, default=4096)
    parser.add_argument("--slide", type=int, default=64)
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--outdir", default=str(REPO / "rebuttal/results/sliding_decode"))
    args = parser.parse_args()

    projs = load_offline_projections(args.proj)
    # Warm up kernels and the prepared-projection memo.
    run(projs, 4096, 256, args.slide, False, args.layers)
    results = {"steps": args.steps, "slide": args.slide, "layers": args.layers, "runs": {}}
    for context in (int(x) for x in args.contexts.split(",")):
        for mode in args.modes.split(","):
            slide, sync = MODES[mode]
            slide = args.slide if slide is None else slide
            samples = [run(projs, context, args.steps, slide, sync, args.layers)
                       for _ in range(args.repeats)]
            best = min(samples, key=lambda s: s["us_per_step"])
            best["us_per_step_all"] = [s["us_per_step"] for s in samples]
            results["runs"][f"{context}_{mode}"] = best
            print(f"{context:>7} {mode:<12} {best['us_per_step']:9.1f} us/step "
                  f"(plain {best['median_plain_step_us']:.1f}, "
                  f"slide +{best['slide_step_extra_us']:.1f} us) "
                  f"dense={best['dense_len']} global={best['global_len']} "
                  f"slides/layer={best['slides_per_layer']}", flush=True)
    out = Path(args.outdir) / f"bench_{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
