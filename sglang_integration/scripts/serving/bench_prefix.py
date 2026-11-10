#!/usr/bin/env python3
"""t50 TTFT with a shared prefix (radix hits) and without reuse, real weights.
usage: bench_prefix.py <pkrx|pk|firx|fi> [reps=5]
pkrx: picokv + radix on (run with PICOKV_RADIX=host); pk: picokv, radix off;
firx / fi: flashinfer radix on / off. SGLang default chunk (8192 here), memf 0.70,
ctx 131072. TTFT = wall time of generate(max_new_tokens=1).
hit cases P+S x bs: bs requests = one shared P-token prefix (sent once untimed to
populate the cache) + a fresh S-token suffix per request and rep.
miss cases (bs:L): fresh random prompts every rep (no reuse; radix overhead)."""
import os, random, sys, time
import sglang as sgl

HIT = [(4096, 256, 1), (16384, 256, 1), (65536, 256, 1), (16384, 256, 8), (16384, 2048, 1)]
MISS = [(1, 512), (24, 512), (1, 8192), (1, 32768)]


def main():
    mode = sys.argv[1]
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    pk = mode.startswith("pk")
    kw = dict(model_path=os.environ.get("BENCH_MODEL", "llama31_8b_instruct"),
              attention_backend="picokv" if pk else "flashinfer", dtype="bfloat16",
              tp_size=1, mem_fraction_static=0.70, max_running_requests=64,
              context_length=131072, trust_remote_code=True, log_level="error",
              disable_prefill_cuda_graph=True,
              cuda_graph_backend_decode="breakable" if pk else "full")
    if not mode.endswith("rx"):
        kw["disable_radix_cache"] = True
    eng = sgl.Engine(**kw)
    rnd = random.Random(0)
    toks = lambda n: [rnd.randint(10, 120000) for _ in range(n)]
    sp1 = {"temperature": 0.0, "max_new_tokens": 1, "ignore_eos": True}
    run = lambda ps: eng.generate(input_ids=ps, sampling_params=[sp1] * len(ps))
    for _ in range(2):
        run([toks(2048) for _ in range(4)])

    def timed(make, n):
        ts, cached = [], []
        for _ in range(n):
            ps = make()
            t0 = time.perf_counter()
            o = run(ps)
            ts.append((time.perf_counter() - t0) * 1e3)
            cached.append(min(x["meta_info"].get("cached_tokens", 0) for x in o))
        ts.sort()
        return ts[len(ts) // 2], ts[0], min(cached)

    only = os.environ.get("PFX_CASES", "hit,miss").split(",")
    for P, S, bs in HIT if "hit" in only else []:
        pre = toks(P)
        run([pre + toks(S)])  # populate (radix modes cache the prefix)
        m, lo, c = timed(lambda: [pre + toks(S) for _ in range(bs)], reps)
        print(f"PFX mode={mode} case=hit P={P} S={S} bs={bs} med_ms={m:.1f} min_ms={lo:.1f} "
              f"cached_min={c}", flush=True)
    for bs, L in MISS if "miss" in only else []:
        m, lo, c = timed(lambda: [toks(L) for _ in range(bs)], reps if bs * L <= 65536 else 3)
        print(f"PFX mode={mode} case=miss bs={bs} L={L} med_ms={m:.1f} min_ms={lo:.1f} "
              f"cached_min={c}", flush=True)
    eng.shutdown()


if __name__ == "__main__":
    main()
