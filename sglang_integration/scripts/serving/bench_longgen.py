#!/usr/bin/env python3
"""Turn-32 long-generation probe: decode tokens stay in picokv's dense window (never
compressed), so the window grows with output length. Streams bs x plen random prompts
(llama31_8b_rand) for GEN tokens and reports the median step (ms) per output-position
bucket. usage: bench_longgen.py <backend> <plen> <bs> <gen>"""
import asyncio, os, random, sys, time
import sglang as sgl

BUCKETS = [0, 512, 2048, 4096, 8192, 12288, 16384]


def main():
    be, plen, bs, gen = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    gkw = dict(cuda_graph_backend_decode=os.environ.get("BENCH_DECODE_GRAPH", "full"),
               disable_prefill_cuda_graph=True)
    eng = sgl.Engine(model_path=os.environ.get("BENCH_MODEL", "llama31_8b_rand"), attention_backend=be, dtype="bfloat16",
                     tp_size=1, mem_fraction_static=0.70, disable_radix_cache=True,
                     chunked_prefill_size=1 << 20, max_running_requests=bs,
                     context_length=plen + gen + 64, trust_remote_code=True,
                     log_level="error", **gkw)
    random.seed(0)
    prompts = [[random.randint(10, 120000) for _ in range(plen)] for _ in range(bs)]
    eng.generate(input_ids=[p[:1024] for p in prompts], sampling_params=[
        {"temperature": 0.0, "max_new_tokens": 8, "ignore_eos": True}] * bs)

    async def one(ids):
        sp = {"temperature": 0.0, "max_new_tokens": gen, "ignore_eos": True}
        g = await eng.async_generate(input_ids=ids, sampling_params=sp, stream=True)
        pts = []
        async for ch in g:
            pts.append((time.perf_counter(), ch["meta_info"]["completion_tokens"]))
        return pts

    async def rep():
        return await asyncio.gather(*[one(p) for p in prompts])

    res = eng.loop.run_until_complete(rep())
    by = {b: [] for b in BUCKETS[:-1]}
    for pts in res:
        for (t0, n0), (t1, n1) in zip(pts[1:], pts[2:]):
            if n1 > n0:
                b = max(x for x in BUCKETS[:-1] if x <= n0)
                by[b].append((t1 - t0) / (n1 - n0) * 1e3)
    for i, b in enumerate(BUCKETS[:-1]):
        xs = sorted(by[b])
        if xs:
            print(f"LONGGEN be={be} plen={plen} bs={bs} out=[{b},{BUCKETS[i+1]}) "
                  f"n={len(xs)} med_ms={xs[len(xs)//2]:.3f}", flush=True)
    eng.shutdown()


if __name__ == "__main__":
    main()
