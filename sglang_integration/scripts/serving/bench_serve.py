#!/usr/bin/env python3
"""Turn-30 serving-pattern speed probes (random-weight llama31_8b_rand, streamed ITL like
scripts/bench_itl.py). usage: bench_serve.py <backend> <plen> <bs> [reps]
fixed : bs x plen, every request gen=GEN                       (t25-style baseline)
churn : bs x plen, request i gen = 64 + 24*i -> bs-1 finishes mid-decode
poison: (bs-1) x plen + 1 x 100-token prompt, gen=GEN
Per scenario: median step (ms) over all requests' streamed gaps, and for the request
that lives longest: its decode wall time, #gaps > 3x median, and their excess ms."""
import asyncio, os, random, sys, time
import sglang as sgl

MODEL = os.environ.get("BENCH_MODEL", "llama31_8b_rand")
GEN = 232


def main():
    be, plen, bs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    reps = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    gkw = {}
    if os.environ.get("BENCH_DECODE_GRAPH"):
        gkw = dict(cuda_graph_backend_decode=os.environ["BENCH_DECODE_GRAPH"],
                   disable_prefill_cuda_graph=True)
    eng = sgl.Engine(model_path=MODEL, attention_backend=be, dtype="bfloat16", tp_size=1,
                     mem_fraction_static=0.70, disable_cuda_graph=not gkw,
                     disable_radix_cache=True, chunked_prefill_size=1 << 20,
                     max_running_requests=bs, trust_remote_code=True, log_level="error", **gkw)
    random.seed(0)
    prompts = [[random.randint(10, 120000) for _ in range(plen)] for _ in range(bs)]
    eng.generate(input_ids=[p[:1024] for p in prompts], sampling_params=[
        {"temperature": 0.0, "max_new_tokens": 8, "ignore_eos": True}] * bs)

    async def one(ids, gen):
        sp = {"temperature": 0.0, "max_new_tokens": gen, "ignore_eos": True}
        g = await eng.async_generate(input_ids=ids, sampling_params=sp, stream=True)
        pts = []
        async for ch in g:
            pts.append((time.perf_counter(), ch["meta_info"]["completion_tokens"]))
        return pts

    scen = {
        "fixed": (prompts, [GEN] * bs),
        "churn": (prompts, [64 + 24 * i for i in range(bs)]),
        "poison": (prompts[:-1] + [prompts[-1][:100]], [GEN] * bs),
    }
    for name, (ps, gens) in scen.items():
        for r in range(reps):
            async def rep():
                return await asyncio.gather(*[one(p, g) for p, g in zip(ps, gens)])
            res = eng.loop.run_until_complete(rep())
            gaps_all, per = [], []
            for pts in res:
                g = [(t1 - t0) / (n1 - n0) * 1e3 for (t0, n0), (t1, n1) in zip(pts[1:], pts[2:]) if n1 > n0]
                gaps_all += g; per.append(g)
            gaps_all.sort(); med = gaps_all[len(gaps_all) // 2]
            li = max(range(bs), key=lambda i: gens[i])
            pts = res[li]; g = per[li]
            spikes = [x for x in g if x > 3 * med]
            print(f"SCEN {name} rep={r} be={be} plen={plen} bs={bs} med={med:.3f} "
                  f"long_decode_s={pts[-1][0] - pts[1][0]:.3f} steps={pts[-1][1] - pts[1][1]} "
                  f"spikes={len(spikes)} excess_ms={sum(x - med for x in spikes):.1f} "
                  f"max_gap={max(g):.1f}", flush=True)
    eng.shutdown()


if __name__ == "__main__":
    main()
