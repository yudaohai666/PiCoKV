"""Per-step decode timing via streamed inter-token latency (robust to spikes).

t(gen)-t(8) pairs are fragile: one host hiccup in either call skews a pair by
whole milliseconds. Here every request streams; the time between consecutive
chunks divided by the tokens they carry is one step sample. Median/trimmed mean
over hundreds of samples per rep.

Usage: python bench_itl.py <backend> <plen> <batch> <gen> [reps] [memf]
"""
import asyncio, os, random, sys, time
import sglang as sgl

# Model dir (timing only: random weights are fine). Override with BENCH_MODEL.
MODEL = os.environ.get("BENCH_MODEL", "llama31_8b_rand")


def stats(xs):
    xs = sorted(xs)
    n = len(xs)
    lo, hi = n // 10, n - n // 10
    tm = sum(xs[lo:hi]) / max(1, hi - lo)
    return xs[n // 2], tm, xs[n // 10], xs[(9 * n) // 10]


def main():
    be, plen, bs, gen = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    reps = int(sys.argv[5]) if len(sys.argv) > 5 else 3
    memf = float(sys.argv[6]) if len(sys.argv) > 6 else 0.70
    gkw = {}
    if os.environ.get("BENCH_DECODE_GRAPH"):
        gkw["cuda_graph_backend_decode"] = os.environ["BENCH_DECODE_GRAPH"]
    graphs_on = os.environ.get("BENCH_GRAPHS", "0") == "1"
    eng = sgl.Engine(
        model_path=MODEL, attention_backend=be, dtype="bfloat16", tp_size=1,
        mem_fraction_static=memf, disable_cuda_graph=not graphs_on, disable_radix_cache=True,
        disable_prefill_cuda_graph=graphs_on, chunked_prefill_size=1 << 20,
        max_running_requests=bs, trust_remote_code=True, log_level="info", **gkw)
    random.seed(0)
    prompts = [[random.randint(10, 120000) for _ in range(plen)] for _ in range(bs)]
    # ITL_WARM_LEN truncates the warmup prompts (JIT/cublas/graph warmup only;
    # saves B full prefills at long contexts). Default: full prompts.
    wl = int(os.environ.get("ITL_WARM_LEN", "0")) or plen
    eng.generate(input_ids=[p[:wl] for p in prompts], sampling_params=[
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

    per_rep = []
    for r in range(reps):
        res = eng.loop.run_until_complete(rep())
        samples = []
        for pts in res:
            # skip the prefill->first-token chunk and the first decode chunk
            for (t0, n0), (t1, n1) in zip(pts[1:], pts[2:]):
                if n1 > n0:
                    samples.append((t1 - t0) / (n1 - n0) * 1e3)
        med, tm, p10, p90 = stats(samples)
        per_rep.append(med)
        print(f"REP {r} n={len(samples)} med={med:.3f} tmean={tm:.3f} "
              f"p10={p10:.3f} p90={p90:.3f}", flush=True)
    per_rep.sort()
    print(f"RESULT backend={be} prompt={plen} batch={bs} gen={gen} "
          f"step_ms_med={per_rep[len(per_rep)//2]:.3f} "
          f"reps=[{', '.join(f'{x:.3f}' for x in per_rep)}]", flush=True)
    eng.shutdown()


if __name__ == "__main__":
    main()
