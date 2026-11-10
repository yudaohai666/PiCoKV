#!/usr/bin/env python3
"""Prefill (TTFT) probe on real weights: wall time of generate(max_new_tokens=1) per case
bs:L, radix off, serving decode graph (picokv breakable, flashinfer full). TTFT includes the
first decode step's host cost under the overlap scheduler (both backends).
usage: bench_ttft.py <backend> [reps=5] [cases=bs:L,...]
env BENCH_MODEL (default llama31_8b_instruct), TTFT_CHUNK (chunked_prefill_size; default
1<<20 = one chunk, 8192 = SGLang's default on an 80 GB GPU)"""
import os, random, sys, time
import sglang as sgl

CASES = ("1:128,1:512,1:2048,1:8192,1:32768,1:65536,1:122880,"
         "4:512,8:1024,24:512,64:256,8:4096,4:16384")


def main():
    be = sys.argv[1]
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    cases = [tuple(map(int, c.split(":"))) for c in (sys.argv[3] if len(sys.argv) > 3 else CASES).split(",")]
    eng = sgl.Engine(model_path=os.environ.get("BENCH_MODEL", "llama31_8b_instruct"),
                     attention_backend=be, dtype="bfloat16",
                     tp_size=1, mem_fraction_static=0.70, disable_radix_cache=True,
                     chunked_prefill_size=int(os.environ.get("TTFT_CHUNK", 1 << 20)),
                     max_running_requests=64, context_length=131072,
                     trust_remote_code=True, log_level="error", disable_prefill_cuda_graph=True,
                     cuda_graph_backend_decode="breakable" if be == "picokv" else "full")
    random.seed(0)
    sp1 = {"temperature": 0.0, "max_new_tokens": 1, "ignore_eos": True}
    warm = [[random.randint(10, 120000) for _ in range(2048)] for _ in range(4)]
    for _ in range(2):
        eng.generate(input_ids=warm, sampling_params=[sp1] * 4)
    for bs, L in cases:
        ps = [[random.randint(10, 120000) for _ in range(L)] for _ in range(bs)]
        eng.generate(input_ids=ps, sampling_params=[sp1] * bs)
        n = reps if bs * L <= 65536 else 3
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            eng.generate(input_ids=ps, sampling_params=[sp1] * bs)
            ts.append((time.perf_counter() - t0) * 1e3)
        ts.sort()
        m = ts[len(ts) // 2]
        print(f"TTFT backend={be} bs={bs} L={L} med_ms={m:.2f} min_ms={ts[0]:.2f} "
              f"prefill_tok_s={bs * L / m * 1e3:.0f} reps={n}", flush=True)
    eng.shutdown()


if __name__ == "__main__":
    main()
