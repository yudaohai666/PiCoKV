#!/usr/bin/env python3
"""Turn-31 prefill (TTFT) probe: wall time of generate(max_new_tokens=1), random-weight
llama31_8b_rand, radix off, chunked_prefill_size=1<<20 (whole prompt in one extend).
usage: bench_ttft.py <backend> [reps]"""
import os, random, sys, time
import sglang as sgl

CASES = [(1, 8192), (1, 32768), (1, 122880), (32, 1024), (64, 512), (64, 256)]


def main():
    be = sys.argv[1]
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    eng = sgl.Engine(model_path=os.environ.get("BENCH_MODEL", "llama31_8b_rand"), attention_backend=be, dtype="bfloat16",
                     tp_size=1, mem_fraction_static=0.70, disable_cuda_graph=True,
                     disable_radix_cache=True, chunked_prefill_size=1 << 20,
                     max_running_requests=64, context_length=131072,
                     trust_remote_code=True, log_level="error")
    random.seed(0)
    sp1 = {"temperature": 0.0, "max_new_tokens": 1, "ignore_eos": True}
    warm = [[random.randint(10, 120000) for _ in range(2048)] for _ in range(4)]
    eng.generate(input_ids=warm, sampling_params=[sp1] * 4)
    for bs, L in CASES:
        ps = [[random.randint(10, 120000) for _ in range(L)] for _ in range(bs)]
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            eng.generate(input_ids=ps, sampling_params=[sp1] * bs)
            ts.append((time.perf_counter() - t0) * 1e3)
        ts.sort()
        print(f"TTFT be={be} bs={bs} len={L} med_ms={ts[len(ts)//2]:.1f} "
              f"all=[{', '.join(f'{x:.1f}' for x in ts)}]", flush=True)
    eng.shutdown()


if __name__ == "__main__":
    main()
