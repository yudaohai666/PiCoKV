"""Memory capacity: N requests of a PLEN-token random prompt submitted at once
(SGLang's default chunked prefill, no radix cache, max_running_requests=N), GEN
generated tokens each (ignore_eos). The scheduler's log reports the pool size
(startup line 'max_total_num_tokens=') and every 8th decode step's
'#running-req' / 'token usage': the largest '#running-req' is how many of the
requests decoded at once. Watch the GPU's memory with nvidia-smi alongside.
  python bench_capacity.py <backend> <plen> <n> [gen=256] [memf=0.9]
picokv runs the breakable decode graph, any other backend the full graph."""
import os, random, sys, time
import sglang as sgl


def main():
    be, plen, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    gen = int(sys.argv[4]) if len(sys.argv) > 4 else 256
    memf = float(sys.argv[5]) if len(sys.argv) > 5 else 0.9
    eng = sgl.Engine(
        model_path=os.environ.get("BENCH_MODEL", "llama31_8b_instruct"), attention_backend=be,
        dtype="bfloat16", tp_size=1, mem_fraction_static=memf, disable_radix_cache=True,
        max_running_requests=n, trust_remote_code=True, log_level="info",
        decode_log_interval=8,
        cuda_graph_backend_decode=("breakable" if be == "picokv" else "full"))
    random.seed(0)
    prompts = [[random.randint(10, 120000) for _ in range(plen)] for _ in range(n)]
    t0 = time.perf_counter()
    out = eng.generate(input_ids=prompts, sampling_params=[
        {"temperature": 0.0, "max_new_tokens": gen, "ignore_eos": True}] * n)
    dt = time.perf_counter() - t0
    toks = sum(o["meta_info"]["completion_tokens"] for o in out)
    print("MEMCAP_RESULT backend=%s plen=%d n=%d gen=%d memf=%.2f done=%d wall_s=%.1f out_tok=%d"
          % (be, plen, n, gen, memf, len(out), dt, toks), flush=True)
    eng.shutdown()


if __name__ == "__main__":
    main()
