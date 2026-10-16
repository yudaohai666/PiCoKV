#!/usr/bin/env python3
"""LongBench infer half (run with .venv_sglang). Reads longbench_records.jsonl,
greedily generates with an SGLang Engine (picokv|flashinfer), writes {id, pred}."""
import argparse, json
import sglang as sgl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", required=True)
    ap.add_argument("--records", default="longbench_records.jsonl")
    ap.add_argument("--preds", default="longbench_preds.jsonl")
    ap.add_argument("--max_length", type=int, default=131072)
    ap.add_argument("--mem_fraction", type=float, default=0.85)
    ap.add_argument("--chunk", type=int, default=16)
    # Decode graph mode. none = eager (the 09-21 setting, kept as default for
    # comparability); bcg = sglang breakable decode graph (picokv speed-bench
    # mode); full = whole-step decode graph (flashinfer speed-bench mode).
    # Prefill graphs stay off in every mode (same as scripts/bench_itl.py).
    ap.add_argument("--graph", choices=["none", "bcg", "full"], default="none")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.records)]
    gkw = {}
    if args.graph != "none":
        gkw = dict(disable_prefill_cuda_graph=True,
                   cuda_graph_backend_decode="breakable" if args.graph == "bcg" else "full")
    eng = sgl.Engine(model_path=args.model, attention_backend=args.backend, dtype="bfloat16",
                     tp_size=1, mem_fraction_static=args.mem_fraction,
                     disable_cuda_graph=args.graph == "none",
                     disable_radix_cache=True, chunked_prefill_size=1 << 20,
                     max_running_requests=64, context_length=args.max_length,
                     trust_remote_code=True, log_level="error", **gkw)
    out = open(args.preds, "w"); done = 0
    for i in range(0, len(rows), args.chunk):
        batch = rows[i:i + args.chunk]
        sp = [{"temperature": 0.0, "max_new_tokens": int(r["max_gen"])} for r in batch]
        res = eng.generate(input_ids=[r["input_ids"] for r in batch], sampling_params=sp)
        for r, o in zip(batch, res):
            out.write(json.dumps({"id": r["id"], "pred": o["text"]}) + "\n")
        out.flush(); done += len(batch)
        print(f"  infer {done}/{len(rows)}", flush=True)
    out.close(); eng.shutdown()
    print(f"INFER_DONE backend={args.backend} n={done} -> {args.preds}", flush=True)


if __name__ == "__main__":
    main()
