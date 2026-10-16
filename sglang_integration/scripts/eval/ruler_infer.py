#!/usr/bin/env python3
"""RULER infer half (run with .venv_sglang: torch 2.13 + SGLang picokv/flashinfer).

Reads ruler_prompts.jsonl (from ruler_io.py --mode capture), generates greedily
via an SGLang Engine with the chosen attention backend, writes ruler_preds.jsonl
{key, pred}. picokv vs flashinfer are compared by running this twice.
"""
import argparse, json
from transformers import AutoTokenizer
import sglang as sgl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", required=True)
    ap.add_argument("--prompts", default="ruler_prompts.jsonl")
    ap.add_argument("--preds", default="ruler_preds.jsonl")
    ap.add_argument("--max_length", type=int, default=131072)
    ap.add_argument("--mem_fraction", type=float, default=0.85)
    ap.add_argument("--chunk", type=int, default=16)
    # Decode graph mode, same meaning as longbench_infer.py --graph.
    ap.add_argument("--graph", choices=["none", "bcg", "full"], default="none")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    rows = [json.loads(l) for l in open(args.prompts)]
    # Encode without an extra BOS: the chat template already begins the sequence.
    for r in rows:
        r["ids"] = tok.encode(r["context"], add_special_tokens=False)

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

    out = open(args.preds, "w")
    done = 0
    for i in range(0, len(rows), args.chunk):
        batch = rows[i:i + args.chunk]
        sp = [{"temperature": 0.0, "max_new_tokens": int(r["max_gen_toks"]),
               "stop": r["until"]} for r in batch]
        res = eng.generate(input_ids=[r["ids"] for r in batch], sampling_params=sp)
        for r, o in zip(batch, res):
            out.write(json.dumps({"key": r["key"], "pred": o["text"]}) + "\n")
        out.flush(); done += len(batch)
        print(f"  infer {done}/{len(rows)}", flush=True)
    out.close()
    eng.shutdown()
    print(f"INFER_DONE backend={args.backend} n={done} -> {args.preds}", flush=True)


if __name__ == "__main__":
    main()
