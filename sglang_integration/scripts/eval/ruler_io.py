#!/usr/bin/env python3
"""RULER capture/score halves (run with .venv_picokv: lm_eval + transformers 4.53).

The picokv accuracy path must run through SGLang (the repo's HF adapter is broken
on this stack). SGLang lives in a different torch env, so we split RULER into:
  capture -> dump the exact (seeded) RULER prompts+gold to prompts.jsonl
  [infer] -> ruler_infer.py (.venv_sglang) generates preds.jsonl via SGLang
  score   -> replay preds.jsonl through lm_eval's official RULER scoring
Capture and score call simple_evaluate with identical seeds/limit/lengths, so the
generated contexts are identical and predictions map back by context hash.
"""
import argparse, hashlib, json, os, sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[3]  # PiCoKV repo root
sys.path.insert(0, str(REPO))
from lm_eval import evaluator          # noqa: E402
from src.lmeval import MyModel         # noqa: E402

SEED = 42


def _key(ctx):
    return hashlib.sha1(ctx.encode("utf-8")).hexdigest()


class CaptureLM(MyModel):
    def __init__(self, tok, max_length, max_gen, sink):
        super().__init__(model=None, tokenizer=tok, model_max_length=max_length,
                         max_gen_toks=max_gen)
        self.sink = sink

    def generate_until(self, requests, disable_tqdm=False):
        out = []
        for req in requests:
            ctx, gk = req.args
            self.sink.append({"key": _key(ctx), "context": ctx,
                              "until": gk.get("until", []) or [],
                              "max_gen_toks": gk.get("max_gen_toks", self._max_gen_toks)})
            out.append("")
        return out


class ReplayLM(MyModel):
    def __init__(self, tok, max_length, max_gen, preds):
        super().__init__(model=None, tokenizer=tok, model_max_length=max_length,
                         max_gen_toks=max_gen)
        self.preds = preds
        self.miss = 0

    def generate_until(self, requests, disable_tqdm=False):
        out = []
        for req in requests:
            ctx, _ = req.args
            p = self.preds.get(_key(ctx))
            if p is None:
                self.miss += 1; p = ""
            out.append(p)
        return out


def evaluate_lengths(lm, model, tasks, lengths, limit, max_gen):
    per = {}
    for length in lengths:
        md = {"max_seq_lengths": [length], "pretrained": model, "tokenizer": model}
        r = evaluator.simple_evaluate(
            model=lm, tasks=tasks, num_fewshot=0, batch_size=1, limit=limit,
            log_samples=True, metadata=md, model_args=f"pretrained={model}",
            apply_chat_template=True, random_seed=SEED, numpy_random_seed=SEED,
            torch_random_seed=SEED, fewshot_random_seed=SEED)
        per[length] = r["results"]
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["capture", "score"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--lengths", required=True)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--max_gen_toks", type=int, default=64)
    ap.add_argument("--prompts", default="ruler_prompts.jsonl")
    ap.add_argument("--preds", default="ruler_preds.jsonl")
    ap.add_argument("--out", default="ruler_scores.json")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tasks = [t.strip() for t in args.tasks.split(",")]
    lengths = [int(x) for x in args.lengths.split(",")]

    if args.mode == "capture":
        sink = []
        lm = CaptureLM(tok, max(lengths), args.max_gen_toks, sink)
        evaluate_lengths(lm, args.model, tasks, lengths, args.limit, args.max_gen_toks)
        # de-dup by key, keep first
        seen, rows = set(), []
        for r in sink:
            if r["key"] in seen:
                continue
            seen.add(r["key"]); rows.append(r)
        with open(args.prompts, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"CAPTURED {len(rows)} prompts -> {args.prompts}", flush=True)
    else:
        preds = {}
        for l in open(args.preds):
            d = json.loads(l); preds[d["key"]] = d["pred"]
        lm = ReplayLM(tok, max(lengths), args.max_gen_toks, preds)
        per = evaluate_lengths(lm, args.model, tasks, lengths, args.limit, args.max_gen_toks)
        json.dump(per, open(args.out, "w"), indent=2, default=str)
        print(f"SCORED (replay miss={lm.miss}) -> {args.out}", flush=True)
        for length, res in per.items():
            for task, m in res.items():
                acc = {k: v for k, v in m.items() if "stderr" not in k and k != "alias"}
                print(f"  len={length} {task}: {acc}", flush=True)


if __name__ == "__main__":
    main()
