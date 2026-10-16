#!/usr/bin/env python3
"""LongBench build/score halves (run with .venv_picokv: transformers 4.53 + metrics deps).

Same SGLang-bridge pattern as RULER (the repo HF adapter is broken on this stack):
  build  -> format+truncate+chat-template each LongBench sample -> records.jsonl
            (input_ids + gold answers + all_classes + max_gen), from the local
            data.zip extract (datasets>=5 dropped THUDM/LongBench's load script).
  [infer]-> longbench_infer.py (.venv_sglang) generates preds via SGLang.
  score  -> group preds by dataset, run the repo's official LongBench scorer.
"""
import argparse, json, os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]  # PiCoKV repo root
HERE = Path(os.environ.get("PICOKV_EVAL_DATA", ".")).resolve()  # dataset dir (default: cwd)
LB = REPO / "benchmarks" / "Longbench"
sys.path.insert(0, str(LB))            # metrics.py / eval.py
DATA = HERE / "longbench_data" / "data"
CFG = LB / "config"
NO_CHAT_TEMPLATE = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}


def build(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = json.load(open(CFG / "dataset2prompt.json"))
    maxlen = json.load(open(CFG / "dataset2maxlen.json"))
    dsets = args.datasets.split(",")
    out = open(args.records, "w")
    n = 0
    for ds in dsets:
        fp = DATA / f"{ds}.jsonl"
        rows = [json.loads(l) for l in open(fp)][: args.limit]
        pf = prompts[ds]; mg = maxlen[ds]
        for i, obj in enumerate(rows):
            prompt = pf.format(**obj)
            ids = tok(prompt, truncation=False, return_tensors="pt").input_ids[0]
            if len(ids) > args.max_length:              # middle-truncate (upstream)
                half = args.max_length // 2
                prompt = (tok.decode(ids[:half], skip_special_tokens=True)
                          + tok.decode(ids[-half:], skip_special_tokens=True))
            if ds not in NO_CHAT_TEMPLATE and tok.chat_template is not None:
                input_ids = tok.apply_chat_template(
                    [{"role": "user", "content": prompt}], tokenize=True,
                    add_generation_prompt=True, enable_thinking=False)
            else:
                input_ids = tok(prompt, truncation=False).input_ids
            out.write(json.dumps({"id": f"{ds}:{i}", "dataset": ds, "input_ids": input_ids,
                                  "max_gen": mg, "answers": obj["answers"],
                                  "all_classes": obj["all_classes"],
                                  "length": obj["length"]}) + "\n")
            n += 1
    out.close()
    print(f"BUILT {n} records ({len(dsets)} datasets) -> {args.records}", flush=True)


def score(args):
    import importlib.util
    spec = importlib.util.spec_from_file_location("lb_eval", str(LB / "eval.py"))
    ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
    recs = {json.loads(l)["id"]: json.loads(l) for l in open(args.records)}
    preds = {json.loads(l)["id"]: json.loads(l)["pred"] for l in open(args.preds)}
    by = {}
    for rid, r in recs.items():
        by.setdefault(r["dataset"], []).append((preds.get(rid, ""), r["answers"], r["all_classes"]))
    scores, tot = {}, []
    for ds, items in sorted(by.items()):
        p = [x[0] for x in items]; a = [x[1] for x in items]; ac = items[0][2]
        s = ev.scorer(ds, p, a, ac)
        scores[ds] = s; tot.append(s)
        print(f"  {ds}: {s}  (n={len(items)})", flush=True)
    scores["avg"] = round(sum(tot) / len(tot), 2)
    json.dump(scores, open(args.out, "w"), indent=2)
    print(f"SCORED avg={scores['avg']} -> {args.out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["build", "score"])
    ap.add_argument("--model")
    ap.add_argument("--datasets", default="")
    ap.add_argument("--limit", type=int, default=15)
    ap.add_argument("--max_length", type=int, default=127500)
    ap.add_argument("--records", default="longbench_records.jsonl")
    ap.add_argument("--preds", default="longbench_preds.jsonl")
    ap.add_argument("--out", default="longbench_scores.json")
    args = ap.parse_args()
    build(args) if args.mode == "build" else score(args)


if __name__ == "__main__":
    main()
