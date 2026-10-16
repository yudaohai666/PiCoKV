#!/usr/bin/env python3
"""LongBench-v2 build/score (run with .venv_picokv). MCQ (A/B/C/D) over long
context; scored by choice accuracy. Same SGLang bridge; reuses longbench_infer.py
for generation (records carry input_ids + max_gen)."""
import argparse, json, os, re, sys
from pathlib import Path

HERE = Path(os.environ.get("PICOKV_EVAL_DATA", ".")).resolve()  # dataset dir (default: cwd)
DATA = HERE / "longbench_v2_data" / "data.json"

TEMPLATE = (
    "Please read the following text and answer the question below.\n\n"
    "{context}\n\n"
    "What is the correct answer to this question: {question}\n"
    "(A) {choice_A}\n(B) {choice_B}\n(C) {choice_C}\n(D) {choice_D}\n\n"
    "Answer with the single letter (A, B, C, or D) of the correct choice. "
    "The correct answer is")


def extract(pred):
    m = re.search(r"answer is\s*\(?([ABCD])\)?", pred)
    if m:
        return m.group(1)
    m = re.search(r"\b([ABCD])\b", pred)
    return m.group(1) if m else ""


def build(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    data = json.load(open(DATA))[: args.limit]
    out = open(args.records, "w")
    for i, o in enumerate(data):
        prompt = TEMPLATE.format(context=o["context"], question=o["question"],
                                 choice_A=o["choice_A"], choice_B=o["choice_B"],
                                 choice_C=o["choice_C"], choice_D=o["choice_D"])
        ids = tok(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(ids) > args.max_length:
            half = args.max_length // 2
            prompt = (tok.decode(ids[:half], skip_special_tokens=True)
                      + tok.decode(ids[-half:], skip_special_tokens=True))
        input_ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                            tokenize=True, add_generation_prompt=True,
                                            enable_thinking=False)
        out.write(json.dumps({"id": f"lbv2:{i}", "input_ids": input_ids, "max_gen": 16,
                              "answer": o["answer"], "difficulty": o["difficulty"],
                              "length": o["length"]}) + "\n")
    out.close()
    print(f"BUILT {min(len(data), args.limit)} records -> {args.records}", flush=True)


def score(args):
    recs = {json.loads(l)["id"]: json.loads(l) for l in open(args.records)}
    preds = {json.loads(l)["id"]: json.loads(l)["pred"] for l in open(args.preds)}
    n = cor = 0
    by_len = {}
    for rid, r in recs.items():
        pl = extract(preds.get(rid, ""))
        ok = int(pl == r["answer"])
        n += 1; cor += ok
        by_len.setdefault(r["length"], [0, 0])
        by_len[r["length"]][0] += ok; by_len[r["length"]][1] += 1
    acc = round(100 * cor / max(n, 1), 2)
    res = {"overall_acc": acc, "n": n,
           "by_length": {k: round(100 * v[0] / v[1], 2) for k, v in by_len.items()}}
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"SCORED acc={acc} (n={n}) by_length={res['by_length']} -> {args.out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["build", "score"])
    ap.add_argument("--model")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--max_length", type=int, default=127500)
    ap.add_argument("--records", default="lbv2_records.jsonl")
    ap.add_argument("--preds", default="lbv2_preds.jsonl")
    ap.add_argument("--out", default="lbv2_scores.json")
    args = ap.parse_args()
    build(args) if args.mode == "build" else score(args)


if __name__ == "__main__":
    main()
