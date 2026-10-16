#!/usr/bin/env python3
"""InfiniteBench build/score (run with .venv_picokv). Self-contained: reads the
downloaded per-task jsonl, builds prompts with the repo's official doc_to_text +
descriptions, and scores with the repo's process_results (utils.py). Reuses
longbench_infer.py for SGLang generation. Contexts > max_length are middle-
truncated identically for both backends (model context = 131072 < InfiniteBench's
~200k), so absolute scores are truncation-limited but the picokv-vs-dense delta
is apples-to-apples."""
import argparse, importlib.util, json, os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]  # PiCoKV repo root
HERE = Path(os.environ.get("PICOKV_EVAL_DATA", ".")).resolve()  # dataset dir (default: cwd)
IBT = REPO / "benchmarks" / "InfiniteBench" / "tasks" / "infinitebench"
DATA = HERE / "ib_data"
spec = importlib.util.spec_from_file_location("ib_utils", str(IBT / "utils.py"))
U = importlib.util.module_from_spec(spec); spec.loader.exec_module(U)

DESC = {
    "passkey": "There is an important info hidden inside a lot of irrelevant text. "
               "Find it and memorize them. I will quiz you about the important "
               "information there.\n\n",
    "kv_retrieval": "Extract the value corresponding to the specified key in the "
                    "JSON object below.\n\n",
    "math_find": "",
}
MAXGEN = {"passkey": 40, "kv_retrieval": 50, "math_find": 40}
SCORER = {"passkey": U.process_results_int_match, "kv_retrieval": U.process_results_kv,
          "math_find": U.process_results_math}


def prompt_of(task, doc):
    if task == "math_find":
        return U.doc_to_text_math_find(doc)
    return DESC[task] + f'{doc["context"]}\n\n{doc["input"]}'


def build(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    out = open(args.records, "w"); n = 0
    for task in args.tasks.split(","):
        rows = []
        with open(DATA / f"{task}.jsonl") as f:
            for i, line in enumerate(f):
                if i >= args.limit:
                    break
                rows.append(json.loads(line))
        for i, doc in enumerate(rows):
            prompt = prompt_of(task, doc)
            ids = tok(prompt, truncation=False, return_tensors="pt").input_ids[0]
            if len(ids) > args.max_length:
                half = args.max_length // 2
                prompt = (tok.decode(ids[:half], skip_special_tokens=True)
                          + tok.decode(ids[-half:], skip_special_tokens=True))
            input_ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                                tokenize=True, add_generation_prompt=True,
                                                enable_thinking=False)
            ans = doc.get("answer", [])
            if not isinstance(ans, list):
                ans = [str(ans)]
            out.write(json.dumps({"id": f"{task}:{i}", "task": task, "input_ids": input_ids,
                                  "max_gen": MAXGEN[task], "answer": ans}) + "\n")
            n += 1
    out.close()
    print(f"BUILT {n} records ({args.tasks}) -> {args.records}", flush=True)


def score(args):
    recs = {json.loads(l)["id"]: json.loads(l) for l in open(args.records)}
    preds = {json.loads(l)["id"]: json.loads(l)["pred"] for l in open(args.preds)}
    by = {}
    for rid, r in recs.items():
        ans = [str(a) for a in r["answer"]]
        sc = SCORER[r["task"]]({"answer": ans}, [preds.get(rid, "")])["score"]
        by.setdefault(r["task"], []).append(sc)
    res = {t: round(100 * sum(v) / len(v), 2) for t, v in sorted(by.items())}
    res["avg"] = round(sum(res.values()) / len(res), 2)
    json.dump(res, open(args.out, "w"), indent=2)
    for t, v in res.items():
        print(f"  {t}: {v}", flush=True)
    print(f"SCORED -> {args.out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["build", "score"])
    ap.add_argument("--model")
    ap.add_argument("--tasks", default="passkey,kv_retrieval,math_find")
    ap.add_argument("--limit", type=int, default=15)
    ap.add_argument("--max_length", type=int, default=127500)
    ap.add_argument("--records", default="ib_records.jsonl")
    ap.add_argument("--preds", default="ib_preds.jsonl")
    ap.add_argument("--out", default="ib_scores.json")
    args = ap.parse_args()
    build(args) if args.mode == "build" else score(args)


if __name__ == "__main__":
    main()
