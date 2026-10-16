#!/usr/bin/env python3
"""Merge NIAH shards into flat {"L-D": pred} results.json and score them with the
show_image_paper.py metric (set word overlap with the needle answer, x100).
usage: score_niah.py name=tag[,tag...] ...   (each tag's preds_<tag>.jsonl files are merged)
"""
import json, os, re, sys

EXP = "eat a sandwich and sit in Dolores Park on a sunny day."
# NIAH_DIR: where preds_<tag>.jsonl live and niah/<name>/results.json are written.
NIAH_DIR = os.environ.get("NIAH_DIR", "niah_runs")
HF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                  "results", "niah", "Llama-3.1-8B-Instruct")  # HF reference runs

def words(t):
    return re.sub(r'[^\w\s]', '', t.lower()).split()

def score(resp):
    e = set(words(EXP))
    return len(set(words(resp or "")) & e) / len(e) * 100

def key(k):
    L, D = k.split("-"); return int(L), int(D)

def load_preds(tags):
    out = {}
    for t in tags:
        for line in open(f"{NIAH_DIR}/preds_{t}.jsonl"):
            r = json.loads(line); out[r["id"]] = r["pred"]
    return dict(sorted(out.items(), key=lambda kv: key(kv[0])))

def summary(name, res):
    s = {k: score(v) for k, v in res.items()}
    bad = [k for k, v in s.items() if v < 100]
    avg = sum(s.values()) / len(s)
    print(f"{name:28s} n={len(s):3d} avg={avg:6.2f} cells<100={len(bad):2d} "
          f"exact0={sum(v == 0 for v in s.values())}")
    return s

def main():
    runs = {}
    for arg in sys.argv[1:]:
        name, tags = arg.split("=")
        res = load_preds(tags.split(","))
        d = f"{NIAH_DIR}/niah/{name}"; os.makedirs(d, exist_ok=True)
        json.dump(res, open(f"{d}/results.json", "w"), indent=4, ensure_ascii=False)
        runs[name] = res
    for tag in ("flash_attention_2", "picokv_offline_k320_v320", "picokv_online_k320_v320"):
        p = f"{HF}/{tag}/results.json"
        if os.path.exists(p):
            runs["HF:" + tag] = dict(sorted(json.load(open(p)).items(), key=lambda kv: key(kv[0])))
    scores = {n: summary(n, r) for n, r in runs.items()}
    names = list(runs)
    print("\n# identical-pred counts (common cells)")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            c = set(runs[a]) & set(runs[b])
            print(f"  {a} vs {b}: {sum(runs[a][k].strip() == runs[b][k].strip() for k in c)}/{len(c)}")
    for n, s in scores.items():
        Ls = sorted({key(k)[0] for k in s}); Ds = sorted({key(k)[1] for k in s})
        print(f"\n# {n}: rows=depth, cols=length(k)")
        print("D\\L " + " ".join(f"{L // 1024:>4d}" for L in Ls))
        for D in Ds:
            print(f"{D:>3d} " + " ".join(f"{s.get(f'{L}-{D}', float('nan')):4.0f}" for L in Ls))
        print("  by-length avg: " + " ".join(f"{L // 1024}k={sum(s[f'{L}-{D}'] for D in Ds if f'{L}-{D}' in s) / max(1, sum(f'{L}-{D}' in s for D in Ds)):.1f}" for L in Ls))
        for k, v in s.items():
            if v < 100:
                print(f"  FAIL {k} {v:5.1f} | {runs[n][k].strip()[:150]!r}")

if __name__ == "__main__":
    main()
