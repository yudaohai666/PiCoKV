#!/usr/bin/env python3
"""Turn-30 serving-feature probes on real weights (NIAH needle score, t28 metric).
usage: probe_quality.py <backend> <radix|chunk> <out.jsonl>
radix: radix cache ON, no chunking. Per cell: A cold, A again (prefix hit),
       B = same document + rephrased question (prefix hit on the whole document).
chunk: radix OFF, sglang's default chunked_prefill_size (8192 on this GPU),
       all 10 depths at L=8192 and L=32768."""
import json, os, re, sys
import sglang as sgl

EXP = "eat a sandwich and sit in Dolores Park on a sunny day."
REPHRASE = ("<|start_header_id|>user<|end_header_id|>\n\nAccording to the document, what is the "
            "best thing to do in San Francisco? Answer in one sentence.<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n")


def score(t):
    w = lambda s: set(re.sub(r'[^\w\s]', '', s.lower()).split())
    e = w(EXP)
    return len(w(t or "") & e) / len(e) * 100


def main():
    be, mode, out = sys.argv[1], sys.argv[2], sys.argv[3]
    rows = {json.loads(l)["id"]: json.loads(l) for l in open(os.environ.get("NIAH_RECORDS", "niah_records.jsonl"))}  # from eval/build_niah.py
    kw = dict(model_path=os.environ.get("NIAH_MODEL", "llama31_8b_instruct"), attention_backend=be, dtype="bfloat16", tp_size=1,
              mem_fraction_static=0.70, disable_cuda_graph=True, max_running_requests=16,
              context_length=131072, trust_remote_code=True, log_level="error")
    if mode == "radix":
        kw.update(disable_radix_cache=False, chunked_prefill_size=1 << 20)
    else:
        kw.update(disable_radix_cache=True)  # chunked_prefill_size left to sglang's default
    eng = sgl.Engine(**kw)
    print(f"server chunked_prefill_size={eng.server_args.chunked_prefill_size} "
          f"radix={'off' if eng.server_args.disable_radix_cache else 'on'}", flush=True)
    tok = eng.tokenizer_manager.tokenizer
    f = open(out, "w")

    def run(tag, ids_list):
        sp = [{"temperature": 0.0, "max_new_tokens": 50}] * len(ids_list)
        res = eng.generate(input_ids=ids_list, sampling_params=sp)
        for t, ids, o in zip(tag, ids_list, res):
            s = score(o["text"])
            rec = {"tag": t, "len": len(ids), "cached": o["meta_info"].get("cached_tokens"),
                   "score": s, "pred": o["text"]}
            f.write(json.dumps(rec) + "\n"); f.flush()
            print(f"{be} {mode} {t:22s} len={len(ids):6d} cached={rec['cached']} "
                  f"score={s:5.1f} pred={o['text'][:70]!r}", flush=True)
        return res

    if mode == "radix":
        for cid in ["8192-22", "32768-22", "32768-55"]:
            a = rows[cid]["input_ids"]
            cut = max(i for i, x in enumerate(a) if x == 128006)      # assistant header
            cut = max(i for i, x in enumerate(a[:cut]) if x == 128006)  # question turn
            b = a[:cut] + tok.encode(REPHRASE, add_special_tokens=False)
            run([f"{cid}/A-cold"], [a])
            run([f"{cid}/A-repeat"], [a])
            run([f"{cid}/B-sameDoc"], [b])
    else:
        for L in (8192, 32768):
            ids = [k for k in rows if k.startswith(f"{L}-")]
            res = run(ids, [rows[k]["input_ids"] for k in ids])
    f.close(); eng.shutdown()
    print("PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
