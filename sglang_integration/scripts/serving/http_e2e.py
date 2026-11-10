"""t56 HTTP end-to-end check of a running sglang server (any attention backend).
usage: http_e2e.py <port> <out.json> [radix]   (radix: also expect prefix-cache hits)"""
import json, os, sys, time, threading, urllib.request, urllib.error, http.client, math, random
port, out = int(sys.argv[1]), sys.argv[2]
radix = len(sys.argv) > 3 and sys.argv[3] == "radix"
urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))
U = f"http://127.0.0.1:{port}"


def req(path, body=None, timeout=900):
    r = urllib.request.Request(U + path, json.dumps(body).encode() if body is not None else None,
                               {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=timeout) as f:
            return f.status, f.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def js(path, body=None):
    s, d = req(path, body)
    try:
        return s, json.loads(d)
    except Exception:
        return s, d


def stream(path, body, stop_after=None):
    """SSE lines (data: ...) of a streaming POST; stop_after: close after n events."""
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
    c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    ev = []
    while True:
        ln = r.readline()
        if not ln:
            break
        ln = ln.decode().strip()
        if ln.startswith("data:"):
            ev.append(ln[5:].strip())
            if stop_after is not None and len(ev) >= stop_after:
                c.sock.close()  # client disconnect mid-stream
                break
    c.close()
    return r.status, ev


R, ok = {}, []


def chk(name, c, info, info_only=False):
    if info_only:
        R[name] = {"ok": bool(c), "info": info}
        print("INFO" if c else "INFO-DIFF", name, str(info)[:240], flush=True)
        return
    ok.append(bool(c))
    R[name] = {"ok": bool(c), "info": info}
    print("PASS" if c else "FAIL", name, str(info)[:240], flush=True)


G = {"temperature": 0, "max_new_tokens": 48}
P = ["The capital of France is", "List three prime numbers:", "def fibonacci(n):",
     "Translate to German: good morning", "Q: What is 17 * 23?\nA:", "Once upon a time,",
     "The chemical symbol for gold is", "Explain recursion in one sentence:"]
s, d = req("/health")
chk("health", s == 200, s)
s, d = js("/get_model_info")
chk("get_model_info", s == 200 and "model_path" in d, d)
s, d = js("/get_server_info")
chk("get_server_info backend", s == 200 and "attention_backend" in d, d.get("attention_backend") if s == 200 else d)
# 1. greedy single + batch: batch rows == their solo outputs
solo = []
for p in P:
    s, d = js("/generate", {"text": p, "sampling_params": G})
    solo.append(d["text"] if s == 200 else None)
chk("generate greedy x8 solo", all(solo), [x[:40] for x in solo if x])
s, d = js("/generate", {"text": P, "sampling_params": G})
bt = [x["text"] for x in d] if s == 200 else []
chk("generate batch-list x8", len(bt) == 8 and all(bt), s)
R["batch_eq_solo"] = sum(a == b for a, b in zip(bt, solo))
print("  batch==solo:", R["batch_eq_solo"], "/ 8")
# 2. parallel sampling n=3
s, d = js("/generate", {"text": P[5], "sampling_params": {"temperature": 0.9, "max_new_tokens": 32, "n": 3}})
chk("n=3 sampling", s == 200 and isinstance(d, list) and len(d) == 3 and all(x["text"] for x in d),
    [x["text"][:30] for x in d] if s == 200 and isinstance(d, list) else d)
# 3. logprobs
s, d = js("/generate", {"text": P[0], "sampling_params": {"temperature": 0, "max_new_tokens": 8},
                        "return_logprob": True, "top_logprobs_num": 3, "logprob_start_len": 0})
mi = d.get("meta_info", {}) if s == 200 else {}
olp, ilp, top = mi.get("output_token_logprobs") or [], mi.get("input_token_logprobs") or [], mi.get("output_top_logprobs") or []
fin = all(x[0] is not None and math.isfinite(x[0]) and x[0] <= 1e-4 for x in olp)
# greedy: the chosen token is the top-1 alternative
top1 = all(t and t[0][1] == o[1] for t, o in zip(top, olp))
chk("logprobs out/in/top", s == 200 and len(olp) == 8 and len(ilp) == mi.get("prompt_tokens") and fin and top1,
    {"n_out": len(olp), "n_in": len(ilp), "prompt": mi.get("prompt_tokens"), "fin": fin, "top1": top1,
     "olp": [round(x[0], 3) for x in olp[:4]]})
# 4. stop string + finish reasons
s, d = js("/generate", {"text": "Count: 1, 2, 3, 4,", "sampling_params": {"temperature": 0, "max_new_tokens": 64, "stop": [" 7"]}})
fr = d.get("meta_info", {}).get("finish_reason") if s == 200 else None
chk("stop string", s == 200 and " 7" not in d["text"] and fr and fr.get("type") == "stop", (d.get("text") if s == 200 else d, fr))
s, d = js("/generate", {"text": P[1], "sampling_params": {"temperature": 0, "max_new_tokens": 5, "ignore_eos": True}})
fr = d.get("meta_info", {}).get("finish_reason") if s == 200 else None
chk("length finish", s == 200 and fr and fr.get("type") == "length" and d["meta_info"]["completion_tokens"] == 5, fr)
# 5. streaming == non-streaming greedy
st, ev = stream("/generate", {"text": P[2], "sampling_params": G, "stream": True})
last = json.loads(ev[-2])["text"] if len(ev) >= 2 else None
chk("stream generate ok", st == 200 and ev[-1] == "[DONE]" and last, (len(ev), (last or "")[:60]))
chk("stream generate == greedy", last == solo[2], (last or "")[:60], True)
# 6. OpenAI chat / completions
s, d = js("/v1/chat/completions", {"model": "m", "temperature": 0, "max_tokens": 32,
                                   "messages": [{"role": "user", "content": "Say hello in French."}]})
chat = d["choices"][0]["message"]["content"] if s == 200 else None
chk("openai chat", s == 200 and chat, (chat or d)[:80] if isinstance(chat or d, str) else d)
st, ev = stream("/v1/chat/completions", {"model": "m", "temperature": 0, "max_tokens": 32, "stream": True,
                                         "messages": [{"role": "user", "content": "Say hello in French."}]})
sc = "".join((json.loads(e)["choices"][0]["delta"].get("content") or "") for e in ev[:-1]
             if json.loads(e).get("choices"))
chk("openai chat stream ok", st == 200 and sc, sc[:80])
chk("openai chat stream == non-stream", sc == chat, sc[:80], True)
s, d = js("/v1/completions", {"model": "m", "prompt": P[4], "temperature": 0, "max_tokens": 16, "logprobs": 2})
lp = d["choices"][0].get("logprobs") if s == 200 else None
chk("openai completions + logprobs", s == 200 and lp and len(lp["token_logprobs"]) == d["usage"]["completion_tokens"],
    (d["choices"][0]["text"][:40] if s == 200 else d))
# 7. client disconnect mid-stream + explicit abort: the server keeps serving
st, ev = stream("/generate", {"text": P[5], "sampling_params": {"temperature": 0, "max_new_tokens": 3000, "ignore_eos": True},
                              "stream": True}, stop_after=5)
time.sleep(2)
res = {}
def _long(): res["r"] = js("/generate", {"text": P[6], "sampling_params": {"temperature": 0, "max_new_tokens": 3000, "ignore_eos": True}, "rid": "t56-abort-me"})
th = threading.Thread(target=_long); th.start(); time.sleep(3)
sa, da = req("/abort_request", {"rid": "t56-abort-me"})
th.join(120)
ra = res.get("r", (None, None))
fra = ra[1].get("meta_info", {}).get("finish_reason") if ra[0] == 200 and isinstance(ra[1], dict) else ra
chk("disconnect + /abort_request", not th.is_alive() and sa == 200, {"abort": sa, "finish": fra})
s, d = js("/generate", {"text": P[0], "sampling_params": G})
chk("after abort: generate ok", s == 200 and d.get("text"), d.get("text", d)[:60] if isinstance(d, dict) else d)
chk("after abort: same greedy", s == 200 and d["text"] == solo[0], d.get("text", d)[:60] if isinstance(d, dict) else d, True)
# 8. concurrency: 96 mixed requests at once (short + 2-8k-token prompts, greedy)
rnd = random.Random(0)
filler = "The grass is green. The sky is blue. The sun is bright. Here we go again. "
jobs = []
for i in range(96):
    if i % 8 == 7:
        jobs.append({"text": filler * rnd.randint(150, 600) + f"\nQ{i}: say the number {i}.\nA:",
                     "sampling_params": {"temperature": 0, "max_new_tokens": rnd.randint(16, 64)}})
    else:
        jobs.append({"text": P[i % 8] + f" (#{i})", "sampling_params": {"temperature": 0, "max_new_tokens": rnd.randint(16, 400)}})
res = [None] * len(jobs)
lat = [None] * len(jobs)
def _one(i):
    t = time.time(); res[i] = js("/generate", jobs[i]); lat[i] = time.time() - t
t0 = time.time()
ths = [threading.Thread(target=_one, args=(i,)) for i in range(len(jobs))]
[t.start() for t in ths]; [t.join(900) for t in ths]
good = [r for r in res if r and r[0] == 200 and r[1].get("text") is not None]
ntok = sum(r[1]["meta_info"]["completion_tokens"] for r in good)
sl = sorted(x for x in lat if x is not None)
chk("concurrency 96 mixed", len(good) == 96, {"ok": len(good), "tok": ntok, "s": round(time.time() - t0, 2),
    "lat_p50": round(sl[len(sl) // 2], 2), "lat_max": round(sl[-1], 2),
    "long_lat": [round(lat[i], 2) for i in range(7, 96, 8)]})
R["conc_texts"] = [r[1]["text"] if r and r[0] == 200 else None for r in res]
# 9. long chunked prompt (~24k tokens > default chunk 8192): passkey retrieval
key = "71432"
half = filler * 640
lp = half + f"The pass key is {key}. Remember it. {key} is the pass key. " + half + "\nWhat is the pass key? The pass key is"
s, d = js("/generate", {"text": lp, "sampling_params": {"temperature": 0, "max_new_tokens": 12}})
pt = d["meta_info"]["prompt_tokens"] if s == 200 else None
chk("long ~24k chunked passkey", s == 200 and key in d["text"], {"prompt": pt, "text": d.get("text") if s == 200 else d})
s2, d2 = js("/generate", {"text": lp, "sampling_params": {"temperature": 0, "max_new_tokens": 12}})
ct = d2["meta_info"].get("cached_tokens") if s2 == 200 else None
chk("long repeat " + ("radix hit" if radix else "same answer"), s2 == 200 and d2["text"] == d["text"] and
    (not radix or (ct or 0) > 0.5 * (pt or 1)), {"cached_tokens": ct, "text": d2.get("text") if s2 == 200 else d2})
# 10. flush_cache, then the same answers
s, d = req("/flush_cache")
chk("flush_cache", s == 200, (s, d[:80]))
s, d = js("/generate", {"text": P[0], "sampling_params": G})
chk("after flush: generate ok", s == 200 and d.get("text"), d.get("text", d)[:60] if isinstance(d, dict) else d)
chk("after flush: same greedy", s == 200 and d["text"] == solo[0], d.get("text", d)[:60] if isinstance(d, dict) else d, True)
s, d = js("/generate", {"text": lp, "sampling_params": {"temperature": 0, "max_new_tokens": 12}})
chk("after flush: long passkey", s == 200 and key in d["text"], d.get("text") if s == 200 else d)
# T56_REP=N: N more full-hit repeats of the long prompt (greedy spread, info only)
nrep = int(os.environ.get("T56_REP", "0"))
if nrep:
    reps = [js("/generate", {"text": lp, "sampling_params": {"temperature": 0, "max_new_tokens": 12}})[1]
            for _ in range(nrep)]
    reps = [x.get("text") if isinstance(x, dict) else str(x) for x in reps]
    R["long_reps"] = reps
    print("  long repeats:", {t: reps.count(t) for t in set(reps)}, flush=True)
# 11. metrics
s, d = req("/metrics")
chk("metrics", s == 200 and "sglang:" in d, (s, len(d)))
# 12. idle: the scheduler's idle memory self-check runs; server still healthy
time.sleep(15)
s, d = req("/health")
chk("health after idle", s == 200, s)
R["solo"] = solo
json.dump(R, open(out, "w"), indent=1, default=str)
print(f"{sum(ok)}/{len(ok)} passed", flush=True)
