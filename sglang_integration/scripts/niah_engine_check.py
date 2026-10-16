"""Engine-level needle-in-haystack check for the PICOKV_WINDOW_KERNEL path.

Real weights (llama31_8b_instruct), 8k-token haystack, two needles at 25%/75%
depth, greedy decode. Run with and without PICOKV_WINDOW_KERNEL=1 and compare
the retrieved numbers + full generated text: the window kernel is exact
softmax (bf16-level diffs vs flashinfer), so both paths must retrieve both
needles and ideally produce identical text.
"""
import os, sys
import torch
from transformers import AutoTokenizer
import sglang as sgl

MODEL = os.environ.get("NIAH_MODEL", "llama31_8b_instruct")  # real weights
TARGET_LEN = 8000

FILLER = [
    "The lighthouse keeper counted seventeen ships on the horizon that morning.",
    "Alpine meadows bloom for only six weeks each year before the frost returns.",
    "A well-organized library catalog saves researchers countless hours.",
    "The city council debated the new bridge proposal for three straight hours.",
    "Marbles found in ancient Rome were often made from recycled glass.",
    "Every autumn the festival draws thousands of visitors to the harbor.",
    "The botanist catalogued forty-two new fern species in a single season.",
    "Train schedules changed twice that year, confusing daily commuters.",
    "Sailors relied on the north star long before compasses became common.",
    "The bakery on Fifth Street opens at four in the morning, every day.",
    "Copper wiring replaced iron in most buildings by the early twentieth century.",
    "She painted the fence a deep shade of green over three weekends.",
    "The museum's newest wing houses fossils from a distant river valley.",
    "Farmers rotated crops to keep the soil rich across generations.",
    "The orchestra rehearsed the difficult passage one final time.",
    "Letters took two weeks to cross the country in those days.",
    "A quiet café at the corner became the neighborhood's unofficial office.",
    "The old mill still turns, though it grinds nothing but flour for tourists.",
    "Cartographers once drew sea monsters where their maps ran out of knowledge.",
    "The twins argued about whose turn it was to water the garden.",
]
NEEDLES = [
    ("Cleopatra", 595951),
    ("Confucius", 838637),
]


def build_prompt(tok, needles=NEEDLES, filler_start=0):
    needle_txt = {
        name: f"One of the special magic numbers for {name} is: {num}. "
              f"Remember it, I will quiz you about the number afterwards."
        for name, num in needles
    }
    # Build a body that contains the filler, with needles at ~25%/75% depth.
    body_parts, n_tok = [], 0
    filler_i = filler_start
    needle_at = {int(TARGET_LEN * f): name for f, name in zip((0.25, 0.75), [NEEDLES[0][0], NEEDLES[1][0]])}
    # Rough plan: pre-count with the tokenizer as we append; insert needles when
    # the running token count crosses their target depth.
    inserted = set()
    sysmsg = (
        "You are a helpful AI bot that answers questions for a user. "
        "Keep your response short and direct"
    )
    ctx_head = ("Some special magic numbers are hidden within the following text. "
                "Make sure to memorize it. I will quiz you about the numbers afterwards.")
    question = ("What are the special magic numbers for Cleopatra and Confucius "
                "mentioned in the provided text?")
    q_tail = " Don't give information outside the document or repeat your findings."

    text_parts = []
    text_parts.append(ctx_head)
    # iterate filler until we hit target length
    cur = " ".join([sysmsg, ctx_head])
    n_tok = len(tok.encode(cur, add_special_tokens=False))
    target_body = TARGET_LEN - n_tok - 64  # leave room for question
    body_tok = 0
    idx = 0
    while body_tok < target_body:
        for d, name in needle_at.items():
            if name not in inserted and body_tok >= d * target_body // TARGET_LEN and body_tok > 0:
                pass  # handled below via fraction of body
        frac = body_tok / max(target_body, 1)
        for depth_frac, name in zip((0.25, 0.75), [NEEDLES[0][0], NEEDLES[1][0]]):
            if name not in inserted and frac >= depth_frac:
                t = needle_txt[name]
                text_parts.append(t)
                body_tok += len(tok.encode(t, add_special_tokens=False))
                inserted.add(name)
        s = f"{idx}. " + FILLER[filler_i % len(FILLER)]
        filler_i += 1
        idx += 1
        text_parts.append(s)
        body_tok += len(tok.encode(s, add_special_tokens=False))
    context = "\n".join(text_parts)
    messages = [
        {"role": "system", "content": sysmsg},
        {"role": "user", "content": context},
        {"role": "user", "content": question + q_tail},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    n = len(tok.encode(prompt, add_special_tokens=False))
    return prompt, n


def main():
    import json
    win = os.environ.get("PICOKV_WINDOW_KERNEL", "auto")  # backend default: auto
    graphs_on = os.environ.get("BENCH_GRAPHS", "0") == "1"
    backend = os.environ.get("NIAH_BACKEND", "picokv")
    nb = int(os.environ.get("NIAH_BATCH", "1"))
    tok = AutoTokenizer.from_pretrained(MODEL)
    # Request 0 is the original prompt; requests 1.. get their own needle
    # numbers and a rotated filler (slightly different lengths -> ragged batch).
    reqs = []
    for i in range(nb):
        nd = [(NEEDLES[0][0], NEEDLES[0][1] + 1117 * i), (NEEDLES[1][0], NEEDLES[1][1] - 2903 * i)]
        prompt, n = build_prompt(tok, nd, filler_start=i)
        reqs.append((prompt, n, nd))
    gkw = {}
    if graphs_on and os.environ.get("BENCH_DECODE_GRAPH"):
        gkw["cuda_graph_backend_decode"] = os.environ["BENCH_DECODE_GRAPH"]
    print(f"[niah-engine] prompt tokens: {[r[1] for r in reqs]}, backend={backend}, "
          f"PICOKV_WINDOW_KERNEL={win}, BENCH_GRAPHS={'1' if graphs_on else '0'} "
          f"decode_graph={gkw.get('cuda_graph_backend_decode', 'default')}", flush=True)
    # BENCH_GRAPHS=1 mirrors bench_sglang_batch.py: decode graphs on,
    # prefill graphs off, memf 0.70.
    eng = sgl.Engine(model_path=MODEL, attention_backend=backend, dtype="bfloat16",
                     tp_size=1, mem_fraction_static=0.70 if graphs_on else 0.80,
                     disable_cuda_graph=not graphs_on,
                     disable_prefill_cuda_graph=graphs_on,
                     disable_radix_cache=True, chunked_prefill_size=1 << 20,
                     max_running_requests=max(4, nb), context_length=16384,
                     trust_remote_code=True, log_level="error", **gkw)
    sp = {"temperature": 0.0, "max_new_tokens": 64}
    # NIAH_LOGPROB=1: also dump the top-2 logprobs of every generated token
    # (to tell a near-tie greedy flip from a real divergence)
    lp = os.environ.get("NIAH_LOGPROB", "0") == "1"
    lkw = {"return_logprob": True, "top_logprobs_num": 2} if lp else {}
    if nb == 1:
        outs = eng.generate(prompt=reqs[0][0], sampling_params=sp, **lkw)
        if isinstance(outs, dict):
            outs = [outs]
    else:
        outs = eng.generate(prompt=[r[0] for r in reqs], sampling_params=[sp] * nb, **lkw)
    texts = [o["text"] if isinstance(o, dict) else o for o in outs]
    ok_all = True
    for i, (text, (_, _, nd)) in enumerate(zip(texts, reqs)):
        print("=== OUTPUT ===" if nb == 1 else f"=== OUTPUT {i} ===")
        print(text)
        hits = {name: str(num) in text for name, num in nd}
        ok = all(hits.values())
        ok_all &= ok
        print("=== CHECK ===")
        print(f"needles found: {'PASS' if ok else 'FAIL'} ("
              + ", ".join(f"{name} {num}: {hits[name]}" for name, num in nd) + ")")
    if os.environ.get("NIAH_OUT"):
        with open(os.environ["NIAH_OUT"], "w") as f:
            json.dump(texts, f, ensure_ascii=False, indent=1)
        if lp:
            tops = [[[(t[0], t[1]) for t in pos] for pos in o["meta_info"]["output_top_logprobs"]]
                    for o in outs]
            with open(os.environ["NIAH_OUT"] + ".top2", "w") as f:
                json.dump(tops, f)
    eng.shutdown()
    print(f"NIAH_ENGINE_DONE win={win} ok={ok_all}")


if __name__ == "__main__":
    main()
