#!/usr/bin/env python3
"""Build the PiCoKV-paper NIAH grid as engine records {id, L, D, input_ids, max_gen}.

Prompt construction is the repo's own: LLMNeedleHaystackTester (PiCoKV/benchmarks/
Needle/needle.py) + the chat messages of run_bin.Transformer.generate_prompt +
apply_chat_template(add_generation_prompt=True); greedy 50 new tokens, like run_bin.py.
usage: build_niah.py <model_dir> <out.jsonl> [lengths] [depths]
"""
import functools, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarks" / "Needle"))
from needle import LLMNeedleHaystackTester, ModelProvider
from transformers import AutoTokenizer

LENGTHS = "8192,16384,24576,32768,40960,49152,57344,65536,73728,81920,90112,98304,106496,114688,122880,131072"
DEPTHS = "0,11,22,33,44,56,67,78,89,100"


class Tok(ModelProvider):
    def __init__(self, path):
        self.tok = AutoTokenizer.from_pretrained(path)

    def evaluate_model(self, prompt, needle=None):
        raise NotImplementedError

    # Same messages as run_bin.Transformer.generate_prompt.
    def generate_prompt(self, context, retrieval_question):
        return [
            {"role": "system", "content": "You are a helpful AI bot that answers questions for a user. "
                                          "Keep your response short and direct"},
            {"role": "user", "content": context},
            {"role": "user", "content": f"{retrieval_question} Don't give information outside the document or repeat your findings"},
        ]

    def encode_text_to_tokens(self, text):
        return self.tok.encode(text, add_special_tokens=False)

    def decode_tokens(self, tokens, context_length=None):
        if context_length is not None:
            tokens = tokens[:context_length]
        return self.tok.decode(tokens, skip_special_tokens=True)


class Tester(LLMNeedleHaystackTester):
    # read+trim depends only on the length; cache it so each length is tokenized once.
    @functools.lru_cache(maxsize=None)
    def trimmed(self, context_length):
        return self.encode_and_trim(self.read_context_files(context_length), context_length)

    def generate_context(self, context_length, depth_percent):
        return self.insert_needle(self.trimmed(context_length), depth_percent, context_length)


def main():
    model, out = sys.argv[1], sys.argv[2]
    lengths = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else LENGTHS).split(",")]
    depths = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else DEPTHS).split(",")]
    mp = Tok(model)
    t = Tester(mp)
    with open(out, "w") as f:
        for L in lengths:
            for D in depths:
                msgs = mp.generate_prompt(t.generate_context(L, D), t.retrieval_question)
                ids = mp.tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
                if isinstance(ids, dict) or hasattr(ids, "input_ids"):
                    ids = ids["input_ids"]
                f.write(json.dumps({"id": f"{L}-{D}", "L": L, "D": D, "input_ids": list(ids), "max_gen": 50}) + "\n")
            print(f"L={L} prompt_tokens={len(ids)}", flush=True)


if __name__ == "__main__":
    main()
