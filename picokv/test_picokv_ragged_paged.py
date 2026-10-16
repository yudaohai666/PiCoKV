"""Correctness test: ragged (unequal dense windows) batched decode paths.

Three passes over IDENTICAL pregenerated inputs (same prompts, same stagger,
same decode tokens):

  A. batched forward_decode with the flashinfer paged dense path (new fast path)
  B. batched forward_decode with the paged path force-disabled (masked-matmul
     fallback — the pre-existing ragged path, now over the token-major pool)
  C. per-request reference decode (PicoKVRequestCache.decode, the verified
     single-request kernel, update_cache=True)

A-vs-B validates the two dense implementations through the same tail kernel;
A-vs-C / B-vs-C validate the token-major pool seeding, in-place writes, paged
tables, and permuted views end-to-end against the reference semantics.

Also checks: the paged path actually engaged (not silently disabled), dlens_py
python bookkeeping stays in sync with the GPU dlen tensor, and an EQUAL-length
batch (SDPA over the permuted view) still matches the reference.

Run under .venv_sglang:

  PYTHONPATH=$PWD PICOKV_PREBUILT_DIR=$PWD/build/picokv_extensions \
  ../.venv_sglang/bin/python picokv/test_picokv_ragged_paged.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

BACKEND = (
    Path(__file__).resolve().parents[2]
    / "sglang/python/sglang/srt/layers/attention/picokv_backend.py"
)
QH, KVH, D = 32, 8, 128
LAYER = SimpleNamespace(layer_id=1)
B = 8
LENGTHS = [400, 800, 1200, 1600, 2400, 3200, 4800, 6400]
STAGGERS = [0, 16, 32, 48, 64, 80, 96, 112]
STEPS = 48


def load_backend():
    spec = importlib.util.spec_from_file_location("picokv_backend", BACKEND)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Cfg:
    num_attention_heads = QH
    head_dim = D

    def get_num_kv_heads(self, tp):
        return KVH


class _Runner:
    device = "cuda"
    model_config = _Cfg()
    tp_size = 1


def _fb(rids, eseq=None):
    fb = SimpleNamespace(req_pool_indices=list(rids))
    if eseq is not None:
        fb.extend_seq_lens = list(eseq)
    return fb


def _fresh(n, gen):
    q = torch.randn(n, QH * D, device="cuda", dtype=torch.bfloat16, generator=gen)
    k = torch.randn(n, KVH * D, device="cuda", dtype=torch.bfloat16, generator=gen)
    v = torch.randn(n, KVH * D, device="cuda", dtype=torch.bfloat16, generator=gen)
    return q, k, v


def main():
    assert torch.cuda.is_available(), "CUDA required"
    mod = load_backend()
    gen = torch.Generator(device="cuda").manual_seed(1234)

    # Pregenerate every input once so all passes see identical tokens.
    prefill_inputs = [_fresh(L, gen) for L in LENGTHS]
    stagger_inputs = [[_fresh(1, gen) for _ in range(s)] for s in STAGGERS]
    step_inputs = [_fresh(B, gen) for _ in range(STEPS)]
    rids = list(range(B))

    def make():
        return mod.PicoKVAttnBackend(_Runner())

    def prefill(be):
        for rid, (q, k, v) in enumerate(prefill_inputs):
            be.forward_extend(q, k, v, LAYER, _fb([rid], [LENGTHS[rid]]))

    def stagger(be):
        for rid, toks in enumerate(stagger_inputs):
            req = be._reqs[rid]
            for q, k, v in toks:
                req.decode(
                    LAYER.layer_id,
                    q.view(1, QH, 1, D), k.view(1, KVH, 1, D), v.view(1, KVH, 1, D),
                )

    # --- pass A: batched decode, paged flashinfer path enabled -----------------
    be = make()
    prefill(be)
    stagger(be)
    # Dense-window length after prefill depends on each prompt's group
    # alignment; read what _fixed_layer will seed from (request caches do NOT
    # grow during batched decode: update_cache=False).
    dlens0 = [int(be._reqs[rid].layers[LAYER.layer_id].dense_key.shape[-2])
              for rid in rids]
    assert len(set(dlens0)) > 1, "expected unequal dense windows for this batch"
    outsA = [
        be.forward_decode(q, k, v, LAYER, _fb(rids)).reshape(B, QH, 1, D)
        for q, k, v in step_inputs
    ]
    assert isinstance(be._fi, dict), "paged flashinfer path did not initialize"
    assert be._fi["wrapper"] is not None, "paged wrapper missing"
    sh = be._batch["shared"]
    assert sh is not None, "shared step state missing"
    expect_dlens = [d + STEPS for d in dlens0]
    assert sh["dlens_py"] == expect_dlens, (sh["dlens_py"], expect_dlens)
    assert sh["lpl"].tolist() == expect_dlens, "GPU lpl out of sync with dlens_py"

    # --- pass B: batched decode, paged path force-disabled ---------------------
    beB = make()
    beB._fi = False
    prefill(beB)
    stagger(beB)
    outsB = [
        beB.forward_decode(q, k, v, LAYER, _fb(rids)).reshape(B, QH, 1, D)
        for q, k, v in step_inputs
    ]
    assert beB._fi is False

    # --- pass C: per-request reference (verified single-request path) ----------
    beC = make()
    prefill(beC)
    stagger(beC)
    outsC = []
    for q, k, v in step_inputs:
        per = []
        for i, rid in enumerate(rids):
            req = beC._reqs[rid]
            per.append(req.decode(
                LAYER.layer_id,
                q[i].view(1, QH, 1, D), k[i].view(1, KVH, 1, D), v[i].view(1, KVH, 1, D),
            ))
        outsC.append(torch.cat(per, 0))

    def compare(name, X, Y):
        xa = torch.stack(X).float()  # [T,B,QH,1,D]
        ya = torch.stack(Y).float()
        assert torch.isfinite(xa).all() and torch.isfinite(ya).all(), f"{name}: non-finite"
        cos = torch.nn.functional.cosine_similarity(
            xa.flatten(), ya.flatten(), dim=0)
        maxabs = (xa - ya).abs().max()
        per_req = torch.nn.functional.cosine_similarity(
            xa.permute(1, 0, 2, 3, 4).flatten(1), ya.permute(1, 0, 2, 3, 4).flatten(1), dim=1)
        print(f"[{name}] cos={cos:.6f}  max|diff|={maxabs:.5f}  "
              f"per-req cos min={per_req.min():.6f}")
        return cos, maxabs

    cosAB, _ = compare("A-paged   vs B-fallback", outsA, outsB)
    cosAC, _ = compare("A-paged   vs C-reference", outsA, outsC)
    cosBC, _ = compare("B-fallback vs C-reference", outsB, outsC)
    assert cosAB >= 0.9995, f"A vs B cos {cosAB:.6f} below tolerance"
    assert cosAC >= 0.999, f"A vs C cos {cosAC:.6f} below tolerance"
    assert cosBC >= 0.999, f"B vs C cos {cosBC:.6f} below tolerance"

    # --- equal-length batch: SDPA over the permuted token-major view -----------
    eq_inputs = [_fresh(L, gen) for L in [800] * 4]
    eq_steps = [_fresh(4, gen) for _ in range(8)]
    beD = make()
    for rid, (q, k, v) in enumerate(eq_inputs):
        beD.forward_extend(q, k, v, LAYER, _fb([rid], [800]))
    outsD = [beD.forward_decode(q, k, v, LAYER, _fb(range(4))).reshape(4, QH, 1, D)
             for q, k, v in eq_steps]
    beE = make()
    for rid, (q, k, v) in enumerate(eq_inputs):
        beE.forward_extend(q, k, v, LAYER, _fb([rid], [800]))
    outsE = []
    for q, k, v in eq_steps:
        per = [beE._reqs[i].decode(
            LAYER.layer_id,
            q[i].view(1, QH, 1, D), k[i].view(1, KVH, 1, D), v[i].view(1, KVH, 1, D))
            for i in range(4)]
        outsE.append(torch.cat(per, 0))
    cosDE, _ = compare("D-equal   vs E-reference", outsD, outsE)
    assert cosDE >= 0.9999, f"equal-path cos {cosDE:.6f} below tolerance"

    print("PASS: ragged paged / fallback / reference agree; equal path unchanged")


if __name__ == "__main__":
    main()
