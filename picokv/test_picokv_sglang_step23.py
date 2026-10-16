"""Step 2 (compressed KV pool: memory + latency) and Step 3 (ragged
continuous-batch decode: correctness + throughput) verification for the
PiCoKV SGLang backend, driven through the backend's own data structures
(``PicoKVRequestCache`` / ``PicoKVBatchDecoder``) loaded from the sglang tree.

Component-level: the full sglang server cannot run in this env (sglang main
pins transformers 5.x, which conflicts with PiCoKV's transformers 4.53, plus
compiled sgl-kernel/flashinfer built against torch 2.13 vs our 2.12). These
tests exercise the exact PiCoKV-specific structures the backend uses.
"""

from __future__ import annotations

import argparse
import importlib.util
import statistics
import time
from pathlib import Path

import torch

BACKEND = (
    Path(__file__).resolve().parents[2] / "sglang"  # sibling sglang checkout
    / "python/sglang/srt/layers/attention/picokv_backend.py"
)
QH, KVH, D = 32, 8, 128
SCALE = 1.0 / (D ** 0.5)
DEV, DT = "cuda", torch.bfloat16


def load_backend():
    spec = importlib.util.spec_from_file_location("picokv_backend", BACKEND)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cos(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-9))


def dense_ms(k_ctx, v_ctx, steps, warmup):
    per = []
    for i in range(warmup + steps):
        q = torch.randn(1, QH, 1, D, device=DEV, dtype=DT)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        torch.nn.functional.scaled_dot_product_attention(q, k_ctx, v_ctx, enable_gqa=True, scale=SCALE)
        torch.cuda.synchronize()
        if i >= warmup:
            per.append((time.perf_counter() - t0) * 1000)
    return statistics.median(per)


def step2_memory_latency(mod, lengths, steps=30, warmup=8):
    from picokv.picokv_standalone import StandaloneConfig
    print("\n===== STEP 2: compressed KV pool — memory + decode latency (1 layer, batch=1) =====")
    print(f"{'L':>8} | {'dense KV MB':>11} {'picokv MB':>10} {'mem save':>8} | "
          f"{'dense ms/tok':>12} {'picokv ms/tok':>13} {'speedup':>8}")
    print("-" * 92)
    for L in lengths:
        req = mod.PicoKVRequestCache(StandaloneConfig(), None)
        kp = torch.randn(1, KVH, L, D, device=DEV, dtype=DT)
        vp = torch.randn_like(kp)
        req.prepare(0, kp, vp)
        pico_mb = req.cache_bytes() / 1024 ** 2
        dense_mb = (kp.numel() + vp.numel()) * kp.element_size() / 1024 ** 2

        # picokv decode latency (steady state)
        per = []
        for i in range(warmup + steps):
            q = torch.randn(1, QH, 1, D, device=DEV, dtype=DT)
            k = torch.randn(1, KVH, 1, D, device=DEV, dtype=DT)
            v = torch.randn_like(k)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            req.decode(0, q, k, v)
            torch.cuda.synchronize()
            if i >= warmup:
                per.append((time.perf_counter() - t0) * 1000)
        p_ms = statistics.median(per)
        d_ms = dense_ms(kp, vp, steps, warmup)
        print(f"{L:>8} | {dense_mb:>11.1f} {pico_mb:>10.1f} {dense_mb/pico_mb:>7.2f}x | "
              f"{d_ms:>12.4f} {p_ms:>13.4f} {d_ms/p_ms:>7.2f}x")
        del req, kp, vp
        torch.cuda.empty_cache()


def step3_ragged(mod, lengths, steps=20, warmup=5):
    from picokv.picokv_standalone import StandaloneConfig
    print("\n===== STEP 3: ragged continuous-batch decode — correctness + throughput =====")
    print(f"requests at INDEPENDENT context lengths: {lengths}")
    dec = mod.PicoKVBatchDecoder(QH, KVH, D, SCALE, StandaloneConfig(), None)
    ctx = {}  # req_id -> growing (k,v) ground-truth context
    for rid, L in enumerate(lengths):
        dec.add_request(rid)
        kp = torch.randn(1, KVH, L, D, device=DEV, dtype=DT)
        vp = torch.randn_like(kp)
        dec.prepare(rid, 0, kp, vp)
        ctx[rid] = [kp, vp]
    req_ids = list(range(len(lengths)))

    # correctness: one ragged decode step, compare each request vs its own dense GT
    q = torch.randn(len(req_ids), QH, 1, D, device=DEV, dtype=DT)
    k = torch.randn(len(req_ids), KVH, 1, D, device=DEV, dtype=DT)
    v = torch.randn_like(k)
    out = dec.decode_batch(0, req_ids, q, k, v)  # [num_reqs, QH, 1, D]
    assert out.shape == (len(req_ids), QH, 1, D), out.shape
    coss = []
    for i, rid in enumerate(req_ids):
        ctx[rid][0] = torch.cat([ctx[rid][0], k[i:i+1]], dim=2)
        ctx[rid][1] = torch.cat([ctx[rid][1], v[i:i+1]], dim=2)
        gt = torch.nn.functional.scaled_dot_product_attention(
            q[i:i+1], ctx[rid][0], ctx[rid][1], enable_gqa=True, scale=SCALE)
        coss.append(cos(out[i:i+1], gt))
    seqs = [dec.reqs[r].seq_lengths[0] for r in req_ids]
    print(f"[correctness] per-request cos vs own dense GT: "
          f"{', '.join(f'{c:.3f}' for c in coss)}")
    print(f"[bookkeeping] per-request seq_len advanced independently: {seqs} "
          f"(expected {[L+1 for L in lengths]})")
    print(f"[memory] total compressed = {dec.total_bytes()/1024**2:.1f} MB vs dense "
          f"{sum(2*KVH*(L+1)*D*2 for L in lengths)/1024**2:.1f} MB")

    # throughput: batched decode steps/s
    per = []
    for i in range(warmup + steps):
        q = torch.randn(len(req_ids), QH, 1, D, device=DEV, dtype=DT)
        k = torch.randn(len(req_ids), KVH, 1, D, device=DEV, dtype=DT)
        v = torch.randn_like(k)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        dec.decode_batch(0, req_ids, q, k, v)
        torch.cuda.synchronize()
        if i >= warmup:
            per.append((time.perf_counter() - t0) * 1000)
    ms = statistics.median(per)
    print(f"[throughput] ragged batch of {len(req_ids)} reqs: {ms:.3f} ms/step "
          f"-> {len(req_ids)*1000/ms:.1f} tok/s (per-request kernel launches; "
          f"fused ragged kernel is the remaining optimization)")
    ok = all(c > 0.5 for c in coss) and seqs == [L + 1 for L in lengths]
    print("\nRESULT:", "PASS" if ok else "FAIL",
          "- ragged batch decodes each request against its own compressed cache; "
          "seq lengths tracked independently.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step2-lengths", default="4096,16384,65536")
    ap.add_argument("--step3-lengths", default="4096,16384,65536")
    args = ap.parse_args()
    assert torch.cuda.is_available()
    mod = load_backend()
    print(f"[harness] loaded backend from sglang tree; sglang glue available={mod._SGLANG_AVAILABLE}")
    step2_memory_latency(mod, [int(x) for x in args.step2_lengths.split(",") if x])
    step3_ragged(mod, [int(x) for x in args.step3_lengths.split(",") if x])


if __name__ == "__main__":
    main()
