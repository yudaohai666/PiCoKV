"""Window-attention kernel correctness + speed test (lever 2).

Validates picokv_window_attn_fn (the custom dense-window decode kernel) against:
  1. a float64 torch reference (out + ln-LSE),
  2. the flashinfer stock paged-decode call it replaces (the backend's exact
     wrapper: BatchDecodeWithPagedKVCacheWrapper(ws, "NHD"), (k, v) pair),
and benches both at the engine-realistic shapes (cap 832, window rows
321..384 growing, ragged mixed).

Also smoke-tests the CUDA-graph replay claim: lpl is read on device at run
time, so one captured launch stays correct as the window grows.

Run under .venv_sglang from 0919_picokv/:
  PICOKV_PREBUILT_DIR=$PWD/PiCoKV/build/picokv_extensions \
  CUDA_VISIBLE_DEVICES=0 .venv_sglang/bin/python -u PiCoKV/picokv/bench_window_attn.py
"""
from __future__ import annotations

import os
import torch

QH, KVH, D = 32, 8, 128
GROUP = QH // KVH
SCALE = 1.0 / (D ** 0.5)
BW = 3005.0  # GB/s measured read bandwidth on this machine

CASES = [
    # (name, B, cap, lpl list)
    ("b1-lpl1",        1, 832, [1]),
    ("b1-win321",      1, 832, [321]),
    ("b1-win384",      1, 832, [384]),
    ("b8-win321",      8, 832, [321] * 8),
    ("b8-ragged",      8, 832, [321, 330, 345, 360, 370, 378, 383, 384]),
    ("b16-win384",    16, 832, [384] * 16),
    ("b16-ragged",    16, 832, [321, 325, 332, 340, 348, 355, 361, 368,
                               372, 376, 379, 381, 383, 384, 384, 384]),
    ("b1-bigcap3584",  1, 3584, [384]),  # >48KB smem: opt-in path
]


def torch_ref(kbuf, vbuf, lpl, q):
    """float64 reference: out [B,QH,D], lse [B,QH] (ln space)."""
    B, cap = kbuf.shape[0], kbuf.shape[1]
    qf = q.double()                                    # [B, QH, D]
    kf = kbuf.double()                                 # [B, cap, KVH, D]
    vf = vbuf.double()
    outs, lses = [], []
    for b in range(B):
        L = int(lpl[b])
        k = kf[b, :L]                                  # [L, KVH, D]
        v = vf[b, :L]                                  # [L, KVH, D]
        # q head h attends kv head h//4 (block-major GQA)
        qg = qf[b].view(KVH, GROUP, D)                 # [KVH, GROUP, D]
        kg = k.permute(1, 0, 2)                        # [KVH, L, D]
        s = torch.einsum("kgd,kld->kgl", qg, kg) * SCALE
        p = torch.softmax(s, dim=-1)                   # [KVH, GROUP, L]
        vg = v.permute(1, 0, 2)                        # [KVH, L, D]
        o = torch.einsum("kgl,kld->kgd", p, vg)        # [KVH, GROUP, D]
        lse = torch.logsumexp(s, dim=-1)                # ln sum exp
        outs.append(o.reshape(QH, D))
        lses.append(lse.reshape(QH))
    return torch.stack(outs), torch.stack(lses)


def main():
    assert torch.cuda.is_available()
    torch.manual_seed(0)
    from picokv.picokv_cuda_extension.picokv_cuda import load_picokv_extension
    ext = load_picokv_extension()
    win = ext.picokv_window_attn_fn
    print(f"[ext] loaded; has picokv_window_attn_fn: {callable(win)}")

    import flashinfer
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    fi = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD")

    all_ok = True
    for name, B, cap, lpls in CASES:
        kbuf = torch.randn(B, cap, KVH, D, device="cuda", dtype=torch.bfloat16)
        vbuf = torch.randn(B, cap, KVH, D, device="cuda", dtype=torch.bfloat16)
        q = torch.randn(B, QH, D, device="cuda", dtype=torch.bfloat16)
        lpl = torch.tensor(lpls, dtype=torch.int32, device="cuda")
        out = torch.empty(B, QH, D, device="cuda", dtype=torch.bfloat16)
        lse = torch.empty(B, QH, device="cuda", dtype=torch.float32)

        win(kbuf, vbuf, lpl, q, out, lse, SCALE)
        torch.cuda.synchronize()

        # --- float64 reference check
        o_ref, lse_ref = torch_ref(kbuf, vbuf, lpl, q)
        o_diff = (out.double() - o_ref).abs().max().item()
        rel_o = o_diff / o_ref.abs().max().item()
        lse_diff = (lse - lse_ref.cuda()).abs().max().item()

        # --- flashinfer contract check (same paged tables the backend builds)
        kt = kbuf.view(B * cap, KVH, D)
        vt = vbuf.view(B * cap, KVH, D)
        indptr = torch.zeros(B + 1, dtype=torch.int32, device="cuda")
        for i, L in enumerate(lpls):
            indptr[i + 1] = indptr[i] + L
        indices = torch.empty(sum(lpls), dtype=torch.int32, device="cuda")
        pos = 0
        for i, L in enumerate(lpls):
            indices[pos:pos + L] = torch.arange(
                i * cap, i * cap + L, dtype=torch.int32, device="cuda")
            pos += L
        ones = torch.ones(B, dtype=torch.int32, device="cuda")
        fi.plan(indptr, indices, ones, QH, KVH, D, 1,
                q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
        fi_out, fi_lse2 = fi.run(q, (kt, vt), out=None, lse=None, return_lse=True)
        fi_diff = (out.float() - fi_out.float()).abs().max().item()
        fi_lse_diff = (lse - fi_lse2 * 0.6931471805559453).abs().max().item()

        ok = rel_o < 5e-3 and lse_diff < 2e-3 and fi_diff < 2e-2 and fi_lse_diff < 2e-2
        all_ok &= ok
        print(f"[{name}] B={B} cap={cap} lpl[{min(lpls)}..{max(lpls)}] "
              f"{'OK ' if ok else 'FAIL'} "
              f"out-vs-ref rel {rel_o:.2e} (abs {o_diff:.2e}) lse-vs-ref {lse_diff:.2e} | "
              f"out-vs-fi abs {fi_diff:.2e} lse-vs-fi {fi_lse_diff:.2e}")

        # --- speed (only the engine-realistic dense cases)
        if name in ("b1-win321", "b8-win321", "b16-win384", "b16-ragged"):
            def t_win():
                win(kbuf, vbuf, lpl, q, out, lse, SCALE)
            def t_fi():
                fi.run(q, (kt, vt), out=None, lse=None, return_lse=True)
            for tag, fn in (("win", t_win), ("fi ", t_fi)):
                for _ in range(5):
                    fn()
                torch.cuda.synchronize()
                ts = []
                for _ in range(5):
                    s = torch.cuda.Event(enable_timing=True)
                    e = torch.cuda.Event(enable_timing=True)
                    s.record()
                    for _ in range(20):
                        fn()
                    e.record()
                    torch.cuda.synchronize()
                    ts.append(s.elapsed_time(e) / 20 * 1e3)
                t = min(ts)
                rows = sum(lpls)
                floor = rows * KVH * D * 2 * 2 / BW / 1e3
                print(f"    {tag}: {t:7.1f} us  (window bytes floor {floor:5.1f} us, "
                      f"{t / floor:5.2f}x)")
        del kbuf, vbuf
        torch.cuda.empty_cache()

    # --- CUDA-graph replay safety: capture once, grow lpl, replay
    print("[graph-replay] capture at lpl=321, replay at lpl=322..325")
    B, cap = 1, 832
    kbuf = torch.randn(B, cap, KVH, D, device="cuda", dtype=torch.bfloat16)
    vbuf = torch.randn(B, cap, KVH, D, device="cuda", dtype=torch.bfloat16)
    q = torch.randn(B, QH, D, device="cuda", dtype=torch.bfloat16)
    lpl = torch.tensor([321], dtype=torch.int32, device="cuda")
    out = torch.empty(B, QH, D, device="cuda", dtype=torch.bfloat16)
    lse = torch.empty(B, QH, device="cuda", dtype=torch.float32)
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3):
            win(kbuf, vbuf, lpl, q, out, lse, SCALE)
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        win(kbuf, vbuf, lpl, q, out, lse, SCALE)
    ok = True
    for L in (321, 322, 323, 324, 325):
        lpl.fill_(L)
        g.replay()
        torch.cuda.synchronize()
        o_ref, lse_ref = torch_ref(kbuf, vbuf, lpl, q)
        rel_o = (out.double() - o_ref).abs().max().item() / o_ref.abs().max().item()
        lse_d = (lse - lse_ref.cuda()).abs().max().item()
        this_ok = rel_o < 5e-3 and lse_d < 2e-3
        ok &= this_ok
        print(f"    lpl={L}: {'OK ' if this_ok else 'FAIL'} rel {rel_o:.2e} lse {lse_d:.2e}")
    all_ok &= ok

    print(f"ALL {'PASS' if all_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
