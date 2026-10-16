"""Component-level split of the PiCoKV decode tail path (issue deep-dive).

For each batch shape it extracts the PRODUCTION 'fixed' batch tensors (built by
one real decode step) and times each sub-op with CUDA events:

  full tail host fn     = query-quant + memset + main scan + fused projection
  query quantize_pack   (its own kernel)
  memset                (glob_code+glob_den zero, same bytes as the host fn)
  main scan kernel      (direct call; equal/single shapes only -- the direct
                         wrapper has no varlen args)
  fused value projection (its own kernel)
  query projection bmm  (the 'proj' segment)

Then:
  * exp_bits sweep {1,2,4,8} on the main kernel -> value-popc/ballot share
  * L sweep (single request, 8k..200k) -> per-token slope vs fixed intercept
  * bandwidth floors (measured 3005 GB/s) for the key+value bytes actually read
  * bmma tensorcore prototype vs popc kernel on a GQA-8/KB4/Cv256 shape
    (production GQA-4/KB5/Cv320 is not supported by the bmma prototype)
  * existing pruned kernel (prune_mode=1) timing on the same tensors
  * kernel launch overhead microbench

Run under .venv_sglang from PiCoKV/:
  PYTHONPATH=$PWD PICOKV_PREBUILT_DIR=$PWD/build/picokv_extensions \
  ../.venv_sglang/bin/python picokv/bench_tail_split.py
"""
from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path
from types import SimpleNamespace

import torch

BACKEND = (
    Path(__file__).resolve().parents[2]
    / "sglang/python/sglang/srt/layers/attention/picokv_backend.py"
)
QH, KVH, D = 32, 8, 128
SCALE = 1.0 / (D ** 0.5)
BW = 3005.0  # GB/s measured read bandwidth on this machine
LN2 = 0.6931471805599453


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


def _fresh(B):
    return (torch.randn(B, QH * D, device="cuda", dtype=torch.bfloat16),
            torch.randn(B, KVH * D, device="cuda", dtype=torch.bfloat16),
            torch.randn(B, KVH * D, device="cuda", dtype=torch.bfloat16))


def bench(fn, iters=20, warmup=5, reps=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) / iters)
    return min(ts)


def get_ext():
    from picokv.picokv_cuda_extension.picokv_cuda import load_picokv_extension
    return load_picokv_extension()


def build_case(be, mod, lengths):
    """Prefill requests, run one decode step, return the production fixed dict."""
    LAYER = SimpleNamespace(layer_id=1)
    for rid, L in enumerate(lengths):
        q, k, v = _fresh(L)
        be.forward_extend(q, k, v, LAYER, _fb([rid], [L]))
    rids = list(range(len(lengths)))
    q, k, v = _fresh(len(rids))
    be.forward_decode(q, k, v, LAYER, _fb(rids))
    fixed = be._batch["fixed"][1]
    return fixed, LAYER


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-bmma", action="store_true")
    ap.add_argument("--skip-lswep", action="store_true")
    args = ap.parse_args()
    assert torch.cuda.is_available()
    torch.manual_seed(0)
    ext = get_ext()
    mod = load_backend()

    CONFIGS = [
        ("equal-b16@8k", [8000] * 16),
        ("single@120k", [120000]),
        ("ragged-b8-mixed", [4000, 8000, 12000, 16000, 24000, 32000, 48000, 64000]),
    ]

    print("=" * 100)
    print("PART 1: tail decomposition on production tensors (ms/call)")
    print("=" * 100)
    for name, lengths in CONFIGS:
        be = mod.PicoKVAttnBackend(_Runner())
        fixed, _ = build_case(be, mod, lengths)
        B = len(lengths)
        batched = fixed["batched"]  # persistent batched cache holds the projections
        Ck = int(batched.key_projection.shape[-2])
        Cv = int(batched.value_projection.shape[-2])
        KB = Ck // 64
        Lmax = int(batched.packed_key.shape[-1])
        Gmax = int(batched.packed_value.shape[2])
        EB = int(fixed["exp_bits"])
        gls = [int(x) for x in fixed["global_lens"].tolist()]
        ragged = len(set(gls)) > 1

        # Own workspaces (same shapes as _get_workspace in decode.py).
        proj_q = torch.randn(B, QH, 1, Ck, device="cuda", dtype=torch.bfloat16)
        pack_q = torch.zeros(B, QH, 1, KB, 8, device="cuda", dtype=torch.long)
        qscale = torch.ones(B, QH, 1, device="cuda", dtype=torch.float32)
        dlse = torch.zeros(B, QH, 1, device="cuda", dtype=torch.float32)
        dout = torch.zeros(B, QH, 1, D, device="cuda", dtype=torch.bfloat16)
        gcode = torch.zeros(B, QH, 1, Cv, device="cuda", dtype=torch.float32)
        gden = torch.zeros(B, QH, 1, device="cuda", dtype=torch.float32)
        out = torch.zeros(B, QH, 1, D, device="cuda", dtype=torch.bfloat16)
        pk, pv = batched.packed_key, batched.packed_value
        vproj = batched.value_projection
        kproj_t = batched.key_projection_t
        gl, sr, st = fixed["global_lens"], fixed["sched_req"], fixed["sched_tile"]
        empty = torch.empty(0, dtype=torch.int32, device="cuda")

        # Query projection bmm (the 'proj' segment).
        gq = torch.randn(B * KVH, 4, D, device="cuda", dtype=torch.bfloat16)
        proj_out = torch.empty(B * KVH, 4, Ck, device="cuda", dtype=torch.bfloat16)
        kproj_t3 = kproj_t.reshape(B * KVH, D, Ck)
        t_bmm = bench(lambda: torch.bmm(gq, kproj_t3, out=proj_out))

        t_quant = bench(lambda: ext.picokv_quantize_pack_projected_query_fn(
            proj_q, pack_q, qscale))
        t_memset = bench(lambda: (gcode.zero_(), gden.zero_()))
        t_full = bench(lambda: ext.picokv_decode_tail_v1_exp8_tiled_gqa4_atomic_fn(
            proj_q, pack_q, pk, pv, qscale, dlse, dout, gcode, gden,
            vproj, out, EB, 256, 1, SCALE, gl, sr, st))
        # Same tail with empty varlen tensors -> equal-path grid inside the main
        # kernel. full - full_eq isolates the scheduled-grid cost (valid when the
        # batch really is equal-length).
        t_full_eq = bench(lambda: ext.picokv_decode_tail_v1_exp8_tiled_gqa4_atomic_fn(
            proj_q, pack_q, pk, pv, qscale, dlse, dout, gcode, gden,
            vproj, out, EB, 256, 1, SCALE, empty, empty, empty))
        t_proj = bench(lambda: ext.picokv_fused_output_projection_fn(
            dout, gcode, gden, vproj, out))
        # Main kernel alone: direct wrapper has no varlen args -> equal/single only.
        t_main = float("nan")
        if not ragged:
            t_main = bench(lambda: ext.picokv_global_code_output_v1_exp8_tiled_gqa4_atomic_fn(
                pack_q, pk, pv, qscale, dlse, gcode, gden, EB, 256, 1, SCALE))

        # exp_bits sweep on the main kernel (equal/single only).
        sweep = {}
        if not ragged:
            for eb in (1, 2, 4, 8):
                sweep[eb] = bench(lambda eb=eb: ext.picokv_global_code_output_v1_exp8_tiled_gqa4_atomic_fn(
                    pack_q, pk, pv, qscale, dlse, gcode, gden, eb, 256, 1, SCALE), iters=20)

        # Pruned kernel (mode 1 = 64-token block pruning), equal/single only.
        t_pruned = float("nan")
        if not ragged:
            t_pruned = bench(lambda: ext.picokv_global_code_output_v1_exp8_tiled_gqa4_atomic_pruned_fn(
                pack_q, pk, pv, qscale, dlse, gcode, gden, EB, 256, 1, SCALE, 1, -10.0))

        kv_bytes = sum(gls) * KVH * (40 + 40)
        floor_us = kv_bytes / (BW * 1e9) * 1e6
        main_us = t_main * 1e3 if t_main == t_main else (t_full - t_quant - t_memset - t_proj) * 1e3

        print(f"\n[{name}] B={B} gls={gls if B <= 2 else f'{min(gls)}..{max(gls)}'} "
              f"Ck={Ck} Cv={Cv} EB={EB} KB={KB} Lmax={Lmax} Gmax={Gmax}")
        print(f"  full tail host fn      : {t_full*1e3:9.1f} us")
        if not ragged:
            print(f"  full tail (equal grid) : {t_full_eq*1e3:9.1f} us"
                  f"  -> scheduled-grid delta = {(t_full - t_full_eq)*1e3:.1f} us")
        print(f"    query quantize_pack  : {t_quant*1e3:9.1f} us")
        print(f"    memset (code+den)    : {t_memset*1e3:9.1f} us")
        print(f"    main scan kernel     : {main_us:9.1f} us  (derived=full-quant-memset-proj)"
              + ("" if t_main != t_main else "  (direct)"))
        print(f"    fused value proj     : {t_proj*1e3:9.1f} us")
        print(f"  query proj bmm         : {t_bmm*1e3:9.1f} us")
        if t_pruned == t_pruned:
            print(f"  pruned main (mode1,e-10): {t_pruned*1e3:8.1f} us")
        print(f"  KV bytes read by main  : {kv_bytes/1e6:.1f} MB -> BW floor {floor_us:.1f} us"
              f"  -> main/floor = {main_us/floor_us:.1f}x")
        if sweep:
            print("  exp_bits sweep (main kernel): " + "  ".join(
                f"eb{eb}={t*1e3:.1f}us" for eb, t in sweep.items()))

        # Dense-equivalent bytes (what full-KV attention would read at these lengths).
        tot_tok = sum(int(c.packed_key.shape[-1]) + 320 for c in []) if False else sum(gls) + 320 * B
        fi_bytes = tot_tok * KVH * 512
        print(f"  full-KV bytes (fi view): {fi_bytes/1e6:.1f} MB -> BW floor {fi_bytes/(BW*1e9)*1e6:.1f} us")

        be._reqs.clear()
        be._batch = None
        torch.cuda.empty_cache()

    if not args.skip_lswep:
        print("\n" + "=" * 100)
        print("PART 2: single-request L sweep (main kernel + full tail), B=1")
        print("=" * 100)
        for L in (8000, 16000, 32000, 64000, 120000, 200000):
            be = mod.PicoKVAttnBackend(_Runner())
            fixed, _ = build_case(be, mod, [L])
            Ck = int(fixed["batched"].key_projection.shape[-2])
            Cv = int(fixed["batched"].value_projection.shape[-2])
            KB = Ck // 64
            EB = int(fixed["exp_bits"])
            B = 1
            proj_q = torch.randn(B, QH, 1, Ck, device="cuda", dtype=torch.bfloat16)
            pack_q = torch.zeros(B, QH, 1, KB, 8, device="cuda", dtype=torch.long)
            qscale = torch.ones(B, QH, 1, device="cuda", dtype=torch.float32)
            dlse = torch.zeros(B, QH, 1, device="cuda", dtype=torch.float32)
            dout = torch.zeros(B, QH, 1, D, device="cuda", dtype=torch.bfloat16)
            gcode = torch.zeros(B, QH, 1, Cv, device="cuda", dtype=torch.float32)
            gden = torch.zeros(B, QH, 1, device="cuda", dtype=torch.float32)
            out = torch.zeros(B, QH, 1, D, device="cuda", dtype=torch.bfloat16)
            pk, pv = fixed["batched"].packed_key, fixed["batched"].packed_value
            gl = fixed["global_lens"]
            g = int(gl.tolist()[0])
            t_main = bench(lambda: ext.picokv_global_code_output_v1_exp8_tiled_gqa4_atomic_fn(
                pack_q, pk, pv, qscale, dlse, gcode, gden, EB, 256, 1, SCALE))
            t_full = bench(lambda: ext.picokv_decode_tail_v1_exp8_tiled_gqa4_atomic_fn(
                proj_q, pack_q, pk, pv, qscale, dlse, dout, gcode, gden,
                fixed["batched"].value_projection, out, EB, 256, 1, SCALE, gl,
                fixed["sched_req"], fixed["sched_tile"]))
            kv_bytes = g * KVH * 80
            print(f"  L={L:7d} gl={g:7d}  main={t_main*1e3:8.1f} us  full={t_full*1e3:8.1f} us"
                  f"  floor={kv_bytes/(BW*1e9)*1e6:6.1f} us  main/floor={t_main*1e3/(kv_bytes/(BW*1e9)*1e6):5.1f}x")
            be._reqs.clear()
            be._batch = None
            torch.cuda.empty_cache()

    print("\n" + "=" * 100)
    print("PART 3: kernel launch overhead")
    print("=" * 100)
    x = torch.zeros(1, device="cuda")
    t_fill = bench(lambda: x.fill_(0.0), iters=400, warmup=50)
    t_zero = bench(lambda: x.zero_(), iters=400, warmup=50)
    print(f"  1-elem fill_ back-to-back : {t_fill*1e3:.2f} us/op (CPU-side dispatch rate, unsync'd)")
    print(f"  1-elem zero_ back-to-back : {t_zero*1e3:.2f} us/op")

    # CPU-side submission rate of the full tail host fn (unsync'd).
    be = mod.PicoKVAttnBackend(_Runner())
    fixed, _ = build_case(be, mod, [8000] * 16)
    Ck = int(fixed["batched"].key_projection.shape[-2]); Cv = int(fixed["batched"].value_projection.shape[-2])
    KB = Ck // 64; EB = int(fixed["exp_bits"]); B = 16
    proj_q = torch.randn(B, QH, 1, Ck, device="cuda", dtype=torch.bfloat16)
    pack_q = torch.zeros(B, QH, 1, KB, 8, device="cuda", dtype=torch.long)
    qscale = torch.ones(B, QH, 1, device="cuda", dtype=torch.float32)
    dlse = torch.zeros(B, QH, 1, device="cuda", dtype=torch.float32)
    dout = torch.zeros(B, QH, 1, D, device="cuda", dtype=torch.bfloat16)
    gcode = torch.zeros(B, QH, 1, Cv, device="cuda", dtype=torch.float32)
    gden = torch.zeros(B, QH, 1, device="cuda", dtype=torch.float32)
    out = torch.zeros(B, QH, 1, D, device="cuda", dtype=torch.bfloat16)
    pk, pv = fixed["batched"].packed_key, fixed["batched"].packed_value
    gl, sr, st = fixed["global_lens"], fixed["sched_req"], fixed["sched_tile"]

    def full_tail():
        ext.picokv_decode_tail_v1_exp8_tiled_gqa4_atomic_fn(
            proj_q, pack_q, pk, pv, qscale, dlse, dout, gcode, gden,
            fixed["batched"].value_projection, out, EB, 256, 1, SCALE, gl, sr, st)
    for _ in range(10):
        full_tail()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    N = 300
    for _ in range(N):
        full_tail()
    t_submit = (time.perf_counter() - t0) / N * 1e6
    torch.cuda.synchronize()
    print(f"  tail host fn CPU submit  : {t_submit:.2f} us/call (python+pybind+4 launches, unsync'd)")
    be._reqs.clear(); be._batch = None; torch.cuda.empty_cache()

    if not args.skip_bmma:
        print("\n" + "=" * 100)
        print("PART 4: bmma tensorcore prototype vs popc (GQA-8/KB4/Cv256/EB4, B=16, L=7872)")
        print("=" * 100)
        B, KVH2, QH2, L2, KB2, CV2, EB2 = 16, 8, 64, 7872, 4, 256, 4
        G2 = (L2 + 63) // 64
        torch.manual_seed(7)
        pq = torch.randint(-2**62, 2**62, (B, QH2, 1, KB2, 8), device="cuda", dtype=torch.long)
        pkey = torch.randint(-2**62, 2**62, (B, KVH2, KB2, L2), device="cuda", dtype=torch.long)
        pval = torch.randint(-2**62, 2**62, (B, KVH2, G2, CV2), device="cuda", dtype=torch.long)
        qs = torch.ones(B, QH2, 1, device="cuda", dtype=torch.float32)
        dl = torch.zeros(B, QH2, 1, device="cuda", dtype=torch.float32)
        gc = torch.zeros(B, QH2, 1, CV2, device="cuda", dtype=torch.float32)
        gd = torch.zeros(B, QH2, 1, device="cuda", dtype=torch.float32)
        kv_bytes = B * KVH2 * L2 * (32 + 32)
        t_popc = bench(lambda: ext.picokv_global_code_output_v1_exp8_tiled_gqa4_atomic_fn(
            pq, pkey, pval, qs, dl, gc, gd, EB2, 256, 1, SCALE))
        t_bmma = bench(lambda: ext.picokv_global_code_output_v1_exp8_tiled_gqa8_bmma_score_atomic_fn(
            pq, pkey, pval, qs, dl, gc, gd, EB2, 256, 1, SCALE))
        print(f"  popc (atomic2_gqa8 kernel)   : {t_popc*1e3:8.1f} us  (floor {kv_bytes/(BW*1e9)*1e6:.1f} us,"
              f" {t_popc*1e3/(kv_bytes/(BW*1e9)*1e6):.1f}x)")
        print(f"  bmma score prototype         : {t_bmma*1e3:8.1f} us  ({t_popc/t_bmma:.2f}x vs popc, same tensors)")
        # same-shape GQA-4 for reference (atomic2 kernel)
        pq4 = pq[:, :32].contiguous(); qs4 = qs[:, :32].contiguous(); dl4 = dl[:, :32].contiguous()
        gc4 = torch.zeros(B, 32, 1, CV2, device="cuda", dtype=torch.float32)
        gd4 = torch.zeros(B, 32, 1, device="cuda", dtype=torch.float32)
        t_popc4 = bench(lambda: ext.picokv_global_code_output_v1_exp8_tiled_gqa4_atomic_fn(
            pq4, pkey, pval, qs4, dl4, gc4, gd4, EB2, 256, 1, SCALE))
        print(f"  popc GQA-4 same KV tensors   : {t_popc4*1e3:8.1f} us  (reference: half the query heads)")


if __name__ == "__main__":
    main()
