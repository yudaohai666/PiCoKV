"""Squeeze-analysis probe: post-optimization per-layer time distribution.

For each shape (engine caliber):
  * prefill: chunked like the engine (<=16384 tokens per extend), timed per
    layer call; the ragged flashinfer attention part timed separately so
    prepare(compression) = total - attn.
  * decode: warm graphs (3 steps), then CUDA-event time of
      (a) full forward_decode per layer-step  (graph replay: copy+scatter+
          dense fi + bmm + fused tail, one launch)
      (b) dense-window paged flashinfer run alone
      (c) query-projection bmm alone
      (d) fused tail host fn alone (quant+memset+main scan+GEMM)
  * window lens after prefill (sink+local+EAS accounting).

Run under .venv_sglang from PiCoKV/:
  CUDA_VISIBLE_DEVICES=6 PYTHONPATH=$PWD \
  PICOKV_PREBUILT_DIR=$PWD/build/picokv_extensions \
  ../.venv_sglang/bin/python picokv/probe_squeeze.py
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
SCALE = 1.0 / (D ** 0.5)
BW = 3005.0  # GB/s measured read bandwidth
CHUNK = 16384  # engine max_prefill_tokens


def load_backend():
    spec = importlib.util.spec_from_file_location("picokv_backend", BACKEND)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Cfg:
    num_attention_heads = QH
    head_dim = D
    # model_path resolves the shipped offline projection .pt exactly like the
    # engine's serving path (append requires offline + eas=0).
    model_path = str(
        (Path(__file__).resolve().parents[2] / "llama31_8b_rand")
    )

    def get_num_kv_heads(self, tp):
        return KVH


class _Runner:
    device = "cuda"
    model_config = _Cfg()
    tp_size = 1


def _fb(rids, eseq=None, plens=None):
    fb = SimpleNamespace(req_pool_indices=list(rids))
    if eseq is not None:
        fb.extend_seq_lens = list(eseq)
    if plens is not None:
        fb.extend_prefix_lens = list(plens)
    return fb


def bench_ev(fn, iters=30, warmup=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(5):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) / iters)
    return min(ts)


def chunk_list(lengths, budget=CHUNK):
    """Engine-style extend packing: whole requests until the token budget."""
    out, cur, cur_n = [], [], 0
    for rid, L in enumerate(lengths):
        pos = 0
        while pos < L:
            take = min(L - pos, budget - cur_n) if cur_n < budget else 0
            if take == 0:
                out.append((cur, cur_n))
                cur, cur_n = [], 0
                take = min(L - pos, budget)
            cur.append(rid)
            cur_n += take
            pos += take
    if cur:
        out.append((cur, cur_n))
    # returns list of (rid_list, ntokens) — but eseq needs per-rid lens
    return out


def build_chunks(lengths, budget=CHUNK):
    """Yield (rids, eseq, prefixes) extends exactly like the engine packs them.

    ``prefixes[i]`` is request ``rids[i]``'s already-prefilled token count
    (the engine's ``extend_prefix_lens``): 0 for a fresh request, >0 for a
    chunked-prefill continuation.
    """
    ext = []
    done = [0] * len(lengths)
    cur_rids, cur_lens, cur_pre = [], [], []
    for rid, L in enumerate(lengths):
        rem = L
        while rem > 0:
            room = budget - sum(cur_lens)
            if room == 0:
                ext.append((cur_rids, cur_lens, cur_pre))
                cur_rids, cur_lens, cur_pre = [], [], []
                room = budget
            take = min(rem, room)
            cur_rids.append(rid)
            cur_lens.append(take)
            cur_pre.append(done[rid])
            done[rid] += take
            rem -= take
    if cur_rids:
        ext.append((cur_rids, cur_lens, cur_pre))
    return ext


def gen_tokens(n, gen):
    q = torch.randn(n, QH * D, device="cuda", dtype=torch.bfloat16, generator=gen)
    k = torch.randn(n, KVH * D, device="cuda", dtype=torch.bfloat16, generator=gen)
    v = torch.randn(n, KVH * D, device="cuda", dtype=torch.bfloat16, generator=gen)
    return q, k, v


def main():
    assert torch.cuda.is_available()
    mod = load_backend()
    from picokv.picokv_cuda_extension.picokv_cuda import load_picokv_extension
    ext = load_picokv_extension()
    LAYER = SimpleNamespace(layer_id=0)

    SHAPES = [
        ("b1@8k", [8192]),
        ("b4@8k", [8192] * 4),
        ("b8@8k", [8192] * 8),
        ("b16@8k", [8192] * 16),
        ("b1@120k", [120000]),
        ("ragged-b8", [4000, 8000, 12000, 16000, 24000, 32000, 48000, 64000]),
    ]

    print("=" * 108)
    print("Per-shape post-optimization profile (backend-direct, graphs ON, µs per layer-step)")
    print("=" * 108)
    for name, lengths in SHAPES:
        torch.manual_seed(0)
        gen = torch.Generator(device="cuda").manual_seed(42)
        be = mod.PicoKVAttnBackend(_Runner())
        rids_all = list(range(len(lengths)))

        # --- prefill (engine-caliber chunking) --------------------------------
        chunks = build_chunks(lengths)
        t_ext = 0.0
        t_attn = 0.0
        for cr, cl, cp in chunks:
            q, k, v = gen_tokens(sum(cl), gen)
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            s.record()
            be.forward_extend(q, k, v, LAYER, _fb(cr, cl, cp))
            e.record()
            torch.cuda.synchronize()
            t_ext += s.elapsed_time(e)
            # ragged attention part alone (same token count)
            qa = q.view(-1, QH, D)
            ka = k.view(-1, KVH, D)
            va = v.view(-1, KVH, D)
            t_attn += bench_ev(
                lambda: be._prefill_attn(qa, ka, va, cl), iters=10, warmup=3)
        n_ext = len(chunks)

        # window lens after prefill (sink+local+EAS)
        wl = [int(be._reqs[r].layers[0].dense_key.shape[-2]) for r in rids_all]
        gl = [int(be._reqs[r].layers[0].packed_key.shape[-1]) for r in rids_all]

        # --- decode: warm graphs, then time step + segments --------------------
        def step():
            q, k, v = gen_tokens(len(rids_all), gen)
            return be.forward_decode(q, k, v, LAYER, _fb(rids_all))

        for _ in range(5):
            step()
        torch.cuda.synchronize()
        t_step = bench_ev(step)

        fixed = be._batch["fixed"][0]
        sh = be._batch["shared"]
        g = be._batch["g"]
        ent = g["by_layer"].get(1)
        graph_on = ent is not None
        batched = fixed["batched"]
        B = len(rids_all)
        Ck = int(batched.key_projection.shape[-2])
        KB = Ck // 64
        EB = int(fixed["exp_bits"])
        kbuf, vbuf = fixed["kbuf"], fixed["vbuf"]
        cap = kbuf.shape[1]
        kt = kbuf.view(B * cap, KVH, D)
        vt = vbuf.view(B * cap, KVH, D)

        # (b) dense paged fi alone (wrapper keeps the step's plan)
        fi = be._fi["wrapper"] if isinstance(be._fi, dict) else None
        q1, k1, v1 = gen_tokens(B, gen)
        qb = q1.view(-1, QH, D)
        t_fi = bench_ev(lambda: fi.run(qb, (kt, vt))) if fi is not None else float("nan")

        # (c) query-projection bmm alone
        gq = qb.reshape(B * KVH, 4, D).contiguous()
        kproj_t3 = batched.key_projection_t.reshape(B * KVH, D, Ck)
        proj_out = torch.empty(B * KVH, 4, Ck, device="cuda", dtype=torch.bfloat16)
        t_bmm = bench_ev(lambda: torch.bmm(gq, kproj_t3, out=proj_out))

        # (d) fused tail host fn alone (production tensors + own workspaces)
        proj_q = torch.randn(B, QH, 1, Ck, device="cuda", dtype=torch.bfloat16)
        pack_q = torch.zeros(B, QH, 1, KB, 8, device="cuda", dtype=torch.long)
        qscale = torch.ones(B, QH, 1, device="cuda", dtype=torch.float32)
        dlse = torch.zeros(B, QH, 1, device="cuda", dtype=torch.float32)
        dout = torch.zeros(B, QH, 1, D, device="cuda", dtype=torch.bfloat16)
        out = torch.zeros(B, QH, 1, D, device="cuda", dtype=torch.bfloat16)
        Cv = int(batched.value_projection.shape[-2])
        gcode = torch.zeros(B, QH, 1, Cv, device="cuda", dtype=torch.float32)
        gden = torch.zeros(B, QH, 1, device="cuda", dtype=torch.float32)
        gl_t, sr, st = fixed["global_lens"], fixed["sched_req"], fixed["sched_tile"]
        t_tail = bench_ev(lambda: ext.picokv_decode_tail_v1_exp8_tiled_gqa4_atomic_fn(
            proj_q, pack_q, batched.packed_key, batched.packed_value, qscale,
            dlse, dout, gcode, gden, batched.value_projection, out,
            EB, 256, 1, SCALE, gl_t, sr, st))

        kv_bytes = sum(int(x) for x in gl_t.tolist()) * KVH * 80
        floor_us = kv_bytes / (BW * 1e9) * 1e6
        attn_sum = t_fi + t_bmm + t_tail
        print(f"\n[{name}] extends={n_ext} window={wl[0]}{'' if len(set(wl))==1 else f'..{max(wl)}'} "
              f"gl={gl[0]}{'' if len(set(gl))==1 else f'..{max(gl)}'} graph={'ON' if graph_on else 'OFF'}")
        print(f"  prefill/layer : {t_ext:8.2f} ms over {n_ext} extends "
              f"(ragged-attn part {t_attn:.2f} ms -> prepare ~{t_ext - t_attn:.2f} ms)")
        print(f"  step (graph)  : {t_step*1e3:8.1f} µs   [fi {t_fi*1e3:.1f} + bmm {t_bmm*1e3:.1f} "
              f"+ tail {t_tail*1e3:.1f} = {attn_sum*1e3:.1f} µs]")
        print(f"  tail vs floor : {t_tail*1e3:8.1f} µs  (KV {kv_bytes/1e6:.1f} MB, floor {floor_us:.1f} µs, "
              f"{t_tail*1e3/floor_us:.1f}x)   step-attn share {100*attn_sum/t_step:.0f}%")

        be._reqs.clear()
        be._batch = None
        torch.cuda.empty_cache()

    print("\nDONE")


if __name__ == "__main__":
    main()
