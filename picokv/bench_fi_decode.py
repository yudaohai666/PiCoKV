"""FlashInfer decode-attention kernel bench at the bench_tail_split shapes.

Mirrors the sglang flashinfer backend decode path (BatchDecodeWithPagedKVCacheWrapper,
fa2 backend, page_size=1, bf16, GQA 32Q/8KV/128d) so the kernel times are directly
comparable to bench_tail_split's PiCoKV tail internals on the same GPU/session.

For each shape it times:
  * fi full attention     — what the sglang fi engine path pays per layer
  * fi windowed attention — window_left=255 (last 256 tokens, i.e. the PiCoKV
                            local window; PiCoKV additionally sinks the first
                            64 tokens, ~20% more windowed bytes)

and prints the full-KV bytes / BW floor (measured 3005 GB/s) like bench_tail_split.

Run under .venv_sglang from PiCoKV/:
  CUDA_VISIBLE_DEVICES=7 ../.venv_sglang/bin/python picokv/bench_fi_decode.py
"""
from __future__ import annotations

import torch
import flashinfer

QH, KVH, D = 32, 8, 128
SCALE = 1.0 / (D ** 0.5)
BW = 3005.0  # GB/s measured read bandwidth on this machine

SHAPES = [
    ("b1@8k", [8000]),
    ("b8@8k", [8000] * 8),
    ("b16@8k", [8000] * 16),
    ("b1@120k", [120000]),
    ("ragged-b8-mixed", [4000, 8000, 12000, 16000, 24000, 32000, 48000, 64000]),
]


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


def main():
    assert torch.cuda.is_available()
    torch.manual_seed(0)
    workspace_buffer = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    # Mirror sglang's FlashInferAttnBackend decode wrapper exactly:
    # backend="fa2", use_tensor_cores=True (GQA group 32/8=4 >= 4), NHD,
    # page_size=1, bf16 q/kv — i.e. the fa2 tensor-core decode kernel the
    # sglang flashinfer engine path actually launches per layer.
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, "NHD", backend="fa2", use_tensor_cores=True
    )

    for name, lens in SHAPES:
        B = len(lens)
        total = sum(lens)
        # sglang passes (k_buffer, v_buffer), each [pool_rows, KVH, D] with
        # page_size=1 (pool row == token); the forked wrapper takes the pair.
        k = torch.randn(total, KVH, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(total, KVH, D, device="cuda", dtype=torch.bfloat16)
        q = torch.randn(B, QH, D, device="cuda", dtype=torch.bfloat16)
        indptr = torch.zeros(B + 1, dtype=torch.int32, device="cuda")
        for i, l in enumerate(lens):
            indptr[i + 1] = indptr[i] + l
        indices = torch.arange(total, dtype=torch.int32, device="cuda")
        last_page_len = torch.ones(B, dtype=torch.int32, device="cuda")

        wrapper.plan(indptr, indices, last_page_len, QH, KVH, D, 1,
                     data_type=torch.bfloat16, q_data_type=torch.bfloat16)
        t_full = bench(lambda: wrapper.forward(q, (k, v), sm_scale=SCALE)) * 1e3

        wrapper.plan(indptr, indices, last_page_len, QH, KVH, D, 1,
                     data_type=torch.bfloat16, q_data_type=torch.bfloat16,
                     window_left=255)
        t_win = bench(lambda: wrapper.forward(q, (k, v), sm_scale=SCALE)) * 1e3

        full_bytes = total * KVH * D * 2 * 2
        win_bytes = sum(min(256, l) * KVH * D * 2 * 2 for l in lens)
        print(f"[{name}] B={B} total_tokens={total}")
        print(f"  fi full attention kernel : {t_full:8.1f} us  "
              f"(full-KV bytes {full_bytes/1e6:.1f} MB -> BW floor {full_bytes/BW/1e3:.1f} us, "
              f"kernel/floor = {t_full/(full_bytes/BW/1e3):.2f}x)")
        print(f"  fi windowed (last-256)   : {t_win:8.1f} us  "
              f"(window bytes {win_bytes/1e6:.1f} MB -> BW floor {win_bytes/BW/1e3:.1f} us, "
              f"kernel/floor = {t_win/(win_bytes/BW/1e3):.2f}x)")
        del k, v, q
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
