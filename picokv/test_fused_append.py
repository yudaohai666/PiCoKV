import torch, time
from sglang.srt.layers.attention.picokv_backend import _append_kv_kernel
torch.manual_seed(0)
for B, cap in [(1, 900), (3, 4096), (16, 8704), (2, 140000)]:
    KVH, D = 8, 128
    qkv = torch.randn(B, (32 + 16) * D, device="cuda", dtype=torch.bfloat16)
    k = qkv[:, 32 * D:40 * D].view(B, KVH, D); v = qkv[:, 40 * D:].view(B, KVH, D)
    kb = torch.randn(B, cap, KVH, D, device="cuda", dtype=torch.bfloat16); vb = torch.randn_like(kb)
    kr, vr = kb.clone(), vb.clone()
    wpos = torch.randint(0, cap, (B,), device="cuda", dtype=torch.int32)
    ar = torch.arange(B, device="cuda")
    kr[ar, wpos] = k; vr[ar, wpos] = v
    _append_kv_kernel[(B,)](kb, vb, k, v, wpos, cap, k.stride(0), v.stride(0), ROW=KVH * D)
    torch.cuda.synchronize()
    print(B, cap, "k equal", torch.equal(kb, kr), "v equal", torch.equal(vb, vr))
    # graph-time comparison (GPU time of N captured launches)
    for name, fn in [("index_put x2", lambda: (kr.__setitem__((ar, wpos), k), vr.__setitem__((ar, wpos), v))),
                     ("triton fused", lambda: _append_kv_kernel[(B,)](kb, vb, k, v, wpos, cap, k.stride(0), v.stride(0), ROW=KVH * D))]:
        g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream()
        with torch.cuda.graph(g):
            for _ in range(100): fn()
        g.replay(); torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record(); [g.replay() for _ in range(10)]; e1.record(); torch.cuda.synchronize()
        print(f"   {name}: {e0.elapsed_time(e1) / 1000 * 1e3:.2f} us/call")
