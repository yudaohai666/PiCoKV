"""Time the FULL tail host fn (quant+memset+main+projection) at several EB.

The eb sweep in bench_tail_split times the MAIN kernel only; the engine A/B
showed eb=2 slower end-to-end, so the slow piece must be elsewhere in the
host fn. This isolates full-tail vs EB on production b16@8k tensors.
"""
import torch
from bench_tail_split import (QH, KVH, D, SCALE, load_backend, bench,
                              get_ext, build_case, _Runner)


def main():
    ext = get_ext()
    mod = load_backend()
    be = mod.PicoKVAttnBackend(_Runner())
    fixed, _ = build_case(be, mod, [8000] * 16)
    batched = fixed["batched"]
    B = 16
    Ck = int(batched.key_projection.shape[-2])
    KB = Ck // 64
    Cv = int(batched.value_projection.shape[-2])
    proj_q = torch.randn(B, QH, 1, Ck, device="cuda", dtype=torch.bfloat16)
    pack_q = torch.zeros(B, QH, 1, KB, 8, device="cuda", dtype=torch.long)
    qscale = torch.ones(B, QH, 1, device="cuda", dtype=torch.float32)
    dlse = torch.zeros(B, QH, 1, device="cuda", dtype=torch.float32)
    dout = torch.zeros(B, QH, 1, D, device="cuda", dtype=torch.bfloat16)
    gcode = torch.zeros(B, QH, 1, Cv, device="cuda", dtype=torch.float32)
    gden = torch.zeros(B, QH, 1, device="cuda", dtype=torch.float32)
    out = torch.zeros(B, QH, 1, D, device="cuda", dtype=torch.bfloat16)
    gl, sr, st = fixed["global_lens"], fixed["sched_req"], fixed["sched_tile"]

    for eb in (1, 2, 4, 8):
        t = bench(lambda eb=eb: ext.picokv_decode_tail_v1_exp8_tiled_gqa4_atomic_fn(
            proj_q, pack_q, batched.packed_key, batched.packed_value, qscale,
            dlse, dout, gcode, gden, batched.value_projection, out,
            eb, 256, 1, SCALE, gl, sr, st))
        print(f"EB={eb}: full tail = {t*1e3:8.1f} us", flush=True)


if __name__ == "__main__":
    main()
