"""Bitwise A/B: popc kernel vs bmma prototype on identical packed tensors.

Sparse bitplane tensors keep dot products finite so real values (not NaNs)
are compared. Float32 outputs compared as raw bit patterns (NaN==NaN).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from picokv.picokv_cuda_extension.picokv_cuda import load_picokv_extension

ext = load_picokv_extension()
# L <= 512 -> ONE tile block per (b,kvh) -> single writer per output element ->
# deterministic atomics -> a bitwise verdict is meaningful (multi-tile L shows
# ulp-level atomic-order noise even between two runs of the SAME kernel).
import sys
B = int(sys.argv[1]) if len(sys.argv) > 1 else 16
L = int(sys.argv[2]) if len(sys.argv) > 2 else 7872
KVH, QH, KB, CV, EB = 8, 64, 4, 256, 4
G = (L + 63) // 64
SCALE = 1.0 / (128 ** 0.5)

def sparse_words(*shape, bits=16):
    """Random 64-bit words with ~`bits` set bits (per 64)."""
    w = torch.zeros(*shape, dtype=torch.long)
    # two bytes set per 8-byte word -> 16/64 bits
    mask = 0x00FF00FF00FF00FF
    r = torch.randint(-2**62, 2**62, shape, device="cuda")
    return (r & mask)

torch.manual_seed(7)
dev = "cuda"
pq = sparse_words(B, QH, 1, KB, 8)
pkey = sparse_words(B, KVH, KB, L)
pval = sparse_words(B, KVH, G, CV)
qs = torch.full((B, QH, 1), 1e-3, device=dev, dtype=torch.float32)
dl = torch.zeros(B, QH, 1, device=dev, dtype=torch.float32)

outs = {}
for which, fn in [
    ("popc", ext.picokv_global_code_output_v1_exp8_tiled_gqa4_atomic_fn),
    ("bmma", ext.picokv_global_code_output_v1_exp8_tiled_gqa8_bmma_score_atomic_fn),
]:
    gc = torch.zeros(B, QH, 1, CV, device=dev, dtype=torch.float32)
    gd = torch.zeros(B, QH, 1, device=dev, dtype=torch.float32)
    fn(pq, pkey, pval, qs, dl, gc, gd, EB, 256, 1, SCALE)
    torch.cuda.synchronize()
    outs[which] = (gc, gd)

gc0, gd0 = outs["popc"]; gc1, gd1 = outs["bmma"]
n_nan = int(torch.isnan(gc0).sum())
code_bits_eq = torch.equal(gc0.view(torch.int32), gc1.view(torch.int32))
den_bits_eq = torch.equal(gd0.view(torch.int32), gd1.view(torch.int32))
print(f"finite values: {gc0.numel()-n_nan}/{gc0.numel()} NaN; "
      f"gcode bitwise: {code_bits_eq}  gden bitwise: {den_bits_eq}")
if not (code_bits_eq and den_bits_eq):
    m = (gc0 != gc1)
    n = int(m.sum())
    print(f"mismatches: {n}")
    for i in m.nonzero()[:5]:
        b, qh, s, c = i.tolist()
        print(f"  [{b},{qh},{s},{c}] popc={gc0[b,qh,s,c].item():.7e} bmma={gc1[b,qh,s,c].item():.7e}")
    dmd = (gd0 != gd1)
    print(f"gden mismatches: {int(dmd.sum())}; max rel gcode diff "
          f"{((gc0-gc1).abs()/gc0.abs().clamp_min(1e-9))[m].max().item():.3e}")
