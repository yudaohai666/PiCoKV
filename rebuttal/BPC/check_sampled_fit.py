"""Pre-check for the sample-matched BPC arm, on real cached key/value states.

Two questions, both cheap, both worth answering before spending GPU-hours:

1. Equivalence. With direction_sample_size >= L, quantize_binarypc_sampled must
   be bit-identical to quantize_binarypc (same global RNG, same input), so the
   new code path cannot silently change the already-measured full-L arm.
2. Sensitivity. How fast does BPC's reconstruction error grow as the fit set
   shrinks? The real campaign samples 8192 of 131072, a 1/16 ratio. The cached
   states are only 8192 long, so the same ratios are emulated by fitting on
   8192/4096/2048/1024/512 tokens and encoding all 8192.

The reported error is the quantizer's own per-token residual norm, which is also
what drives EAS, so it is the quantity the pipeline actually consumes.
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rebuttal.BPC.encoding import quantize_binarypc, quantize_binarypc_sampled

BITS = 320
CACHE = ROOT / "draw/cache/kv_L10_ctx8192_d50_fp16.pt"


def load_states():
    blob = torch.load(CACHE, map_location="cpu")
    if isinstance(blob, dict):
        print("cache keys:", {k: tuple(v.shape) for k, v in blob.items() if torch.is_tensor(v)})
        for name in ("key", "key_states", "k"):
            if name in blob:
                return name, blob[name]
        name, tensor = next((k, v) for k, v in blob.items() if torch.is_tensor(v) and v.dim() == 4)
        return name, tensor
    return "tensor", blob


def main():
    name, states = load_states()
    states = states.cuda().to(torch.bfloat16)
    if states.dim() == 3:
        states = states.unsqueeze(0)
    print(f"using '{name}' {tuple(states.shape)} {states.dtype}")
    length = states.shape[-2]

    torch.manual_seed(42)
    full = quantize_binarypc(states, BITS)
    torch.manual_seed(42)
    same = quantize_binarypc_sampled(states, BITS, direction_sample_size=length * 2)
    identical = torch.equal(full.codes, same.codes) and torch.equal(full.decoder, same.decoder)
    print(f"\n[1] sample_size >= L reproduces full-L arm exactly: {identical}")
    if not identical:
        print("    codes equal:", torch.equal(full.codes, same.codes))
        print("    decoder max abs diff:", (full.decoder - same.decoder).abs().max().item())

    base = full.err.float()
    denom = torch.linalg.vector_norm(states.float(), dim=-1).mean().item()
    print(f"\n[2] fit-set sensitivity (encode always on all {length} tokens)")
    print(f"    ||state|| mean = {denom:.4f}")
    print(f"    {'fit tokens':>12} {'ratio':>8} {'mean err':>10} {'rel err':>9} {'vs full':>9}")
    print(f"    {length:>12} {'1/1':>8} {base.mean().item():>10.4f} "
          f"{base.mean().item()/denom:>9.4f} {'1.000x':>9}")
    for size in (4096, 2048, 1024, 512):
        torch.manual_seed(42)
        out = quantize_binarypc_sampled(states, BITS, direction_sample_size=size)
        err = out.err.float().mean().item()
        print(f"    {size:>12} {'1/' + str(length // size):>8} {err:>10.4f} "
              f"{err/denom:>9.4f} {err/base.mean().item():>8.3f}x")


if __name__ == "__main__":
    main()
