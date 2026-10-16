"""Chunk-append == full-prepare bitwise equality (chunked-prefill fix gate).

Compares, per layer, (a) prepare(chunk_1) + append(chunk_2..n) against
(b) prepare(whole sequence) with eas=0 + offline projections. Every packed
tensor, dense window and recorded length must be BITWISE identical: the
append path reuses the same fixed-projection cascade kernel per token, so
results must not depend on how the sequence was sliced.
"""
import importlib.util
import os
import sys

import torch

PICOKV_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PICOKV_ROOT)

BACKEND = os.path.join(
    os.path.dirname(PICOKV_ROOT), "sglang", "python", "sglang",
    "srt", "layers", "attention", "picokv_backend.py",
)
spec = importlib.util.spec_from_file_location("picokv_backend_standalone", BACKEND)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)  # sglang import fails -> AttentionBackend=object guard

from picokv.picokv_standalone import StandaloneConfig  # noqa: E402

PROJ = os.path.join(PICOKV_ROOT, "offline_projection", "projections",
                    "Qwen3-8B_k320_v320.pt")
H, D = 8, 128


def run_case(name, total, chunks, seed=0):
    torch.manual_seed(seed)
    k = torch.randn(1, H, total, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, H, total, D, device="cuda", dtype=torch.bfloat16)
    cfg = StandaloneConfig(eas=0.0)
    offline = mod.load_offline_projections_pure(PROJ)

    ca = mod.PicoKVRequestCache(cfg, offline)
    cb = mod.PicoKVRequestCache(cfg, offline)
    off = 0
    for i, c in enumerate(chunks):
        kc = k[:, :, off:off + c].contiguous()
        vc = v[:, :, off:off + c].contiguous()
        if i == 0:
            assert not ca.can_append(0), "fresh request must not append"
            ca.prepare(0, kc, vc)
        else:
            assert ca.can_append(off), f"{name}: can_append failed at chunk {i} (prefix={off})"
            assert ca.append(0, kc, vc), f"{name}: append failed at chunk {i}"
        off += c
    cb.prepare(0, k, v)

    A, B = ca.layers[0], cb.layers[0]
    checks = {
        "packed_key": torch.equal(A.packed_key, B.packed_key),
        "packed_value": torch.equal(A.packed_value, B.packed_value),
        "dense_key": torch.equal(A.dense_key, B.dense_key),
        "dense_value": torch.equal(A.dense_value, B.dense_value),
        "seq_len": ca.seq_lengths[0] == cb.seq_lengths[0] == total,
    }
    print(f"[{name}] gl={A.packed_key.shape[-1]}/{B.packed_key.shape[-1]} "
          f"groups={A.packed_value.shape[2]}/{B.packed_value.shape[2]} "
          f"dense={A.dense_key.shape[-2]}/{B.dense_key.shape[-2]}")
    for field, ok in checks.items():
        print(f"  {field:12s}: {'BITWISE EQUAL' if ok else '*** MISMATCH ***'}")
    return all(checks.values())


ok = True
# engine-shaped: 16384 budget then odd sizes -> every carry path (gl%64)
ok &= run_case("120k/chunked", 100000, [16384, 1000, 777, 50123, 77, 3639, 28000])
# tiny chunks: dense fallback -> promote -> compressed appends, C < local
ok &= run_case("dense-promote", 900, [200, 300, 100, 300])
# single continuation aligned to 64
ok &= run_case("aligned", 40000, [16384, 23616])
print("APPEND == FULL-PREPARE (bitwise)" if ok else "FAILED")
sys.exit(0 if ok else 1)
