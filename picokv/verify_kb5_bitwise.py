"""Bitwise A/B: generic runtime-KBlocks vs KBlocks=5-specialized scan kernel.

The KBlocksT template specialization only removes dead `kb < KBlocks` guarded
iterations and constant-folds the key strides, so on the same tensors its
outputs must be BIT-IDENTICAL to the generic kernel. The env switch
(PICOKV_NO_KBSPEC) is cached per process, so each run is its own process:

  CUDA_VISIBLE_DEVICES=7 python picokv/verify_kb5_bitwise.py --out spec.pt
  PICOKV_NO_KBSPEC=1 ... --out gen1.pt     # generic kernel
  PICOKV_NO_KBSPEC=1 ... --out gen2.pt     # determinism control
  python picokv/verify_kb5_bitwise.py --cmp gen1.pt gen2.pt   # must be equal
  python picokv/verify_kb5_bitwise.py --cmp spec.pt gen1.pt   # the actual A/B
"""
from __future__ import annotations

import argparse
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


def get_ext():
    from picokv.picokv_cuda_extension.picokv_cuda import load_picokv_extension
    return load_picokv_extension()


def build_case(be, mod, lengths):
    LAYER = SimpleNamespace(layer_id=1)
    for rid, L in enumerate(lengths):
        q, k, v = _fresh(L)
        be.forward_extend(q, k, v, LAYER, _fb([rid], [L]))
    rids = list(range(len(lengths)))
    q, k, v = _fresh(len(rids))
    be.forward_decode(q, k, v, LAYER, _fb(rids))
    return be._batch["fixed"][1]


def run_all() -> dict:
    ext = get_ext()
    mod = load_backend()
    outs = {}

    # --- Cases A (equal b16@8k, direct main-kernel call, production shape) ---
    be = mod.PicoKVAttnBackend(_Runner())
    fixed = build_case(be, mod, [8000] * 16)
    batched = fixed["batched"]
    B = 16
    Ck = int(batched.key_projection.shape[-2])
    Cv = int(batched.value_projection.shape[-2])
    KB = Ck // 64
    proj_q = torch.randn(B, QH, 1, Ck, device="cuda", dtype=torch.bfloat16)
    pack_q = torch.zeros(B, QH, 1, KB, 8, device="cuda", dtype=torch.long)
    qscale = torch.ones(B, QH, 1, device="cuda", dtype=torch.float32)
    dlse = torch.zeros(B, QH, 1, device="cuda", dtype=torch.float32)
    pk, pv = batched.packed_key, batched.packed_value
    pk_nt = pk.transpose(-1, -2).contiguous()  # [B,KVH,L,KB] for key_transposed=0
    ext.picokv_quantize_pack_projected_query_fn(proj_q, pack_q, qscale)
    torch.cuda.synchronize()
    for tag, eb, kt, keyt in (
        ("eb4_kt1", 4, 1, pk),      # production instantiation
        ("eb2_kt1", 2, 1, pk),      # EB=2 log-threshold instantiation
        ("eb4_kt0", 4, 0, pk_nt),   # KEY_T=false instantiation
    ):
        gcode = torch.zeros(B, QH, 1, Cv, device="cuda", dtype=torch.float32)
        gden = torch.zeros(B, QH, 1, device="cuda", dtype=torch.float32)
        ext.picokv_global_code_output_v1_exp8_tiled_gqa4_atomic_fn(
            pack_q, keyt, pv, qscale, dlse, gcode, gden, eb, 256, kt, SCALE)
        torch.cuda.synchronize()
        outs[f"A_{tag}_gcode"] = gcode.cpu().clone()
        outs[f"A_{tag}_gden"] = gden.cpu().clone()

    # --- Case S (L=384: ONE tile-block per (b,kvh) -> single writer per
    # output element -> no atomic-order nondeterminism; spec vs generic MUST be
    # bitwise identical here or the specialization changed the arithmetic) ---
    be3 = mod.PicoKVAttnBackend(_Runner())
    fixed3 = build_case(be3, mod, [384] * 4)
    batched3 = fixed3["batched"]
    B3 = 4
    Ck3 = int(batched3.key_projection.shape[-2])
    Cv3 = int(batched3.value_projection.shape[-2])
    KB3 = Ck3 // 64
    proj_q3 = torch.randn(B3, QH, 1, Ck3, device="cuda", dtype=torch.bfloat16)
    pack_q3 = torch.zeros(B3, QH, 1, KB3, 8, device="cuda", dtype=torch.long)
    qscale3 = torch.ones(B3, QH, 1, device="cuda", dtype=torch.float32)
    dlse3 = torch.zeros(B3, QH, 1, device="cuda", dtype=torch.float32)
    ext.picokv_quantize_pack_projected_query_fn(proj_q3, pack_q3, qscale3)
    torch.cuda.synchronize()
    gcode3 = torch.zeros(B3, QH, 1, Cv3, device="cuda", dtype=torch.float32)
    gden3 = torch.zeros(B3, QH, 1, device="cuda", dtype=torch.float32)
    ext.picokv_global_code_output_v1_exp8_tiled_gqa4_atomic_fn(
        pack_q3, batched3.packed_key, batched3.packed_value, qscale3, dlse3,
        gcode3, gden3, 4, 256, 1, SCALE)
    torch.cuda.synchronize()
    outs["S_384_gcode"] = gcode3.cpu().clone()
    outs["S_384_gden"] = gden3.cpu().clone()

    # --- Case D (ragged b8, full tail host fn = scheduled varlen path) ---
    be2 = mod.PicoKVAttnBackend(_Runner())
    lengths = [4000, 8000, 12000, 16000, 24000, 32000, 48000, 64000]
    fixed2 = build_case(be2, mod, lengths)
    batched2 = fixed2["batched"]
    B2 = len(lengths)
    Ck2 = int(batched2.key_projection.shape[-2])
    Cv2 = int(batched2.value_projection.shape[-2])
    KB2 = Ck2 // 64
    proj_q2 = torch.randn(B2, QH, 1, Ck2, device="cuda", dtype=torch.bfloat16)
    pack_q2 = torch.zeros(B2, QH, 1, KB2, 8, device="cuda", dtype=torch.long)
    qscale2 = torch.ones(B2, QH, 1, device="cuda", dtype=torch.float32)
    dlse2 = torch.zeros(B2, QH, 1, device="cuda", dtype=torch.float32)
    dout2 = torch.zeros(B2, QH, 1, D, device="cuda", dtype=torch.bfloat16)
    gcode2 = torch.zeros(B2, QH, 1, Cv2, device="cuda", dtype=torch.float32)
    gden2 = torch.zeros(B2, QH, 1, device="cuda", dtype=torch.float32)
    out2 = torch.zeros(B2, QH, 1, D, device="cuda", dtype=torch.bfloat16)
    EB2 = int(fixed2["exp_bits"])
    gl2, sr2, st2 = fixed2["global_lens"], fixed2["sched_req"], fixed2["sched_tile"]
    ext.picokv_decode_tail_v1_exp8_tiled_gqa4_atomic_fn(
        proj_q2, pack_q2, batched2.packed_key, batched2.packed_value,
        qscale2, dlse2, dout2, gcode2, gden2, batched2.value_projection, out2,
        EB2, 256, 1, SCALE, gl2, sr2, st2)
    torch.cuda.synchronize()
    outs["D_ragged_gcode"] = gcode2.cpu().clone()
    outs["D_ragged_gden"] = gden2.cpu().clone()
    outs["D_ragged_out"] = out2.cpu().clone()
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str)
    ap.add_argument("--cmp", nargs=2, metavar=("A.pt", "B.pt"))
    args = ap.parse_args()
    assert torch.cuda.is_available()
    torch.manual_seed(0)
    if args.out:
        outs = run_all()
        torch.save(outs, args.out)
        print(f"saved {len(outs)} tensors -> {args.out}")
        for k, v in outs.items():
            print(f"  {k}: {tuple(v.shape)} {v.dtype} "
                  f"min={v.float().min().item():.4e} max={v.float().max().item():.4e}")
    if args.cmp:
        a = torch.load(args.cmp[0])
        b = torch.load(args.cmp[1])
        all_eq = True
        for k in a:
            eq = torch.equal(a[k], b[k])
            if not eq:
                d = (a[k].float() - b[k].float()).abs()
                rel = d / a[k].float().abs().clamp_min(1e-30)
                print(f"  {k}: MISMATCH  n={int((a[k] != b[k]).sum())}/{a[k].numel()}  "
                      f"max_abs={d.max().item():.3e}  max_rel={rel.max().item():.3e}")
            else:
                print(f"  {k}: bitwise EQUAL")
            all_eq &= bool(eq)
        print("RESULT:", "PASS (bitwise identical)" if all_eq else "FAIL")


if __name__ == "__main__":
    main()
