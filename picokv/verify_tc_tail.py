"""Turn-24: extension-level A/B, tensor-core tail kernel (default) vs popc
(PICOKV_NO_TC=1), on production tensors built by the real backend.

  python picokv/verify_tc_tail.py --out tc.pt               # default = tc kernel
  PICOKV_NO_TC=1 python picokv/verify_tc_tail.py --out popc1.pt
  PICOKV_NO_TC=1 python picokv/verify_tc_tail.py --out popc2.pt
  python picokv/verify_tc_tail.py --cmp popc1.pt popc2.pt   # atomic-order noise floor
  python picokv/verify_tc_tail.py --cmp tc.pt popc1.pt      # S_* keys must be bitwise
"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_kb5_bitwise as v  # noqa: E402


def _fb(rids, eseq=None):  # backend reads req_pool_indices.shape[0]
    fb = SimpleNamespace(req_pool_indices=torch.tensor(list(rids), dtype=torch.int64))
    if eseq is not None:
        fb.extend_seq_lens = torch.tensor(list(eseq), dtype=torch.int32)
    return fb


v._fb = _fb
QH, D, SCALE = v.QH, v.D, v.SCALE


def tensors(B, fixed):
    bt = fixed["batched"]
    Ck = int(bt.key_projection.shape[-2]); Cv = int(bt.value_projection.shape[-2])
    proj_q = torch.randn(B, QH, 1, Ck, device="cuda", dtype=torch.bfloat16)
    pack_q = torch.zeros(B, QH, 1, Ck // 64, 8, device="cuda", dtype=torch.long)
    qscale = torch.ones(B, QH, 1, device="cuda", dtype=torch.float32)
    # Realistic LSE: weights neither all-underflowed nor saturated.
    dlse = (torch.rand(B, QH, 1, device="cuda") * 4.0 + 6.0).float()
    return bt, Cv, proj_q, pack_q, qscale, dlse


def run_extra(ext, mod, outs):
    # Single-writer shapes (one block per (b, kvh) under the unscheduled grid):
    # L=384/512 -> BT2 (grid < 2*SMs picks BT1 -> L=200/256 single tile), so
    # both BT1 and BT2 tc instantiations are covered. Direct main-kernel call.
    for L in (520, 576, 704, 832):  # global = L - 320 window = 200/256/384/512
        be = mod.PicoKVAttnBackend(v._Runner())
        fixed = v.build_case(be, mod, [L] * 4)
        print(f"  L={L} global_lens={fixed['global_lens'].tolist() if fixed['global_lens'] is not None else None} pk={tuple(fixed['batched'].packed_key.shape)}", flush=True)
        bt, Cv, proj_q, pack_q, qscale, dlse = tensors(4, fixed)
        ext.picokv_quantize_pack_projected_query_fn(proj_q, pack_q, qscale)
        pk = bt.packed_key
        pk_nt = pk.transpose(-1, -2).contiguous()
        for eb in (1, 2, 4, 8):
            for kt, keyt in ((1, pk), (0, pk_nt)):
                gcode = torch.zeros(4, QH, 1, Cv, device="cuda", dtype=torch.float32)
                gden = torch.zeros(4, QH, 1, device="cuda", dtype=torch.float32)
                ext.picokv_global_code_output_v1_exp8_tiled_gqa4_atomic_fn(
                    pack_q, keyt, bt.packed_value, qscale, dlse, gcode, gden, eb, 256, kt, SCALE)
                torch.cuda.synchronize()
                outs[f"S_L{L}_eb{eb}_kt{kt}_gcode"] = gcode.cpu().clone()
                outs[f"S_L{L}_eb{eb}_kt{kt}_gden"] = gden.cpu().clone()
        be._reqs.clear(); be._batch = None
    # Full tail host fn (quant + memset + main + projection), equal batches
    # (unscheduled grid -> BT1 / BT2 / BT4 by grid size) at realistic LSE.
    for B, L in ((1, 8000), (16, 8000), (1, 120000), (16, 32000)):
        be = mod.PicoKVAttnBackend(v._Runner())
        fixed = v.build_case(be, mod, [L] * B)
        bt, Cv, proj_q, pack_q, qscale, dlse = tensors(B, fixed)
        dout = torch.zeros(B, QH, 1, D, device="cuda", dtype=torch.bfloat16)
        gcode = torch.zeros(B, QH, 1, Cv, device="cuda", dtype=torch.float32)
        gden = torch.zeros(B, QH, 1, device="cuda", dtype=torch.float32)
        out = torch.zeros(B, QH, 1, D, device="cuda", dtype=torch.bfloat16)
        ext.picokv_decode_tail_v1_exp8_tiled_gqa4_atomic_fn(
            proj_q, pack_q, bt.packed_key, bt.packed_value, qscale, dlse, dout, gcode, gden,
            bt.value_projection, out, int(fixed["exp_bits"]), 256, 1, SCALE,
            fixed["global_lens"], fixed["sched_req"], fixed["sched_tile"])
        torch.cuda.synchronize()
        outs[f"F_b{B}_L{L}_gcode"] = gcode.cpu().clone()
        outs[f"F_b{B}_L{L}_gden"] = gden.cpu().clone()
        outs[f"F_b{B}_L{L}_out"] = out.cpu().clone()
        be._reqs.clear(); be._batch = None
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--cmp", nargs=2)
    a = ap.parse_args()
    if a.out:
        torch.manual_seed(0)
        ext = v.get_ext(); mod = v.load_backend()
        outs = v.run_all()
        print("run_all done", flush=True)
        torch.manual_seed(1)
        run_extra(ext, mod, outs)
        torch.save(outs, a.out)
        nonfinite = [k for k, t in outs.items() if not torch.isfinite(t.float()).all()]
        print(f"saved {len(outs)} tensors -> {a.out}; non-finite: {nonfinite}")
    if a.cmp:
        x = torch.load(a.cmp[0]); y = torch.load(a.cmp[1])
        nbit = nclose = nfail = 0
        for k in x:
            if torch.equal(x[k], y[k]):
                nbit += 1
                print(f"  {k:28s} bitwise EQUAL")
                continue
            xf, yf = x[k].double(), y[k].double()
            rel = ((xf - yf).abs().max() / yf.abs().max().clamp_min(1e-30)).item()
            must_bitwise = k.startswith("S_")
            ok = (not must_bitwise) and rel < (1e-2 if k.endswith("_out") else 1e-5)
            nclose += ok; nfail += not ok
            print(f"  {k:28s} n_diff={int((x[k] != y[k]).sum())}/{x[k].numel()} "
                  f"maxrel(vs max|ref|)={rel:.2e}  {'CLOSE-OK' if ok else 'FAIL'}")
        print(f"RESULT: {nbit} bitwise, {nclose} close, {nfail} FAIL")


if __name__ == "__main__":
    main()
