"""Turn-22: main/tail kernel timing at long context, achieved popcll rate vs
measured ISA limit (isa_rate.cu: 7.64 popcll/SM/clk = ~2.0e12/s on this GPU)."""
import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_tail_split as bts
from types import SimpleNamespace

def _fb(rids, eseq=None):  # backend now reads req_pool_indices.shape[0]
    fb = SimpleNamespace(req_pool_indices=torch.tensor(list(rids), dtype=torch.int64))
    if eseq is not None:
        fb.extend_seq_lens = torch.tensor(list(eseq), dtype=torch.int32)
    return fb
bts._fb = _fb

POPC_RATE = 2.0e12  # measured popcll/s (isa_rate.cu)
CASES = [(s.split("@")[0], s) for s in sys.argv[1:]] or [("b16", "16@120000")]

def main():
    torch.manual_seed(0)
    ext = bts.get_ext(); mod = bts.load_backend()
    for _, spec in CASES:
        B, L = (int(x) for x in spec.split("@"))
        be = mod.PicoKVAttnBackend(bts._Runner())
        fixed, _ = bts.build_case(be, mod, [L] * B)
        bt = fixed["batched"]
        Ck = int(bt.key_projection.shape[-2]); Cv = int(bt.value_projection.shape[-2]); KB = Ck // 64
        EB = int(fixed["exp_bits"]); QH, KVH, D = bts.QH, bts.KVH, bts.D
        pk, pv = bt.packed_key, bt.packed_value
        nbytes = pk.numel() * pk.element_size() + pv.numel() * pv.element_size()
        proj_q = torch.randn(B, QH, 1, Ck, device="cuda", dtype=torch.bfloat16)
        pack_q = torch.zeros(B, QH, 1, KB, 8, device="cuda", dtype=torch.long)
        qscale = torch.ones(B, QH, 1, device="cuda", dtype=torch.float32)
        dlse = torch.zeros(B, QH, 1, device="cuda", dtype=torch.float32)
        dout = torch.zeros(B, QH, 1, D, device="cuda", dtype=torch.bfloat16)
        gcode = torch.zeros(B, QH, 1, Cv, device="cuda", dtype=torch.float32)
        gden = torch.zeros(B, QH, 1, device="cuda", dtype=torch.float32)
        out = torch.zeros(B, QH, 1, D, device="cuda", dtype=torch.bfloat16)
        gl, sr, st = fixed["global_lens"], fixed["sched_req"], fixed["sched_tile"]
        ext.picokv_quantize_pack_projected_query_fn(proj_q, pack_q, qscale)
        t_full = bts.bench(lambda: ext.picokv_decode_tail_v1_exp8_tiled_gqa4_atomic_fn(
            proj_q, pack_q, pk, pv, qscale, dlse, dout, gcode, gden,
            bt.value_projection, out, EB, 256, 1, bts.SCALE, gl, sr, st), iters=10)
        mains = {}
        for eb in (1, 2, 4):
            mains[eb] = bts.bench(lambda eb=eb: ext.picokv_global_code_output_v1_exp8_tiled_gqa4_atomic_fn(
                pack_q, pk, pv, qscale, dlse, gcode, gden, eb, 256, 1, bts.SCALE), iters=10)
        tk = B * L * KVH  # (token, kv-head) pairs
        sc_popc = tk * (QH // KVH) * 7 * KB          # score: heads x mag-planes x key words
        val_popc = lambda eb: tk * (QH // KVH) * eb * Cv / 64
        t4 = mains[4] * 1e-3
        print(f"[{spec}] B={B} L={L} Ck={Ck} Cv={Cv} EB={EB} KV={nbytes/1e9:.3f}GB/layer")
        print(f"  tail(full)={t_full*1e3:.1f}us  main eb1/2/4={mains[1]*1e3:.1f}/{mains[2]*1e3:.1f}/{mains[4]*1e3:.1f}us"
              f"  per-plane={(mains[4]-mains[1])/3*1e3:.1f}us")
        print(f"  BW floor={nbytes/bts.BW/1e6*1e3:.1f}us ({mains[4]/(nbytes/bts.BW/1e6):.2f}x)"
              f"  popc floor={(sc_popc+val_popc(4))/POPC_RATE*1e6:.1f}us"
              f"  achieved={(sc_popc+val_popc(4))/t4/1e12:.2f}e12 popcll/s"
              f" ({(sc_popc+val_popc(4))/t4/POPC_RATE*100:.0f}% of ISA peak)")
        print(f"  x32 layers: tail={t_full*32:.2f}ms main={mains[4]*32:.2f}ms", flush=True)
        del be, fixed, bt, pk, pv
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
