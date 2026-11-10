"""``picokv_mla`` path B decode (int8 code space): attention over the PiCoKV latent pool without a bf16 scratch.

Per request the compressed rows are the positions ``[SINK, SINK + nc)``, ``nc = max(min(ce, L - 1) - SINK, 0)``; the
dense rows are the sink ``[0, min(SINK, L))`` and the tail ``[SINK + nc, L)``. Per layer
(``picokv_mla_sm90/decode_i8.cuh``):

* dense: stock fa3 over the dense rows (``pki8_plan`` builds their page table once per forward), with lse;
* qproj: ``q' = q_nope @ dec^T`` quantised per head to int8 hi / lo terms (the q record);
* attn: int8 score per compressed entry, ``P = fp16 2^(s - R)``, ``Y = sum P c`` per (CTA, row) segment;
* combine: segments -> ``A'' = (2 y - L) / L`` (fp16), lse_c; merge: ``o_c = A'' @ dec`` merged with the dense part.

Every grid depends only on the batch size and every count lives on the device, so the launches can be captured in a
CUDA graph."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.ops.attention.flash_attention import flash_attn_with_kvcache

H = 20  # q heads (GLM-4.7-Flash)
HN = 24
BITS = 1280
NOPE = 512
ROPE = 64
SINK = 64
UE = 128  # compressed entries per unit
QREC_B = 10 * 40 * 128 + HN * 128 + 2 * HN * 4  # q record per row (decode_i8.cuh)
MERGE_KS = 0  # merge split-K ranks; 0 = by the row count
DENSE_SPLITS = 0  # fa3 num_splits of the dense part (sink + tail); 0 = 1 at B >= 32 (measured), else fa3's


@cache_once
def _module():
    return load_jit(
        "picokv_mla_i8",
        cuda_files=["picokv_mla_sm90/decode_i8.cuh"],
        cuda_wrappers=[
            ("qproj", "sglang::pki8_qproj_dispatch"),
            ("attn", "sglang::pki8_attn_dispatch"),
            ("combine", "sglang::pki8_combine_dispatch"),
            ("merge", "sglang::pki8_merge_dispatch"),
        ],
        extra_cuda_cflags=["-DNDEBUG", "--use_fast_math"],
    )


@triton.jit
def _pki8_plan_kernel(seq_ptr, ce_ptr, rpi_ptr, plan_ptr, B, SINK: tl.constexpr, UE: tl.constexpr,
                      BP: tl.constexpr):
    i = tl.arange(0, BP)
    m = i < B
    L = tl.load(seq_ptr + i, mask=m, other=1).to(tl.int32)
    rpi = tl.load(rpi_ptr + i, mask=m, other=0).to(tl.int32)
    ce = tl.load(ce_ptr + rpi, mask=m, other=SINK)
    nc = tl.maximum(tl.minimum(ce, L - 1) - SINK, 0)  # the current token stays dense
    nu = (nc + UE - 1) // UE
    ct = tl.cumsum((nc + 63) // 64, 0)
    cu = tl.cumsum(nu, 0)
    tl.store(plan_ptr + i + 1, ct, mask=m)
    tl.store(plan_ptr + (B + 1) + i + 1, cu, mask=m)
    tl.store(plan_ptr + 2 * (B + 1) + i, nc, mask=m)
    tl.store(plan_ptr + 3 * (B + 1) + i, rpi, mask=m)
    z = i == 0
    tl.store(plan_ptr + i, tl.zeros([BP], tl.int32), mask=z)
    tl.store(plan_ptr + (B + 1) + i, tl.zeros([BP], tl.int32), mask=z)


@triton.jit
def _pki8_dense_pt_kernel(seq_ptr, ce_ptr, rpi_ptr, r2t_ptr, r2t_ld, s2d_ptr, dpt_ptr, dlen_ptr, DW,
                          SINK: tl.constexpr, BLK: tl.constexpr):
    """Dense page table of row b: the dense rows of positions [0, min(SINK, L)) + [SINK + nc, L) (page 1)."""
    b = tl.program_id(0)
    j = tl.program_id(1) * BLK + tl.arange(0, BLK)
    L = tl.load(seq_ptr + b).to(tl.int32)
    rpi = tl.load(rpi_ptr + b).to(tl.int64)
    ce = tl.load(ce_ptr + rpi)
    nc = tl.maximum(tl.minimum(ce, L - 1) - SINK, 0)
    nd = L - nc
    live = j < nd
    pos = tl.where(j < tl.minimum(L, SINK), j, j + nc)
    slot = tl.load(r2t_ptr + rpi * r2t_ld + pos, mask=live, other=0)
    row = tl.maximum(tl.load(s2d_ptr + slot, mask=live, other=0), 0)
    tl.store(dpt_ptr + b * DW + j, tl.where(live, row, 0), mask=j < DW)
    if tl.program_id(1) == 0:
        tl.store(dlen_ptr + b, tl.minimum(nd, DW))


def _sm_count(dev) -> int:
    return torch.cuda.get_device_properties(dev).multi_processor_count


def _ws_layout(max_bs: int, dw: int, sms: int):
    return [("plan", (4, max_bs + 1), torch.int32), ("dpt", (max_bs, dw), torch.int32),
            ("dlen", (max_bs,), torch.int32), ("cuq", (max_bs + 1,), torch.int32),
            ("qrec", (max_bs, QREC_B), torch.uint8), ("yp", (sms + max_bs, H, BITS), torch.float32),
            ("mlp", (sms + max_bs, 2, HN), torch.float32), ("a2", (max_bs, H, BITS), torch.float16),
            ("lsec", (max_bs, H), torch.float32)]


def _ws_offsets(max_bs: int, dw: int, sms: int):
    off, out = 0, []
    for name, shape, dt in _ws_layout(max_bs, dw, sms):
        out.append((name, shape, dt, off))
        off += -(-math.prod(shape) * dt.itemsize // 256) * 256
    return out, off


class Pki8Workspace:
    """Fixed buffers of the path B decode for batches up to ``max_bs`` rows with at most ``dw`` dense rows each, carved
    from one uint8 allocation (``bytes_for``)."""

    def __init__(self, max_bs: int, dw: int, device):
        self.device = torch.device(device)
        self.G = _sm_count(self.device)
        self.max_bs, self.dw = max_bs, dw
        lay, n = _ws_offsets(max_bs, dw, self.G)
        self.buf = torch.zeros((n,), dtype=torch.uint8, device=self.device)
        for name, shape, dt, off in lay:
            setattr(self, name, self.buf[off:off + math.prod(shape) * dt.itemsize].view(dt).view(shape))
        self.dlen.fill_(1)
        self.cuq.copy_(torch.arange(max_bs + 1, dtype=torch.int32, device=self.device))

    def nbytes(self) -> int:
        return self.buf.numel()

    @staticmethod
    def bytes_for(max_bs: int, dw: int, sms: int) -> int:
        return _ws_offsets(max_bs, dw, sms)[1]


def pki8_plan(ws: Pki8Workspace, ce, seq_lens, req_pool_indices, r2t, s2d, B: int):
    """The per-forward plan of ``B`` rows from seq_lens, the req pool indices, ``ce`` (int32 compressed end per req pool
    slot), req_to_token and slot2dense (device tensors; graph-safe)."""
    assert B <= ws.max_bs, (B, ws.max_bs)
    _pki8_plan_kernel[(1,)](seq_lens, ce, req_pool_indices, ws.plan, B, SINK=SINK, UE=UE,
                            BP=max(16, triton.next_power_of_2(B)))
    _pki8_dense_pt_kernel[(B, triton.cdiv(ws.dw, 128))](seq_lens, ce, req_pool_indices, r2t, r2t.stride(0), s2d,
                                                        ws.dpt, ws.dlen, ws.dw, SINK=SINK, BLK=128)


def pki8_decode(ws: Pki8Workspace, q_nope, q_rope, arena, dense, dec, r2t, B: int, scale: float, out,
                pdl: bool = True):
    """Absorbed MLA decode of one layer over the PiCoKV pool for the plan in ``ws`` (``pki8_plan``).
    q_nope [B, H, 512], q_rope [B, H, 64] (bf16, last dim contiguous); arena: the layer's uint8 arena; dense: its
    [rows, 576] bf16 view; dec [1280, 512] bf16; out [B, H, 512] bf16."""
    mod = _module()
    stream = torch.cuda.current_stream(q_nope.device).cuda_stream
    od, lse = flash_attn_with_kvcache(
        q=q_rope, k_cache=dense[:, NOPE:].view(-1, 1, 1, ROPE), v_cache=dense[:, :NOPE].view(-1, 1, 1, NOPE),
        qv=q_nope, page_table=ws.dpt[:B], cache_seqlens=ws.dlen[:B], cu_seqlens_q=ws.cuq[:B + 1], max_seqlen_q=1,
        softmax_scale=scale, causal=True, num_splits=DENSE_SPLITS or (1 if B >= 32 else 0), return_softmax_lse=True)[:2]
    od = od.view(B, H, NOPE)
    p = int(pdl)
    mod.qproj(q_nope, q_rope, dec, ws.qrec, B, q_nope.stride(0), q_nope.stride(1), q_rope.stride(0),
              q_rope.stride(1), p, stream)
    mod.attn(ws.qrec, arena, r2t, ws.plan, ws.yp, ws.mlp, B, r2t.stride(0), scale * math.log2(math.e), ws.G, p,
             stream)
    mod.combine(ws.plan, ws.yp, ws.mlp, ws.a2, ws.lsec, B, ws.G, p, stream)
    mod.merge(ws.a2, ws.lsec, dec, od, lse, out, B, od.stride(0), od.stride(1), lse.stride(0), lse.stride(1),
              out.stride(0), out.stride(1), MERGE_KS, p, stream)
    return out
