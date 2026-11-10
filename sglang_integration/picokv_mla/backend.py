"""``picokv_mla``: fa3 MLA backend over the PiCoKV latent pool (``pool.py``).

Extend (path A): the batch's latent rows are filled into a bf16 scratch and stock fa3 runs on it. Per extend the
backend builds a flat plan of the batch's KV slots (request-major, CSR by ``cu_seqlens_k``) and points the fa3 page
table at the scratch rows (``page_size`` 1 inside fa3; the allocator keeps pages of 4).

Decode (path B, ``decode_i8.py``; ``SGLANG_PICOKV_MLA_PATH_B=0`` keeps path A): stock fa3 over the dense rows (sink +
tail) with lse, attention over the codes of the compressed rows in int8 code space, merged per head. Runs in a full
decode CUDA graph: the host part (slide, checks) runs before the replay, the plan is captured.

After the extend forward that reaches a request's fire point F (``picokv_mla_fire_point``), positions
``[SINK, F - LOCAL)`` of every layer are compressed; before each decode forward, rows that left the local window are
compressed in whole C-row chunks. Rope, sink, local window and positions >= F stay exact. The compressed end of each
request is kept per req pool slot (host mirror + device copy read by the decode plan).

Radix (``SGLANG_PICOKV_MLA_RADIX=1``, ``radix.py``): before an extend the backend copies the private rows of new hits,
fires a hit whose prefix already reaches its fire point and marks [SINK, c) of hits before their fire point to be read
exact from the host mirror; after the forward it unmarks."""

from __future__ import annotations

import logging

import numpy as np
import torch
import triton

from sglang.kernels.ops.kvcache.kv_indices import create_flashinfer_kv_indices_triton
from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import SharedReadEnds
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.layers.attention.picokv_mla import (
    CHUNK,
    DW,
    LOCAL,
    PATH_B,
    SINK,
    picokv_mla_radix_on,
    picokv_mla_ws_rows,
)
from sglang.srt.layers.attention.picokv_mla.radix import _ADM, _BACKEND
from sglang.srt.layers.attention.picokv_mla.pool import UNITS_PER_ROW, PicoKVMLATokenToKVPool
from sglang.srt.runtime_context import get_memory, get_schedule

logger = logging.getLogger(__name__)


def picokv_mla_fire_point(req) -> int:
    """F: the logprob start of a teacher-forced request, else the prompt length."""
    n = len(req.origin_input_ids)
    s = getattr(req, "logprob_start_len", -1)
    return s if getattr(req, "return_logprob", False) and 0 < s < n else n


def _cpu(x):
    return None if x is None else (x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)).astype(np.int64)


class PicoKVMLABackend(FlashAttentionBackend):
    def __init__(self, model_runner, *args, **kwargs):
        radix = picokv_mla_radix_on()
        if not get_memory().disable_radix_cache and not radix:
            # a stock radix hit would re-fire on compressed rows
            raise ValueError("picokv_mla needs --disable-radix-cache or SGLANG_PICOKV_MLA_RADIX=1")
        if PATH_B and envs.SGLANG_ENABLE_METADATA_GLUE_GRAPH.get():
            # the glue graph freezes the replay prep, which runs the slide and reads host counts
            raise ValueError("picokv_mla path B does not support SGLANG_ENABLE_METADATA_GLUE_GRAPH")
        if PATH_B and CHUNK <= 0:
            raise ValueError("picokv_mla path B needs SGLANG_PICOKV_MLA_CHUNK > 0 (the slide bounds the dense rows)")
        super().__init__(model_runner, *args, **kwargs)
        pool = model_runner.token_to_kv_pool
        assert isinstance(pool, PicoKVMLATokenToKVPool), type(pool)
        assert self.page_size == UNITS_PER_ROW, self.page_size
        assert not (self.kv_index_translator is not None and self.kv_index_translator.is_translating)
        self.pool = pool
        if not get_schedule().disable_overlap_schedule:
            # the scheduler's WAR barrier orders its allocs / frees after this forward's reads (``shared_read_ends``);
            # only the invariant checks would sync the host in the forward path
            pool.defer_checks()
        # fa3 reads the scratch row by row
        self.page_size = 1
        self.max_num_pages = self.max_context_len
        # compressed end per req pool slot; -1 = no fire since the slot's request started
        n = self.req_to_token.shape[0]
        self._ce = np.full((n,), -1, dtype=np.int64)
        self._ce_dev = torch.full((n,), -1, dtype=torch.int32, device=self.device)
        # rid whose first extend ran in each req pool slot (radix.py trusts _ce of that request only)
        self._rid = [None] * n
        self._radix = radix
        self._late = 0  # fires of an extend that ran past the fire point
        if radix:
            _BACKEND[0] = self
        self._ws = None
        if PATH_B:
            from sglang.srt.layers.attention.picokv_mla import decode_i8

            assert decode_i8.SINK == SINK, (decode_i8.SINK, SINK)
            self._i8 = decode_i8
            self._ws = decode_i8.Pki8Workspace(max(picokv_mla_ws_rows(), 1), DW, self.device)

    # -- compressed-end bookkeeping --
    def shared_read_ends(self, fm):
        # The scheduler stream writes the pool itself (codes and slot2dense when a request finishes or fires, the private
        # rows of a radix hit, slot2dense on alloc / free) while the decode kernels read codes, dense rows and
        # req_to_token: the base IN_REPLAY end (after the plan) let those writes race the replay (garbage outputs after
        # radix hits under the overlap scheduler). Fence on the whole replay.
        if fm.is_decode():
            return SharedReadEnds.POST_REPLAY
        return super().shared_read_ends(fm)

    def _set_ce(self, rpi, val):
        """ce[rpi] = val (numpy int64 arrays without duplicates), host mirror and device copy."""
        self._ce[rpi] = val
        dev = self._ce_dev.device
        i = torch.from_numpy(rpi).pin_memory().to(dev, non_blocking=True)
        v = torch.from_numpy(val.astype(np.int32)).pin_memory().to(dev, non_blocking=True)
        self._ce_dev[i] = v

    @staticmethod
    def _rpi_cpu(fb, n):
        x = getattr(fb, "req_pool_indices_cpu", None)
        return _cpu(x)[:n] if x is not None else _cpu(fb.req_pool_indices[:n].cpu())

    @staticmethod
    def _seq_cpu(fb, n):
        x = fb.seq_lens_cpu
        return _cpu(x)[:n] if x is not None else _cpu(fb.seq_lens[:n].cpu())

    def _slide(self, fb):
        """Before a decode forward: compress whole CHUNK-row runs that left the local window; with path B also check
        that every row fired and keeps at most DW dense rows."""
        n = fb.batch_size - (getattr(fb, "num_padding", 0) or 0)
        if n <= 0:
            return
        rpi, seq = self._rpi_cpu(fb, n), self._seq_cpu(fb, n)
        ce = self._ce[rpi]
        if bool((ce < 0).any()):
            if PATH_B:
                raise RuntimeError(f"picokv_mla: {int((ce < 0).sum())} decode rows without a fire")
            return
        if CHUNK > 0:
            k = np.maximum((seq - 1 - LOCAL - ce) // CHUNK * CHUNK, 0)  # rows [0, seq - 1) are written
            sel = np.nonzero(k)[0]
            if sel.size:
                ks = k[sel]
                tot = int(ks.sum())
                st = np.repeat(np.cumsum(ks) - ks - ce[sel], ks)
                pos = np.arange(tot, dtype=np.int64) - st
                ii = np.repeat(rpi[sel], ks)
                dev = self.req_to_token.device
                ii_t = torch.from_numpy(ii).pin_memory().to(dev, non_blocking=True)
                pos_t = torch.from_numpy(pos).pin_memory().to(dev, non_blocking=True)
                self.pool.compress(self.req_to_token[ii_t, pos_t].long())
                ce = ce + k
                self._set_ce(rpi[sel], ce[sel])
        if PATH_B:
            nd = seq - np.maximum(np.minimum(ce, seq - 1) - SINK, 0)
            if bool((nd > DW).any()):
                raise RuntimeError(f"picokv_mla: decode row with {int(nd.max())} dense rows > {DW}")

    def after_forward(self, forward_batch):
        """Called by ModelRunner after the forward: fire for requests whose fire point this extend reached."""
        self.pool.end_forward()
        self.pool.unmark_x()
        fires = forward_batch.pk_fire_points
        fm = forward_batch.forward_mode
        if not fires or not fm.is_extend() or fm.is_decode_or_idle():
            return
        nb = len(fires)
        seqs = _cpu(forward_batch.seq_lens_cpu)
        pres = np.asarray([int(x) for x in forward_batch.extend_prefix_lens_cpu], dtype=np.int64)
        rpi = self._rpi_cpu(forward_batch, nb)
        rids = forward_batch.rids
        adm = _ADM if self._radix else {}
        r2t = self.req_to_token
        parts, ri, rv = [], [], []
        for i, f in enumerate(fires):
            a = adm.get(rids[i])
            # first chunk of a request: a miss starts at 0, a hit's record says; a hit fires from its cut c on
            first = a["first"] if a is not None else pres[i] == 0
            lo = max(SINK, a["c"]) if a is not None else SINK
            if a is not None:
                a["first"] = False
            if first:
                self._rid[int(rpi[i])] = rids[i]
            if pres[i] < f <= int(seqs[i]):
                if f < int(seqs[i]):
                    # tokens [f, seq) of this extend read [SINK, f - LOCAL) exact (FIRE_CAP off / no chunked prefill)
                    self._late += 1
                    if self._late & (self._late - 1) == 0:
                        logger.warning("picokv_mla: %d fires after tokens past the fire point", self._late)
                if f - LOCAL > lo:
                    parts.append(r2t[int(rpi[i]), lo:f - LOCAL])
                ri.append(i)
                rv.append(max(SINK, f - LOCAL))
            elif first and pres[i] < f:
                # first chunk before the fire point
                ri.append(i)
                rv.append(-1)
        if parts:
            self.pool.compress(torch.cat(parts).long())
        if ri:
            self._set_ce(rpi[ri], np.asarray(rv, dtype=np.int64))

    def _radix_pre(self, fb):
        """Before an extend forward with radix hits (radix.py): the private-row copies of new hits, the fire of a hit
        whose prefix already reaches its fire point, exact reads of [SINK, c) for hits before their fire point."""
        if not _ADM or not fb.rids:
            return
        rids, fires, pres = fb.rids, fb.pk_fire_points, fb.extend_prefix_lens_cpu
        rpi = None
        r2t = self.req_to_token
        src, dst, comp, marks, ri, rv = [], [], [], [], [], []
        for i, rid in enumerate(rids):
            a = _ADM.get(rid)
            if a is None:
                continue
            if rpi is None:
                rpi = self._rpi_cpu(fb, len(rids))
            p, f, c, r = int(pres[i]), fires[i], a["c"], int(rpi[i])
            if a["copy"] is not None:
                src.append(a["copy"][0])
                dst.append(a["copy"][1])
                a["copy"] = None
            if p >= f:
                if a["first"]:
                    lo = max(SINK, c)
                    if f - LOCAL > lo:
                        comp.append(r2t[r, lo:f - LOCAL])
                    ri.append(r)
                    rv.append(max(SINK, f - LOCAL))
            elif c > SINK:
                marks.append(r2t[r, SINK:c])
        if src:
            self.pool.priv_copy(torch.cat(src), torch.cat(dst))
        if comp:
            self.pool.compress(torch.cat(comp).long())
        if ri:
            self._set_ce(np.asarray(ri, dtype=np.int64), np.asarray(rv, dtype=np.int64))
        if marks:
            self.pool.mark_x(torch.cat(marks).long())

    # -- forward metadata --
    def init_forward_metadata(self, forward_batch):
        fm = forward_batch.forward_mode
        if PATH_B and fm.is_decode():
            self._slide(forward_batch)
            self._decode_plan(forward_batch)
            return
        if fm.is_decode():
            self._slide(forward_batch)
        elif self._radix and fm.is_extend() and not fm.is_decode_or_idle():
            self._radix_pre(forward_batch)
        super().init_forward_metadata(forward_batch)
        if fm.is_idle():
            return
        m = self.forward_metadata
        n = int(forward_batch.seq_lens_cpu.sum())
        bs = forward_batch.batch_size
        plan = torch.empty((max(n, 1),), dtype=torch.int32, device=self.device)
        nb = max(1, min(32, triton.cdiv(int(forward_batch.seq_lens_cpu.max()), 4096)))
        create_flashinfer_kv_indices_triton[(bs, nb)](
            self.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            m.cu_seqlens_k,
            None,
            plan,
            self.req_to_token.stride(0),
            TOKEN_BLOCK_PARALLEL=nb > 1,
        )
        width = m.page_table.shape[1]
        m.page_table = torch.clamp_max(
            m.cu_seqlens_k[:-1, None] + torch.arange(width, dtype=torch.int32, device=self.device)[None, :],
            max(n - 1, 0),
        ).to(torch.int32)
        loc = forward_batch.out_cache_loc
        rows = self.pool.slot2dense[loc.long()]
        self.pool._check_rows(rows, loc)
        self.pool.begin_forward(plan, n, loc, rows.to(loc.dtype))

    def _decode_plan(self, fb):
        """Path B decode plan (device ops only: eager, or recorded in the decode graph)."""
        self.forward_metadata = None
        loc = fb.out_cache_loc
        # the write rows of this forward's tokens (set_mla_kv_buffer reuses them for this loc tensor)
        self.pool.begin_forward(None, 0, loc, self.pool.slot2dense[loc.long()].to(loc.dtype))
        self._i8.pki8_plan(self._ws, self._ce_dev, fb.seq_lens, fb.req_pool_indices, self.req_to_token,
                           self.pool.slot2dense, fb.batch_size)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        if not PATH_B:
            raise NotImplementedError("picokv_mla path A runs without cuda graphs (SGLANG_PICOKV_MLA_PATH_B=1)")
        if max_bs > self._ws.max_bs:
            self._ws = self._i8.Pki8Workspace(max_bs, DW, self.device)

    def init_forward_metadata_out_graph(self, forward_batch, in_capture: bool = False):
        if not (PATH_B and forward_batch.forward_mode.is_decode()):
            raise NotImplementedError(f"picokv_mla graphs: decode only, got {forward_batch.forward_mode}")
        if not in_capture:
            self._slide(forward_batch)

    def init_forward_metadata_in_graph(self, forward_batch):
        self._decode_plan(forward_batch)

    def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True, q_rope=None, k_rope=None, **kwargs):
        if not PATH_B:
            return super().forward_decode(q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, q_rope=q_rope,
                                          k_rope=k_rope, **kwargs)
        i8 = self._i8
        h, dv = layer.tp_q_head_num, layer.v_head_dim
        assert h == i8.H and dv == i8.NOPE and layer.head_dim == i8.NOPE + i8.ROPE and not layer.logit_cap, (
            h, dv, layer.head_dim, layer.logit_cap)
        if k is not None and save_kv_cache:
            self.token_to_kv_pool.set_mla_kv_buffer(layer, forward_batch.out_cache_loc, k, k_rope)
        if q_rope is not None:
            q_nope = q.view(-1, h, dv)
            q_rope = q_rope.view(-1, h, i8.ROPE)
        else:
            q_all = q.contiguous().view(-1, h, layer.head_dim)
            q_nope, q_rope = q_all[:, :, :dv], q_all[:, :, dv:]
        b = q_nope.shape[0]
        out = torch.empty((b, h, dv), dtype=torch.bfloat16, device=q.device)
        l = layer.layer_id - self.pool.start_layer
        pool = self.pool
        i8.pki8_decode(self._ws, q_nope, q_rope, pool._arena[l], pool._dense[l], pool.dec[l], self.req_to_token, b,
                       layer.scaling, out)
        return out.view(-1, h * dv)
