"""PiCoKV MLA token pool and allocator (design B: one page-4 arena per layer).

Every allocator page holds either 4 token slots (288 B each per layer: 40 code words + exact rope 64) or one dense
bf16 latent row (576 elements = 1152 B), so dense row p is page p. A token gets its slot from the usual paged
allocation plus one more page for its dense row; ``slot2dense[slot]`` names that row, -1 once the token is compressed
(the row page goes back to the free list). Slots 0..3 map to row 0, the padding row.

Reads go through a fill kernel: ``get_mla_kv_buffer`` (MHA prefix reads) copies dense rows and decodes compressed
ones; ``get_key_buffer`` (fa3 absorbed MLA) fills a shared bf16 scratch in the order of the backend's flat plan
(``backend.py`` rewrites the fa3 page table to index it).

Radix (``radix.py``): every compress also writes the rows' exact nope to a pinned host mirror (one tensor per layer,
indexed by slot); slots the backend marks -2 in ``slot2dense`` are read exact from it (nope; rope from the slot)."""

from __future__ import annotations

import hashlib
import logging
import os
import time
from contextlib import nullcontext

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.layers.attention.picokv_mla import DW, PATH_B, picokv_mla_radix_on, picokv_mla_ws_rows
from sglang.srt.layers.attention.picokv_mla.codec import (
    BITS,
    NOPE,
    ROPE,
    ROPE_OFF,
    UNIT,
    UNITS_PER_ROW,
    pk_encode_layers,
    pk_fill,
    pk_mirror_layers,
    pk_priv_layers,
)
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.utils import get_num_new_pages

logger = logging.getLogger(__name__)

ROW = NOPE + ROPE
# pages kept back from the scheduler-visible count: page rounding of up to this many requests per admission
RESERVE_PAGES = 256
CHECK = os.environ.get("SGLANG_PICOKV_MLA_CHECK", "1") == "1"


def picokv_mla_cell_size_bytes(layers: int) -> int:
    """Bytes per pool slot: the coded slot of every layer, one scratch row and its ``slot2dense`` entry."""
    return UNIT * layers + ROW * 2 + 4


def picokv_mla_fixed_bytes(layers: int) -> int:
    """Pool bytes independent of its size: the bf16 enc/dec projections plus the 2 MiB rounding of each of the five
    allocations (arena, scratch, slot2dense, enc, dec); with path B also the decode workspace (one allocation)."""
    n = 2 * layers * BITS * NOPE * 2 + 5 * (2 << 20)
    if PATH_B:
        from sglang.srt.layers.attention.picokv_mla.decode_i8 import Pki8Workspace

        sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        n += Pki8Workspace.bytes_for(picokv_mla_ws_rows(), DW, sms) + (2 << 20)
    return n


class PicoKVMLATokenToKVPool(MLATokenToKVPool):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._alloc = None
        self._plan = None
        self._plan_n = 0
        self._plan_ver = 0
        self._filled = [-1] * self.layer_num
        self._wloc = None
        self._wrows = None
        self._nbad = None
        self._xmark = None
        self._mirror = None
        self._load_proj()
        if picokv_mla_radix_on():
            t = time.time()
            n = self.num_rows * self.page_size
            self._mirror = [torch.empty((n, NOPE), dtype=torch.bfloat16, pin_memory=True) for _ in range(self.layer_num)]
            logger.info("picokv_mla radix: host mirror %d layers x %d slots (%.2f GiB pinned) in %.1f s",
                        self.layer_num, n, self.layer_num * n * NOPE * 2 / 2**30, time.time() - t)

    # -- storage --
    def _create_buffers(self):
        assert self.page_size == UNITS_PER_ROW, f"picokv_mla needs page_size {UNITS_PER_ROW}, got {self.page_size}"
        assert self.kv_cache_dim == ROW and self.store_dtype == self.dtype == torch.bfloat16, (
            "picokv_mla needs a bf16 512+64 MLA latent pool")
        self.num_rows = self.size // self.page_size + 1  # pages 0..num_pages, page 0 = padding
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (torch.cuda.use_mem_pool(self.custom_mem_pool) if self.custom_mem_pool else nullcontext()):
                # one allocation for all layers (per-layer tensors would each round up to 2 MiB)
                self._arena_all = torch.zeros(
                    (self.layer_num, self.num_rows * ROW * 2), dtype=torch.uint8, device=self.device)
                self._arena = list(self._arena_all.unbind(0))
                self.scratch = torch.zeros((self.size, 1, ROW), dtype=torch.bfloat16, device=self.device)
                self.slot2dense = torch.full(
                    (self.num_rows * self.page_size,), -1, dtype=torch.int32, device=self.device)
        self.slot2dense[: self.page_size] = 0
        self.kv_buffer = [a.view(torch.bfloat16).view(-1, 1, ROW) for a in self._arena]
        self._dense = [a.view(torch.bfloat16).view(-1, ROW) for a in self._arena]
        self._codes = [a.view(torch.int32) for a in self._arena]
        self._rope = [a.view(torch.bfloat16)[ROPE_OFF // 2:] for a in self._arena]
        self._scratch_rope = self.scratch.view(-1)[NOPE:]

    def _clear_buffers(self):
        del self.kv_buffer, self._arena_all, self._arena, self._dense, self._codes, self._rope
        del self.scratch, self._scratch_rope

    def get_kv_size_bytes(self):
        return self._arena_all.numel() + self.scratch.numel() * 2 + self.slot2dense.numel() * 4

    def _load_proj(self):
        d = os.environ.get("SGLANG_PICOKV_MLA_PROJ")
        assert d, "picokv_mla: set SGLANG_PICOKV_MLA_PROJ to the dir of proj_L{layer}_b1280.pt"
        enc, dec, sha = [], [], hashlib.sha256()
        for lid in range(self.start_layer, self.start_layer + self.layer_num):
            path = os.path.join(d, f"proj_L{lid}_b{BITS}.pt")
            raw = open(path, "rb").read()
            sha.update(raw)
            if lid == self.start_layer:
                sha0 = hashlib.sha256(raw).hexdigest()[:16]
            p = torch.load(path, weights_only=False, map_location="cpu")
            enc.append(p["enc"][0])
            dec.append(p["dec"][0])
        self.enc = torch.stack(enc).to(self.device, torch.bfloat16).contiguous()
        self.dec = torch.stack(dec).to(self.device, torch.bfloat16).contiguous()
        assert self.enc.shape == self.dec.shape == (self.layer_num, BITS, NOPE), self.enc.shape
        logger.info("picokv_mla: %d slots (%d rows), proj %s sha0 %s all %s", self.size, self.num_rows, d, sha0,
                    sha.hexdigest()[:16])

    # -- per-forward plan (backend.py) --
    def begin_forward(self, plan, n, loc, rows):
        """plan: int32 [n] slots in flat (request, position) order; loc/rows: this forward's write slots and their
        dense rows (``set_mla_kv_buffer`` reuses rows when it gets the same loc tensor)."""
        self._plan, self._plan_n = plan, n
        self._plan_ver += 1
        self._wloc, self._wrows = loc, rows

    def end_forward(self):
        self._plan = self._wloc = self._wrows = None

    def defer_checks(self):
        """Overlap schedule: count invariant violations on the device and read the counts back asynchronously (a
        violation raises a few forwards late) instead of syncing the host in the forward path."""
        self._nbad = torch.zeros((3,), dtype=torch.int64, device=self.device)
        self._nbad_host = torch.zeros((3,), dtype=torch.int64).pin_memory()
        self._nbad_ev = None

    def _flag(self, bad, k):
        what = ("write slots without a dense row", "compress of slots without a dense row",
                "radix prefix slots not compressed")
        if self._nbad is None:
            if bool(bad.any()):
                raise RuntimeError(f"picokv_mla: {int(bad.sum())} {what[k]}")
            return
        self._nbad[k:k + 1].add_(bad.sum())
        ev = self._nbad_ev
        if ev is not None and ev.query():
            if bool((self._nbad_host > 0).any()):
                raise RuntimeError(f"picokv_mla: {self._nbad_host.tolist()} {what}")
            ev = None
        if ev is None:
            self._nbad_host.copy_(self._nbad, non_blocking=True)
            self._nbad_ev = torch.cuda.Event()
            self._nbad_ev.record()

    def _check_rows(self, rows, loc):
        if CHECK:
            self._flag((rows < 0) | ((rows == 0) & (loc != 0)), 0)

    def _l(self, layer_id):
        return layer_id - self.start_layer

    def get_key_buffer(self, layer_id: int):
        """The plan's latent rows of this layer, filled into the shared scratch: [n, 1, 576]."""
        if self._plan is None:
            raise RuntimeError("picokv_mla: get_key_buffer outside a planned forward")
        l = self._l(layer_id)
        n = self._plan_n
        if self._filled[l] != self._plan_ver:
            pk_fill(self._plan, self.slot2dense, self._dense[l], self._codes[l], self._rope[l], self.dec[l],
                    self.scratch, ROW, self._scratch_rope, ROW, x=self._x(l))
            self._filled = [-1] * self.layer_num
            self._filled[l] = self._plan_ver
        return self.scratch[:n]

    def get_value_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id)[..., : self.kv_lora_rank]

    def get_mla_kv_buffer(self, layer, loc, dst_dtype=None):
        l = self._l(layer.layer_id)
        n = loc.shape[0]
        nope = torch.empty((n, 1, NOPE), dtype=torch.bfloat16, device=self.device)
        rope = torch.empty((n, 1, ROPE), dtype=torch.bfloat16, device=self.device)
        pk_fill(loc.contiguous(), self.slot2dense, self._dense[l], self._codes[l], self._rope[l], self.dec[l],
                nope, NOPE, rope, ROPE, x=self._x(l))
        if dst_dtype is not None and dst_dtype != torch.bfloat16:
            nope, rope = nope.to(dst_dtype), rope.to(dst_dtype)
        return nope, rope

    def set_mla_kv_buffer(self, layer, loc, cache_k_nope, cache_k_rope, layer_id_override=None):
        layer_id = layer_id_override if layer_id_override is not None else layer.layer_id
        if loc is self._wloc:
            rows = self._wrows
        else:
            rows = self.slot2dense[loc.long()]
            self._check_rows(rows, loc)
            rows = rows.to(loc.dtype)
        self._filled[self._l(layer_id)] = -1
        self._write_mla_kv_buffer(self.kv_buffer[self._l(layer_id)], rows, cache_k_nope, cache_k_rope)

    def set_kv_buffer(self, *args, **kwargs):
        raise NotImplementedError("picokv_mla writes go through set_mla_kv_buffer")

    def compress(self, slots):
        """Encode the dense rows of int64 ``slots`` (every layer) into their coded slots and free the rows."""
        if slots.numel() == 0:
            return
        rows = self.slot2dense[slots]
        if CHECK:
            self._flag(rows <= 0, 1)
        if self._mirror is not None:
            pk_mirror_layers(self._dense, rows, self._mirror, slots)
        pk_encode_layers(self._dense, rows, self._codes, self._rope, self.enc, self.dec, slots)
        self.slot2dense[slots] = -1
        self._filled = [-1] * self.layer_num
        self._alloc.release_dense(rows)

    # -- radix (radix.py, backend.py) --
    def _x(self, l):
        return self._mirror[l] if self._xmark is not None else None

    def priv_copy(self, src, dst):
        """Exact latent of slots ``src`` (tree slots: host mirror nope + slot rope) into the dense rows of the
        request's private slots ``dst``, every layer."""
        pk_priv_layers(src, dst, self.slot2dense, self._dense, self._rope, self._mirror)
        self._filled = [-1] * self.layer_num

    def mark_x(self, slots):
        """For this forward, read the compressed tree slots ``slots`` (int64) exact from the mirror."""
        if CHECK:
            self._flag(self.slot2dense[slots] != -1, 2)
        self.slot2dense[slots] = -2
        self._xmark = slots
        self._filled = [-1] * self.layer_num

    def unmark_x(self):
        if self._xmark is not None:
            self.slot2dense[self._xmark] = -1
            self._xmark = None
            self._filled = [-1] * self.layer_num

    def _unsupported(self, *args, **kwargs):
        raise NotImplementedError("picokv_mla: KV transfer / offload / spec relocation are not supported")

    move_kv_cache = get_cpu_copy = load_cpu_copy = get_contiguous_buf_infos = _unsupported


class PicoKVMLAPagedAllocator(PagedTokenToKVPoolAllocator):
    """Paged allocator whose pages also serve as dense rows: every allocated token slot takes one more free page for
    its dense row (``slot2dense``), released when the token is compressed or freed. Scheduler-visible capacity counts
    a dense token as 5 slots (its slot + a 4-slot row), so ``available_size`` is ``free_pages * 4 // 5`` less a
    reserve for page rounding."""

    def __init__(self, size, page_size, dtype, device, kvcache, need_sort):
        assert page_size == UNITS_PER_ROW and isinstance(kvcache, PicoKVMLATokenToKVPool)
        self.s2d = kvcache.slot2dense
        super().__init__(size, page_size, dtype, device, kvcache, need_sort)
        kvcache._alloc = self

    def clear(self):
        super().clear()
        self.s2d.fill_(-1)
        self.s2d[: self.page_size] = 0

    def _free_pages_now(self):
        return len(self.free_pages) + self.num_staged_pages

    def available_size(self):
        return max(0, self._free_pages_now() - RESERVE_PAGES) * self.page_size // (self.page_size + 1)

    def schedulable_tokens(self):
        return max(0, self.num_pages - RESERVE_PAGES) * self.page_size // (self.page_size + 1)

    def _ensure(self, num_pages):
        if num_pages > len(self.free_pages):
            self.merge_and_sort_free()
        return num_pages <= len(self.free_pages)

    def _set_dense(self, out):
        if out is None or out.numel() == 0:
            return out
        n = out.numel()
        rows = self.free_pages[:n]
        self.free_pages = self.free_pages[n:]
        self.s2d[out] = rows.to(torch.int32)
        return out

    def alloc(self, need_size: int):
        if not self._ensure(need_size // self.page_size + need_size):
            return None
        return self._set_dense(super().alloc(need_size))

    def alloc_extend(self, prefix_lens, prefix_lens_cpu, seq_lens, seq_lens_cpu, last_loc, extend_num_tokens,
                     num_new_pages=None):
        if num_new_pages is None:
            num_new_pages = get_num_new_pages(seq_lens=seq_lens_cpu, page_size=self.page_size,
                                              prefix_lens=prefix_lens_cpu)
        if not self._ensure(num_new_pages + extend_num_tokens):
            return None
        out = super().alloc_extend(prefix_lens, prefix_lens_cpu, seq_lens, seq_lens_cpu, last_loc,
                                   extend_num_tokens, num_new_pages=num_new_pages)
        return self._set_dense(out)

    def alloc_decode(self, seq_lens, seq_lens_cpu, last_loc):
        nnp = get_num_new_pages(seq_lens=seq_lens_cpu, page_size=self.page_size, decode=True)
        if not self._ensure(nnp + len(seq_lens)):
            return None
        return self._set_dense(super().alloc_decode(seq_lens, seq_lens_cpu, last_loc))

    def _drop_dense(self, pages):
        """Release the dense rows still held by the slots of these slot pages."""
        slots = (pages.long()[:, None] * self.page_size + torch.arange(self.page_size, device=self.device)).reshape(-1)
        d = self.s2d[slots]
        d = d[d > 0]
        self.s2d[slots] = -1
        self.s2d[: self.page_size] = 0
        if d.numel():
            self._release_page_ids(d.long())

    def release_dense(self, rows):
        self._release_page_ids(rows.long())

    def free(self, free_index: torch.Tensor):
        # in a free group the slot pages wait for free_group_end (which calls free again); their rows go now, and the
        # second call finds them already dropped
        if free_index.numel() == 0:
            return
        self._drop_dense(torch.unique(free_index // self.page_size))
        super().free(free_index)

    def free_page_ids(self, page_ids: torch.Tensor):
        if page_ids.numel() == 0:
            return
        self._drop_dense(page_ids)
        super().free_page_ids(page_ids)

    def free_page_reps(self, page_reps: torch.Tensor):
        if page_reps.numel() == 0:
            return
        self._drop_dense(page_reps // self.page_size)
        super().free_page_reps(page_reps)

    def check_decode_capacity(self, *, num_tokens, tree_cache, requests=None, spec_algorithm=None):
        self.evict_to_free_tokens(tree_cache, num_tokens)
        rows = self.page_size * (len(requests) if requests is not None else num_tokens)
        return self._free_pages_now() * self.page_size >= num_tokens + rows

    def get_cpu_copy(self, *args, **kwargs):
        raise NotImplementedError("picokv_mla: no CPU copy of the PiCoKV pool")

    load_cpu_copy = get_cpu_copy
