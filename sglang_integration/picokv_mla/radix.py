"""R-exact radix tree for picokv_mla (``SGLANG_PICOKV_MLA_RADIX=1`` with the radix cache on): SGLang's RadixCache over
the PiCoKV latent pool (``pool.py``).

Tree slots are always compressed (codes + exact rope) and the pool's host mirror holds their exact nope (every compress
writes it). A hit of prefix length m by a request with fire point F (cut c = page_floor(max(SINK, F - LOCAL)), at most
m) gets private slots with dense rows for positions [0, SINK) and [c, m), copied exactly from the tree slots (mirror
nope + slot rope) before its first extend; [SINK, c) stay the tree's slots. Until the fire the backend reads [SINK, c)
exact from the mirror (``slot2dense`` -2); the fire compresses [max(SINK, c), F - LOCAL) of the request's own rows;
after it [SINK, c) read their codes. A hit thus reads what the request reads without one and no compressed row is
fired again. At finish the request's own dense rows that the insert can take (sink, local window) are compressed
first. An unfinished request inserts its prompt up to page_floor(F - LOCAL) once it fired (its sink is dense in the tree
until it finishes, [SINK, F - LOCAL) is compressed); before the fire it inserts nothing, so a waiting request whose
prompt shares at least ``SGLANG_PICOKV_MLA_HOLD`` (default the chunked prefill size) more tokens with that insert than
its current match is held back until the insert (scheduling order only; ``SGLANG_PICOKV_MLA_HOLD_MAX_S`` bounds the
wait), else it prefills the shared prefix again.

Sizes: an allocator page holds 4 tree tokens or one dense row, so tree sizes are reported in ``available_size`` units
(u(P) = max(0, P - RESERVE_PAGES) * 4 // 5 for P free pages) and ``evict`` turns a shortfall in those units into tree
tokens."""

from __future__ import annotations

import logging
import os
import time

import torch

from sglang.srt.layers.attention.picokv_mla import LOCAL, SINK, picokv_mla_fire_point
from sglang.srt.layers.attention.picokv_mla.pool import RESERVE_PAGES, UNITS_PER_ROW
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey

logger = logging.getLogger(__name__)

P = UNITS_PER_ROW
# rid -> admission record of a hit (picokv_admit): c, m, private head slots, pending copy, first extend not run yet
_ADM = {}
# the PicoKVMLABackend of this process (compressed ends per req pool slot)
_BACKEND = [None]
# rid -> (tree prefix length before, end) of a fired request's insert; None: fired, nothing inserted
_FINS = {}
# hold-back owners: hash of the first h prompt tokens -> {rid: [req, admitted at, insert end, {waiter rid: shared}]}
_HOLD = {}
HOLD = int(os.environ.get("SGLANG_PICOKV_MLA_HOLD", "-1"))
HOLD_MAX_S = float(os.environ.get("SGLANG_PICOKV_MLA_HOLD_MAX_S", "60"))


def _u(pages: int) -> int:
    return max(0, pages - RESERVE_PAGES) * P // (P + 1)


def _lcp(a, b, n: int) -> int:
    """Length of the common prefix of ``a[:n]`` and ``b[:n]`` (lists)."""
    i = 0
    while i < n:
        j = min(i + 4096, n)
        if a[i:j] != b[i:j]:
            while i < j and a[i] == b[i]:
                i += 1
            return i
        i = j
    return n


def _hold_release(req) -> None:
    k = getattr(req, "_pk_hkey", None)
    owners = _HOLD.get(k) if k is not None else None
    if owners and owners.pop(req.rid, None) is not None and not owners:
        del _HOLD[k]


class PicoKVMLARadixCache(RadixCache):
    def __init__(self, params):
        super().__init__(params)
        assert self.page_size == P and not self.is_eagle, (self.page_size, self.is_eagle)
        cps = params.chunked_prefill_size
        self._hold_h = HOLD if HOLD >= 0 else (cps if cps and cps > 0 else 8192)
        self.hold_stats = {"held": 0, "passes": 0, "timeouts": 0}
        logger.info("picokv_mla radix: R-exact tree (sink %d, local %d, hold %d)", SINK, LOCAL, self._hold_h)

    def reset(self):
        super().reset()
        _ADM.clear()
        _FINS.clear()
        _HOLD.clear()

    # -- admission (schedule_policy add_one_req) --
    @staticmethod
    def _cut(req, m: int) -> int:
        return min(max(SINK, picokv_mla_fire_point(req) - LOCAL) // P * P, m)

    def picokv_charge(self, req) -> int:
        """Dense tokens of req's private rows for its matched prefix."""
        m = len(req.prefix_indices)
        if m == 0:
            return 0
        return m if m <= SINK else SINK + m - self._cut(req, m)

    def picokv_admit(self, req, n) -> None:
        """Allocate the private slots (evicting first like alloc_for_extend) and put them into req's prefix."""
        pi = req.prefix_indices
        m = len(pi)
        alloc = self.token_to_kv_pool_allocator
        if alloc.available_size() < n:
            self.evict(EvictParams(num_tokens=n - alloc.available_size()))
        idx = alloc.alloc(n)
        if idx is None:
            raise RuntimeError(f"picokv_mla radix: no {n} slots for the private rows of {req.rid}")
        idx = idx.to(pi.dtype)
        if m <= SINK:
            c, head, src, pre = 0, None, pi, idx
        else:
            c = self._cut(req, m)
            head, src = idx[:SINK], torch.cat([pi[:SINK], pi[c:]])
            pre = torch.cat([idx[:SINK], pi[SINK:c], idx[SINK:]])
        req.prefix_indices = pre
        req.kv.cache_protected_len = c
        _ADM[req.rid] = {"c": c, "m": m, "head": head, "copy": (src, idx), "first": True}

    # -- hold-back (schedule_policy add_one_req) --
    @staticmethod
    def _ins_len(req) -> int:
        """What the fire of req inserts: its prompt up to page_floor(max(SINK, F - LOCAL))."""
        return min(max(SINK, picokv_mla_fire_point(req) - LOCAL), len(req.origin_input_ids)) // P * P

    def _hkey(self, req) -> int:
        k = getattr(req, "_pk_hkey", None)
        if k is None:
            k = req._pk_hkey = hash(tuple(req.origin_input_ids[: self._hold_h]))
        return k

    def picokv_admitted(self, req) -> None:
        """An admitted prefill whose fire inserts at least h tokens past its match becomes a hold-back owner."""
        x = self._ins_len(req)
        if self._hold_h <= 0 or getattr(req, "skip_radix_cache_insert", False) or x - len(req.prefix_indices) < self._hold_h:
            return
        now = time.time()
        for k in list(_HOLD):  # owners that ended before their insert (or stuck) hold nobody
            owners = _HOLD[k]
            for r in [r for r, e in owners.items() if e[0].finished() or now - e[1] > HOLD_MAX_S]:
                del owners[r]
            if not owners:
                del _HOLD[k]
        _HOLD.setdefault(self._hkey(req), {})[req.rid] = [req, now, x, {}]

    def picokv_hold(self, req) -> bool:
        """True: req waits for an in-flight prefill whose fire insert would extend its match by at least h tokens."""
        if not _HOLD:
            return False
        _hold_release(req)  # an owner back in the waiting queue holds nobody
        h, ids = self._hold_h, req.origin_input_ids
        m, cap = len(req.prefix_indices), len(ids) - 1
        if h <= 0 or cap - m < h:
            return False
        k = self._hkey(req)
        owners = _HOLD.get(k)
        if not owners:
            return False
        now, found = time.time(), False
        for rid, (o, t, x, shared) in list(owners.items()):
            if o.finished() or now - t > HOLD_MAX_S:
                self.hold_stats["timeouts"] += int(not o.finished())
                del owners[rid]
                continue
            s = shared.get(req.rid)
            if s is None:
                s = shared[req.rid] = min(_lcp(ids, o.origin_input_ids, min(len(ids), x)), cap)
            if s - m >= h:
                found = True
                break
        if not owners:
            _HOLD.pop(k, None)
        if found:
            self.hold_stats["passes"] += 1
            if not getattr(req, "_pk_held", False):
                req._pk_held = True
                self.hold_stats["held"] += 1
                if self.hold_stats["held"] % 32 == 1:
                    logger.info("picokv_mla radix hold-back: %s", self.hold_stats)
        return found

    # -- finish / unfinished --
    def cache_finished_req(self, req, is_insert: bool = True, *, owned_kv_len: int):
        a = _ADM.pop(req.rid, None)
        fi = _FINS.pop(req.rid, None)
        _hold_release(req)
        b = _BACKEND[0]
        rpi = req.kv.req_pool_idx
        if b is None or b._rid[rpi] != req.rid:
            # no extend of this request ran: its rows are in an unknown state, keep them out of the tree
            is_insert = False
        ins = is_insert
        if ins:
            self._compress_own(b, rpi, a["m"] if a is not None else 0, owned_kv_len)
        if fi is not None:
            # the fire insert put [pf, x) into the tree; [protected, pf) are this request's duplicates
            pf, x = fi
            assert owned_kv_len >= x, (owned_kv_len, x)
            r2t = self.req_to_token_pool.req_to_token
            if not ins and pf < SINK:
                b.pool.compress(r2t[rpi, pf:SINK].long())  # the tree's sink rows stay dense until the request ends
            lo = req.kv.cache_protected_len
            if pf > lo:
                self.token_to_kv_pool_allocator.free_segment(r2t[rpi, lo:pf], start_pos=lo)
            req.kv.cache_protected_len = x
        super().cache_finished_req(req, is_insert, owned_kv_len=owned_kv_len)
        if a is not None and a["head"] is not None:
            self.token_to_kv_pool_allocator.free_segment(a["head"], start_pos=0)

    def _compress_own(self, b, rpi, m, kv_len):
        """Compress the request's own dense rows in [m, page_floor(kv_len)): the rows the insert can take."""
        end = kv_len // P * P
        ce = int(b._ce[rpi])
        rng = [(m, end)] if ce < 0 else [(m, min(SINK, end)), (max(m, ce), end)]
        r2t = self.req_to_token_pool.req_to_token
        parts = [r2t[rpi, lo:hi] for lo, hi in rng if hi > lo]
        if parts:
            b.pool.compress(torch.cat(parts).long())

    def cache_unfinished_req(self, req, chunked=False):
        """Before the fire no insert (tree slots are compressed, the request still needs its rows dense); the first
        call after it inserts the compressed prompt (_fire_insert). The request keeps its own slots; keep its prefix
        current."""
        rpi = req.kv.req_pool_idx
        if req.rid not in _FINS and not self.disable:
            b = _BACKEND[0]
            if b is not None and b._rid[rpi] == req.rid and b._ce[rpi] >= 0:
                self._fire_insert(req, b, rpi, chunked)
        n = len(req.get_fill_ids())
        req.prefix_indices = self.req_to_token_pool.req_to_token[rpi, :n].to(torch.int64, copy=True)

    def _fire_insert(self, req, b, rpi, chunked):
        """Insert [0, x), x = page_floor(max(SINK, F - LOCAL)) of a fired request: [SINK, x) are compressed, the sink is
        dense (the request reads it) until the request ends. Locks the new path; cache_finished_req frees the
        request's duplicates of what the tree already had."""
        _hold_release(req)
        x = min(self._ins_len(req), int(b._ce[rpi]) // P * P)
        a = _ADM.get(req.rid)
        if x <= max(SINK, req.kv.cache_protected_len, a["m"] if a is not None else 0):
            _FINS[req.rid] = None
            return
        key = RadixKey(req.origin_input_ids[:x], req.extra_key, is_bigram=False, cache_salt=req.cache_salt)
        vals = self.req_to_token_pool.req_to_token[rpi, :x].to(torch.int64, copy=True)
        res = self.insert(InsertParams(key=key, value=vals, chunked=chunked, priority=getattr(req, "priority", 0) or 0))
        mr = self.match_prefix(MatchPrefixParams(key=key))
        assert len(mr.device_indices) == x, (len(mr.device_indices), x)
        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(mr.last_device_node)
        req.last_node = mr.last_device_node
        _FINS[req.rid] = (res.prefix_len, x)

    # -- sizes in available_size units --
    def _free_pages(self) -> int:
        return self.token_to_kv_pool_allocator._free_pages_now()

    def evictable_size(self):
        p = self._free_pages()
        return _u(p + self.evictable_size_ // P) - _u(p)

    def protected_size(self):
        p = self._free_pages() + self.evictable_size_ // P
        return _u(p + self.protected_size_ // P) - _u(p)

    def evict(self, params: EvictParams):
        """``params.num_tokens`` in available_size units -> the tree tokens whose pages raise available_size by it."""
        k = params.num_tokens
        if k <= 0:
            return super().evict(params)
        p = self._free_pages()
        target = _u(p) + k
        pages = RESERVE_PAGES + -(-target * (P + 1) // P) - p
        return super().evict(EvictParams(num_tokens=max(pages, 1) * P))
