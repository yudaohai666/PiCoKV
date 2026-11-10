"""PiCoKV radix prefix reuse (PICOKV_RADIX=host): SGLang's RadixCache whose
nodes also own host K/V slots.

The picokv backend keeps no per-token device KV (the token pool is a no-op;
its indices are virtual) and compresses each request's KV, so a radix hit
cannot read a cached prefix from the pool. With PICOKV_RADIX=host the
backend writes every prefilled token's raw post-RoPE K/V (all layers) into a
pinned host store (_ChunkStage pages) and this tree records, per node, the
store slots of its tokens (``node.pk``, int32, one per key token). A hit
streams the prefix back layer by layer and attends prefix ++ suffix exactly
(the same machinery as exact chunked prefill), then builds the request's
compressed cache over the whole prompt.

Bookkeeping: the backend notes each prefilled row's slots under the request
id (``pk_note``: start position, slots); ``cache_unfinished_req`` /
``cache_finished_req`` take that note, insert the covered tokens only
(prompt-only: decode tokens have no host K/V and are freed as SGLang frees
an unaligned tail), and free the host slots the tree did not take (tokens
another request inserted first). Evicting a leaf frees its slots; the store
evicts the tree (LRU) when it is full, then drops the host K/V of nodes held
only by decoding requests (drop_host), so the store stays near its cap.
Admission: a hit whose prefix another running request holds also builds a
compressed cache over that prefix, so schedule_policy charges it
(picokv_charge) and holds those accounting tokens until it finishes
(picokv_admit; the tree counts shared tokens once).
GPU hot tier (PICOKV_RADIX_HOT, the backend's _ChunkStage): device mirrors of
host pages, so a hit copies its prefix device to device instead of over PCIe.
A mirror holds accounting tokens of its bytes (pk_hot_hold, only from the
free pool), counted as evictable: evict() frees mirrors first (their host
K/V stays), so admission sees the capacity it sees without them.
page_size 1, no EAGLE bigram keys.
"""

from __future__ import annotations

import bisect
import dataclasses
import logging
import os
import weakref
from typing import Optional

import numpy as np
import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    EvictResult,
    InsertParams,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey

logger = logging.getLogger(__name__)

_LIVE = None  # weakref to this process's tree (the backend finds it here)
# PICOKV_RADIX_ADMIT=0: no shared-prefix charge / stage reserve at admission (A/B)
_ADMIT = os.environ.get("PICOKV_RADIX_ADMIT", "1") != "0"


def live_tree() -> Optional["PicoKVRadixCache"]:
    return _LIVE() if _LIVE is not None else None


class SlotRuns:
    """Free host slots as sorted disjoint runs [s[i], e[i]). First-fit
    allocation (one run when one is long enough, else several), merging
    free."""

    def __init__(self):
        self.s, self.e = [], []
        self.cap = self.nfree = 0

    def grow(self, n):
        if n <= 0:  # (an empty run [cap, cap) would refuse the next grow)
            return
        self.cap += n
        self._put(self.cap - n, self.cap)

    def shrink(self, n):
        """Undo grow(n): drop the top n slots (all free)."""
        if n <= 0:
            return
        a = self.cap - n
        if not self.s or self.e[-1] != self.cap or self.s[-1] > a:
            raise RuntimeError(f"picokv radix: shrink of host slots in use [{a}, {self.cap})")
        if self.s[-1] == a:
            del self.s[-1], self.e[-1]
        else:
            self.e[-1] = a
        self.cap, self.nfree = a, self.nfree - n

    def alloc(self, n) -> Optional[np.ndarray]:
        if n <= 0:
            return np.empty(0, dtype=np.int32)
        if n > self.nfree:
            return None
        for i in range(len(self.s)):
            if self.e[i] - self.s[i] >= n:
                a = self.s[i]
                self._cut(i, n)
                return np.arange(a, a + n, dtype=np.int32)
        out, k = [], n
        while k > 0:
            a, b = self.s[0], self.e[0]
            t = min(k, b - a)
            out.append(np.arange(a, a + t, dtype=np.int32))
            self._cut(0, t)
            k -= t
        return np.concatenate(out)

    def _cut(self, i, n):
        self.s[i] += n
        self.nfree -= n
        if self.s[i] == self.e[i]:
            del self.s[i], self.e[i]

    def free_in(self, a, b) -> int:
        """Free slots in [a, b)."""
        i, n = bisect.bisect_right(self.e, a), 0
        while i < len(self.s) and self.s[i] < b:
            n += min(b, self.e[i]) - max(a, self.s[i])
            i += 1
        return n

    def free(self, slots):
        if slots is None or len(slots) == 0:
            return
        x = np.sort(np.asarray(slots, dtype=np.int64))
        cut = np.flatnonzero(np.diff(x) != 1) + 1
        for a, b in zip(np.r_[x[:1], x[cut]], np.r_[x[cut - 1], x[-1:]] + 1):
            self._put(int(a), int(b))

    def _put(self, a, b):
        i = bisect.bisect_left(self.s, a)
        if (a < 0 or b > self.cap or (i > 0 and self.e[i - 1] > a)
                or (i < len(self.s) and self.s[i] < b)):
            raise RuntimeError(f"picokv radix: double free of host slots [{a}, {b})")
        self.nfree += b - a
        if i > 0 and self.e[i - 1] == a:
            i -= 1
            self.e[i] = b
        else:
            self.s.insert(i, a)
            self.e.insert(i, b)
        if i + 1 < len(self.s) and self.s[i + 1] == self.e[i]:
            self.e[i] = self.e[i + 1]
            del self.s[i + 1], self.e[i + 1]


def _path_pk(node, root):
    """Host slots of the tokens on root -> node (None: a node without)."""
    parts = []
    while node is not root and node is not None:
        pk = getattr(node, "pk", None)
        if pk is None:
            return None
        parts.append(pk)
        node = node.parent
    if not parts:
        return np.empty(0, dtype=np.int32)
    return np.concatenate(parts[::-1])


class PicoKVRadixCache(RadixCache):
    """RadixCache + per-node host K/V slots (module docstring). ``slots``:
    the host store's free slots (the backend's _ChunkStage grows it with
    its pinned pages and evicts this tree when it is full)."""

    def __init__(self, params):
        global _LIVE
        if params.page_size != 1 or params.is_eagle:
            raise ValueError("picokv radix cache needs page_size 1 and no EAGLE")
        self.slots = SlotRuns()
        self._notes = {}  # rid -> [start position, host slots of the prefilled tokens]
        self._reqs = weakref.WeakValueDictionary()  # rid -> Req (its last_node)
        self._pf = weakref.WeakValueDictionary()  # rid -> Req whose prefill may still read its path
        self._ins_host = self._ins_taken = None
        self.picokv_rtok = None  # the backend's stage reserve (tokens), set at attach
        self._pk_hot = None  # the backend's _ChunkStage when its GPU hot tier is on
        self._hot_n = 0  # accounting tokens the hot tier's mirrors hold
        # PICOKV_RADIX_HOT_FREE: a mirror is taken only while this fraction of
        # the pool stays free after it
        self._hot_free = float(os.environ.get("PICOKV_RADIX_HOT_FREE", "0.1"))
        super().__init__(params)
        _LIVE = weakref.ref(self)

    # -- host slot bookkeeping (backend side) --------------------------------
    def pk_note(self, rid, start, slots):
        """The backend wrote K/V of tokens [start, start + len(slots)) of
        request ``rid`` into ``slots``; the next cache_*_req takes them."""
        old = self._notes.get(rid)
        if old is not None and old[0] + len(old[1]) == start:
            old[1] = np.concatenate([old[1], slots])
            return
        if old is not None:
            self.slots.free(old[1])
        self._notes[rid] = [int(start), slots]

    def pk_unnote(self, rid, start, slots):
        """Undo pk_note(rid, start, slots) of an extend that failed before
        writing them: the note keeps only its tokens before ``start`` (an
        earlier chunk's) and ``slots`` go back to the free list."""
        note = self._notes.get(rid)
        if note is not None and note[0] <= start <= note[0] + len(note[1]):
            if start > note[0]:
                note[1] = note[1][:start - note[0]]
            else:
                del self._notes[rid]
        self.slots.free(slots)

    def pk_prefix(self, rid, plen) -> Optional[np.ndarray]:
        """Host slots of request ``rid``'s first ``plen`` tokens: its tree
        path (req.last_node), then its untaken note. None: not all held."""
        req = self._reqs.get(rid)
        node = getattr(req, "last_node", None) if req is not None else None
        path = (_path_pk(node, self.root_node) if node is not None
                else np.empty(0, dtype=np.int32))
        if path is None:
            return None
        if len(path) >= plen:
            return path[:plen]
        note = self._notes.get(rid)
        if note is None or not note[0] <= len(path) or note[0] + len(note[1]) < plen:
            return None
        return np.concatenate([path, note[1][len(path) - note[0]:plen - note[0]]])

    def evict_host(self, n) -> None:
        """Evict LRU unlocked leaves until n host slots are free. evict()
        counts device tokens, which a leaf without host K/V (drop_host) has
        too while freeing no slot: evict until the slots are there or
        nothing evictable is left. (The tree's own eviction: the hot tier's
        mirrors free no host slot.)"""
        while self.slots.nfree < n and RadixCache.evictable_size(self) > 0:
            before = RadixCache.evictable_size(self)
            RadixCache.evict(self, EvictParams(num_tokens=n - self.slots.nfree))
            if RadixCache.evictable_size(self) == before:
                break

    # -- GPU hot tier accounting (the backend's _ChunkStage mirrors) ---------
    def pk_hot_hold(self, n):
        """n accounting tokens for a device mirror, taken only from the free
        pool while PICOKV_RADIX_HOT_FREE of it stays free (a mirror never
        costs admission: evict() gives them back first). None: not now."""
        alloc = self.token_to_kv_pool_allocator
        if alloc.available_size() - n < self._hot_free * alloc.size:
            return None
        idx = alloc.alloc(n)
        if idx is not None:
            self._hot_n += n
        return idx

    def pk_hot_unhold(self, idx):
        self._hot_n -= len(idx)
        self.token_to_kv_pool_allocator.free(idx)

    def evictable_size(self):
        return super().evictable_size() + self._hot_n

    def evict(self, params: EvictParams) -> EvictResult:
        """The hot tier's mirrors first (LRU; their prefixes stay on the
        host), then RadixCache.evict for the rest."""
        hot, n = self._pk_hot, params.num_tokens
        if hot is None or self._hot_n <= 0 or n <= 0:
            return super().evict(params)
        got = hot.hot_evict(n)
        if got >= n:
            return EvictResult(num_tokens_evicted=got)
        res = super().evict(dataclasses.replace(params, num_tokens=n - got))
        return dataclasses.replace(res, num_tokens_evicted=res.num_tokens_evicted + got)

    def drop_host(self, n) -> int:
        """Free >= n host slots (when evict() cannot: the rest is locked by
        running requests) by dropping the host K/V of nodes no pending
        prefill reads, least recently used subtrees first. The nodes stay
        (SGLang's lock_ref guards the device side, which picokv does not
        read); match_prefix ends at a node without pk, so a subtree goes with
        its root. Pending: matched requests until their prefill is done
        (cache_unfinished_req not chunked / cache_finished_req). Returns the
        slots freed."""
        prot = set()
        for req in list(self._pf.values()):
            node = getattr(req, "last_node", None)
            while node is not None and id(node) not in prot:
                prot.add(id(node))
                node = node.parent
        cand, stack = [], [self.root_node]
        while stack:
            for c in stack.pop().children.values():
                if id(c) in prot:
                    stack.append(c)
                elif getattr(c, "pk", None) is not None:
                    cand.append(c)
        cand.sort(key=lambda c: c.last_access_time)
        freed = 0
        for c in cand:
            if freed >= n:
                break
            st = [c]
            while st:
                y = st.pop()
                st.extend(y.children.values())
                if getattr(y, "pk", None) is not None:
                    freed += len(y.pk)
                    self.slots.free(y.pk)
                    y.pk = None
        return freed

    # -- admission accounting (schedule_policy add_one_req) -------------------
    def picokv_charge(self, req) -> int:
        """Tokens of req's matched prefix that a running request already holds.
        picokv builds every request's compressed cache over its whole prompt,
        so a concurrent hit costs its prefix again, while the tree counts the
        shared tokens once (they cover the first holder)."""
        if not _ADMIT:
            return 0
        n, node = 0, req.last_node
        while node is not None and node is not self.root_node:
            if node.lock_ref > 0:
                n += len(node.key)
            node = node.parent
        return min(n, len(req.prefix_indices))

    def picokv_admit(self, req, n) -> None:
        """Hold n accounting tokens (picokv_charge) for an admitted request,
        evicting first like alloc_for_extend; cache_finished_req frees them."""
        self._free_hold(req)
        if n <= 0:
            return
        alloc = self.token_to_kv_pool_allocator
        if alloc.available_size() < n:
            self.evict(EvictParams(num_tokens=n - alloc.available_size()))
        idx = alloc.alloc(n)
        if idx is None:
            logger.warning("picokv radix: no %d tokens to hold for %s's shared prefix", n, req.rid)
            return
        req._picokv_hold = idx

    def picokv_stage_ok(self, req, staged, can_run_list) -> bool:
        """The backend reserves its stage buffers (G: each prefix row's prefix
        ++ new tokens, two layers) for ``picokv_rtok`` tokens: a row with a
        prefix (``staged`` = prefix + this extend) that would take the batch
        past them waits a round; the batch's first prefix row always goes."""
        cap = self.picokv_rtok
        if cap is None or not _ADMIT or len(req.prefix_indices) == 0:
            return True
        used = [r.extend_range.end for r in can_run_list if len(r.prefix_indices) > 0]
        return not used or sum(used) + staged <= cap

    def _free_hold(self, req):
        h = getattr(req, "_picokv_hold", None)
        if h is not None:
            req._picokv_hold = None
            self.token_to_kv_pool_allocator.free(h)

    def _cover(self, prot, note, n):
        """Insert length (tokens [0, hw) have host K/V: the protected prefix
        is in the tree, the note continues it) and the host slots aligned to
        the key (-1 below the note)."""
        if note is None:
            return prot, None
        p0, s = note
        hw = min(p0 + len(s), n)
        if p0 > prot or hw <= prot:
            return prot, None
        host = np.full(hw, -1, dtype=np.int32)
        host[p0:hw] = s[:hw - p0]
        return hw, host

    def _free_note(self, note, taken=()):
        """Free a note's slots except positions in the ``taken`` ranges
        [lo, hi) (now held by tree nodes)."""
        if note is None:
            return
        p0, s = note
        keep = np.zeros(len(s), dtype=bool)
        for lo, hi in taken:
            keep[lo - p0:hi - p0] = True
        self.slots.free(s[~keep])

    # -- tree structure: keep node.pk aligned with node.key -------------------
    def reset(self):
        hot = getattr(self, "_pk_hot", None)
        if hot is not None:
            hot.hot_reset()
        root = getattr(self, "root_node", None)
        if root is not None:
            stack = list(root.children.values())
            while stack:
                n = stack.pop()
                stack.extend(n.children.values())
                if getattr(n, "pk", None) is not None:
                    self.slots.free(n.pk)
                    n.pk = None
        for _, s in self._notes.values():
            self.slots.free(s)
        self._notes.clear()
        super().reset()
        self.root_node.pk = np.empty(0, dtype=np.int32)

    def _split_node(self, key, child, split_len):
        pk = getattr(child, "pk", None)
        new_node = super()._split_node(key, child, split_len)
        new_node.pk = None if pk is None else pk[:split_len]
        child.pk = None if pk is None else pk[split_len:]
        return new_node

    def _insert_helper(self, node, key, value, priority=0, chunked=False):
        """A new leaf takes the host slots of key[plen:]; a path node without
        host K/V (drop_host) takes them back when this insert has them."""
        n = len(key)
        plen, last = super()._insert_helper(node, key, value, priority, chunked)
        h = self._ins_host
        if plen < n:
            last.pk = None  # a new leaf holds key[plen:]
        if h is None or len(h) < n or self._ins_taken is None:
            return plen, last
        y, b = last, n
        while y is not None and y is not self.root_node and b > 0:
            a = b - len(y.key)
            if getattr(y, "pk", None) is None and a >= 0 and (h[a:b] >= 0).all():
                y.pk = h[a:b]
                self._ins_taken.append((a, b))
            y, b = y.parent, a
        return plen, last

    def _delete_leaf(self, node):
        super()._delete_leaf(node)
        pk = getattr(node, "pk", None)
        if pk is not None:
            self.slots.free(pk)
            node.pk = None

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        """RadixCache.match_prefix, cut at the first node without host K/V
        (none in practice: every insert carries slots); records the request
        so the backend can find its matched path (pk_prefix)."""
        res = super().match_prefix(params)
        if params.req is not None:
            self._reqs[params.req.rid] = self._pf[params.req.rid] = params.req
        node, cut = res.last_device_node, None
        while node is not self.root_node and node is not None:
            if getattr(node, "pk", None) is None:
                cut = node
            node = node.parent
        if cut is None:
            return res
        keep, vals, n = cut.parent, [], cut.parent
        while n is not self.root_node:
            vals.append(n.value)
            n = n.parent
        dev = (torch.cat(vals[::-1]) if vals
               else self._empty_match_result.device_indices)
        return res._replace(device_indices=dev, last_device_node=keep,
                            last_host_node=keep, best_match_node=keep)

    # -- request caching: prompt tokens with host K/V only -------------------
    def _insert_host(self, host, params):
        self._ins_host, self._ins_taken = host, []
        try:
            res = self.insert(params)
        finally:
            taken, self._ins_host, self._ins_taken = self._ins_taken, None, None
        return res, taken

    def cache_finished_req(self, req, is_insert: bool = True, *, owned_kv_len: int):
        self._free_hold(req)
        note = self._notes.pop(req.rid, None)
        self._pf.pop(req.rid, None)
        if self.disable or not is_insert:
            self._free_note(note)
            return super().cache_finished_req(req, is_insert, owned_kv_len=owned_kv_len)
        hw, host = self._cover(req.kv.cache_protected_len, note, owned_kv_len)
        self._ins_host, self._ins_taken = host, []
        try:
            # tokens [hw, owned) (decode tokens: no host K/V) leave like an
            # unaligned tail
            super().cache_finished_req(req, True, owned_kv_len=hw)
        finally:
            taken, self._ins_host, self._ins_taken = self._ins_taken, None, None
        self._free_note(note, taken)
        if hw < owned_kv_len:
            self.token_to_kv_pool_allocator.free_segment(
                self.req_to_token_pool.req_to_token[req.kv.req_pool_idx, hw:owned_kv_len],
                start_pos=hw)

    def cache_unfinished_req(self, req, chunked=False):
        """RadixCache.cache_unfinished_req over the tokens with host K/V
        (_cover); the rest stays the request's own, as a partial page."""
        if self.disable:
            return
        self._reqs[req.rid] = req
        if chunked:
            self._pf[req.rid] = req
        else:
            self._pf.pop(req.rid, None)  # prefill done: its path is droppable (drop_host)
        note = self._notes.pop(req.rid, None)
        prot = req.kv.cache_protected_len
        token_ids = req.get_fill_ids()
        kv_indices = self.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, : len(token_ids)
        ]
        hw, host = self._cover(prot, note, len(token_ids))
        radix_key = RadixKey(token_ids[:hw], req.extra_key, cache_salt=req.cache_salt)
        values = kv_indices[:hw].to(dtype=torch.int64, copy=True)
        result, taken = self._insert_host(host, InsertParams(
            key=radix_key, value=values, chunked=chunked,
            priority=getattr(req, "priority", 0) or 0))
        self._free_note(note, taken)
        new_prefix_len = result.prefix_len
        self.token_to_kv_pool_allocator.free_segment(
            kv_indices[prot:new_prefix_len], start_pos=prot)

        match_result = RadixCache.match_prefix(self, MatchPrefixParams(key=radix_key))
        new_indices, new_last_node = (
            match_result.device_indices, match_result.last_device_node)
        assert len(new_indices) == hw, f"{len(new_indices)=}, {hw=}"
        self.req_to_token_pool.write(
            (req.kv.req_pool_idx, slice(prot, hw)), new_indices[prot:])
        req.kv.cache_protected_len = hw
        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)
        if hw < len(kv_indices):
            req.prefix_indices = torch.cat([new_indices, kv_indices[hw:]])
        else:
            req.prefix_indices = new_indices
        req.last_node = new_last_node


def _factory(ctx):
    if ctx.disable_radix_cache or ctx.params.disable:
        from sglang.srt.mem_cache.registry import default_radix_cache_factory

        return default_radix_cache_factory(ctx)
    return PicoKVRadixCache(ctx.params)


def register():
    from sglang.srt.mem_cache.registry import (
        get_radix_cache_factory,
        register_radix_cache_backend,
    )

    if get_radix_cache_factory("picokv") is None:
        register_radix_cache_backend("picokv", _factory)


register()
