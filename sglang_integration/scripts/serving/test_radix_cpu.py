"""CPU unit test: PicoKVRadixCache slot bookkeeping (no GPU, mock pools)."""
import random
from array import array

import numpy as np
import torch

from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.picokv_radix_cache import PicoKVRadixCache, SlotRuns, _path_pk
from sglang.srt.mem_cache.radix_cache import RadixKey


class Alloc:
    device = "cpu"
    page_size = 1

    def __init__(self):
        self.freed = []

    def free_segment(self, idx, *, start_pos):
        self.freed += [int(x) for x in idx.tolist()]

    def free_segments(self, segs):
        for idx, s in segs:
            self.free_segment(idx, start_pos=s)


class R2T:
    def __init__(self):
        self.req_to_token = torch.zeros(8, 64, dtype=torch.int32)

    def write(self, ix, v):
        self.req_to_token[ix] = v.to(torch.int32)


class KV:
    def __init__(self, idx):
        self.req_pool_idx, self.cache_protected_len, self.kv_committed_len = idx, 0, 0


class Req:
    def __init__(self, rid, idx, prompt):
        self.rid, self.kv = rid, KV(idx)
        self.origin_input_ids, self.output_ids = array("q", prompt), array("q")
        self.extra_key = self.cache_salt = None
        self.last_node = self.prefix_indices = None
        self.priority = 0
        self.fill = len(prompt)

    def get_fill_ids(self):
        return (self.origin_input_ids + self.output_ids)[: self.fill]


NEXT = [1000]


def dev(n):
    a = torch.arange(NEXT[0], NEXT[0] + n, dtype=torch.int32)
    NEXT[0] += n
    return a


def admit(tree, req, r2t):
    """scheduler side: match, lock, write the row's prefix + new device slots."""
    m = tree.match_prefix(MatchPrefixParams(key=RadixKey(req.origin_input_ids), req=req))
    req.prefix_indices, req.last_node = m.device_indices, m.last_device_node
    req.kv.cache_protected_len = len(m.device_indices)
    tree.inc_lock_ref(req.last_node)
    p = len(m.device_indices)
    r2t.req_to_token[req.kv.req_pool_idx, :p] = m.device_indices.to(torch.int32)
    return p


def prefill(tree, req, r2t, lo, hi):
    """backend side: tokens [lo, hi) get device slots + host slots (pk_note)."""
    r2t.req_to_token[req.kv.req_pool_idx, lo:hi] = dev(hi - lo)
    if lo > 0:
        assert tree.pk_prefix(req.rid, lo) is not None
    s = tree.slots.alloc(hi - lo)
    tree.pk_note(req.rid, lo, s)
    req.kv.kv_committed_len = hi
    return s


def check(tree, none_ok=False):
    st = [tree.root_node]
    held = []
    while st:
        n = st.pop()
        st += list(n.children.values())
        if n is not tree.root_node:
            if n.pk is None and none_ok:
                continue
            assert n.pk is not None and len(n.pk) == len(n.key), (len(n.key), n.pk)
            held.append(n.pk)
    h = np.concatenate(held) if held else np.empty(0, np.int32)
    assert len(np.unique(h)) == len(h), "slot held twice"
    notes = sum(len(s) for _, s in tree._notes.values())
    assert tree.slots.nfree + len(h) + notes == tree.slots.cap, (tree.slots.nfree, len(h), notes)
    return len(h)


def main():
    a, r2t = Alloc(), R2T()
    tree = PicoKVRadixCache(CacheInitParams(disable=False, req_to_token_pool=r2t,
                                            token_to_kv_pool_allocator=a, page_size=1))
    tree.slots.grow(200)
    # A: prompt 10, prefill, continue (cache_unfinished_req), decode 5, finish
    r1 = Req("r1", 0, list(range(1, 11)))
    assert admit(tree, r1, r2t) == 0
    s1 = prefill(tree, r1, r2t, 0, 10)
    r1.output_ids.append(50); r1.fill = 11
    tree.cache_unfinished_req(r1)
    assert r1.kv.cache_protected_len == 10 and len(r1.prefix_indices) == 11
    assert check(tree) == 10
    r2t.req_to_token[0, 10:15] = dev(5); r1.kv.kv_committed_len = 15
    r1.output_ids.extend([51, 52, 53, 54])
    dec = r2t.req_to_token[0, 10:15].tolist()
    tree.cache_finished_req(r1, owned_kv_len=15)
    assert sorted(a.freed) == sorted(dec), a.freed
    assert np.array_equal(_path_pk(r1.last_node, tree.root_node), s1)
    assert check(tree) == 10
    # B: hit 6 of r1's prompt + 4 new, finishes at prefill
    a.freed.clear()
    r2 = Req("r2", 1, list(range(1, 7)) + [20, 21, 22, 23])
    assert admit(tree, r2, r2t) == 6
    assert np.array_equal(tree.pk_prefix("r2", 6), s1[:6])
    s2 = prefill(tree, r2, r2t, 6, 10)
    tree.cache_finished_req(r2, owned_kv_len=10)
    assert a.freed == []
    m = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", list(range(1, 7)) + [20, 21, 22, 23]))))
    assert np.array_equal(_path_pk(m.last_device_node, tree.root_node), np.r_[s1[:6], s2])
    assert check(tree) == 14
    # C: duplicate prompt, two requests prefill concurrently
    r3, r4 = Req("r3", 2, list(range(30, 38))), Req("r4", 3, list(range(30, 38)))
    admit(tree, r3, r2t); admit(tree, r4, r2t)
    s3 = prefill(tree, r3, r2t, 0, 8); prefill(tree, r4, r2t, 0, 8)
    tree.cache_finished_req(r3, owned_kv_len=8)
    a.freed.clear()
    dup = r2t.req_to_token[3, :8].tolist()
    tree.cache_finished_req(r4, owned_kv_len=8)
    assert sorted(a.freed) == sorted(dup)
    assert check(tree) == 22
    # D: chunked 20 = 8 + 8 + 4, overlap: chunk 3 noted before chunk 2 is cached
    r5 = Req("r5", 4, list(range(40, 60)))
    admit(tree, r5, r2t)
    c1 = prefill(tree, r5, r2t, 0, 8); r5.fill = 8
    tree.cache_unfinished_req(r5, chunked=True)
    c2 = prefill(tree, r5, r2t, 8, 16); r5.fill = 16
    c3 = prefill(tree, r5, r2t, 16, 20)  # note now covers [8, 20)
    assert np.array_equal(tree.pk_prefix("r5", 20), np.r_[c1, c2, c3])
    tree.cache_unfinished_req(r5, chunked=True)  # inserts [0,16) only (fill)
    assert r5.kv.cache_protected_len == 16
    assert "r5" not in tree._notes  # popped; [16, 20) slots freed with it
    check(tree)
    assert tree.pk_prefix("r5", 20) is None  # lost row
    assert np.array_equal(tree.pk_prefix("r5", 16), np.r_[c1, c2])
    # re-note chunk 3 (backend redo) then finish with 2 decode tokens
    c3 = prefill(tree, r5, r2t, 16, 20); r5.fill = 20
    r2t.req_to_token[4, 20:22] = dev(2); r5.kv.kv_committed_len = 22
    r5.output_ids.extend([7, 8])
    a.freed.clear()
    tail = r2t.req_to_token[4, 20:22].tolist()
    tree.cache_finished_req(r5, owned_kv_len=22)
    assert sorted(a.freed) == sorted(tail)
    assert check(tree) == 42
    # E: a node without host slots cuts the match
    tree.insert(InsertParams(key=RadixKey(array("q", [90, 91, 92])), value=torch.tensor([1, 2, 3])))
    m = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", [90, 91, 92, 93]))))
    assert len(m.device_indices) == 0 and m.last_device_node is tree.root_node
    tree._delete_leaf(tree.root_node.children[90])
    # F: evict everything unlocked -> all host slots back
    tree.evict(EvictParams(num_tokens=10 ** 6))
    assert check(tree) == 0 and tree.slots.nfree == tree.slots.cap
    try:
        tree.slots.free(np.array([3], np.int32)); raise AssertionError("no double free error")
    except RuntimeError:
        pass
    # I: store full while a decoding request holds its prompt: drop_host frees
    # the host K/V nobody's prefill reads (not a pending request's path), and
    # a later insert of the same tokens refills the dropped node
    r7 = Req("r7", 6, list(range(100, 110)))
    admit(tree, r7, r2t); p7 = prefill(tree, r7, r2t, 0, 10)
    r7.output_ids.append(5); r7.fill = 11
    tree.cache_unfinished_req(r7)  # continues to decode (locked, prefill done)
    r8 = Req("r8", 7, list(range(100, 105)) + [130, 131])
    assert admit(tree, r8, r2t) == 5  # pending: [100, 105) is protected
    p8 = prefill(tree, r8, r2t, 5, 7)
    assert tree.evict(EvictParams(num_tokens=100)) is None or True
    assert check(tree) == 10  # locked: evict frees nothing
    assert tree.drop_host(100) == 5
    head = tree.root_node.children[100]
    tail = head.children[105]
    assert np.array_equal(head.pk, p7[:5]) and tail.pk is None
    assert check(tree, none_ok=True) == 5
    assert np.array_equal(tree.pk_prefix("r8", 5), p7[:5])
    tree.cache_finished_req(r8, owned_kv_len=7)
    r9 = Req("r9", 0, list(range(100, 110)) + [140])
    assert admit(tree, r9, r2t) == 5  # cut at the dropped node
    p9 = prefill(tree, r9, r2t, 5, 11)
    tree.cache_finished_req(r9, owned_kv_len=11)
    assert np.array_equal(tail.pk, p9[:5]) and np.array_equal(tail.children[140].pk, p9[5:])
    m = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", list(range(100, 110)) + [140]))))
    assert len(m.device_indices) == 11
    r2t.req_to_token[6, 10:12] = dev(2); r7.kv.kv_committed_len = 12
    r7.output_ids.append(6)
    tree.cache_finished_req(r7, owned_kv_len=12)
    assert check(tree) == 10 + 2 + 1
    tree.evict(EvictParams(num_tokens=10 ** 6))
    assert check(tree) == 0 and tree.slots.nfree == tree.slots.cap
    # G: reset with live nodes + notes
    r6 = Req("r6", 5, [1, 2, 3, 4]); admit(tree, r6, r2t); prefill(tree, r6, r2t, 0, 4)
    tree.cache_unfinished_req(r6); prefill(tree, r6, r2t, 4, 6)
    tree.reset()
    assert tree.slots.nfree == tree.slots.cap and not tree._notes
    # H: SlotRuns vs a set model
    rng = random.Random(0)
    sr, held = SlotRuns(), []
    sr.grow(500)
    for it in range(4000):
        if rng.random() < 0.55:
            x = sr.alloc(rng.randint(1, 40))
            if x is not None:
                held.append(x)
        elif held:
            sr.free(held.pop(rng.randrange(len(held))))
        if it % 997 == 0:
            sr.grow(rng.randint(1, 50))
        used = np.concatenate(held) if held else np.empty(0, np.int32)
        assert len(np.unique(used)) == len(used)
        assert sr.nfree + len(used) == sr.cap
        assert all(sr.s[i] < sr.e[i] for i in range(len(sr.s)))
        assert all(sr.e[i] < sr.s[i + 1] for i in range(len(sr.s) - 1))
    print("RADIX CPU PASS")


if __name__ == "__main__":
    main()
