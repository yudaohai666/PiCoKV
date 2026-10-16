"""PiCoKV attention backend for SGLang.

Layers (bottom-up):

* ``PicoKVRequestCache`` — one request's per-layer compressed KV state, built
  directly on PiCoKV's **pure** ``prepare_kv_cache`` / ``decode_attention`` API
  (no ``transformers`` import — so it is compatible with whatever transformers
  version the serving engine pins, and unit-testable standalone).
* ``PicoKVCore`` — single-request (batch=1) engine in sglang's flattened
  ``[num_tokens, heads*head_dim]`` layout: ``extend`` (prefill: exact attention
  + compress) and ``decode`` (one PiCoKV CUDA decode step).
* ``PicoKVBatchDecoder`` — step 3: a continuous-batch decoder holding one
  ``PicoKVRequestCache`` per running request at *independent* context lengths,
  so a ragged batch decodes correctly. Fuses per-request results; the current
  build launches the verified single-request kernel per request (a fused ragged
  kernel is the remaining optimization).
* ``PicoKVAttnBackend`` — the ``AttentionBackend`` glue (defined only when
  sglang imports). Registered as backend name ``picokv``.

Decode fast path (all branches unified): a persistent token-major dense pool
``[B, cap, KVH, D]`` per layer, written in place each step at device-side
positions; exact dense-window attention via flashinfer's paged decode kernel
(one plan per step, page_size = cap so every kernel input lives at a stable
address); then the fused PiCoKV tail. When shapes are stable (fixed batch
membership), the whole per-layer op sequence is replayed through a CUDA graph
(``PICOKV_GRAPHS=0`` disables).

Hard requirements of the PiCoKV decode kernel: GQA=4, head_dim=128, bf16.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

# Pure PiCoKV API — imports NO transformers/numpy (verified), unlike the HF
# adapter in picokv_standalone.transformers.
from picokv.picokv_standalone import (
    PreparedKVCache,
    StandaloneConfig,
    decode_attention,
    prepare_kv_cache,
)

# Incremental chunk-append path (eas=0 + offline projections). Both come from
# the standalone/prefill extension; when unavailable, continuations fall back
# to re-prepare (old behavior) instead of failing.
try:
    from picokv.picokv_prefill_cuda_extension.picokv_prefill_cuda import (
        blockk_fixed_cuda,
    )
    from picokv.picokv_standalone.prepare import _prepare_fixed_block32_cached
except Exception:  # pragma: no cover - append path degrades to re-prepare
    blockk_fixed_cuda = None
    _prepare_fixed_block32_cached = None

# Fused decode append (PICOKV_FUSED_APPEND=0 disables): one launch writes this
# step's K and V rows into the dense pools, replacing two index_put kernels.
try:
    import triton
    import triton.language as tl

    @triton.jit
    def _append_kv_kernel(kbuf, vbuf, k, v, wpos, cap, k_s0, v_s0,
                          ROW: tl.constexpr):
        # one program per request; a row is KVH*D contiguous elements
        b = tl.program_id(0).to(tl.int64)
        off = tl.arange(0, ROW)
        dst = (b * cap + tl.load(wpos + b).to(tl.int64)) * ROW + off
        tl.store(kbuf + dst, tl.load(k + b * k_s0 + off))
        tl.store(vbuf + dst, tl.load(v + b * v_s0 + off))
except Exception:  # pragma: no cover - falls back to index_put
    _append_kv_kernel = None


def load_offline_projections_pure(path: str) -> dict:
    """Transformers-free loader for the shipped block-32 encoder/decoder .pt.

    Returns a dict with ``key_encoder``/``value_encoder`` (and matching
    ``*_decoder``) each shaped ``[num_layers, num_kv_heads, bits, head_dim]``.
    """
    data = torch.load(path, map_location="cpu")
    for side in ("key", "value"):
        if f"{side}_encoder" not in data and side in data:
            data[f"{side}_encoder"] = data.pop(side)
    if "key_encoder" not in data or "value_encoder" not in data:
        raise ValueError(f"{path}: missing key_encoder/value_encoder")
    for side in ("key", "value"):
        if f"{side}_decoder" not in data:
            raise ValueError(
                f"{path} stores encoders only; PiCoKV serving needs the folded "
                "encoder+decoder pair (rebuild with create_offline_projections.py)"
            )
    return data


def _slice_layer(stack: Optional[torch.Tensor], layer_idx: int, heads: int):
    """Slice ``[layers, heads, bits, dim]`` to one layer, or return None."""
    if stack is None:
        return None
    if layer_idx >= stack.shape[0]:
        raise ValueError(f"projection has {stack.shape[0]} layers; got {layer_idx}")
    layer = stack[layer_idx]
    if layer.shape[0] != heads:
        raise ValueError(
            f"projection layer {layer_idx} has {layer.shape[0]} heads; model has {heads}"
        )
    return layer


class PicoKVRequestCache:
    """One request's per-layer PiCoKV cache (pure API; mirrors the HF adapter's
    prepare/decode logic without importing transformers)."""

    def __init__(self, config: StandaloneConfig, offline: Optional[dict] = None) -> None:
        self.config = config
        self._offline = offline  # dict of full [layers,heads,bits,dim] stacks or None
        self.layers: Dict[int, PreparedKVCache] = {}
        # Layers whose prompt <= sink+local have no global segment to compress:
        # keep them fully dense and serve decode with SDPA (same math).
        self._dense: Dict[int, tuple] = {}
        self.seq_lengths: Dict[int, int] = {}
        # Set per extend at layer 0: this continuation chunk merges incrementally
        # (append) instead of re-preparing from the chunk alone.
        self._append_mode = False

    def _prepare_kwargs(self, heads: int) -> dict:
        kwargs = self.config.prepare_kwargs()
        if self._offline is not None:
            kwargs["key_encoders"] = _slice_layer(self._offline["key_encoder"], self._layer, heads)
            kwargs["value_encoders"] = _slice_layer(self._offline["value_encoder"], self._layer, heads)
            kwargs["key_decoders"] = _slice_layer(self._offline["key_decoder"], self._layer, heads)
            kwargs["value_decoders"] = _slice_layer(self._offline["value_decoder"], self._layer, heads)
        return kwargs

    def prepare(self, layer_idx: int, key_bhsd: torch.Tensor, value_bhsd: torch.Tensor) -> None:
        layer_idx = int(layer_idx)
        total_len = int(key_bhsd.shape[-2])
        if total_len <= self.config.sink_size + self.config.local_size:
            self._dense[layer_idx] = (key_bhsd.contiguous(), value_bhsd.contiguous())
            self.seq_lengths[layer_idx] = total_len
            return
        self._layer = layer_idx
        heads = int(key_bhsd.shape[1])
        self.layers[layer_idx] = prepare_kv_cache(
            key_bhsd, value_bhsd, **self._prepare_kwargs(heads)
        )
        self.seq_lengths[layer_idx] = total_len

    # ------------------------------------------------------------------
    # incremental chunk-append (chunked prefill continuation, eas=0)
    # ------------------------------------------------------------------

    def can_append(self, prefix_len: int) -> bool:
        """Whether a continuation chunk (``prefix_len`` tokens already cached)
        can merge incrementally instead of re-preparing from the chunk alone.

        Requirements: the recorded layer-0 length matches the engine's prefix
        length (same request, no decode tokens since — also rules out recycled
        req-pool indices), and either the dense fallback (raw prefix retained,
        exact by construction) or the compressed path with offline projections
        (deterministic fixed cascade), eas=0 (kept = every global token; no
        cross-chunk error selection) and pristine sink+local windows.
        """
        if prefix_len <= 0 or int(self.seq_lengths.get(0, -1)) != prefix_len:
            return False
        if 0 in self._dense:
            return True  # dense merge: raw prefix K/V is retained exactly
        if (
            blockk_fixed_cuda is None
            or _prepare_fixed_block32_cached is None
            or self._offline is None
            or float(self.config.eas) != 0.0
            or 0 not in self.layers
        ):
            return False
        window = int(self.config.sink_size) + int(self.config.local_size)
        return all(c.dense_key.shape[-2] == window for c in self.layers.values())

    def append(self, layer_idx: int, key_bhsd: torch.Tensor, value_bhsd: torch.Tensor) -> bool:
        """Merge one continuation chunk into this layer's cache.

        Exact equivalent of re-preparing over prefix+chunk: the outgoing window
        tokens ``[L_prev-local, L_new-local)`` leave the dense window and are
        quantized with the same fixed offline projections (per-token
        deterministic cascade — slice-independent), then packed onto the global
        segment. Key pack is per-token columns (concat); value pack is 64-token
        groups (rebuild the boundary group with carry extraction). Dense-fallback
        layers keep raw K/V and promote to compressed via a full prepare once
        the sequence outgrows sink+local. Returns False when this layer's state
        does not allow a merge (caller falls back to prepare)."""
        layer_idx = int(layer_idx)
        key_bhsd = key_bhsd.contiguous()
        value_bhsd = value_bhsd.contiguous()
        C = int(key_bhsd.shape[-2])
        if C <= 0:
            return True
        cfg = self.config
        sink, local = int(cfg.sink_size), int(cfg.local_size)

        if layer_idx in self._dense:
            dk, dv = self._dense[layer_idx]
            dk = torch.cat((dk, key_bhsd), dim=-2)
            dv = torch.cat((dv, value_bhsd), dim=-2)
            total = int(dk.shape[-2])
            if total <= sink + local:
                self._dense[layer_idx] = (dk.contiguous(), dv.contiguous())
                self.seq_lengths[layer_idx] = total
                return True
            del self._dense[layer_idx]  # exact: full raw sequence available
            self.prepare(layer_idx, dk, dv)
            return True

        cache = self.layers.get(layer_idx)
        if cache is None or cache.dense_key.shape[-2] != sink + local:
            return False
        heads = int(key_bhsd.shape[1])
        dim = int(key_bhsd.shape[-1])
        device, dtype = key_bhsd.device, key_bhsd.dtype
        kb = int(cfg.key_bits)
        vb = int(cfg.value_bits)

        # Outgoing tokens [L_prev-local, L_new-local): old dense local tail
        # followed by the chunk, minus the C tokens that stay in the new window.
        tail_k = cache.dense_key[:, :, sink:]
        tail_v = cache.dense_value[:, :, sink:]
        out_k = torch.cat((tail_k, key_bhsd), dim=-2)[:, :, :C]
        out_v = torch.cat((tail_v, value_bhsd), dim=-2)[:, :, :C]

        # Quantize + pack with the same fixed projections prepare used (same
        # kernel, window at 0 — per-token results are slice-independent).
        enc_k = _prepare_fixed_block32_cached(
            _slice_layer(self._offline["key_encoder"], layer_idx, heads), 1, heads, kb, dim, device, dtype
        ).view(1, heads, kb, dim)
        dec_k = _prepare_fixed_block32_cached(
            _slice_layer(self._offline["key_decoder"], layer_idx, heads), 1, heads, kb, dim, device, dtype
        ).view(1, heads, kb, dim)
        enc_v = _prepare_fixed_block32_cached(
            _slice_layer(self._offline["value_encoder"], layer_idx, heads), 1, heads, vb, dim, device, dtype
        ).view(1, heads, vb, dim)
        dec_v = _prepare_fixed_block32_cached(
            _slice_layer(self._offline["value_decoder"], layer_idx, heads), 1, heads, vb, dim, device, dtype
        ).view(1, heads, vb, dim)
        packed_key_new, _ = blockk_fixed_cuda(
            out_k, enc_k, dec_k, 0, C,
            zerorope=int(cfg.zerorope), apply_zerorope=True, pack_key_layout=True,
        )
        packed_value_tok, _ = blockk_fixed_cuda(
            out_v, enc_v, dec_v, 0, C,
            zerorope=0, apply_zerorope=False, pack_key_layout=False,
        )

        # Key pack [1,H,KB,gl]: per-token columns — plain concat.
        cache.packed_key = torch.cat((cache.packed_key, packed_key_new), dim=-1)

        # Value pack [1,H,groups,Cv]: 64-token groups. Rebuild from the last
        # partial group: extract its per-token words (carry), prepend them to
        # the new token words, re-pack whole groups.
        VB = vb // 64
        gl_old = int(cache.packed_key.shape[-1]) - C
        rem = gl_old % 64
        words = packed_value_tok  # [1,H,C,VB] per-token words
        if rem:
            gw = cache.packed_value[:, :, -1]  # [1,H,Cv]: bit i = token i's code bit c
            shifts_i = torch.arange(rem, device=device, dtype=torch.int64)
            bits = (gw.view(1, heads, VB, 64).unsqueeze(-1) >> shifts_i) & 1  # [1,H,VB,64,rem]
            shifts_s = torch.arange(64, device=device, dtype=torch.int64)
            carry = (bits << shifts_s.view(1, 1, 1, 64, 1)).sum(dim=3)  # [1,H,VB,rem]
            words = torch.cat((carry.permute(0, 1, 3, 2).contiguous(), words), dim=2)
        T = int(words.shape[2])
        G_tail = (T + 63) // 64
        s64 = torch.arange(64, device=device, dtype=torch.int64)
        cb = ((words.unsqueeze(-1) >> s64) & 1).view(1, heads, T, vb)  # code c = j*64+s
        Tp = G_tail * 64
        if Tp != T:
            cb = torch.nn.functional.pad(cb, (0, 0, 0, Tp - T))  # missing tokens: zero codes
        cb = cb.view(1, heads, G_tail, 64, vb)
        groups = (cb << s64.view(1, 1, 1, 64, 1)).sum(dim=3)  # [1,H,G_tail,Cv]
        keep = int(cache.packed_value.shape[2]) - (1 if rem else 0)
        cache.packed_value = torch.cat((cache.packed_value[:, :, :keep], groups), dim=2)

        # Dense window slides by exactly C tokens: sink + last `local` of
        # [old local ++ chunk].
        comb_k = torch.cat((tail_k, key_bhsd), dim=-2)
        comb_v = torch.cat((tail_v, value_bhsd), dim=-2)
        cache.dense_key = torch.cat((cache.dense_key[:, :, :sink], comb_k[:, :, -local:]), dim=-2)
        cache.dense_value = torch.cat((cache.dense_value[:, :, :sink], comb_v[:, :, -local:]), dim=-2)
        self.seq_lengths[layer_idx] += C
        return True

    def decode(self, layer_idx: int, q_bhsd, k_bhsd, v_bhsd) -> torch.Tensor:
        layer_idx = int(layer_idx)
        if layer_idx in self._dense:
            dk, dv = self._dense[layer_idx]
            dk = torch.cat((dk, k_bhsd.contiguous()), dim=-2)
            dv = torch.cat((dv, v_bhsd.contiguous()), dim=-2)
            self._dense[layer_idx] = (dk, dv)
            out = torch.nn.functional.scaled_dot_product_attention(
                q_bhsd, dk, dv, enable_gqa=True
            )
            self.seq_lengths[layer_idx] += int(k_bhsd.shape[-2])
            return out
        cache = self.layers[layer_idx]
        # Backend dedup leaves ragged packed tensors as non-contiguous views
        # into a decode batch's copy; the per-request kernels need them dense.
        for f in ("packed_key", "packed_value"):
            if not getattr(cache, f).is_contiguous():
                setattr(cache, f, getattr(cache, f).contiguous())
        out = decode_attention(
            q_bhsd, cache, decode_key=k_bhsd, decode_value=v_bhsd,
            update_cache=True, **self.config.decode_kwargs(),
        )
        self.seq_lengths[layer_idx] += int(k_bhsd.shape[-2])
        return out

    def cache_bytes(self) -> int:
        seen, total = set(), 0

        def add(o):
            nonlocal total
            if torch.is_tensor(o):
                if o.is_cuda and o.data_ptr() not in seen:
                    seen.add(o.data_ptr())
                    total += o.numel() * o.element_size()
            elif isinstance(o, (list, tuple)):
                [add(x) for x in o]
            elif isinstance(o, dict):
                [add(x) for x in o.values()]
            elif hasattr(o, "__dict__"):
                [add(x) for x in vars(o).values()]

        add(self.layers)
        add(self._dense)
        return total


class _FlatMixin:
    """Shared flattened<->[1,H,S,D] helpers for sglang's per-token layout."""

    def _to_bhsd(self, x: torch.Tensor, heads: int) -> torch.Tensor:
        s = x.shape[0]
        return x.view(s, heads, self.head_dim).transpose(0, 1).unsqueeze(0).contiguous()

    def _from_bhsd(self, o: torch.Tensor) -> torch.Tensor:
        s = o.shape[2]
        return o.squeeze(0).transpose(0, 1).reshape(s, self.num_q_heads * self.head_dim)


class PicoKVCore(_FlatMixin):
    """Single-request (batch=1) PiCoKV engine in sglang's flattened layout."""

    def __init__(
        self,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        scaling: float,
        config: Optional[StandaloneConfig] = None,
        offline_proj_path: Optional[str] = None,
    ) -> None:
        if num_q_heads // num_kv_heads != 4:
            raise ValueError(f"PiCoKV needs GQA=4; got QH={num_q_heads}, KVH={num_kv_heads}")
        if head_dim != 128:
            raise ValueError(f"PiCoKV needs head_dim==128; got {head_dim}")
        self.num_q_heads, self.num_kv_heads, self.head_dim = num_q_heads, num_kv_heads, head_dim
        self.scaling = float(scaling)
        self.config = config or StandaloneConfig()
        self._offline = load_offline_projections_pure(offline_proj_path) if offline_proj_path else None
        self.req = PicoKVRequestCache(self.config, self._offline)

    def reset(self) -> None:
        self.req = PicoKVRequestCache(self.config, self._offline)

    def extend(self, layer_id, q, k, v) -> torch.Tensor:
        q4 = self._to_bhsd(q, self.num_q_heads)
        k4 = self._to_bhsd(k, self.num_kv_heads)
        v4 = self._to_bhsd(v, self.num_kv_heads)
        self.req.prepare(layer_id, k4, v4)
        o = torch.nn.functional.scaled_dot_product_attention(
            q4, k4, v4, is_causal=True, enable_gqa=True, scale=self.scaling
        )
        return self._from_bhsd(o)

    def decode(self, layer_id, q, k, v) -> torch.Tensor:
        q4 = self._to_bhsd(q, self.num_q_heads)
        k4 = self._to_bhsd(k, self.num_kv_heads)
        v4 = self._to_bhsd(v, self.num_kv_heads)
        return self._from_bhsd(self.req.decode(layer_id, q4, k4, v4))


class PicoKVBatchDecoder(_FlatMixin):
    """Step 3: continuous-batch decode over requests at INDEPENDENT context
    lengths. Each request keeps its own ``PicoKVRequestCache``; a batched decode
    step dispatches per request over the verified single-request kernel and
    stacks the results, so a ragged batch is correct. A fused ragged/paged
    kernel (one launch for the whole batch) is the remaining optimization.
    """

    def __init__(self, num_q_heads, num_kv_heads, head_dim, scaling,
                 config=None, offline_proj_path=None):
        self.num_q_heads, self.num_kv_heads, self.head_dim = num_q_heads, num_kv_heads, head_dim
        self.scaling = float(scaling)
        self.config = config or StandaloneConfig()
        self._offline = load_offline_projections_pure(offline_proj_path) if offline_proj_path else None
        self.reqs: Dict[int, PicoKVRequestCache] = {}

    def add_request(self, req_id: int) -> None:
        self.reqs[req_id] = PicoKVRequestCache(self.config, self._offline)

    def remove_request(self, req_id: int) -> None:
        self.reqs.pop(req_id, None)

    def prepare(self, req_id: int, layer_id: int, k_bhsd, v_bhsd) -> None:
        self.reqs[req_id].prepare(layer_id, k_bhsd, v_bhsd)

    def decode_batch(self, layer_id: int, req_ids: List[int], q_bhsd, k_bhsd, v_bhsd):
        """q/k/v: [num_reqs, heads, 1, D] aligned with ``req_ids``. Returns
        [num_reqs, QH, 1, D] — each row decoded against that request's cache."""
        outs = []
        for i, rid in enumerate(req_ids):
            outs.append(self.reqs[rid].decode(
                layer_id, q_bhsd[i:i + 1], k_bhsd[i:i + 1], v_bhsd[i:i + 1]))
        return torch.cat(outs, dim=0)

    def total_bytes(self) -> int:
        return sum(r.cache_bytes() for r in self.reqs.values())


# --- sglang glue: only defined when sglang is importable --------------------
try:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

    _SGLANG_AVAILABLE = True
except Exception:  # pragma: no cover - lets the core be imported standalone
    AttentionBackend = object  # type: ignore
    _SGLANG_AVAILABLE = False

try:
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
        eager_on_graph,
        is_in_breakable_cuda_graph,
    )
except Exception:  # pragma: no cover - standalone import / older sglang

    def eager_on_graph(enable, capture_stub=None):
        return lambda fn: fn

    def is_in_breakable_cuda_graph():
        return False


class PicoKVAttnBackend(AttentionBackend):
    """SGLang attention backend backed by per-request PiCoKV caches.

    Supports a batch of requests at INDEPENDENT context lengths (continuous
    batching): each request slot (``forward_batch.req_pool_indices``) owns a
    :class:`PicoKVRequestCache`. ``forward_extend`` splits the concatenated
    extend tokens by ``extend_seq_lens`` and compresses each request's prompt;
    ``forward_decode`` decodes one token per running request against its own
    compressed cache. K/V are NOT written to the stock dense pool — PiCoKV keeps
    only packed codes + sink/local window.
    """

    def __init__(self, model_runner):
        super().__init__()
        import os
        from picokv.picokv_standalone import StandaloneConfig

        self.device = model_runner.device
        cfg = model_runner.model_config
        tp = getattr(model_runner, "tp_size", 1)
        self.num_q_heads = cfg.num_attention_heads // tp
        self.num_kv_heads = cfg.get_num_kv_heads(tp)
        self.head_dim = cfg.head_dim
        self.scaling = 1.0 / (self.head_dim ** 0.5)
        # PICOKV_EXP_BITS: attention-weight bitplanes in the tail kernel
        # (values stay packed codes; only decode-side weight quantization
        # changes). 4 = default; 2 saves ~13% of the main scan (quality-gated
        # by NIAH before flipping). PICOKV_EAS: dense retention of the
        # highest-error global tokens (StandaloneConfig default 0.0; the HF
        # runner's default is 0.01). Disables incremental chunk-append when > 0.
        self._scfg = StandaloneConfig(
            exp_bits=int(os.environ.get("PICOKV_EXP_BITS", "4")),
            eas=float(os.environ.get("PICOKV_EAS", "0.0")))
        # Offline projections are the DEFAULT serving path: prefill then runs
        # the fused blockk prepare (one kernel per side) instead of per-request
        # online direction learning. Resolution order: PICOKV_OFFLINE_PROJ
        # (explicit path) > the conventional shipped .pt for this model >
        # online. PICOKV_ONLINE=1 forces the online path (A/B / fallback).
        path = os.environ.get("PICOKV_OFFLINE_PROJ")
        if path is None and os.environ.get("PICOKV_ONLINE", "0") != "1":
            path = self._default_offline_proj(cfg, self._scfg)
        self._offline = load_offline_projections_pure(path) if path else None
        self._reqs = {}  # req_pool_index -> PicoKVRequestCache (ragged fallback)
        # Finished requests are pruned from _reqs once their req-pool row is
        # free again (see _prune_finished); without it a cache lived until its
        # row was reused by a later request.
        self._mr = model_runner
        # PICOKV_KV_DEDUP=1 (default): after a layer's batched copy is built,
        # each request's packed/projection tensors become views into it, so
        # decode holds the compressed cache ONCE instead of per-request + batched
        # (b16@120k OOM'd at 78.6 GB with 15 requests; ~4 GB/request vs 2.46).
        self._dedup = os.environ.get("PICOKV_KV_DEDUP", "1") == "1"
        self._batch = None       # live decode batch: rids + per-layer fixed tensors
        self._pools = None       # stashed pools from the last decode batch (carry-over)
        self._cont_warned = False  # chunked-prefill continuation note printed once
        # flashinfer wrappers: {"wrapper":..., "planned":...} or False=disabled.
        # _fi = paged decode (dense window); _fi_pre = ragged prefill.
        self._fi = None

        # Custom dense-window kernel. PICOKV_WINDOW_KERNEL=auto (default) uses
        # it for batches of <= PICOKV_WINDOW_MAX_BS (8) requests and the
        # flashinfer split-kv tensor-core kernel above that; 1 = always,
        # 0 = never. BCG ITL @8k prompt (ms/step, custom vs fi): b1 7.16/7.41,
        # b4 8.21/8.86, b8 9.80/9.84, b16 12.46/12.37. The custom path also
        # skips the per-step fi wrapper.plan and keeps one graph signature as
        # the window grows. The choice is fixed per batch (sh["win"]).
        self._win = None  # None = not requested; False = off/failed; else fn
        _wmode = os.environ.get("PICOKV_WINDOW_KERNEL", "auto").strip().lower()
        self._win_max_bs = (1 << 30) if _wmode == "1" else int(
            os.environ.get("PICOKV_WINDOW_MAX_BS", "8"))
        if _wmode != "0":
            try:
                from picokv.picokv_cuda_extension.picokv_cuda import (
                    load_picokv_extension,
                )
                self._win = getattr(load_picokv_extension(), "picokv_window_attn_fn")
            except Exception as e:
                print(f"[PicoKVAttnBackend] window kernel unavailable ({e!r})", flush=True)
                self._win = False
        if self._win:
            print("[PicoKVAttnBackend] window kernel ON for batch <= "
                  f"{self._win_max_bs} (custom dense-window path, no fi plan)",
                  flush=True)
        self._fi_pre = None
        self._prefill = os.environ.get("PICOKV_PREFILL", "fi").strip().lower()
        # CUDA-graph decode (item 1): PICOKV_GRAPHS=0 disables; the first
        # PICOKV_GRAPH_WARMUP steps of each batch run eager (cublas/JIT warmup)
        # before per-layer graphs are captured.
        self._graphs_on = os.environ.get("PICOKV_GRAPHS", "1") != "0"
        self._graph_warmup = max(0, int(os.environ.get("PICOKV_GRAPH_WARMUP", "3")))
        # Eager decode reads req ids (a device->host sync) once per step, at
        # the step's first layer, instead of at every layer call (-1.4 to
        # -2.4 ms/step measured). PICOKV_RIDS_PER_STEP=0 restores per-layer
        # reads. The breakable engine graph reads them in replay-prep instead.
        self._rids_per_step = os.environ.get("PICOKV_RIDS_PER_STEP", "1") == "1"
        # Host req ids come from the scheduler's CPU mirror
        # (forward_batch.req_pool_indices_cpu) when present: no device->host
        # sync, so the overlap scheduler keeps the GPU queue full across
        # steps. Spot-checked against the device copy on every batch change
        # and every 256 reads; a mismatch disables the mirror for good.
        # PICOKV_RIDS_CPU=0 always reads the device copy.
        self._rids_cpu = os.environ.get("PICOKV_RIDS_CPU", "1") == "1"
        self._rids_n = 0
        self._fused_append = os.environ.get("PICOKV_FUSED_APPEND", "1") == "1"
        # sglang breakable decode graph state (see init_forward_metadata_out_graph)
        self._eng_capture = False  # BCG warmup/capture on dummy requests
        self._eng_rids = None      # real req ids of the step being replayed
        self._eng_bs = 0           # padded bucket size of that step
        # BCG replay: the per-layer graph writes the attention output straight
        # into the break's captured output buffer (recorded per (layer, bucket)
        # at capture), so the break's output copy is a self-copy no-op.
        # PICOKV_ENG_DIRECT_OUT=0 restores the copy.
        self._eng_direct_out = os.environ.get("PICOKV_ENG_DIRECT_OUT", "1") == "1"
        self._eng_outs = {}
        # Per-layer graph: the dense-window branch (K/V append + window or fi
        # attention) runs on a forked stream, overlapped with the query
        # projection/quantize/pack; the global kernel joins both.
        # PICOKV_GRAPH_FORK=0 keeps the serial order.
        self._graph_fork = os.environ.get("PICOKV_GRAPH_FORK", "1") == "1"
        self._fork_stream = None
        self._graph_pool = None
        self._n_extend = 0
        self._n_decode = 0
        print(f"[PicoKVAttnBackend] ACTIVE (multi-request) qh={self.num_q_heads} "
              f"kvh={self.num_kv_heads} hd={self.head_dim} "
              f"mode={'offline (' + path + ')' if self._offline else 'online'} "
              f"prefill={self._prefill} graphs={'on' if self._graphs_on else 'off'}", flush=True)

    @staticmethod
    def _default_offline_proj(cfg, scfg) -> Optional[str]:
        """Resolve the shipped offline projection .pt for the served model.

        Conventional location: ``<PiCoKV repo>/offline_projection/projections/
        {model_dir_name}_k{key_bits}_v{value_bits}.pt`` (produced by
        ``offline_projection/create_offline_projections.py``). Returns None
        when the model has no shipped projection (caller falls back to online).
        """
        try:
            model_path = getattr(cfg, "model_path", None)
            if not model_path:
                return None
            from pathlib import Path

            import picokv

            proj_dir = (
                Path(picokv.__file__).resolve().parent.parent
                / "offline_projection"
                / "projections"
            )
            tag = Path(str(model_path).rstrip("/")).name
            cand = proj_dir / f"{tag}_k{scfg.key_bits}_v{scfg.value_bits}.pt"
            if cand.is_file():
                return str(cand)
            print(f"[PicoKVAttnBackend] no offline projection at {cand}; "
                  f"falling back to online", flush=True)
        except Exception as e:  # pragma: no cover - defensive
            print(f"[PicoKVAttnBackend] offline default lookup failed ({e!r}); "
                  f"using online", flush=True)
        return None

    def init_forward_metadata(self, forward_batch):
        # eager forward (no engine graph): forward_decode reads req ids itself
        self._eng_capture = False
        self._eng_rids = None

    # ------------------------------------------------------------------
    # engine-level CUDA graph hooks. Only the BREAKABLE decode graph is
    # supported (cuda_graph_backend_decode="breakable"): forward_decode is an
    # eager break (eager_on_graph), so replay runs the graphed embedding /
    # proj / MLP segments and calls this backend's Python once per layer —
    # the per-request cache state and the per-layer graphs keep working. A
    # whole-step ("full") graph would skip that Python entirely (capture runs
    # before any prefill, so replay would never run PiCoKV attention) and is
    # rejected at capture.
    # ------------------------------------------------------------------

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        pass  # no engine-graph buffers: the break owns its inputs/outputs

    def init_forward_metadata_out_graph(self, forward_batch, in_capture: bool = False):
        if in_capture:
            if not is_in_breakable_cuda_graph():
                raise NotImplementedError(
                    "picokv backend supports only the breakable decode CUDA "
                    "graph (cuda_graph_backend_decode='breakable'); a full "
                    "graph replay would skip the host-side per-request cache "
                    "state. Use breakable, or disable_cuda_graph=True.")
            # warmup + capture run on dummy requests: forward_decode returns
            # zeros (the break output only needs its shape/address recorded)
            self._eng_capture = True
            self._eng_rids = None
            return
        # replay-prep (outside the graph): the step's only read of the real
        # (unpadded) req ids; every layer's break reuses it.
        if torch.cuda.is_current_stream_capturing():
            # the metadata glue graph (SGLANG_ENABLE_METADATA_GLUE_GRAPH)
            # would freeze this host state at capture: refuse so it falls back
            raise RuntimeError("picokv replay-prep sets host state; not capturable")
        self._eng_capture = False
        raw = forward_batch.batch_size - forward_batch.num_padding
        self._eng_rids = self._host_rids(forward_batch, raw)
        self._eng_bs = forward_batch.batch_size

    def init_forward_metadata_in_graph(self, forward_batch):
        pass

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def _eng_pad(self, o):
        # BCG replay: the break output buffer has the padded bucket's rows
        pad = self._eng_bs - o.shape[0]
        return o if pad == 0 else torch.nn.functional.pad(o, (0, 0, 0, pad))

    def _warn_continuation(self, appended: bool) -> None:
        """One-time note when a chunked-prefill continuation is detected."""
        if self._cont_warned:
            return
        self._cont_warned = True
        if appended:
            print(
                "[PicoKVAttnBackend] chunked-prefill continuation: cache merged "
                "incrementally (all chunks retained), but prefill OUTPUT of "
                "chunks after the first attends only the current chunk — set "
                "chunked_prefill_size >= prompt length for exact hidden states.",
                flush=True,
            )
        else:
            print(
                "[PicoKVAttnBackend] WARNING: chunked-prefill continuation "
                "re-prepared from the current chunk only — earlier chunks' "
                "compressed KV LOST (incremental append needs eas=0 default + "
                "offline projections). Set chunked_prefill_size >= prompt length.",
                flush=True,
            )

    @staticmethod
    def _ints(x):
        if hasattr(x, "tolist"):
            return [int(v) for v in x.tolist()]
        return [int(v) for v in x]

    def _host_rids(self, fb, n):
        """The batch's first n req pool indices as host ints: the CPU mirror
        when it is present and unpadded, else a device read (D2H sync)."""
        m = getattr(fb, "req_pool_indices_cpu", None) if self._rids_cpu else None
        if m is None or m.shape[0] != n:
            return self._ints(fb.req_pool_indices[:n])
        r = self._ints(m)
        self._rids_n += 1
        if self._rids_n == 1:
            print("[PicoKVAttnBackend] decode req ids from the CPU mirror "
                  "(no D2H sync)", flush=True)
        if (self._batch is None or self._batch.get("rids") != r
                or self._rids_n % 256 == 0):
            d = self._ints(fb.req_pool_indices[:n])
            if d != r:
                print(f"[PicoKVAttnBackend] req_pool_indices_cpu mirror {r} != "
                      f"device {d}; reading the device copy from now on", flush=True)
                self._rids_cpu = False
                return d
        return r

    def _log_l0(self, path, msg):
        # step-first line: once per path change within a batch, then every
        # 256 steps (a per-step print+flush is host time on the critical path)
        b = self._batch
        b["nl0"] = b.get("nl0", 0) + 1
        if b.get("last_path") != path or b["nl0"] % 256 == 0:
            b["last_path"] = path
            print(msg, flush=True)

    def _bhsd(self, chunk, heads):
        # [n, heads, D] -> [1, heads, n, D]
        return chunk.reshape(-1, heads, self.head_dim).permute(1, 0, 2).unsqueeze(0).contiguous()

    # ------------------------------------------------------------------
    # prefill
    # ------------------------------------------------------------------

    def forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kw):
        import torch as _t
        rids = self._ints(forward_batch.req_pool_indices)
        eseq = self._ints(forward_batch.extend_seq_lens)
        q_ = q.view(-1, self.num_q_heads, self.head_dim)
        k_ = k.view(-1, self.num_kv_heads, self.head_dim)
        v_ = v.view(-1, self.num_kv_heads, self.head_dim)

        # A prefill invalidates any live decode batch (membership may change):
        # stash its pools so continuing requests carry their appended tokens.
        if self._batch is not None:
            self._stash_pools()
        self._batch = None
        if layer.layer_id == 0:
            self._prune_finished(rids)
        # extend_prefix_lens > 0 marks a chunked-prefill continuation: the
        # request already holds its prefix's compressed cache.
        _epl = getattr(forward_batch, "extend_prefix_lens", None)
        _epl_cpu = getattr(forward_batch, "extend_prefix_lens_cpu", None)
        plens = (
            self._ints(_epl_cpu)
            if _epl_cpu is not None
            else (self._ints(_epl) if _epl is not None else [0] * len(rids))
        )
        off = 0
        for rid, ln, plen in zip(rids, eseq, plens):
            cache = self._reqs.get(rid)
            if layer.layer_id == 0:
                # Layer 0 starts every extend: decide append vs fresh prepare
                # once (can_append validates prefix length + prerequisites).
                cont = plen > 0 and cache is not None and cache.can_append(plen)
                if not cont:
                    if plen > 0:
                        self._warn_continuation(False)
                    cache = PicoKVRequestCache(self._scfg, self._offline)
                    self._reqs[rid] = cache
                elif not self._cont_warned:
                    self._warn_continuation(True)
                cache._append_mode = cont
            elif cache is None:
                cache = PicoKVRequestCache(self._scfg, self._offline)
                self._reqs[rid] = cache
            kb = self._bhsd(k_[off:off + ln], self.num_kv_heads)
            vb = self._bhsd(v_[off:off + ln], self.num_kv_heads)
            if cache._append_mode and cache.append(layer.layer_id, kb, vb):
                pass  # merged incrementally (exact for the cache state)
            else:
                cache.prepare(layer.layer_id, kb, vb)
            if self._pools is not None:
                # this request re-prefilled: its stashed pool data is stale
                self._pools["idx"].pop(rid, None)
            off += ln
        # Prefill attention: ONE batched ragged flashinfer call over the
        # concatenated token-major q/k/v (exact causal attention — replaces the
        # per-request SDPA loop). PICOKV_PREFILL=sdpa forces the old path.
        o = self._prefill_attn(q_, k_, v_, eseq) if self._prefill != "sdpa" else None
        if o is not None:
            self._n_extend += 1
            if layer.layer_id == 0:
                print(f"[PicoKVAttnBackend] forward_extend L0 RAGGED reqs={len(rids)} "
                      f"tokens={eseq} attn=flashinfer", flush=True)
            return o.reshape(-1, self.num_q_heads * self.head_dim)
        F = _t.nn.functional
        outs, off = [], 0
        for rid, ln in zip(rids, eseq):
            qb = self._bhsd(q_[off:off + ln], self.num_q_heads)
            kb = self._bhsd(k_[off:off + ln], self.num_kv_heads)
            vb = self._bhsd(v_[off:off + ln], self.num_kv_heads)
            o = F.scaled_dot_product_attention(
                qb, kb, vb, is_causal=True, enable_gqa=True, scale=self.scaling)
            outs.append(o.squeeze(0).permute(1, 0, 2).reshape(ln, self.num_q_heads * self.head_dim))
            off += ln
        self._n_extend += 1
        if layer.layer_id == 0:
            print(f"[PicoKVAttnBackend] forward_extend L0 RAGGED reqs={len(rids)} "
                  f"tokens={eseq} attn=sdpa", flush=True)
        return outs[0] if len(outs) == 1 else _t.cat(outs, dim=0)

    def _prefill_attn(self, q_, k_, v_, eseq):
        """One batched exact causal prefill attention via flashinfer's ragged
        prefill kernel over the concatenated token-major q/k/v. Returns
        [ΣL,QH,D] or None when unavailable (caller falls back to per-request
        SDPA — same math)."""
        if self._fi_pre is False:
            return None
        try:
            if self._fi_pre is None:
                import flashinfer

                ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=self.device)
                self._fi_pre = {
                    "wrapper": flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, kv_layout="NHD"),
                    "planned": None,  # eseq tuple of the last plan() (skip re-plan)
                }
            fi = self._fi_pre
            key = tuple(eseq)
            if fi["planned"] != key:
                import numpy as np

                ip = np.zeros(len(eseq) + 1, dtype=np.int32)
                ip[1:] = np.cumsum(eseq).astype(np.int32)
                indptr = torch.from_numpy(ip).to(q_.device)
                fi["wrapper"].plan(
                    indptr, indptr,
                    self.num_q_heads, self.num_kv_heads, self.head_dim,
                    causal=True, sm_scale=self.scaling,
                    q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
                )
                fi["planned"] = key
            return fi["wrapper"].run(q_, k_, v_)
        except Exception as e:
            if isinstance(self._fi_pre, dict):
                print(f"[PicoKVAttnBackend] ragged prefill DISABLED ({e!r}); "
                      f"using SDPA", flush=True)
            self._fi_pre = False
            return None

    # ------------------------------------------------------------------
    # decode: batch/pool construction
    # ------------------------------------------------------------------

    def _prune_finished(self, live):
        """Drop the caches of finished requests: a req-pool row back in the
        pool's free list belongs to no live request. ``live`` (this forward's
        rids) is never pruned — under the overlap scheduler a request's row
        can be freed while its last step is still in flight. Besides freeing
        memory early, this releases the batched storage that dedup views
        (``_fixed_layer``) of finished requests would otherwise pin."""
        if not self._reqs:
            return
        fs = getattr(getattr(self._mr, "req_to_token_pool", None), "free_slots", None)
        if not isinstance(fs, list):
            return
        free, live = set(fs), set(live)
        for rid in [r for r in self._reqs if r in free and r not in live]:
            del self._reqs[rid]

    def _stash_pools(self):
        """Snapshot the live decode batch's pools + per-request window lengths
        so the NEXT batch can carry decode-appended tokens across the rebuild:
        a request's cache ``dense_key`` stays the prefill-time window; only the
        pool holds the tokens appended since."""
        b = self._batch
        if b is None:
            return
        sh = b.get("shared")
        pools = {lid: (fx["kbuf"], fx["vbuf"]) for lid, fx in b["fixed"].items() if fx}
        if sh is None or not pools:
            self._pools = None
            return
        self._pools = {
            "idx": {rid: i for i, rid in enumerate(b["rids"])},
            "lens": list(sh["dlens_py"]),  # per-request window len incl. last append
            "pools": pools,
        }

    def _fixed_layer(self, rids, layer_id):
        """Stack the length-invariant + compressed-global tensors for one layer
        across the batch, padding packed_key/value to the batch's max global
        length (done ONCE per batch; these never change during decode), and
        build that layer's persistent token-major dense pool ``[B,cap,KVH,D]``
        (cap leaves headroom for generated tokens; doubled on overflow).
        Returns a dict of fixed tensors, or None if any request lacks a
        compressed cache for this layer (short-prompt dense fallback)."""
        import torch as _t
        caches = []
        for rid in rids:
            rc = self._reqs.get(rid)
            if rc is None or layer_id not in rc.layers:
                return None
            caches.append(rc.layers[layer_id])
        for f in ("key_projection", "value_projection", "key_projection_t"):
            if len({tuple(getattr(c, f).shape) for c in caches}) != 1:
                return None
        gls = [int(c.packed_key.shape[-1]) for c in caches]
        gvs = [int(c.packed_value.shape[-2]) for c in caches]
        max_gl = max(gls)
        gmax = (max_gl + 63) // 64
        B = len(caches)
        # Zero-padded batched copies, filled per request (no padded temporaries:
        # the build's transient peak stays at one layer's batched copy).
        pk0, pv0 = caches[0].packed_key, caches[0].packed_value
        pk = _t.zeros((B,) + tuple(pk0.shape[1:-1]) + (max_gl,), dtype=pk0.dtype, device=pk0.device)
        pv = _t.zeros((B,) + tuple(pv0.shape[1:-2]) + (gmax, pv0.shape[-1]),
                      dtype=pv0.dtype, device=pv0.device)
        for i, c in enumerate(caches):
            pk[i:i + 1, ..., :gls[i]] = c.packed_key
            pv[i:i + 1, ..., :gvs[i], :] = c.packed_value
        dev = caches[0].packed_key.device
        dt = caches[0].dense_key.dtype
        # Seed each request's dense window. Requests carried over from the last
        # decode batch keep their appended tokens (their cache's dense_key is
        # the stale prefill window); freshly prefilled requests seed from it.
        pools = self._pools
        carry = pools["idx"] if pools is not None else {}
        dlens = []
        for i, rid in enumerate(rids):
            j = carry.get(rid)
            dlens.append(int(pools["lens"][j]) if j is not None
                         else int(caches[i].dense_key.shape[-2]))
        cap = max(dlens) + 512
        kbuf = _t.zeros(B, cap, self.num_kv_heads, self.head_dim, device=dev, dtype=dt)
        vbuf = _t.zeros_like(kbuf)
        opool = pools["pools"].get(layer_id) if pools is not None else None
        for i, c in enumerate(caches):
            ln = dlens[i]
            j = carry.get(rids[i])
            if j is not None and opool is not None:
                kbuf[i, :ln] = opool[0][j, :ln]
                vbuf[i, :ln] = opool[1][j, :ln]
            else:
                kbuf[i, :ln] = c.dense_key[0].transpose(0, 1)  # [len,KVH,D]
                vbuf[i, :ln] = c.dense_value[0].transpose(0, 1)
        # Block schedule for the tail kernel's paged grid: map each grid.x block
        # -> (request, local tile) so ONLY real blocks launch. Block covers
        # BlockTiles(2)*kTileSize(256) = 512 global tokens.
        BLK = 512
        sr, stl = [], []
        for b, g in enumerate(gls):
            nb = (g + BLK - 1) // BLK
            sr.extend([b] * nb)
            stl.extend(range(nb))
        # Persistent batched cache: with the dense attention precomputed
        # (dense_attn) the dense_key/value views are never read, so bind the
        # FULL pool view once and keep the tail workspaces alive across steps
        # (the no-dense-attn fallback builds a temporary sliced cache instead).
        batched = PreparedKVCache(
            dense_key=kbuf.permute(0, 2, 1, 3),
            dense_value=vbuf.permute(0, 2, 1, 3),
            key_projection=_t.cat([c.key_projection for c in caches], 0).contiguous(),
            value_projection=_t.cat([c.value_projection for c in caches], 0).contiguous(),
            key_projection_t=_t.cat([c.key_projection_t for c in caches], 0).contiguous(),
            packed_key=pk,
            packed_value=pv,
            exp_bits=caches[0].exp_bits,
        )
        if self._dedup:
            # Re-point each request at its slice of the batched copy so the
            # per-request originals are freed. Ragged packed slices are
            # non-contiguous; PicoKVRequestCache.decode re-materializes them
            # if the per-request path ever touches this layer again.
            for i, c in enumerate(caches):
                c.packed_key = pk[i:i + 1, ..., :gls[i]]
                c.packed_value = pv[i:i + 1, ..., :gvs[i], :]
                c.key_projection = batched.key_projection[i:i + 1]
                c.value_projection = batched.value_projection[i:i + 1]
                c.key_projection_t = batched.key_projection_t[i:i + 1]
        return dict(
            batched=batched,
            kbuf=kbuf,
            vbuf=vbuf,
            exp_bits=caches[0].exp_bits,
            global_lens=_t.tensor(gls, dtype=_t.int32, device=dev),
            gl_max=max(gls),
            sched_req=_t.tensor(sr, dtype=_t.int32, device=dev),
            sched_tile=_t.tensor(stl, dtype=_t.int32, device=dev),
            dlens0=dlens,
            equal=(len(set(gls)) == 1 and len(set(dlens)) == 1),
            dlens_equal=(len(set(dlens)) == 1),
        )

    # ------------------------------------------------------------------
    # decode: per-step bookkeeping
    # ------------------------------------------------------------------

    def _step_begin(self, fixed):
        """Per-STEP bookkeeping, once at the step's first layer call (all
        layers of a step share this state): advance the shared write positions
        (one device add — no per-layer tensor churn), grow every layer's pool
        on overflow, and re-plan the paged dense kernel (one host sync per
        step)."""
        import torch as _t
        sh = self._batch["shared"]
        if sh is None:
            dl = fixed["dlens0"]
            dev = fixed["kbuf"].device
            B = len(dl)
            cap = fixed["kbuf"].shape[1]
            sh = {
                # write position of this step (device-side: scatter + graphs
                # read VALUES, never python ints)
                "wpos": _t.tensor(dl, dtype=_t.int32, device=dev),
                # per-request dense kv lens INCLUDING this step's token
                "lpl": _t.tensor([d + 1 for d in dl], dtype=_t.int32, device=dev),
                "dlens_py": [d + 1 for d in dl],
                "cur": max(dl) + 1,
                "ar": _t.arange(B, device=dev),
                "ar32": _t.arange(B, dtype=_t.int32, device=dev),
                "cap": cap,
                # Paged tables (page_size=1 — the only page size this
                # flashinfer build runs correctly): persistent buffers whose
                # CONTENTS are rebuilt on device each step (stable kernel-input
                # addresses, CUDA-graph-friendly, no per-step H2D).
                "indptr": _t.zeros(B + 1, dtype=_t.int32, device=dev),
                "indices": _t.empty(B * cap, dtype=_t.int32, device=dev),
                "ones": _t.ones(B, dtype=_t.int32, device=dev),
                "T": 0,
                # custom window kernel for this batch (else flashinfer)
                "win": bool(self._win) and B <= self._win_max_bs,
            }
            self._batch["shared"] = sh
        else:
            sh["wpos"].add_(1)
            sh["lpl"].add_(1)
            sh["dlens_py"] = [d + 1 for d in sh["dlens_py"]]
            sh["cur"] += 1
        if sh["cur"] > sh["cap"]:  # grow every layer's pool together (rare)
            newcap = sh["cap"] * 2
            for fx in self._batch["fixed"].values():
                if fx is None:
                    continue
                kbuf, vbuf = fx["kbuf"], fx["vbuf"]
                nk = _t.zeros(kbuf.shape[0], newcap, kbuf.shape[2], kbuf.shape[3],
                              device=kbuf.device, dtype=kbuf.dtype)
                nv = _t.zeros_like(nk)
                nk[:, : kbuf.shape[1], :] = kbuf
                nv[:, : vbuf.shape[1], :] = vbuf
                fx["kbuf"], fx["vbuf"] = nk, nv
                fx["batched"].dense_key = nk.permute(0, 2, 1, 3)
                fx["batched"].dense_value = nv.permute(0, 2, 1, 3)
            sh["cap"] = newcap
            sh["indices"] = _t.empty(
                kbuf.shape[0] * newcap, dtype=_t.int32, device=sh["wpos"].device)
            self._batch["g"]["by_layer"].clear()  # pool/table addresses changed
        if not sh["win"]:  # the window kernel reads lpl only, never the tables
            self._rebuild_tables(sh)
        self._plan_paged(sh)
        g = self._batch["g"]
        g["n"] += 1
        if g["bufs"] is None and (self._graphs_on or sh["win"]):
            dev = sh["wpos"].device
            B = len(sh["dlens_py"])
            g["bufs"] = {
                "q": _t.empty(B, self.num_q_heads, self.head_dim, device=dev, dtype=torch.bfloat16),
                "k": _t.empty(B, self.num_kv_heads, self.head_dim, device=dev, dtype=torch.bfloat16),
                "v": _t.empty(B, self.num_kv_heads, self.head_dim, device=dev, dtype=torch.bfloat16),
                "fi_out": _t.empty(B, self.num_q_heads, self.head_dim, device=dev, dtype=torch.bfloat16),
                "fi_lse": _t.empty(B, self.num_q_heads, device=dev, dtype=torch.float32),
                # custom window kernel output (ln-space LSE, unlike fi_lse)
                "win_lse": _t.empty(B, self.num_q_heads, device=dev, dtype=torch.float32),
            }
        return sh

    def _append_kv(self, kbuf, vbuf, sh, k_, v_):
        """Write this step's K/V [B,KVH,D] at each request's wpos in the dense
        pools: one fused Triton launch when every row is dense, else two
        index_put kernels."""
        D = self.head_dim
        row = self.num_kv_heads * D
        if (self._fused_append and _append_kv_kernel is not None
                and row & (row - 1) == 0
                and k_.stride(2) == 1 and k_.stride(1) == D
                and v_.stride(2) == 1 and v_.stride(1) == D
                and kbuf.is_contiguous() and vbuf.is_contiguous()):
            _append_kv_kernel[(k_.shape[0],)](
                kbuf, vbuf, k_, v_, sh["wpos"], kbuf.shape[1],
                k_.stride(0), v_.stride(0), ROW=row)
            return
        kbuf[sh["ar"], sh["wpos"]] = k_
        vbuf[sh["ar"], sh["wpos"]] = v_

    def _rebuild_tables(self, sh):
        """Rebuild the page_size=1 paged tables in place on device (once per
        step at the step's first layer): ``indices`` = each request's live pool
        positions (request-major), ``indptr`` = cumulative live lengths."""
        import torch as _t
        l = sh["lpl"]
        T = int(sum(sh["dlens_py"]))
        cum = _t.cumsum(l, 0, dtype=_t.int32)
        sh["indptr"][1:].copy_(cum)
        pos = _t.arange(T, device=l.device) - _t.repeat_interleave(
            cum - l, l, output_size=T)
        req = _t.repeat_interleave(sh["ar32"], l, output_size=T)
        sh["indices"][:T] = req * sh["cap"] + pos
        sh["T"] = T

    def _plan_paged(self, sh):
        """Plan the paged dense-window kernel for this step (once per step at
        the step's first layer). Every step advances the window lengths, so
        plan runs every step; its launch config is recorded as the graph
        signature (a change re-captures the per-layer graphs instead of
        replaying a stale config)."""
        if sh["win"]:
            # Custom kernel path: no fi plan (no per-step host work). The
            # kernel reads lpl from device memory, so one captured launch
            # serves every step — the signature only changes with the batch
            # or a pool grow, never as the window grows.
            self._batch["g"]["sig"] = ("win", len(sh["dlens_py"]), sh["cap"])
            return
        if self._fi is False:
            return
        try:
            if self._fi is None:
                import flashinfer

                ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
                self._fi = {"wrapper": flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, kv_layout="NHD")}
            self._fi["wrapper"].plan(
                sh["indptr"], sh["indices"], sh["ones"],
                self.num_q_heads, self.num_kv_heads, self.head_dim, 1,
                q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
            )
            pi = getattr(self._fi["wrapper"], "_plan_info", None)
            try:
                # flashinfer 0.6.18 returns a tvm_ffi Array (not list/tuple);
                # convert so it can key the CUDA-graph store.
                sig = tuple(pi) if pi is not None else None
            except Exception:
                sig = None
            self._batch["g"]["sig"] = sig
            if sig is None and self._graphs_on:
                print("[PicoKVAttnBackend] plan sig unavailable; CUDA graphs "
                      "cannot engage (running eager)", flush=True)
        except Exception as e:
            print(f"[PicoKVAttnBackend] paged dense path DISABLED ({e!r}); "
                  f"using fallback", flush=True)
            self._fi = False

    _LN2 = 0.6931471805599453  # flashinfer returns log2 LSE; tail kernel wants ln

    def _paged_dense_attn(self, sh, fixed, q_):
        """Exact dense-window attention via flashinfer's paged decode kernel
        over the token-major pool (one fused kernel per layer; the plan ran
        once at layer 0). All branches (equal / dlens_equal / ragged) route
        through here. Returns (out [B,QH,1,D], ln-LSE [B,QH,1]) or None when
        flashinfer is unavailable (caller falls back)."""
        if sh["win"]:
            # Custom kernel path: same contract, one fused kernel + merge,
            # ln-LSE directly (no log2 conversion).
            kbuf, vbuf = fixed["kbuf"], fixed["vbuf"]
            B = kbuf.shape[0]
            g = self._batch["g"]
            bufs = g["bufs"] if g is not None else None
            if bufs is not None:
                out, lse = bufs["fi_out"], bufs["win_lse"]
            else:
                out = torch.empty(B, self.num_q_heads, self.head_dim,
                                  device=kbuf.device, dtype=torch.bfloat16)
                lse = torch.empty(B, self.num_q_heads,
                                  device=kbuf.device, dtype=torch.float32)
            # q_ is a view of the qkv projection and non-contiguous for B>1.
            if not q_.is_contiguous():
                q_ = q_.contiguous()
            self._win(kbuf, vbuf, sh["lpl"], q_, out, lse, self.scaling)
            return (
                out.view(B, self.num_q_heads, 1, self.head_dim),
                lse.unsqueeze(-1),
            )
        if self._fi is False:
            return None
        try:
            kbuf, vbuf = fixed["kbuf"], fixed["vbuf"]
            B, cap = kbuf.shape[0], kbuf.shape[1]
            kt = kbuf.view(B * cap, self.num_kv_heads, self.head_dim)
            vt = vbuf.view(B * cap, self.num_kv_heads, self.head_dim)
            out, lse2 = self._fi["wrapper"].run(q_, (kt, vt), out=None, lse=None, return_lse=True)
            return (
                out.view(B, self.num_q_heads, 1, self.head_dim),
                (lse2 * self._LN2).unsqueeze(-1),
            )
        except Exception as e:
            print(f"[PicoKVAttnBackend] paged dense path DISABLED ({e!r}); "
                  f"using fallback", flush=True)
            self._fi = False
            return None

    # ------------------------------------------------------------------
    # decode: CUDA graph (item 1)
    # ------------------------------------------------------------------

    def _layer_graph(self, fixed, sh, layer_id, q_, k_, v_, clone=True, direct=False,
                     eng_out=None):
        """CUDA-graph the per-layer decode op sequence: scatter-append this
        step's K/V, paged dense attention, query-projection bmm, fused tail —
        ~10 kernel launches whose configs depend only on shapes that are stable
        within a batch. Inputs are copied into fixed per-batch buffers; every
        kernel-side input (wpos, lpl, paged tables, pools) is a persistent
        device tensor whose VALUES advance per step, so one captured graph
        serves every step. The flashinfer launch config is pinned by keying on
        the plan signature; a signature change re-captures. ``direct`` (BCG
        replay, where q/k/v sit at the segment's fixed output addresses every
        step): the graph reads them in place, keyed on those addresses, and
        stages a non-contiguous q (the kernels need dense q) inside the graph.
        ``eng_out`` (BCG replay): the break's captured output buffer
        [bucket, QH*D]; the tail writes rows [:B] in place and that buffer is
        returned. Otherwise returns the output view [B, QH*D]. None when
        graphs are off / capture failed (caller runs the eager path)."""
        import torch as _t
        g = self._batch["g"]
        if g is None or g["bufs"] is None or g["sig"] is None:
            return None
        bufs = g["bufs"]
        B = fixed["kbuf"].shape[0]
        ov = None
        q_stage = False
        if direct:
            q_stage = not q_.is_contiguous()
            qs = bufs["q"] if q_stage else q_
            ks, vs = k_, v_
            if (eng_out is not None and eng_out.is_contiguous()
                    and eng_out.dtype == _t.bfloat16 and eng_out.shape[0] >= B
                    and eng_out.shape[1] == self.num_q_heads * self.head_dim):
                ov = eng_out[:B].view(B, self.num_q_heads, 1, self.head_dim)
            ptrs = (q_.data_ptr(), tuple(q_.stride()), ks.data_ptr(), vs.data_ptr(),
                    None if ov is None else ov.data_ptr())
        else:
            bufs["q"].copy_(q_)
            bufs["k"].copy_(k_)
            bufs["v"].copy_(v_)
            qs, ks, vs = bufs["q"], bufs["k"], bufs["v"]
            ptrs = None
        ent = g["by_layer"].get(layer_id)
        if (ent is not None and ent["sig"] == g["sig"] and ent["cap"] == sh["cap"]
                and ent["ptrs"] == ptrs):
            try:
                ent["graph"].replay()
                if ov is not None:
                    return eng_out
                # clone: the captured output buffer is persistent (reused next
                # step); callers may hold the returned tensor across steps.
                o = ent["out"].view(ent["out"].shape[0], -1)
                return o.clone() if clone else o
            except Exception as e:
                print(f"[PicoKVAttnBackend] graph replay failed ({e!r}); graphs off", flush=True)
                self._graphs_off(g)
                return None
        if g["ncap"] >= 160:  # circuit breaker: sig churn would recapture forever
            print("[PicoKVAttnBackend] graph capture budget exhausted; graphs off", flush=True)
            self._graphs_off(g)
            return None
        try:
            if self._graph_pool is None:
                self._graph_pool = _t.cuda.graph_pool_handle()
            kbuf, vbuf = fixed["kbuf"], fixed["vbuf"]
            cap = kbuf.shape[1]
            kt = kbuf.view(B * cap, self.num_kv_heads, self.head_dim)
            vt = vbuf.view(B * cap, self.num_kv_heads, self.head_dim)
            gl = None if fixed["equal"] else fixed["global_lens"]
            sr = None if fixed["equal"] else fixed["sched_req"]
            st = None if fixed["equal"] else fixed["sched_tile"]
            batched = fixed["batched"]

            def dense_fn():
                # append this step's K/V, then exact dense-window attention
                self._append_kv(kbuf, vbuf, sh, ks, vs)
                if sh["win"]:
                    self._win(kbuf, vbuf, sh["lpl"], qs,
                              bufs["fi_out"], bufs["win_lse"], self.scaling)
                    return (
                        bufs["fi_out"].view(B, self.num_q_heads, 1, self.head_dim),
                        bufs["win_lse"].unsqueeze(-1),
                    )
                fi = self._fi["wrapper"]
                out, lse2 = fi.run(qs, (kt, vt),
                                   out=bufs["fi_out"], lse=bufs["fi_lse"], return_lse=True)
                return (out.view(B, self.num_q_heads, 1, self.head_dim),
                        (lse2 * self._LN2).unsqueeze(-1))

            fork = self._graph_fork and not self._scfg.use_pruning
            if fork and self._fork_stream is None:
                self._fork_stream = _t.cuda.Stream(device=kbuf.device)
            cg = _t.cuda.CUDAGraph()
            with _t.cuda.graph(cg, pool=self._graph_pool):
                if q_stage:
                    bufs["q"].copy_(q_)
                o = decode_attention(
                    qs.unsqueeze(2), batched, update_cache=False,
                    global_lens=gl, dense_lens=None,
                    sched_req=sr, sched_tile=st,
                    dense_attn=None if fork else dense_fn(),
                    dense_fn=dense_fn if fork else None,
                    dense_stream=self._fork_stream if fork else None,
                    out=ov,
                    **self._scfg.decode_kwargs(),
                )
            cg.replay()  # capture records only — execute this step's work now
            g["by_layer"][layer_id] = {"graph": cg, "out": o, "sig": g["sig"],
                                       "cap": sh["cap"], "ptrs": ptrs}
            g["ncap"] += 1
            if ov is not None:
                return eng_out
            o = o.view(o.shape[0], -1)
            return o.clone() if clone else o
        except Exception as e:
            print(f"[PicoKVAttnBackend] graph capture failed ({e!r}); graphs off", flush=True)
            self._graphs_off(g)
            return None

    def _graphs_off(self, g):
        # Keep the batch's graph-state dict (its bufs back the window path and
        # _step_begin still counts steps on it); drop only the graphs.
        self._graphs_on = False
        g["by_layer"].clear()

    # ------------------------------------------------------------------
    # decode
    # ------------------------------------------------------------------

    # Breakable decode graph: each call is an eager break between captured
    # segments; replay re-calls this with the capture-time args (q/k/v at the
    # same addresses, refilled by the preceding segment) and copies the
    # return value into the captured output buffer. Plain eager: a no-op.
    @eager_on_graph(True)
    def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kw):
        import torch as _t
        import os as _os
        import time as _time
        from picokv.picokv_standalone import PreparedKVCache
        if self._eng_capture:
            z = q.new_zeros(q.shape[0], self.num_q_heads * self.head_dim)
            if self._eng_direct_out:
                # the last capture-time return per (layer, bucket) is the
                # break's captured output buffer for that bucket's graph
                self._eng_outs[(layer.layer_id, q.shape[0])] = z
            return z
        _prof = _os.environ.get("PICOKV_DECPROF")
        b = self._batch
        # BCG replay: rows past the real batch are bucket padding
        eng = self._eng_rids is not None and is_in_breakable_cuda_graph()
        eo = None
        if eng:
            rids = self._eng_rids
            if self._eng_direct_out:
                eo = self._eng_outs.get((layer.layer_id, q.shape[0]))
            q, k, v = q[:len(rids)], k[:len(rids)], v[:len(rids)]
        elif (self._rids_per_step and b is not None and b["min_layer"] is not None
                and layer.layer_id > b["min_layer"]):
            # same step as the min-layer call that read them
            rids = b["rids"]
        else:
            rids = self._host_rids(forward_batch, forward_batch.req_pool_indices.shape[0])
        q_ = q.view(-1, self.num_q_heads, self.head_dim)
        k_ = k.view(-1, self.num_kv_heads, self.head_dim)
        v_ = v.view(-1, self.num_kv_heads, self.head_dim)

        if self._batch is None or self._batch.get("rids") != rids:
            if self._batch is not None:
                self._stash_pools()
            self._prune_finished(rids)
            self._batch = {
                "rids": rids, "fixed": {}, "shared": None, "min_layer": None,
                "g": {"n": 0, "bufs": None, "by_layer": {}, "ncap": 0, "sig": None},
            }
        if layer.layer_id not in self._batch["fixed"]:
            self._batch["fixed"][layer.layer_id] = self._fixed_layer(rids, layer.layer_id)
        fixed = self._batch["fixed"][layer.layer_id]
        # A decode step's bookkeeping runs once, at the step's FIRST layer call
        # (the lowest layer id the batch ever calls — layer 0 in the engine; a
        # unit test may drive a single arbitrary layer, where every call is a
        # step).
        if self._batch["min_layer"] is None or layer.layer_id < self._batch["min_layer"]:
            self._batch["min_layer"] = layer.layer_id
        _step_first = layer.layer_id == self._batch["min_layer"]

        if fixed is not None:
            B = len(rids)
            if _prof:
                _t.cuda.synchronize(); _tp0 = _time.perf_counter()
            if _step_first:
                self._step_begin(fixed)
            sh = self._batch["shared"]
            # CUDA graph: after per-batch warmup steps, replay the captured
            # per-layer op sequence (one launch) instead of ~10 eager launches.
            g = self._batch["g"]
            if (self._graphs_on and g["n"] > self._graph_warmup
                    and (self._fi not in (None, False) or sh["win"])):
                o = self._layer_graph(fixed, sh, layer.layer_id, q_, k_, v_,
                                      clone=not eng, direct=eng, eng_out=eo)
                if o is not None:
                    self._n_decode += 1
                    if _step_first:
                        self._log_l0(
                            "GRAPH",
                            f"[PicoKVAttnBackend] forward_decode L0 GRAPH{'/BCG' if eng else ''} B={B} "
                            f"gl_max={fixed['gl_max']} d_max={sh['cur']} "
                            f"(decode#{self._n_decode})")
                    if o is eo:  # written in place into the break output
                        return o
                    return self._eng_pad(o) if eng else o
            # Persistent paged pool: append this step's token in place at each
            # request's write position (device-side wpos — same op every step).
            kbuf, vbuf = fixed["kbuf"], fixed["vbuf"]
            self._append_kv(kbuf, vbuf, sh, k_, v_)
            # Tail-kernel varlen schedule: only the GLOBAL (compressed) lens
            # can be ragged; equal-global batches take the uniform fast path.
            if fixed["equal"]:
                gl = sr = st = None
            else:
                gl = fixed["global_lens"]
                sr, st = fixed["sched_req"], fixed["sched_tile"]
            # Dense-window attention: one paged flashinfer kernel for every
            # branch (equal batches included); fallback = internal paths.
            dense_attn = self._paged_dense_attn(sh, fixed, q_)
            dl = None
            if dense_attn is None and not fixed["dlens_equal"]:
                dl = sh["wpos"] + 1  # ragged fallback: per-request dense lens
            batched = (fixed["batched"] if dense_attn is not None else PreparedKVCache(
                dense_key=kbuf[:, :sh["cur"]].permute(0, 2, 1, 3),
                dense_value=vbuf[:, :sh["cur"]].permute(0, 2, 1, 3),
                key_projection=fixed["batched"].key_projection,
                value_projection=fixed["batched"].value_projection,
                key_projection_t=fixed["batched"].key_projection_t,
                packed_key=fixed["batched"].packed_key,
                packed_value=fixed["batched"].packed_value,
                exp_bits=fixed["exp_bits"],
            ))
            qB = q_.unsqueeze(2).contiguous()  # [B,QH,1,D]
            if _prof:
                _t.cuda.synchronize(); _tp1 = _time.perf_counter()
            o = decode_attention(
                qB, batched, update_cache=False,
                global_lens=gl, dense_lens=dl,
                sched_req=sr, sched_tile=st,
                dense_attn=dense_attn,
                **self._scfg.decode_kwargs(),
            )
            self._n_decode += 1
            if _prof:
                _t.cuda.synchronize(); _tp2 = _time.perf_counter()
                d = self.__dict__.setdefault("_decprof", {"prep": 0.0, "kern": 0.0, "n": 0})
                d["prep"] += _tp1 - _tp0; d["kern"] += _tp2 - _tp1; d["n"] += 1
                if d["n"] % (32 * 8) == 0:
                    tot = d["prep"] + d["kern"]
                    print(f"[PICOKV_DECPROF] per-layer-call n={d['n']} "
                          f"prep={1e3*d['prep']/d['n']:.4f}ms({100*d['prep']/tot:.1f}%) "
                          f"kernel={1e3*d['kern']/d['n']:.4f}ms({100*d['kern']/tot:.1f}%)", flush=True)
            if _step_first:
                _path = "PAGED" if dense_attn is not None else ("EQUAL" if fixed["equal"] else "VARLEN")
                self._log_l0(
                    _path,
                    f"[PicoKVAttnBackend] forward_decode L0 {_path}{'/BCG' if eng else ''} "
                    f"B={B} gl_max={fixed['gl_max']} d_max={sh['cur']} (decode#{self._n_decode})")
            # clone: the tail writes into the persistent cache workspace output
            # (reused next step); callers may hold the returned tensor. Under
            # BCG replay the break copies it into its own buffer right away.
            if eng:
                return self._eng_pad(o.reshape(B, self.num_q_heads * self.head_dim))
            return o.clone().reshape(B, self.num_q_heads * self.head_dim)

        # per-request fallback (short-prompt / dense-only layers). Every
        # decoded request must have been prefilled through this backend.
        missing = [rid for rid in rids if rid not in self._reqs]
        if missing:
            raise KeyError(f"decode for requests with no PiCoKV cache: {missing}")
        outs = []
        for i, rid in enumerate(rids):
            cache = self._reqs[rid]
            qb = q_[i:i + 1].permute(1, 0, 2).unsqueeze(0).contiguous()  # [1,QH,1,D]
            kb = k_[i:i + 1].permute(1, 0, 2).unsqueeze(0).contiguous()
            vb = v_[i:i + 1].permute(1, 0, 2).unsqueeze(0).contiguous()
            o = cache.decode(layer.layer_id, qb, kb, vb)  # [1,QH,1,D]
            outs.append(o.reshape(1, self.num_q_heads * self.head_dim))
        self._n_decode += 1
        if layer.layer_id == 0:
            print(f"[PicoKVAttnBackend] forward_decode L0 RAGGED-fallback reqs={len(rids)} (decode#{self._n_decode})", flush=True)
        o = _t.cat(outs, dim=0)
        return self._eng_pad(o) if eng else o

    def support_triton(self):
        return False
