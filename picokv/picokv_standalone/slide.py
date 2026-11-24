"""Incremental decode for the offline path: periodic local-window slide.

After prefill the dense retained cache is laid out as ``[sink | EAS | local]``
and decode appends generated tokens to its end.  Every ``slide_tokens`` decode
steps the oldest ``slide_tokens`` tokens of the recent region (local window +
generated tokens) are quantized with the stored offline encoder/decoder pair,
appended to the packed global cache, and removed from the dense cache.  The
recent region therefore stays between ``local_size`` and
``local_size + slide_tokens`` tokens.  See :class:`SlidingWindow` for how the
per-step and per-slide costs are kept off the critical path.

Slid tokens skip EAS: all of them are quantized.  Attention is permutation
invariant over cached tokens (RoPE is already applied), so new tokens are
appended to the packed global cache without re-sorting.

Packed layouts produced by the prefill kernels:

* ``packed_key``   ``[B, H, Ck/64, N]`` int64; one word = 64 code bits of a token.
* ``packed_value`` ``[B, H, ceil(N/64), Cv]`` int64; bit ``t`` of word ``(g, c)``
  is code ``c`` of token ``64*g + t``.  Padding bits of the last group are 0.
"""

from __future__ import annotations

import torch

from .cache import PreparedKVCache

try:
    from picokv.picokv_prefill_cuda_extension.picokv_prefill_cuda import blockk_fixed_cuda
except Exception:  # pragma: no cover - reported by append_to_global_
    blockk_fixed_cuda = None

_WORD_BITS = 64
_SHIFTS: dict[torch.device, torch.Tensor] = {}


def _bit_shifts(device: torch.device) -> torch.Tensor:
    shifts = _SHIFTS.get(device)
    if shifts is None:
        shifts = torch.arange(_WORD_BITS, device=device, dtype=torch.long)
        _SHIFTS[device] = shifts
    return shifts


def token_words_to_bits(token_words: torch.Tensor) -> torch.Tensor:
    """``[B,H,T,C/64]`` token-major words -> ``[B,H,T,C]`` 0/1 int64 codes."""
    shifts = _bit_shifts(token_words.device)
    return ((token_words.unsqueeze(-1) >> shifts) & 1).flatten(-2)


def unpack_value_groups(packed_value: torch.Tensor, length: int) -> torch.Tensor:
    """``[B,H,G,Cv]`` group-major words -> first ``length`` tokens ``[B,H,T,Cv]``."""
    shifts = _bit_shifts(packed_value.device).view(_WORD_BITS, 1)
    bits = (packed_value.unsqueeze(-2) >> shifts) & 1  # [B,H,G,64,Cv]
    return bits.flatten(2, 3)[:, :, : int(length)]


def pack_value_groups(bits: torch.Tensor, offset: int = 0) -> torch.Tensor:
    """``[B,H,T,Cv]`` 0/1 codes -> ``[B,H,ceil((offset+T)/64),Cv]`` group words.

    Token ``j`` lands at bit ``offset + j``; all other bits are 0.  Bits of one
    word are distinct powers of two, so ``sum`` equals bitwise OR and cannot
    overflow int64 (only bit 63 is negative).
    """
    batch, heads, tokens, code_dim = bits.shape
    back = (-(offset + tokens)) % _WORD_BITS
    if offset or back:
        bits = torch.nn.functional.pad(bits, (0, 0, int(offset), back))
    groups = bits.shape[2] // _WORD_BITS
    shifts = _bit_shifts(bits.device).view(_WORD_BITS, 1)
    bits = bits.view(batch, heads, groups, _WORD_BITS, code_dim)
    return (bits << shifts).sum(dim=-2)


class SlidingWindow:
    """Sliding state of all compressed layers, slid together in one batch.

    Every layer of one prompt has the same geometry (sink, EAS count, kept
    count), so all layers slide in the same decode step.  The slide is done
    once for the whole stack instead of once per layer: the per-layer version
    issued ~25 small torch ops per layer, and in the host-bound decode loop
    those launches (not GPU work) were the slide cost.

    Dense K/V of all layers live in one ``[layers, B, H, capacity, D]`` buffer
    (capacity ``recent_start + local_size + slide_tokens``); each decode token
    is written in place and ``cache.dense_key``/``dense_value`` are views of
    the filled prefix, which flash attention reads directly.

    A slide has two halves:

    * launch (first layer of the slide step, after its attention): on
      ``stream``, quantize the oldest ``slide_tokens`` recent tokens of every
      layer with one batched kernel call per side, and build each layer's new
      packed tensors.  The old packed tensors and the dense tokens stay
      published, so the remaining layers of this step attend exactly as before
      and the side-stream work overlaps them.
    * commit (first append of the next step): wait for the side stream, shift
      every layer's recent region left in one batched copy, and publish the new
      packed tensors.
    """

    def __init__(
        self,
        caches: dict[int, PreparedKVCache],
        key_encoders: dict[int, torch.Tensor],
        value_encoders: dict[int, torch.Tensor],
        local_size: int,
        slide_tokens: int,
        zerorope: int,
        stream: torch.cuda.Stream,
    ) -> None:
        if blockk_fixed_cuda is None:
            raise RuntimeError("PiCoKV prefill CUDA extension is unavailable")
        self.index = {layer: i for i, layer in enumerate(sorted(caches))}
        self.caches = [caches[layer] for layer in sorted(caches)]
        first = self.caches[0]
        for cache in self.caches[1:]:
            if (
                cache.dense_key.shape != first.dense_key.shape
                or cache.packed_key.shape != first.packed_key.shape
                or cache.packed_value.shape != first.packed_value.shape
            ):
                raise ValueError("sliding requires every compressed layer to share one geometry")
        self.local_size = int(local_size)
        self.slide_tokens = int(slide_tokens)
        self.recent_start = first.dense_len - self.local_size
        self.capacity = self.recent_start + self.local_size + self.slide_tokens
        self.zerorope = int(zerorope)
        self.stream = stream
        self.slides = 0
        self.lengths = [first.dense_len] * len(self.caches)
        self.global_len = first.global_len

        batch, heads, length, dim = first.dense_key.shape
        shape = (len(self.caches), batch, heads, self.capacity, dim)
        self.key_buf = first.dense_key.new_empty(shape)
        self.value_buf = first.dense_value.new_empty(shape)
        self._key_layers = self.key_buf.unbind(0)
        self._value_layers = self.value_buf.unbind(0)
        for i, cache in enumerate(self.caches):
            # Layer by layer, so each prefill copy is freed right after.
            self._key_layers[i].narrow(2, 0, length).copy_(cache.dense_key)
            self._value_layers[i].narrow(2, 0, length).copy_(cache.dense_value)
            self._publish(i)

        # Batched kernel operands: layers stacked into the batch dimension.
        order = sorted(caches)
        self.key_encoder = torch.cat([key_encoders[layer] for layer in order])
        self.value_encoder = torch.cat([value_encoders[layer] for layer in order])
        self.key_decoder = torch.cat([c.key_projection for c in self.caches])
        self.value_decoder = torch.cat([c.value_projection for c in self.caches])
        # Each layer's decoder becomes a (contiguous) slice of the stack, so
        # the stack does not hold a second copy of every projection.
        n = len(self.caches)
        for cache, key_proj, value_proj in zip(
            self.caches,
            self.key_decoder.unflatten(0, (n, -1)).unbind(0),
            self.value_decoder.unflatten(0, (n, -1)).unbind(0),
        ):
            cache.key_projection, cache.value_projection = key_proj, value_proj
        # Every slide appends a multiple of 64 tokens, so the fill of the last
        # value group is the same after each slide.  Its words are kept here
        # (stacked) so the merge is one batched OR, not a per-layer gather.
        full_groups, self._tail = divmod(self.global_len, _WORD_BITS)
        self._value_tail = None
        if self._tail:
            self._value_tail = torch.cat(
                [c.packed_value[:, :, full_groups] for c in self.caches]
            )
            # Last used (and freed) on the side stream by the first launch.
            self._value_tail.record_stream(stream)
        self._done: torch.cuda.Event | None = None
        self._pending: list[tuple[torch.Tensor, torch.Tensor]] | None = None

    def _publish(self, i: int) -> None:
        cache, length = self.caches[i], self.lengths[i]
        cache.dense_key = self._key_layers[i].narrow(2, 0, length)
        cache.dense_value = self._value_layers[i].narrow(2, 0, length)

    def layer_encoders(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """``[B,H,bits,D]`` key/value encoders of one layer."""
        i, n = self.index[layer_idx], len(self.caches)
        return (
            self.key_encoder.unflatten(0, (n, -1))[i],
            self.value_encoder.unflatten(0, (n, -1))[i],
        )

    def append(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor) -> None:
        """Write ``[B,H,T,D]`` decode K/V in place; caller then runs attention."""
        i = self.index[layer_idx]
        if self._pending is not None and self.lengths[i] == self.capacity:
            self._commit()
        start = self.lengths[i]
        tokens = int(key.shape[-2])
        if start + tokens > self.capacity:
            raise RuntimeError("sliding dense buffer overflow")
        self._key_layers[i].narrow(2, start, tokens).copy_(key)
        self._value_layers[i].narrow(2, start, tokens).copy_(value)
        self.lengths[i] = start + tokens
        self._publish(i)

    def maybe_slide(self, layer_idx: int) -> bool:
        """Launch a slide when the first layer's recent region is full."""
        if self._pending is not None or self.lengths[self.index[layer_idx]] < self.capacity:
            return False
        self._launch()
        return True

    def _launch(self) -> None:
        rs, step, n = self.recent_start, self.slide_tokens, len(self.caches)
        self.stream.wait_stream(torch.cuda.current_stream(self.key_buf.device))
        with torch.cuda.stream(self.stream):
            # Oldest recent tokens of every layer as one [layers*B,H,step,D]
            # batch (the kernel wrapper makes it contiguous on this stream).
            # Main-stream appends only write slots past rs+local, and nothing
            # overwrites [rs, rs+step) before the commit waits for this stream.
            key_words, _ = blockk_fixed_cuda(
                self.key_buf.narrow(3, rs, step).flatten(0, 1),
                self.key_encoder, self.key_decoder, 0, step,
                zerorope=self.zerorope, apply_zerorope=True, pack_key_layout=True,
            )
            value_words, _ = blockk_fixed_cuda(
                self.value_buf.narrow(3, rs, step).flatten(0, 1),
                self.value_encoder, self.value_decoder, 0, step,
                zerorope=0, apply_zerorope=False, pack_key_layout=False,
            )
            # New tokens' value bits go straight to their final bit offset;
            # the partially used last group is OR-ed in, nothing is unpacked.
            full_groups, tail = divmod(self.global_len, _WORD_BITS)
            groups = pack_value_groups(token_words_to_bits(value_words), offset=tail)
            if tail:
                groups[:, :, 0] |= self._value_tail
            self.global_len += step
            self._value_tail = groups[:, :, -1] if self.global_len % _WORD_BITS else None
            # The decode kernel needs contiguous packed tensors, so each layer
            # still gets its own concatenation.
            self._pending = [
                (
                    torch.cat((cache.packed_key, key), dim=-1),
                    torch.cat(
                        (cache.packed_value.narrow(2, 0, full_groups) if tail else cache.packed_value, value),
                        dim=2,
                    ),
                )
                for cache, key, value in zip(
                    self.caches,
                    key_words.unflatten(0, (n, -1)).unbind(0),
                    groups.unflatten(0, (n, -1)).unbind(0),
                )
            ]
            self._done = torch.cuda.Event()
            self._done.record(self.stream)
        # No record_stream is needed.  Old packed tensors are dropped only
        # after the main stream waited for this stream.  Side-stream blocks
        # published to main are reused only by the next launch, which first
        # waits for all main-stream work queued so far.
        self.slides += 1

    def _commit(self) -> None:
        if any(length != self.capacity for length in self.lengths):
            raise RuntimeError("every sliding layer must decode the slide step before the next step")
        torch.cuda.current_stream(self.key_buf.device).wait_event(self._done)
        rs, mid = self.recent_start, self.recent_start + self.slide_tokens
        keep = self.capacity - mid
        for buf in (self.key_buf, self.value_buf):
            buf.narrow(3, rs, keep).copy_(buf.narrow(3, mid, keep).clone())
        for cache, (packed_key, packed_value) in zip(self.caches, self._pending):
            cache.packed_key, cache.packed_value = packed_key, packed_value
        # Dense views are republished by each layer's next append.
        self.lengths = [rs + keep] * len(self.caches)
        self._pending = None
        self._done = None

    def finish(self) -> None:
        """Commit an in-flight slide and publish every layer's dense view."""
        if self._pending is not None:
            self._commit()
            for i in range(len(self.caches)):
                self._publish(i)


__all__ = [
    "SlidingWindow",
    "pack_value_groups",
    "token_words_to_bits",
    "unpack_value_groups",
]
