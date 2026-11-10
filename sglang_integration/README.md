# PiCoKV attention backend for SGLang

This directory adds `picokv` as an SGLang attention backend, so PiCoKV's
compressed KV cache can be served through `sgl.Engine` / `sglang.launch_server`
with continuous batching and decode CUDA graphs.

- Prefill: exact dense attention (FlashInfer ragged), then the prompt K/V is
  packed into PiCoKV codes (320 bits per head-token by default) with **offline
  projections** and `eas=0`. One fused prepare kernel per side; a prefill
  batch of several prompts quantizes them together in one launch per side. No
  per-request online fitting. Chunked prefill (SGLang's default chunk size)
  is exact: earlier chunks' raw K/V wait in pinned host memory and each later
  chunk attends prefix + chunk.
- Decode: a dense bf16 window is attended exactly (FA3
  `flash_attn_with_kvcache` on SM90, else FlashInfer paged decode or a fused
  window kernel at small batch). The window holds the first 64 tokens
  (sink), the last 256 tokens (prompt or generated) and up to 64 tokens of the
  current slide epoch; every 64 decode steps older generated tokens are
  compressed too, at every batch size (`PICOKV_DENSE_LOCAL` can give
  batches of <= 8 rows a longer dense local for speed; off by default).
  The rest goes through the PiCoKV tail kernel, which runs
  the `m16n8k256` tensor-core path by default (`PICOKV_NO_TC=1` uses the popc
  path). The per-layer ops are replayed through CUDA graphs when batch membership
  is stable.
- The SGLang KV pool is replaced by a no-op pool. PiCoKV owns its memory, and
  the pool sizing is told the compressed cell size, so far more tokens fit:
  about 6.4x the dense ceiling. The decode rows' dense windows and slide
  headroom are reserved from the KV budget first (see Limitations).
- Each decode row keeps its compressed codes at its own length
  (`PICOKV_ROWS`), so a long request in a batch of short ones does not pad
  every row to its length.

Hard requirements of the decode kernel: bf16, GQA = 4, head_dim = 128 (for
example Llama-3.1-8B). Attention TP (`tp_size` > 1) is supported.

## Contents

| Path | What |
|---|---|
| `picokv_backend.py` | The backend (`python/sglang/srt/layers/attention/picokv_backend.py` in the SGLang tree) |
| `picokv_radix_cache.py` | Radix cache backend `picokv` for `PICOKV_RADIX=host` (`python/sglang/srt/mem_cache/picokv_radix_cache.py`) |
| `sglang_picokv.patch` | Changes to 15 SGLang files, against upstream `sgl-project/sglang@567d5925` |
| `apply.sh` | Install the patch and both modules into an SGLang checkout |
| `export_from_sglang.sh` | Maintainer helper: regenerate the patch and module copies from a working checkout |
| `run_itl.sh`, `scripts/bench_itl.py` | Decode inter-token-latency benchmark (picokv vs flashinfer) |
| `scripts/niah_engine_check.py` | Quick real-weight needle check through the engine |
| `scripts/eval/` | LongBench / RULER / InfiniteBench / LongBench-v2 / NIAH-grid drivers |
| `scripts/serving/` | Serving-behaviour probes (throughput, TTFT, long generation, radix/chunk quality) |

What the patch changes in SGLang:

- `attention_registry.py`: registers backend `picokv`;
  `arg_groups/choices.py` adds it to the `--attention-backend` CLI choices.
- `arg_groups/attention_hook.py`, `arg_groups/pipeline.py`: the serving
  guard (`handle_picokv_serving_constraints`, see Running; with
  `PICOKV_RADIX=host` it selects radix cache backend `picokv`).
- `kv_cache_configurator.py`: uses a no-op MHA token pool for `picokv`,
  reserves the decode rows' dense windows + slide headroom from the KV budget
  and caps `max_running_requests` so that reserve stays <= half of it.
- `pool_configurator.py`: sizes the pool by the backend's per-token cell
  (`picokv_cell_size_bytes`: the K/V code bits of the offline projection,
  20 KiB per token for Llama-3.1-8B k320/v320; dense bf16 when the decode
  slide is off, e.g. online projections).
- `beam_search/coordinator.py`: rejects beam search (`beam_width > 1`) with
  picokv (the per-request compressed caches are not forked).
- `managers/scheduler_components/batch_result_processor.py`: a request whose
  prompt prefix was lost (see Limitations, lost prefixes) finishes with an
  abort (HTTP 503) instead of decoding from a truncated context.
- `forward_batch_info.py`, `decode_cuda_graph_runner.py`: carry
  `req_pool_indices_cpu`, so the backend reads request ids without a
  device-to-host sync. Set only when the prefill or decode attention backend
  is picokv: other backends' forward batches stay as in base SGLang (two-batch
  overlap's `filter_batch` rejects a field it does not know that holds a value).
- `breakable_cuda_graph_backend.py`: lets breakable graphs return
  dataclass outputs.
- `breakable_cuda_graph/breakable_cuda_graph.py`: composed replay
  (`PICOKV_COMPOSE`). A break marked `composable` may hand its own captured
  graph over instead of replaying it (`defer_graph`); when every break of a
  replay does, the replay launches one cached graph made of child-graph nodes
  (segment 0, layer 0, segment 1, ...) instead of 65 separate graph launches.
  A set of graphs is composed only after it repeats on two consecutive
  replays; anything else (a break that runs eagerly, `defer_flush`) replays
  in order as before. Segments are kept as raw graphs for this (host memory
  only). Off unless a backend enables it.
- `managers/schedule_batch.py`: `orig_seq_lens` (read only by picokv and
  dual-chunk attention) is the request's full fill length, so a retracted
  request that re-prefills prompt + output in chunks stages every non-final
  chunk (it used to take a chunk past the prompt as the last one, and the
  next chunk attended only itself).
- `managers/schedule_policy.py`: three admission hooks for the `picokv`
  radix cache (`PICOKV_RADIX=host`; other radix caches have none of these
  attributes and are untouched): a hit is charged the prefix tokens that a
  running request already holds (`picokv_charge`, held until the request
  finishes: `picokv_admit`), and a prefix row that would overflow the
  stage's GPU layer buffers waits a round (`picokv_stage_ok`). See
  Limitations, radix.
- `models/utils.py`: the fused RoPE + KV-store path
  (`enable_fused_set_kv_buffer`, used by e.g. Qwen3-MoE / GPT-OSS) is off for
  the no-op pool, whose one-page placeholder buffer it would write out of
  bounds.

## Setup

The scripts assume a sibling layout. The backend also resolves `picokv` from
`PYTHONPATH`.

```
<WS>/
  PiCoKV/     # this repo (branch picokv_sg)
  sglang/     # upstream sglang @ 567d5925, patched by apply.sh
```

```bash
cd <WS>
git clone https://github.com/sgl-project/sglang.git
git -C sglang checkout 567d5925fe896a7b70043271f5621d6fe6e650d4
PiCoKV/sglang_integration/apply.sh sglang          # add --link to symlink the backend

# environment (tested: Python 3.12, torch 2.13.0+cu130, flashinfer-python 0.6.18,
# sglang-kernel 0.4.7, transformers 5.17.0, triton 3.7.1, CUDA 13.0, sm_90 80 GB GPU)
uv venv --python 3.12 .venv_sglang && source .venv_sglang/bin/activate
uv pip install -e sglang/python

# build the PiCoKV CUDA extensions with this torch (writes PiCoKV/build/picokv_extensions)
cd PiCoKV && TORCH_CUDA_ARCH_LIST=9.0 python picokv/build_cuda_extensions.py && cd ..
```

The extensions must be built against the serving venv's torch. Use
`--output-dir DIR` together with `PICOKV_PREBUILT_DIR=DIR` to keep several
builds side by side. The PiCoKV repo's own `.venv` (torch 2.6) is a separate
environment. It is still what the `*_io.py` build/score halves of the eval
scripts use.

## Running

```bash
export PYTHONPATH=<WS>/PiCoKV                      # `import picokv`
export PICOKV_OFFLINE_PROJ=<WS>/PiCoKV/offline_projection/projections/Llama-3.1-8B-Instruct_k320_v320.pt
# export PICOKV_PREBUILT_DIR=...                    # only if built with --output-dir
```

```python
import sglang as sgl

eng = sgl.Engine(
    model_path="<path>/Llama-3.1-8B-Instruct",
    attention_backend="picokv",
    dtype="bfloat16", tp_size=1,           # tp_size > 1 works too (attention TP)
    mem_fraction_static=0.70,
)
out = eng.generate(prompt, {"temperature": 0, "max_new_tokens": 256})
```

The same fields work as `sglang.launch_server` flags (`--attention-backend
picokv`; the patch adds it to the CLI choices). All numbers below were
measured through `sgl.Engine`.

**Serving guard.** With `attention_backend="picokv"`, server-argument
resolution (`handle_picokv_serving_constraints`, before the memory settings)
turns off what would silently lose context, with one warning each: the radix
cache (kept with `PICOKV_RADIX=host`, which selects the `picokv` radix cache
backend; see Limitations, radix), mixed-chunk prefill, a non-breakable decode
graph (it switches to `breakable`) and the prefill graph. Chunked prefill stays on at SGLang's chunk
size (8192 by default on this GPU): the backend's chunk stage makes every
chunk exact (see Limitations, chunked prefill). Only without the stage
(`PICOKV_CHUNK_STAGE=off` or `PICOKV_PREFILL=sdpa`) does the guard turn
chunked prefill off, unless `chunked_prefill_size >= context_len` or
`PICOKV_ALLOW_CHUNKED_PREFILL=1`. It rejects
explicitly-set combinations that have no working path (decode graph backend
other than `breakable`/`disabled`, a prefill graph) and speculative decoding,
DP attention, pipeline parallelism, prefill and decode context parallelism,
diffusion-LLM decoding, the hierarchical cache, PD disaggregation, PD
multiplexing, MoE all-to-all backends, `--moe-dense-tp-size`, two-batch
overlap, the KV canary (it audits the token pool, which picokv does not
write), beam search and streaming sessions. `PICOKV_PREFILL`,
`PICOKV_CHUNK_STAGE`, `PICOKV_RADIX` and `PICOKV_LOST` are checked against
their allowed values (a typo is a `ValueError`, not a silent default).
`enable_torch_compile` is turned off with a warning (not
validated with the per-layer breakable graph), and `enable_memory_saver`
warns that `release_memory_occupation` frees only the weights. A
`kv_cache_dtype` other than bf16 is ignored with a warning (the backend keeps
a bf16 window and its own compressed codes); `PICOKV_RADIX=host` needs
`page_size=1`. At backend init every `RadixAttention` layer is checked
against what the kernels compute; a model is rejected (`ValueError`) for
GQA != 4, head_dim != 128, non-bf16, a `scaling` other than
`head_dim ** -0.5` (e.g. Granite's attention multiplier), a sliding window
(including hybrid-SWA models), logit soft-capping, cross-attention, a
non-decoder attention type, iRoPE or attention temperature. The old explicit flags
(`disable_radix_cache=True, chunked_prefill_size=1 << 20,
cuda_graph_backend_decode="breakable", disable_prefill_cuda_graph=True`) are
still accepted; they give one-chunk prefill (same outputs, more prefill
activation memory on long prompts).

**Offline projections are required by default.** On init the backend logs
(info level) a line containing `mode=offline (<path>)`. If `PICOKV_OFFLINE_PROJ`
is unset, it looks for
`PiCoKV/offline_projection/projections/<model_dir_name>_k320_v320.pt`; if that
file is missing, init fails with a `ValueError` naming both ways out: set
`PICOKV_OFFLINE_PROJ`, or `PICOKV_ONLINE=1` for per-request online
projections (slower prefill, decode slide off, lower long-context retrieval
quality on RULER/passkey; it logs a warning). A projection that does not fit
the model (key/value encoder shape other than
`[layers, kv_heads, bits, head_dim]`, meta `bits`/`sink`/`local` other than
the backend's, incomplete calibration) is also a `ValueError`. Models with
random weights are fine for timing only.

Use `cuda_graph_backend_decode="breakable"`. The `full` decode graph is not
supported (`NotImplementedError`), and a full graph would buy <= 1% anyway
because the backend already graphs its own per-layer ops.

## Environment knobs

All are read at backend init. The defaults are the tested serving
configuration.

| Variable | Default | Meaning |
|---|---|---|
| `PICOKV_OFFLINE_PROJ` | (auto) | Offline projection `.pt`; see above |
| `PICOKV_ONLINE` | `0` | `1` allows serving without an offline projection: per-request online projection fitting, decode slide off (A/B / fallback only) |
| `PICOKV_EAS` | `0.0` | Fraction of error-selected prompt tokens kept dense. `0` means the window is exactly sink + local. Offline + `eas=0` is the supported path (incremental chunk append needs it) |
| `PICOKV_EXP_BITS` | `4` | In-kernel quantization bits of attention weights (1/2/4/8) |
| `PICOKV_LOST` | `abort` | A request whose prompt prefix could not be staged (see Limitations, lost prefixes) finishes with an abort, HTTP 503 `ServiceUnavailableError`. `keep` = old behaviour: it decodes from the chunks it still has, with a warning |
| `PICOKV_KV_DEDUP` | `1` | Free the per-request copy once a layer's batched copy exists |
| `PICOKV_SLIDE` | `1` | Decode slide: every 64 decode steps the dense tokens past sink + local are appended to the compressed global codes (quantized ahead, spread over the epoch), so the window stays <= sink + local + 64 during generation. Needs offline projections + `eas=0`. `0` keeps every decoded token dense (A/B only; memory and window cost grow with output length) |
| `PICOKV_SLIDE_LOCAL` | prefill local (`256`) | Dense tail kept during decode (multiple of 32, `>=` the prefill local). Longer = more generated text exact, +4 KB per token per layer per row |
| `PICOKV_SLIDE_HEADROOM` | `1024` | Compressed-global capacity (tokens) reserved per row at each batch build; growth past it regrows that layer's packed copy (realloc + graph recapture) |
| `PICOKV_PREFILL` | `fi` | Prefill dense attention: `fi` = FlashInfer ragged, `sdpa` = torch SDPA fallback |
| `PICOKV_WINDOW_KERNEL` | `auto` | Fused dense-window kernel for small batches (`<= PICOKV_WINDOW_MAX_BS`, default 8); `0`/`1` force it off/on. Superseded by `PICOKV_DENSE_FA3` on SM90 |
| `PICOKV_DENSE_FA3` | `1` | SM90: dense-window attention with FA3 `flash_attn_with_kvcache` from `sgl_kernel` at every batch size (no FlashInfer plan, stable graph signature). `0` falls back to FlashInfer / the window kernel |
| `PICOKV_FA3_META` | `1` | FA3 dense window: the tile-scheduler metadata (`get_scheduler_metadata`) is computed once per decode step and shared by every layer, instead of FA3 preparing it inside each layer's call (one launch fewer per layer; bitwise the same output). `0` = per call |
| `PICOKV_SHARED_PROJ` | `1` | Decode reads one offline projection copy per layer (Triton query/value projection kernels around the global-only tail op `picokv_decode_tail_global_fn`) instead of per-row replicas (~63 MB per running request). Needs an extension rebuilt from this tree; with an older build it switches itself off. `0` keeps replicas + cuBLAS |
| `PICOKV_SHARED_ONE` | `1` | A one-row batch also takes the shared-projection Triton kernels (its projections are trivially shared): one launch fewer per layer than the cuBLAS GEMM + epilogue pair, about 3 us per layer. Same bf16 operands and fp32 accumulation, but not bitwise the cuBLAS output. `0` = cuBLAS at batch 1 |
| `PICOKV_PROJ_TILES` | `1` | Shared-projection tile shapes picked by batch size: narrow column tiles for small batches, tall row tiles for large ones, which re-read each kv head's matrix fewer times (160 rows: query projection 7.4 -> 4.0 us, value projection 7.9 -> 4.8 us per layer). The reduction order is the same at every shape, so the output is bitwise unchanged. Every tile variant is compiled on the first decode call, outside graph capture, so a batch growing into a new size range does not stall a serving step on a 160-510 ms Triton compile. `0` = fixed 16 x 64 tiles |
| `PICOKV_FAST_BUILD` | `1` | Incremental batch rebuild on a batch change: carried rows are copied on device from the previous batch, only new rows are read from their request cache. `0` rebuilds every row (A/B only) |
| `PICOKV_FAST_BUILD2` | `1` | With the fast build: a graph-bucket change computes its per-build work once instead of once per layer (row mapping, copy runs, row geometry, the slot schedule), hands the kept rows' storage over in numpy, and gathers the carried rows' dense windows straight into unzeroed buffers. Same batch state, bitwise (unit test against `0` over shrinks, growth, re-added and fresh rows, rows to rectangle; greedy engine runs identical). Host time per bucket change at 104-152 rows: 10-13 ms instead of 19-45 ms (32 layers). `0` = the per-layer path |
| `PICOKV_WARM_INIT` | `1` | With `PICOKV_FAST_BUILD2`: launches the window gather's two `index_select` kernels (<= 16 rows and more) once at startup, so the first bucket change that gathers <= 16 carried rows does not pay CUDA's lazy module load (about 49 ms, once). `0` = load on first use |
| `PICOKV_EXT_FAST` | `1` | Prefill host path: request ids and extend lengths are read once per forward from the scheduler's CPU mirrors (no device-to-host sync per layer), and a prefill whose every prompt fits the dense window (`<=` sink + local) stores its K/V as views into one per-layer copy instead of two copies per request. `0` = old path (A/B only) |
| `PICOKV_FAST_PREPARE` | `1` | Prefill pack (offline + `eas=0`): calls the fused quantize kernels directly with memoized projections and index maps instead of the generic `prepare_kv_cache` path (bitwise the same codes). `0` = old path (A/B only) |
| `PICOKV_PREP_BATCH` | `1` | A prefill with >= 2 compressed prompts of global length <= `PICOKV_PREP_BATCH_GL` (`4096`) quantizes them in one launch per side, in groups of <= `PICOKV_PREP_BATCH_TOKENS` (`65536`) tokens, instead of one small launch per prompt. A group of fewer than `PICOKV_PREP_BATCH_MIN` (`3`) prompts with at most `PICOKV_PREP_BATCH_MIN_TOKENS` (`768`) prompt tokens packs each prompt on its own instead: ~3 ms less host time per prefill, ~1-2 ms more GPU time. TTFT, the two modes alternating within one engine, 50 ms idle before each request: 2 x 384 / 448 / 512: 25.5 / 27.9 / 27.6 vs 28.4 / 29.8 / 28.8 ms batched, 2 x 576: 31.6 vs 31.1; requests back to back (three alternating serial runs): 2 x 352 / 384: 29.2 / 29.2 vs 31.7 / 31.8 ms, but 2 x 416-640 per-request read +0.5..+1.1 ms (2 x 512: 34.0 vs 33.0), the extra GPU time delaying the next request, hence the token cap. The group's dense windows are gathered into one buffer (`picokv_prefill_gather_pack_from_packed_out_fn`): each prompt's own ~640 KiB pair would come from the CUDA caching allocator's small-block pool, which no other size reuses (+78 MiB peak on the 160-request MATH run). A prefill extension built without that op keeps such groups batched. Bitwise the same codes |
| `PICOKV_PREP_OVERLAP_MIN` | `832` | Prompts of at least this many tokens pack V on a side stream, overlapped with K (a batched quantize group: `PICOKV_PREP_OVERLAP_MIN_GROUP`, `1200`, over its prompts' tokens together; 2 x 512 / 3 x 384: 31.5 / 34.5 vs 34.0 / 37.2 ms with the side stream, from 1280 tokens the side stream is 0.5-0.8 ms faster). Below it both sides run on the main stream: a prompt that short prefills host-bound, and the event pair plus stream switch (~18 us per layer) costs more than the overlap saves (TTFT, the two modes alternating within one engine: 1 x 384 / 512 / 640 / 704 / 768: 25.1 / 24.7 / 24.5 / 26.1 / 26.6-27.1 vs 26.5 / 25.9 / 26.0 / 27.6 / 27.3-27.7 ms with the side stream, even at 800; from 896 tokens the side stream is 0.3-0.7 ms faster). Same kernels and inputs, bitwise the same codes. `0` = always the side stream |
| `PICOKV_FIXED_BAND` | `1` | Prefill pack (fixed-cascade quantizer, d=128, bf16): one warp per 16-token band, `NB` bands per CTA (tensor-core `mma.sync` m16n8k16, the residual kept in register fragments, the cascade's E/D tiles double-buffered with `cp.async`) instead of the fragment kernel. `NB` is the smallest of 1 / 2 / 4 whose grid fits one wave, else 8 up to `PICOKV_FIXED_BAND_WIDE` (`1024`) CTAs, else 4. Same accumulation order: bitwise the same codes and errors (kernel test over 1770 shapes, engine outputs identical). Kernel time at 8 kv heads: 512-token prompt 40.6 -> 14.8 us, 8k 98 -> 51 us, 32k 362 -> 203 us, 128k 1327 -> 743 us, decode slide (1-16 rows) 39-41 -> 15-16 us. fp32 inputs, unaligned bases or partial phase masks keep the fragment kernel. `0` = fragment kernel |
| `PICOKV_PREP_FUSED` | `1` | A compressed prompt below `PICOKV_PREP_OVERLAP_MIN` tokens (prefilled on its own or in a solo group, see `PICOKV_PREP_BATCH`) prepares each layer in one call of `picokv_prefill_prepare_fast_fn`: one kernel reads the prompt's K/V rows in place from the qkv buffer and splits them into the dense window and a buffer holding only the global tokens, then the V and K fixed-cascade launches and the value regroup run inside the same call, instead of a permute copy, two launch calls, the key gather and the value regroup issued one by one from Python. Same kernels on the same rows (the quantizer masks loads past the global length, its token offset only addresses memory): bitwise the same codes (op test over 468 shapes and offsets, engine outputs identical to the separate calls and to the previous build). Same memory: the buffers are allocated in the separate path's order and sizes (the global tokens are carved from a block the size of the K/V copy it replaces, so a short prompt's buffer does not land in the caching allocator's small pool), peak allocated / reserved and nvidia-smi peaks equal to `0` in the TTFT and engine runs. Prefill step (extend start to the next forward, median of 20 over two alternating serial runs, chunk 8192): 1 x 384 / 512 / 640 / 768: 14.1 / 14.1 / 13.9 / 14.1 vs 15.8 / 15.9 / 15.6 / 15.5 ms; 2 x 352 / 384: 16.7 / 16.5 vs 19.3 / 19.4 ms; TTFT (median of 9, same runs): 23.1 / 23.0 / 24.0 / 26.8 vs 24.9 / 24.9 / 24.7 / 27.2 ms; 28.5 / 28.1 vs 29.6 / 29.3 ms; the shapes on the unchanged path within noise (prefill step -0.1 to -0.3 ms, TTFT -0.6 to +0.2 ms). Batched groups and prompts from 832 tokens keep the previous path. An extension built without the op uses the separate calls. `0` = separate calls |
| `PICOKV_FILL_KERNEL` | `1` | A batch build or in-place slot change writes its new rows (packed codes + dense window, all layers) with one Triton launch instead of four copies per row per layer. `0` = row-wise copies (A/B only) |
| `PICOKV_SLOTS` | `1` | Slot mode (breakable decode graph + FA3/window path + fast build): the decode batch owns the graph bucket's rows (live rows first, vacant rows after), so a request leaving or joining within the bucket updates rows in place and keeps every tensor, table and captured per-layer graph. A bucket change still rebuilds. `0` rebuilds on every batch change (A/B only) |
| `PICOKV_SLOT_MEMO` | `1` | With `PICOKV_SLOTS`: an in-place slot change keeps the fill launch's per-layer layout checks and row bases with the batch (weak references, so an entry holds only while its tensors live; per-row storage rows are still checked each time) instead of redoing them for every layer, builds the fill jobs without per-row helper calls, and rewrites the rows' write positions / dense lens / live mask with one host-to-device copy (they are rows of one small buffer, each row 16-byte aligned like a separate tensor) instead of three. The same fill table and the same values reach the same kernels: bitwise (the launch table is compared entry by entry against the previous code over the slot and build unit tests; greedy engine dumps identical to the previous build), same memory (nvidia-smi peaks equal in every TTFT / engine / MATH run). Host time of one slot change at one row: about 0.84 ms instead of 1.04 ms; first decode step after a request joins (1 x 512, three alternating serial runs each, median): 2.24 vs 2.41 ms host. `0` = recomputed per change, three copies |
| `PICOKV_FILL_TAB` | `1` | With `PICOKV_SLOTS` and the fill kernel: the fill launch's table (17 int64 per job, 32 jobs per fresh row) is built by one call of the prefill extension (`picokv_fill_table_fn`: the same checks and entries in C++, the same pinned buffer + non-blocking host-to-device copy) instead of about 8 us of Python per job. A job it does not take (unexpected type / rank, a layout check that fails, a Python error) returns None and the Python loop decides as before. Same table (compared entry by entry against the Python table over the slot and build unit tests), same launch: bitwise, same memory. Table of one fresh row: about 56 us instead of 196 us. `0` (or an extension built without it) = the Python loop |
| `PICOKV_VIEW_MEMO` | `1` | Breakable decode graph: replay calls every layer's break with the same captured q / k / v tensor objects and output buffer each step, so their row slices / head views and the graph's address key are kept per (layer, bucket) and reused while those objects are the ones passed in (strong references, so an id is never reused while kept) instead of being rebuilt per layer per step (about 12 us of host time each). Same views, same addresses: bitwise, no device memory. With `PICOKV_FILL_TAB` (1 x 512, serial alternating runs, median): first decode step after a request joins 1.80 vs 2.29 ms host (2 x 384: 2.32 vs 3.18, 4 x 512: 2.81 vs 4.10), steady step 0.65-0.68 vs 1.08-1.14 ms host; TTFT 1 x 512 24.5 vs 25.7 ms, 2 x 352 28.5 vs 30.6 (FlashInfer 21.1 / 25.8); prompts of 640 tokens or more and every steady decode are GPU-bound and unchanged (ITL, 160 concurrent requests). `0` = rebuild every call |
| `PICOKV_SLOT_FAST` | `1` | With `PICOKV_SLOTS`: an in-place slot change of a batch whose layers all keep packed rectangles (graph bucket 1, or `PICOKV_ROWS=0`) and whose new rows need no window catch-up builds the per-row lists every layer shares (eas=0: one global length / window length / capacity per row at every layer) once and copies them per layer, instead of rebuilding them layer by layer. The same checks, copies, fill launch and table writes: bitwise (unit tests in both storage modes; `PICOKV_DET=1` slot-change replays at 1 / 16 / 64 rows identical to the previous code), same memory. Host time of one slot change at one row (Llama-3.1-8B-Instruct, 1 x 512, four alternating serial runs, median per run): 0.54-0.57 instead of 0.65-0.69 ms; first decode step after a request joins 1.49-1.58 vs 1.59-1.73 ms host (FlashInfer 0.44); steady decode unchanged (512 x 1 6.49 ms/step, 160 rows 11.33). Batches with per-row storage (2 or more rows by default) or catch-up rows take the layer-by-layer path. `0` = layer by layer |
| `PICOKV_SLOT_COMPOSE` | `0` | Opt-in. With `PICOKV_SLOTS` and `PICOKV_COMPOSE`: a slot change stays inside the composed replay instead of flushing it (it touches only picokv state, never the step's q / k / v or another segment's buffers): bitwise. Off by default because the step after a slot change then sees a new graph set and instantiates a composed exec: the first request per bucket pays about 3 ms TTFT and the peak is about 14 MiB higher, and the steady decode it saves is GPU-bound. `0` = flush first |
| `PICOKV_STABLE_ORDER` | `1` | Read by the scheduler (`schedule_batch.py`; picokv only, not with speculative decoding). When requests leave the decode batch, every surviving row keeps its position and rows past the new size fill the vacated positions, instead of SGLang's order-preserving filter, which shifts every row after a finished one (each shifted row moves its dense window and table entries in place). Same rows; the request-side state (sampling, penalties) is filtered with the same index list. `0` = SGLang's order |
| `PICOKV_SLOTS_MAX_WASTE_MB` | `1024` | Slot-mode memory guard: a build or in-place change whose packed copy would exceed the rebuilt (exact-fit, unslotted) batch's by more than this (vacant rows, width kept after a long request left; ~20 KB per token-row over 32 layers) runs unslotted / rebuilds instead |
| `PICOKV_ROWS` | `1` | Per-row packed storage: each decode row keeps its compressed codes in its own buffer at its own capacity (global length + slide headroom), read by the tensor-core tail through per-layer row-address tables, instead of a `[B, H, KW, L]` rectangle padded to the longest row. Carried rows keep their buffers across batch and slot changes; a regrow reallocates only the overflowing row. Same codes, same kernel math (bitwise-equal packed rows vs the rectangle in the unit tests). Needs an extension with `picokv_decode_tail_global_rows_fn` (rebuilt from this tree), the fast build, shared projections and the tc tail; otherwise it logs `per-row packed storage: off`. `0` = rectangle (A/B only) |
| `PICOKV_CHUNK_STAGE` | `host` | Exact chunked prefill: every chunk that is not its request's last leaves its raw K/V in a pinned host block pool, and a continuation chunk streams its prefix back per layer (overlapped with compute) and attends prefix + chunk. The compressed cache is merged chunk by chunk (bitwise the one-chunk codes). GPU cost: two layer buffers of `context + chunk` tokens while a request is staged (1.1 GB at 128k), reserved from the KV budget. `off` = continuation chunks attend only themselves (then the guard turns chunked prefill off) |
| `PICOKV_STAGE_BLOCK` | `8192` | Host pool block (tokens x all layers; 1 GiB at 8 kv heads) |
| `PICOKV_STAGE_PREALLOC` | `1` | Pin a context-sized host pool at init (16 GiB at 128k). `0` pins blocks on first use (~0.45 s per block, once) |
| `PICOKV_RADIX` | `off` | `host`: keep SGLang's radix cache on. Tree nodes (`mem_cache/picokv_radix_cache.py`, radix cache backend `picokv`) own the raw post-RoPE K/V of their prompt tokens in the chunk stage's pinned host pages; a hit streams the prefix back per layer and attends prefix + suffix exactly, then compresses the whole prompt. Needs the chunk stage; the guard sets `radix_cache_backend="picokv"` (another explicit backend is rejected) |
| `PICOKV_RADIX_HOST_GB` | `32` | Radix mode: host K/V store cap (128 KB per token over 32 layers at 8 kv heads: 32 GiB = 256k tokens), the total over the attention-TP ranks (each rank pins its share), pinned at init with `PICOKV_STAGE_PREALLOC=1`. When full it evicts the tree (LRU, past leaves that hold no host K/V), then drops the host K/V of nodes held only by decoding requests; past that (running prefills alone exceed it) it grows with a warning |
| `PICOKV_RADIX_ADMIT` | `1` | Radix mode, `0`: no shared-prefix charge and no stage-reserve check at admission (A/B only: concurrent hits on one long prefix then run out of memory) |
| `PICOKV_RADIX_HOT` | `1` | Radix mode: GPU hot tier. Every host page (8192 tokens) may have a device mirror (`[L, 8192, 2, H, D]`, 1.07 GB, plus 0.17 GB of code planes with `PICOKV_RADIX_CODES`); a prefix slot whose mirror is valid is copied device to device instead of over PCIe, and the extend's own K/V writes update the mirror on the way (the device-to-host copy already runs). A mirror holds its bytes as token-pool accounting tokens (`pk_hot_hold`, only while `PICOKV_RADIX_HOT_FREE` of the pool stays free) and the tree counts them evictable: `evict()` frees mirrors first, their host K/V stays, so admission sees the capacity it sees without them. `0` = every hit streams the prefix over PCIe |
| `PICOKV_RADIX_HOT_GB` | `8` | Radix mode: device memory cap of the hot tier's mirrors, per attention rank (a page is 1.2 GB: about 6 pages = 54k prefix tokens). Past it the least recently used page without a live slot loses its mirror (its host K/V stays, a later hit streams it again). Larger caps buy longer mirrored prefixes for hit latency; they are device memory the pool held anyway, so they cost the compressed cache's capacity |
| `PICOKV_RADIX_HOT_FREE` | `0.1` | Radix mode: a mirror is taken only while this fraction of the token pool stays free after it |
| `PICOKV_RADIX_CODES` | `1` | With the hot tier and the fast prepare: mirrors keep their tokens' codes too (`[L,H,5,8192]` / `[L,H,8192,5]` int64 per page, +16% mirror bytes), so a hit recomputes nothing for its cached span: the leading run of a hit's global tokens whose codes a mirror holds is read back and only its new tokens' codes are computed (blockk on the suffix), written to their mirrors on the way. Codes are per token, so the cache is bitwise the one a full recompress builds (`PICOKV_RADIX_CODES_CHECK=1` compares every prepare). `0` = a hit recompresses its whole prompt |
| `PICOKV_RADIX_CODES_MIN` | `1024` | Radix mode: prompts shorter than this (tokens) take no code plan (a recompress is cheaper) |
| `PICOKV_RADIX_STATS` | `0` | One stdout line per radix extend: tokens copied by path (mirror, host, dedup, window, new) and the mirror count |
| `PICOKV_RADIX_CODES_CHECK` | `0` | Debug: every code-reusing prepare is compared field by field (and its full-capacity tail) with `_prepare_fast` over the same stage tokens; a mismatch is logged, the counts print at the last layer |
| `PICOKV_ALLOW_CHUNKED_PREFILL` | `0` | Only without the stage: `1` keeps a `chunked_prefill_size` smaller than the context length (each later chunk is compressed without the earlier chunks' KV: quality drops on long prompts) |
| `PICOKV_FILL_GROUP_MB` | `512` | Per-row storage: a rebuild or slot change fills its new rows in layer groups, each allocating at most this many MB of new row storage before those layers' request copies are freed, so new long rows never hold two copies of every layer at once (16 new 120k rows in one group needed +38 GB) |
| `PICOKV_ROWS_SHARE_CAP` | `2048` | Per-row storage: rows of the same capacity `<=` this many tokens share one buffer per side per 16 rows, across all layers (fewer allocations when many short rows join: about 2 tensors per new short row instead of 2 per row and layer; a survivor pins at most 15 other short rows) |
| `PICOKV_PREALLOC` | `1` | With dedup and the fast prepare: at a long request's first compressed chunk, each compressed layer's packed codes move into one key and one value buffer at the capacity its decode row will need (the whole prompt, from the scheduler's `orig_seq_lens`, plus the window catch-up and slide headroom, rounded up to 512 tokens); the request cache holds prefix views of them. Later chunks append their codes in place (no regrown copy per chunk and layer), and the decode batch adopts the buffers as the row's storage (per-row storage, the one-row rectangle and in-place slot changes: no copy, no free). Requests up to `PICOKV_ROWS_SHARE_CAP` tokens keep the old path (short rows share decode storage); prompts up to `sink + local` stay dense as before and compress once they reach that length. Same codes and kernels: bitwise (unit tests: chunk sizes not multiples of 64, many chunks, dense and short requests, decode batch changes, re-added rows, late joins, several slide epochs; greedy engine runs with `PICOKV_DET=1` identical to `0` at chunk 2048 and 8192). Prefill at SGLang's default chunk size 1.3-2.6% faster on 32k-120k prompts; decode within 0.2%. `0` = grow a copy per chunk (previous behavior) |
| `PICOKV_KV_POOL` | `1` | With `PICOKV_PREALLOC`: the full-capacity buffers come from their own PyTorch caching-allocator pool (`torch.cuda.MemPool(use_on_oom=True)`), outside graph capture. Allocated mid-forward from the default pool they carve the blocks the extend's activations just freed, and the remainders that no later activation or buffer fits pile up (reserved minus allocated grew about 5 GB per million prompt tokens). The pool's cached free blocks still serve other allocations as a last resort. Needs torch with `MemPool` (otherwise the default pool). See the memory capacity table. `0` = default pool |
| `PICOKV_PREALLOC_MAX_TOKENS` | `16384` | `PICOKV_PREALLOC` only in extends of at most this many tokens (chunked prefill; SGLang's default chunk on this GPU is 8192). A larger extend (chunking off, or a larger chunk size) keeps the previous path, whose decode batch builds the row capacity after the extend from its freed activations: there the buffers would be allocated while the extend's activations peak (`chunked_prefill_size=1 << 20`, 8000 x 16, three prompts per extend: peak +0.6 GB, +1.0 GB with the pool; 32k x 8: +0.8 GB with the pool). A request whose first chunk lands in such an extend runs exactly as with `PICOKV_PREALLOC=0` |
| `PICOKV_FI_NOSYNC` | `1` | FlashInfer decode re-plan without the device-to-host sync (`fast_decode_plan`); only used when FA3 is off |
| `PICOKV_GRAPHS` | `1` | Per-layer CUDA-graph replay (`PICOKV_GRAPH_WARMUP=3`). A batch that captures more than its budget (signature churn: 160, or 16 per layer with the slide) runs eager; recaptures forced by a schedule, pool or FA3-metadata reallocation (e.g. a one-row batch's schedule grows every ~2 x `PICOKV_SLIDE_HEADROOM` generated tokens) are not charged to it. A failed capture runs that batch eager; three failures in a row (a successful capture resets the count) turn graphs off for 1, 2, 4, ... 32 minutes, after which the next batch change (a rebuild or an in-place slot change) re-arms them |
| `PICOKV_GRAPH_FORK` | `auto` | Per-layer graph runs the dense-window branch (append + dense attention) on a forked stream, overlapped with the query projection + pack. `auto`: without FA3 always; with FA3 only on a layer whose compressed tail grid is above `PICOKV_TAIL_QUANT_BLOCKS` (the fused query quantizer is off there, so the query projection + pack are a serial ~13 us per layer at 160 rows) and only at a batch outside `PICOKV_FORK_B`; elsewhere the serial PDL chain is faster. Only the order of independent kernels changes: bitwise (greedy engine runs with `PICOKV_DET=1`: forked == serial at 1-16 rows, dense -> slide, mixed lengths). Gain (Llama-3.1-8B-Instruct, sm_90, real weights, ms/step serial -> auto): 122880x1 8.19 -> 8.10, 122880x2 9.62 -> 9.56, 65536x4 9.74 -> 9.69, 16384x8 8.39 -> 8.34, 4096x48 9.95 -> 9.92, 4096x64 10.89 -> 10.85, MATH-500 at 160 concurrent step p50 at 97-160 rows -1.6%; short prompts and small grids unchanged (same graphs). Memory: one fork event pair per graph, measured peak +4 to +6 MiB. `1`/`0` force fork / serial |
| `PICOKV_FORK_B` | `8,48` | `lo,hi`: with `PICOKV_GRAPH_FORK=auto` and FA3, a large-grid layer forks only at batch `<= lo` or `>= hi` rows (forked at 16-32 rows with 8k prompts: 8192x16 8.49 -> 8.53, 8192x32 10.23 -> 10.29 ms/step) |
| `PICOKV_Q_DIRECT` | `1` | Breakable decode graph: FA3 + the shared-projection Triton qproj read q in place as a strided view of the qkv projection (no per-layer staging copy). `0` restores the copy |
| `PICOKV_DENSE_ONLY` | `1` | A layer whose batch has no compressed token at all (every row gl = 0: short prompts, nothing slid yet) runs only append + dense-window attention, in its graph and in the eager steps before capture, skipping qproj / pack / tail / vproj (the tail would merge in exactly zero, so the output is bitwise unchanged). The first compressed token in the batch re-captures the full sequence. `0` always runs the full sequence |
| `PICOKV_DENSE_LOCAL` | `0` | Opt-in speed knob, off by default: slide local of a small decode batch (`<= PICOKV_DENSE_LOCAL_MAX_BS` rows), e.g. `4096`. Off, every batch slides at `PICOKV_SLIDE_LOCAL` (window `W = sink + local` = 320), so generated tokens are compressed the same way at every batch size. On, a short-prompt request in a batch of <= 8 rows keeps up to that many generated tokens dense, so how much of a request is compressed depends on its batch size (quality numbers are then not the method's). What it buys (Llama-3.1-8B-Instruct, sm_90, real weights, one serial run of the previous code version, off / `4096` / flashinfer): MATH-500 at 8 concurrent 1016 / 1109 / 1120 tok/s; 64-token prompts, generated tokens [512, 2048): 7.11 / 6.81 / 6.90 ms/step at 8 rows, 6.90 / 6.25 / 6.22 at 1 row; prompts >= 32k unchanged (within 1%). A batch keeps its window until the next rebuild; a rebuild to a larger batch compresses the excess once (catch-up, about 80 ms for 8 rows x 4096 tokens over 32 layers). Memory: about `B x (W + 64) x 128 KB` (4.3 GB at 8 rows with `4096`), reserved from the KV budget (see Limitations, memory accounting) |
| `PICOKV_DENSE_LOCAL_MAX_BS` | `8` | Largest decode batch that uses `PICOKV_DENSE_LOCAL` |
| `PICOKV_STEPPROF` | unset | `1` logs (info level) host wall time per engine step, bucketed into build / capture / prefill / replay / warm-up, every `PICOKV_STEPPROF_EVERY` (`2048`) steps |
| `PICOKV_RIDS_CPU`, `PICOKV_RIDS_PER_STEP`, `PICOKV_FUSED_APPEND`, `PICOKV_ENG_DIRECT_OUT` | `1` | Host-overhead fast paths; set `0` only to debug |
| `PICOKV_NO_TC` | unset | `1` uses the popc tail instead of the tensor-core tail |
| `PICOKV_QQ_WARP` | `1` | Decode query quantize + bit-plane pack with one warp per row (ballot-packed planes; bitwise the old kernel's codes, 0.80 -> 0.29 ms per step at 160 rows over 32 layers). Needs an extension rebuilt from this tree. `0` = old one-thread-per-word kernel |
| `PICOKV_TAIL_QUANT` | `1` | Shared tail without the query prep op: the Triton query projection also zeroes the tail accumulators, and each tensor-core tail block quantizes and bit-packs its 4 query heads itself from the bf16 projected query (`picokv_decode_tail_global_q_fn`). Bitwise the prep op's codes; one launch fewer per layer. Used only for tail grids of `<= PICOKV_TAIL_QUANT_BLOCKS` (`1024`) blocks (schedule entries x kv heads): every block pays the quantizer, which past about 2 waves of blocks costs more than the launch it saves (8 rows x 32k: +9.8 us per layer), so long contexts keep the prep op. Not with `PICOKV_DET`. Needs an extension rebuilt from this tree. `0` = prep op always |
| `PICOKV_PDL` | `1` | Programmatic dependent launch (sm_90, Triton with `gdc_wait`) in the per-layer decode graph. On the fused-quantizer path (`PICOKV_TAIL_QUANT`, serial graph) the layer runs query projection -> append -> FA3 -> tail -> value projection: the query projection triggers the append's early launch at its entry (the append reads nothing it writes and waits for it before exiting, so FA3 starts after both), and the value projection launches as a dependent of the tail and waits for it before any access. On the other paths only the value projection is a dependent launch. Same kernels and arithmetic: bitwise the serial order's outputs (300 graph replays with fresh inputs at 1-160 rows). FA3 with scheduler metadata cannot be launched as a dependent, and an early trigger inside the tail measured +1.6-3% per step at 4-16 rows, so the tail does not trigger. Gain: -0.7 to -0.8% per step at 512 x 1 / 4096 x 4, neutral (within 0.1%) from 8k x 16 up. Both launch variants of each Triton kernel are compiled on the first eager decode call (the per-layer graphs capture with no eager step before them). `0` = serial order, no PDL |
| `PICOKV_TC_SCHED_SPLIT` | auto | Tensor-core tail on a schedule table (slot mode, per-row storage): below 2 blocks per SM (entries x kv heads < 2 x SM count, 264 on H100) each 512-token entry runs as 2 one-tile blocks, the width the unscheduled grid already picks for small grids (global kernel, 1 row x 1k compressed tokens: 7.5 -> 5.0 us per layer). The float atomics then add 2 partials per entry instead of 1 (last-bit differences, like any tile split). `1` always, `0` never |
| `PICOKV_COMPOSE` | `1` | Composed decode replay (breakable decode graph, needs `cuda-python` and the patched `breakable_cuda_graph.py`): a decode step whose 32 layer breaks all replay an existing per-layer graph runs SGLang's 33 graph segments and those 32 graphs as one composed CUDA graph (child-graph nodes in order) instead of 65 graph launches; the layer boundaries' launch gaps go away. The layers' eager step work (step begin, slide merge / quantize / regrow) touches only picokv state and is enqueued before the launch. A set of graphs is composed after it repeats on two consecutive steps; a step that captures, rebuilds or runs a layer eagerly replays in order as before (a fixed batch composes every step after warm-up; 160 concurrent mixed-length requests: 80% of steps compose, 19% replay in order around captures, rebuilds and slot changes). Same kernels in the same dependency order: bitwise (greedy engine runs with `PICOKV_DET=1` give identical tokens at 1-16 rows, dense -> slide, compressed prefixes, staggered finishes). Device memory: one composed exec per graph bucket, at most 2 kept (about 2-3 MiB each; measured peak +4 to +6 MiB at fixed batches, +14 MiB at 160 concurrent requests; KV pool unchanged); host: the raw graphs are kept. Gain (Llama-3.1-8B-Instruct, sm_90, graph decode): -1.2 to -2.0% per step (512x1 6.67 -> 6.58 ms, 8192x16 8.66 -> 8.56, 122880x1 8.37 -> 8.25). `0` = per-graph replay |
| `PICOKV_FILL_OFFPATH` | `1` | With `PICOKV_COMPOSE`: each per-layer graph is composed as a clone in which FA3's semaphore reset (the one-int zero fill of the scheduler metadata's tile counter after the FA3 kernel, read only by the next layer's FA3) runs beside the tail instead of before it: the tail depends on the FA3 kernel directly. Matched by kernel name, fill size/value and the counter's address; a graph without that pattern is composed as captured. Bitwise (same kernels and inputs). Gain on top of `PICOKV_COMPOSE`: -0.4 to -0.8% per step (512x1 6.58 -> 6.53 ms, 8192x16 8.56 -> 8.51, 122880x1 8.25 -> 8.21). `0` = compose the graphs as captured |
| `PICOKV_COMPOSE_STATS` | `0` | `N`: print the composed-replay counters (composed / in order / flushed / built / evicted steps, graphs with an off-path variant) to stderr every `N` decode steps |
| `PICOKV_DET` | `0` | Deterministic decode tail: each tensor-core tail block stores its partial sums to a per-layer scratch buffer and a Triton kernel adds them in a fixed order, instead of float `atomicAdd` across tile blocks (whose order varies run to run). Same inputs give bitwise the same output, also at batch 1 (which then leaves the fused single-row op). Scratch: about 10 B per compressed token per kv head per layer (80 B at 8 kv heads), transient (157 MB at 16 x 120k; in graph mode it lives in the graph pool, shared by the layers). Needs the tc tail, shared projections, 320-bit codes and an extension with `picokv_decode_tail_global_part_fn`. Cost (Llama-3.1-8B-Instruct, sm_90, graph decode, 1024 generated tokens): +1.3 to +3.1% per step (8192x16 8.74 / 8.93 ms, 32768x8 10.19 / 10.50, 65536x4 10.03 / 10.29, 122880x1 8.50 / 8.61, 122880x2 9.81 / 10.08, off / on), +1.6-2.8% on 24 x 4096-token generations, -1.5 to -1.9% throughput at 160 concurrent MATH requests (6536-6550 / 6428-6455 tok/s); a batch with no compressed token yet (short prompts, nothing slid) runs dense only and pays nothing. With it 40 of 40 repeated serving waves give identical text (off: 22-24 of 40) |
| `PICOKV_DECPROF` | unset | Per-step decode profiling; it perturbs timings, so never set it for A/B runs |
| `PICOKV_PREBUILT_DIR` | `PiCoKV/build/picokv_extensions` | Where the built extensions are loaded from. A build whose `.so` was compiled for another GPU architecture (e.g. a stale sm_100 build on sm_90) is rejected at load with the path and both archs |

Kernel-tuning switches (`PICOKV_TC_BT`, `PICOKV_TAIL_BT1`, `PICOKV_WIN_CH`,
`PICOKV_NO_KBSPEC`, `PICOKV_SCORE_BMMA`) are for experiments. Leave them
unset.

## Results (Llama-3.1-8B-Instruct, one sm_90 80 GB GPU)

Decode inter-token latency, in ms/step (median, idle GPU, 128 generated
tokens, real weights). picokv uses the breakable decode graph. flashinfer uses
its best mode, the full decode graph.

| Prompt | Batch | picokv | flashinfer |
|---|---|---|---|
| 512 | 1 | 6.51 | 6.17 |
| 4096 | 4 | 7.05 * | 7.12 |
| 8k | 1 | 7.04 * | 6.62 * |
| 8k | 16 | 8.41 | 11.85 |
| 32k | 8 | 9.76 | 17.31 |
| 64k | 4 | 9.69 | 17.25 |
| 120k | 1 | 8.10 | 11.39 |
| 120k | 2 | 9.57 | 16.59 |
| 120k | 16 | 25.45 ** | does not fit (max 2 concurrent: 16.30) |

The unstarred rows come from one serial run on one idle GPU
(`ITL_REPS=3 ITL_WARM_LEN=1024`, 128 generated tokens), picokv with and without
`PICOKV_PREALLOC` and flashinfer interleaved per shape (with vs without: within
0.15% at every shape). The benchmark prefills without chunking, so these
extends exceed `PICOKV_PREALLOC_MAX_TOKENS` (512 x 1: below the share cap) and
the default takes the path without; the table lists that run. The 512 x 1,
8k x 16, 32k x 8 and 120k x 1 rows are from a later serial run with the band
quantizer (`PICOKV_FIXED_BAND`), the previous build and flashinfer interleaved
per shape (previous build: 6.53 / 8.44 / 9.78 / 8.12 ms, same memory).
* an earlier run (previous code version).
** 1024 generated tokens, `mem_fraction_static=0.70` (earlier code: 26.76);
with 128 tokens at 0.9 it reads 28.37 ms (the shorter run weighs the first
steps after the batch build more).

Long generation (24 rows, 256-token prompts, 4096 generated tokens): 8.33
ms/step for picokv vs 9.91 for flashinfer. A single short row is the slow
case (512 x 1: +9%; a 64-token prompt generating 2048 tokens at 1 row: 6.71
vs 6.22 ms/step past 512 tokens, previous code 6.76): the backend's fixed
per-layer cost is larger than flashinfer's whole decode attention at that
size. Once a row has a compressed part, each layer adds the query projection
and the tail kernel, which quantizes the projected query in its prologue
(together ~8 us at 1 row, ~1k compressed tokens; 12.4 us with the separate
quantize op), and the value projection (~4 us): about 0.38 ms per step over
32 layers.

At 120k, peak decode throughput is about 600 tok/s for picokv versus 123 tok/s
for flashinfer (4.9x); 16 x 120k with per-row storage peaks at 69 GB (the
batch build fills new rows in layer groups, `PICOKV_FILL_GROUP_MB`). To reproduce:
`BENCH_MODEL=<path>/Llama-3.1-8B-Instruct ITL_REPS=2 ITL_WARM_LEN=1024 ./run_itl.sh <gpu> pk_8k_b16 picokv 8000 16 128 bcg`
(and `flashinfer ... full`). The default random-weight model gives the same
decode ITL within 1-2%, but it has no projection, so its prefill runs online.

Time to first token, ms (median of 5, of 3 above 64k tokens; real weights;
`scripts/serving/bench_ttft.py`, `TTFT_CHUNK=8192` for the chunked columns).
"8192" is SGLang's default chunked prefill on this GPU (picokv through its
chunk stage), "one chunk" is `chunked_prefill_size=1 << 20`:

| Batch x prompt | picokv 8192 | flashinfer 8192 | picokv one chunk | flashinfer one chunk |
|---|---|---|---|---|
| 1 x 512 | 23.0 * | 21.2 | | |
| 1 x 2048 | 55 | 55 | | |
| 1 x 8192 | 221 | 227 | 223 | 224 |
| 1 x 32k | 1236 | 1521 | 1171 | 1217 |
| 1 x 64k | 3360 | 4721 | 3197 | 3322 |
| 1 x 120k | 9195 | 14303 | 8739 | 9626 |
| 4 x 16k | 2023 | 2200 | 1925 | 1991 |
| 8 x 4096 | 829 | 817 | 816 | 804 |
| 2 x 384 | 28.1 | 26.0 | | |
| 2 x 512 | 32.2 | 30.8 | | |
| 3 x 384 | 36.6 | 34.5 | | |
| 24 x 512 | 297 | 295 | | |
| 64 x 256 | 392 | 387 | | |

The 8192 columns come from one serial run (picokv, the previous build without
the band quantizer, flashinfer; the previous build read 1249 / 3381 / 9254 /
2045 / 847 ms at 32k / 64k / 120k / 4 x 16k / 8 x 4096, within 0.6% elsewhere,
same peak memory; the 1 x 512 / 2 x 384 / 2 x 512 / 3 x 384 rows from two
later serial picokv runs (median of 9 each, alternating with `PICOKV_PREP_FUSED=0`)
and one flashinfer run over the same case list, none of them among the first two
cases (1 x 512 also as the first case: 23.2-26.2 ms, flashinfer 21.3;
such small shapes move by 1-3 ms with their place in
the case list, see the knob table for the alternating comparison), the one-chunk columns from an earlier run (previous code
version). * not the first case of its run (as the first case it reads about
26-27 ms, flashinfer 21.2: the decode path's warm-up, see Limitations / known
issues). At SGLang's
default chunk size picokv takes 0.65-0.82x flashinfer's TTFT on 32k-120k
prompts: flashinfer attends each chunk's prefix through its paged kernel,
about 2x slower than its ragged one, while picokv's continuation chunks attend
the staged prefix with a ragged plan. Chunking costs picokv about 6-7% over one
chunk; its peak memory over this run is 22.3 GiB (one chunk, earlier run:
about 36 GB; flashinfer 57.4 GiB, its preallocated pool).

Quality (greedy decoding, offline projections, `eas=0`):

| Benchmark | flashinfer | picokv |
|---|---|---|
| LongBench, all 3750 samples | 49.62-49.74 | 49.30-49.43 |
| Needle-in-a-haystack, 160 cells (8k-128k x 10 depths) | 98.12 | 97.56-98.47 |
| RULER 4k / 16k / 64k | 1.000 / 1.000 / 0.991 | 1.000 / 1.000 / 0.981 |

The current code at SGLang's default chunk size (8192, chunk stage) scores
LongBench 49.34 and NIAH 98.47 (previous code: 49.30-49.43 / 97.78-98.47;
flashinfer at 8192: 98.12).
Every needle length up to 96k scores 100 for both. All misses are at
>= 104k with the needle at depth <= 22%, where flashinfer also misses 4 cells.
The base model is weak there, and HF FlashAttention-2 scores 97.56. The
picokv range is over code versions and runs; the current code misses the
same 4 cells as flashinfer. Some earlier versions also miss part of the cell
at 104k, depth 11%; that partial score depends on a near-tie greedy split.

AIME (30 problems each, up to 16384 generated tokens; sampled = temperature
0.6, top-p 0.95, 16 samples per problem):

| Benchmark | flashinfer | picokv |
|---|---|---|
| AIME 2024 greedy / avg@16 / pass@16 / maj@16 | 6.7 / 6.5-6.9 / 30.0-33.3 / 13.3-16.7 | 3.3 / 5.0-7.9 / 23.3-36.7 / 10.0-23.3 |
| AIME 2025 greedy / avg@16 / pass@16 / maj@16 | 3.3 / 0.8 / 10.0 / 0.0 | 0.0-3.3 / 0.0-1.5 / 0.0-10.0 / 0.0-6.7 |

One problem is 3.3 points. Llama-3.1-8B-Instruct is near the floor on AIME,
so these rows only show that nothing breaks over 16k-token generations.
Ranges are over runs (sampling is unseeded: four picokv AIME 2024 runs of one
code version gave avg@16 5.0 / 7.3 / 7.3 / 7.7; on AIME 2025 four runs of the
recent code had 0, 3, 2, 4 and 7 correct samples out of 480). With 64 sampled requests in
flight, picokv generates 3683-3904 / 3206-3947 tok/s (AIME 2024 / 2025)
against 1936-2033 / 2011-2055 for flashinfer, whose pool cannot hold 64 x 16k
tokens.

Short-context serving, MATH-500 (500 problems, greedy, up to 4096 generated
tokens, real weights, `mem_fraction_static=0.6`). Throughput is generated
tokens per second of wall time:

| Concurrency | picokv tok/s | flashinfer tok/s | picokv acc | flashinfer acc |
|---|---|---|---|---|
| 8 | 1034-1069 | 1112-1128 | 49.6-51.0 | 52.2 |
| 24 | 2513-2587 | 2581-2582 | 50.6-52.6 | 53.2 |

With `PICOKV_DENSE_LOCAL=4096` (the default before) the 8-concurrent row reads
1095-1109 tok/s and 51.8: a 4096-token dense local keeps almost every MATH
generation (mean 815 tokens) uncompressed, so it measured mostly dense
attention. Now every batch size compresses generated tokens past the 320-token
window; at 24 concurrent that was already the case. The 24-concurrent gap is
numeric (greedy splits at near-tie tokens; a Triton backend lands at the same
level), not a retrieval loss. At 8 concurrent four runs (two code versions;
the previous one read 1016 tok/s) sit 0.8-2.2 points under the dense-local
run, inside the 24-concurrent run-to-run spread. picokv's greedy output
differs from run to run (two runs give 78-83% identical answers at 8 concurrent,
64-70% at 24). flashinfer's
output is identical across runs.

Memory capacity (`scripts/serving/bench_capacity.py`: N random prompts of
one length submitted at once, SGLang's default chunked prefill (8192) and its
default CUDA allocator (no `expandable_segments`), no radix cache,
`mem_fraction_static=0.9`, 256 generated tokens each). "Concurrent" is the
largest `#running-req` the scheduler logged; peak is the GPU's memory over the
run (nvidia-smi); time is the whole run, all N requests:

| Prompt | N | picokv pool (tokens) | picokv concurrent | picokv peak | picokv time | flashinfer pool | flashinfer concurrent | flashinfer peak | flashinfer time |
|---|---|---|---|---|---|---|---|---|---|
| 32k | 96 | 2.33M | 72 | 70.2 GiB | 131 s | 456k | 14 | 74.4 GiB | 188 s |
| 64k | 48 | 2.57M | 39 | 70.5 GiB | 171 s | 456k | 7 | 74.3 GiB | 263 s |
| 120k | 24 | 2.69M | 22 | 72.9 GiB | 228 s | 456k | 3 | 74.3 GiB | 374 s |

picokv admits 5.1-7.3x as many long requests and finishes the batch 1.4-1.6x
sooner. Before `PICOKV_PREALLOC` this allocator ran out of memory at 64k x 48
and 120k x 24 (holes left by the per-chunk regrown copies: up to 10 GiB
reserved but unallocated) and 32k x 96 peaked at 78.3 GiB; with
`expandable_segments:True` those runs peaked at 75.3-78.5 GiB, now 69.9-71.1
GiB. Both backends stop at their token pool (picokv usage 0.98-1.00, the pool
depends slightly on `max_running_requests`, which picokv reserves per row;
flashinfer 0.98, at 120k 0.79: a fourth prompt does not fit) and every request
completes, without an out-of-memory error. flashinfer's peak is its
preallocated pool. picokv's pool is an accounting budget (about 23 KB per
token over 32 layers vs 128 KB of bf16 K/V) and its peak is what the rows
actually hold.

Mixed-length decode batch (`PICOKV_ROWS`): one 122,739-token NIAH prompt and
N MATH-500 prompts in one `generate` call (`mem_fraction_static=0.9`,
`max_total_tokens=200000`). Peak = this process's GPU memory:

| N (graph bucket) | per-row storage | rectangle (`PICOKV_ROWS=0`) |
|---|---|---|
| 23 (24) | needle found, peak 36.1 GB | needle found, peak 78.5 GB, same tokens |
| 31 (32) | needle found, peak 36.5 GB, 1011 tok/s | CUDA out of memory |

At equal lengths per-row storage costs +0.2-0.4% per step (8k x 16:
8.77 vs 8.76 ms; 32k x 8: 10.17-10.20 vs 10.16 ms) and about 1 GB more peak
memory (32.3 vs 31.3 GB at 32k x 8). MATH-500 at 24 concurrent (same
settings as the mixed run): 51.6-52.0 / 2446-2472 tok/s per-row vs 51.0 /
2523 tok/s rectangle (84-85% identical answers, the run-to-run level at 24
concurrent; the per-row runs also generated 7-9% fewer tokens, so tok/s is
not a clean speed A/B).

Tensor parallelism (`tp_size=2`, two GPUs): NIAH 80 cells 8k-120k, picokv
78/80 = flashinfer 78/80 (the same two misses at 112k / 120k, depth 0);
MATH-500 at 24 concurrent 49.0 / 3612 tok/s vs flashinfer 51.2 / 3804 tok/s.

High concurrency without `max_total_tokens` (320 MATH-500 prompts, up to 2048
generated tokens, `max_running_requests=160`, `mem_fraction_static=0.9`):
picokv runs at 6927-6970 tok/s (peak 33 GB; it used to crash on admission;
earlier code 6102-6793), flashinfer at 7366-7412 tok/s (peak 75 GB)
(0.94x). The gap is in the decode steps at 129-160 rows: at ~600-token contexts picokv's
dense window (~320 tokens) plus the compressed tail costs more than
flashinfer's whole dense attention.

Mixed HTTP serving (`scripts/serving/http_e2e.py`, 96 concurrent requests:
84 short prompts and 12 prompts of 3k-12k tokens sharing a repeated filler,
16-400 generated tokens, `mem_fraction_static=0.6`): all 96 complete for both
backends; picokv takes 6.2 s vs flashinfer's 5.7 s without the radix cache
(1.08x the wall time), 4.25-4.3 vs 3.7 s with it (picokv `PICOKV_RADIX=host`,
1.15x).
The script's 23 checks (greedy solo == batch, `n=3`, logprobs, stop / length,
streaming, OpenAI chat / completions, client disconnect + `/abort_request`,
25k-token chunked passkey + repeat, `/flush_cache`, `/metrics`) pass for both
backends with and without radix, except that in 3 of 14 picokv runs the
repeated 25k prompt kept the key but continued differently after it (" The
pass key is" vs " The grass is"; a near-tie, see Limitations, determinism).

## Limitations / known issues

- **Radix prefix reuse is host-staged and off by default.** The compressed
  cache cannot be shared across requests: the prompt's split into sink /
  compressed global / local window depends on the full prompt length, and a
  plain radix hit skipped the prefill that builds it (NIAH about 18). Without
  `PICOKV_RADIX=host` the guard turns the radix cache off. With it, the tree
  keeps each cached prompt token's raw K/V on the host (see the knob) and a hit
  recomputes attention over prefix + suffix exactly, so the compressed cache is
  the one a full prefill builds. Real weights, Llama-3.1-8B-Instruct:
  - Outputs, 48 greedy requests (doc + questions, 20k shared prefix, multi-turn,
    duplicates, mixed batches; chunk 2048): radix vs no radix 31/48 identical
    48-token outputs, first token 48/48 (first-token |dlogprob| median 0.004).
    That is picokv's own run-to-run level (no radix vs no radix 32/48; the
    decode batch layout varies). Host-store eviction (`PICOKV_RADIX_HOST_GB=1`)
    and drop under decode pressure: first token 48/48 and 20/20.
  - NIAH 160 cells (chunk 8192, one engine per 80-cell shard, greedy): 97.84
    with radix (hot tier + codes) vs 97.50 without radix; 152 of the 160 cells
    are the same text, one cell scores differently (114k: 72.7 vs 18.2, radix
    higher) and the rest are the 106k-131k tail cells where the no-radix runs
    themselves move by a cell or two. LongBench 3750: 49.35 vs 49.28 (+0.07;
    run-to-run 49.28-49.43); NIAH before the hot tier: 98.35 vs 98.41
    (flashinfer radix 98.01, no radix 98.12).
  - Exactness: the code planes are per token, so a hit's compressed cache is
    bitwise the one a full prefill builds (`PICOKV_RADIX_CODES_CHECK=1`: 480
    prepares, 0 mismatched; `PICOKV_DET=1` replays, hot tier on vs off:
    16/16 bitwise, including hits that are not chunk-aligned and a row reading
    codes an earlier row of the same extend wrote).
  - TTFT (ms, median of 7, P-token shared prefix cached + S-token new suffix,
    one engine per row, SGLang's default chunk 8192; `scripts/serving/
    bench_prefix.py` measures the same cases with the default cache off):

    | case | radix | radix, hot tier off | radix, hot tier without codes | flashinfer radix |
    |---|---|---|---|---|
    | P 4k + S 256 | 35.3 | 39.0 | 40.7 | 27.5 |
    | P 16k + S 256 | 35.0 | 50.0 | 43.3 | 27.0 |
    | P 64k + S 256 | 81.5 | 184.3 | 105.2 | 47.7 |
    | P 16k + S 256, 8 requests | 130.0 | 200.9 | 206.8 | 113.8 |

    Without radix the same prompts are full prefills: 114 / 540 / 3501 / 4158 ms
    (flashinfer 117 / 570 / 4739 / 4566).
  Costs and limits: without the hot tier a hit is bound by the host-to-device
  copy (128 KB per prefix token, about 45 GB/s: 184 ms at 64k vs flashinfer's
  48 ms) plus compressing the whole prompt; the hot tier mirrors the prefix on
  the device (`PICOKV_RADIX_HOT_GB`, 8 GiB per rank: about 54k tokens) and its
  code planes let a hit skip the recompress of its cached span, which brings
  64k to 81 ms and 8 x 16k to 130 ms, the rest still over PCIe (a longer
  prefix than the cap streams its tail every hit). A miss pays the
  device-to-host copy, one
  `cudaMemcpy2DAsync` per layer and slot run straight from the layer's K/V
  (no device copy; +2 ms at 1 x 512, host-side: the CUDA calls cost ~4 us
  in a loop but ~17 us inside the forward; +0.5-1.6% at 24 x 512, noise at
  8k, +2% at 32k); only prompt tokens are cached (decode tokens have no
  host K/V, so a multi-turn follow-up re-prefills the previous answer);
  every hit row
  rebuilds its own compressed cache over its prefix (no sharing between
  concurrent hits on one prefix: 48 hits on a 48k document take 4.9 s vs
  flashinfer radix's 2.0 s); `PICOKV_RADIX_HOST_GB` of host memory is pinned
  at init (~0.4 s per GiB; pinning holds the CUDA driver lock, so pinning on
  demand or from a background thread stalls the forward instead).
  Admission (`mem_cache/picokv_radix_cache.py` + three hooks in
  `managers/schedule_policy.py`): a hit on a prefix that a running request
  holds is charged the prefix tokens (each holder has its own compressed
  cache over it; SGLang counts the shared prefix once) and holds them until
  it finishes, and the prefix rows of one prefill share the stage's GPU
  layer buffers (`context + chunk` tokens): 48 concurrent hits on a 48k
  document serve 48/48 at 56.7 GB peak (flashinfer radix 58.7 GB); without
  it (`PICOKV_RADIX_ADMIT=0`, A/B only) the same run dies of OOM.
  The hit rows of one prefill are staged together (one gather per layer
  across rows, dense attention of all rows in one ragged call): a 49-request
  hit prefill takes 129 ms instead of 202 (flashinfer radix about 40 ms),
  with bitwise the same outputs (48/48).
- **Chunked prefill goes through a host stage.** The compressed cache keeps
  no raw prefix K/V, so a continuation chunk could not attend its prefix (at
  SGLang's default 8192 that dropped 32k-NIAH to about 15.5). The chunk stage
  (`PICOKV_CHUNK_STAGE=host`) keeps every non-final chunk's raw K/V in a
  pinned host pool, streams layer l's prefix back two layers ahead (about
  8.5 ms per layer at a 120k prefix, hidden under about 28 ms of compute) and
  attends prefix + chunk with a second ragged plan; the codes are merged chunk
  by chunk and equal the one-chunk codes bitwise (unit test). Costs: about 7%
  TTFT over one chunk on long prompts, a context-sized pinned host pool
  (16 GiB at 128k, pinned at init), and two GPU layer buffers of
  `context + chunk` tokens (1.1 GB at 128k) reserved from the KV budget
  (`picokv_stage_reserve_bytes`).
- **Memory accounting.** The packed codes are what SGLang's token pool
  counts (the projection's key + value code bits: 80 B per token per kv head
  per layer at k320/v320, ~20 KB per token over 32 layers at 8 kv heads; with
  the decode slide off, e.g. online projections, decoded tokens stay dense
  and the pool counts dense bf16, 128 KB per token). What the per-token size does not cover is reserved
  from the KV budget before the pool is sized: every decode row's dense window
  (`B x (sink + local + 64)` tokens at ~128 KB per token, `local` = 256, or
  `PICOKV_DENSE_LOCAL` for graph buckets <= 8 rows when set) plus its packed slide headroom
  (`<= 2 x PICOKV_SLIDE_HEADROOM + 511` tokens), for `max_running_requests`
  rows rounded up to SGLang's largest decode graph bucket (the aligned request count, not a power of 2), plus one dead slot batch
  (`PICOKV_SLOTS_MAX_WASTE_MB`) (`picokv_fixed_reserve_bytes`; 1.8 GiB
  at 8 rows, 3.3 GiB at 24, 16.3 GiB at 160; with `PICOKV_DENSE_LOCAL=4096`
  5.5 / 6.3 / 16.3 GiB). If that reserve would take more than half the
  budget, `max_running_requests` is capped (one warning, e.g. 256 -> 64 at
  `mem_fraction_static=0.4`). The bound needs per-row storage
  (`PICOKV_ROWS=1`): a rectangle holds B x its longest row, so a long request
  next to short ones pins far more than the pool counts (see the mixed-batch
  table; with per-row storage off the backend logs this at init). A prefill frees a kept decode batch none of whose rows is still
  running once its packed codes exceed `PICOKV_SLOTS_MAX_WASTE_MB`.
- **Small batches are slower than flashinfer end to end.** Steady decode of
  short prompts is at parity until the rows generate past the 320-token
  window (8 rows, 64-token prompts: 6.46 / 7.01 vs 6.52 / 6.89 ms/step for
  flashinfer over generated tokens [0, 512) / [512, 2048); 1 row: 6.30 / 6.71
  vs 6.17 / 6.22). Past it every layer runs the compressed tail as well
  (per-layer fixed cost, see the decode latency notes). MATH-500 throughput is
  0.92-0.96x flashinfer at 8 concurrent (1034-1069 vs 1112-1128 tok/s over
  runs; this version 1061 vs 1119, 0.95x, with 8% fewer generated tokens it
  finishes the 500 problems in 389 vs 402 s) and 0.97-1.00x at 24;
  `PICOKV_DENSE_LOCAL=4096` gives 0.99x at 8 by leaving those generations
  dense. Prefill is ~3% of wall time at 8 concurrent, in-place slot changes
  ~1.6%. TTFT at SGLang's default chunk size (with
  SGLang's overlap scheduler it includes the host cost of the first decode
  step, for flashinfer too) is within 3-6% of flashinfer at 4x512 / 8x1024 /
  24x512 / 64x256 (58.0 / 203.5 / 302.7 / 395.8 vs 54.7 / 198.6 / 293.4 /
  385.7 ms). A single 128-token prompt takes 21.5 vs 20.6 ms. Short
  compressed prompts are the worst case: 1 x 512 23.0 vs 21.2 ms (23.2-26.2
  as the first case of a run), 2 x 384 28.1 vs 26.0 ms. Such a
  prefill is host-bound (1 x 512, before `PICOKV_PREP_FUSED`: forward
  host time 16.0 vs 12.9 ms, GPU busy 12.5 vs 11.3 ms; the compression path
  added about 100 us of host work per layer, the fused prepare takes 1.4-2.9 ms
  off the prefill step and 0.4-1.9 ms off TTFT), and the first decode step, which
  the overlap scheduler launches before the prefill result is read, spends
  about 1.9 ms more host time admitting the new row. The first requests after
  startup also pay the decode path's warm-up (eager steps, then the per-layer
  graph capture: 9-22 ms on the first 3), so a benchmark that starts with
  this shape reads about 28 ms. From 1k up it is within 3% or faster
  (1x1024: 32.0 vs 31.0 ms, 1x2048: 55.9 vs 54.9 ms; see the TTFT table).
- **Batch-membership changes.** A request joining or leaving within the decode
  graph bucket updates rows in place (`PICOKV_SLOTS`, about 6 ms per change at
  24 rows; host time at 129-160 rows about 2.5 ms when requests only leave,
  5 ms when new ones join: the layers' row tables live in one block per batch,
  permuted in one numpy pass and copied to the GPU in one transfer, and new
  short rows share buffers across layers). With `PICOKV_STABLE_ORDER` (default) the rows past the new size
  fill the finished requests' positions, so a change moves one row (its dense
  window and table entries) per finished request instead of shifting every
  later row. A bucket change rebuilds the batch and re-captures the per-layer
  graphs: with `PICOKV_FAST_BUILD2` the build step takes about 24-30 ms at
  8-152 rows (32 layers; 35-63 ms before at 104-152 rows), then two eager
  steps (about 1.5x a replay) and one capture step (about 3.5x).
- **Parallelism.** Attention tensor parallelism works (`tp_size` > 1: each
  rank keeps its kv heads' compressed cache and its slice of the offline
  projections). The ranks agree on every staging verdict (chunk stage, radix
  host store: a MIN all-reduce on the attention-TP CPU group), so a pinned or
  device allocation that fails on one rank fails the extend on all of them
  instead of letting the ranks' caches diverge. Speculative decoding, DP attention, pipeline parallelism,
  decode context parallelism, the hierarchical cache and PD disaggregation
  are rejected by the guard; mixed-chunk prefill+decode is turned off.
  Supported models are bf16 with GQA 4 and head_dim 128.

- **Lost prefixes abort the request.** A continuation chunk whose prefix the
  chunk stage could not hold (pinned host or device memory ran out, or a
  radix hit's host K/V went missing) still runs on the chunk alone, and the
  backend logs a warning with the running count of such extends and unseen
  prefix tokens (at 1, 2, 4, 8, ... occurrences). At the request's last
  prefill chunk the scheduler then finishes it with an abort, HTTP 503
  `ServiceUnavailableError` ("N prompt-prefix tokens could not be attended
  ...; retry the request"; a stream ends with an abort carrying status 503),
  instead of answering from a truncated context. `PICOKV_LOST=keep` restores
  decoding from what is left. Without the stage (`PICOKV_CHUNK_STAGE=off` +
  `PICOKV_ALLOW_CHUNKED_PREFILL=1`) the loss is opted into and only counted.
  With the default stage and the `schedule_batch.py` fix the count stays 0,
  forced retraction included (`SGLANG_TEST_RETRACT=1`; unpatched sglang: 4
  extends, 1792 tokens).
- **Determinism.** `PICOKV_DET=1` (off by default, see the knob table) makes a run repeat bitwise when the same requests arrive in the same batches. It is not batch invariance: the slide epoch (one 64-step clock per decode batch; an in-place slot change keeps it, a rebuild restarts it, so a row's tokens can slide at a different step), the tail's tile split and, when set, `PICOKV_DENSE_LOCAL` depend on the batch, so a request can decode differently next to other requests. SGLang's `--enable-deterministic-inference` (batch-invariant kernels) rejects picokv; the flag stays unsupported. Off, the tail's float atomics make the last bits vary, and greedy text diverges at near-ties (22-24 of 40 repeated serving waves identical; flashinfer 40 of 40).
- **Fallback and internal limits.** If FlashInfer's ragged prefill runs out
  of memory (that extend) or fails for good (the server), prefill attention
  falls back to torch SDPA per request. Its memory is not reserved from the
  KV budget: a stage continuation builds a boolean mask of up to
  `1024 x (prefix + chunk)` per query block, and SDPA with an explicit mask
  may materialize the scores of that block (not measured; the fallback was
  never reached in the runs above). The tail schedule's padding entries use
  tile index `2^20` (an early exit for any compressed length below 2^29
  tokens, the kernels' int32 range). `PICOKV_TAIL_QUANT_BLOCKS` counts
  schedule entries x kv heads, not launched blocks (with
  `PICOKV_TC_SCHED_SPLIT` an entry launches 2), so the gate is a speed
  heuristic only; either branch gives the same codes.
- **SGLang control paths the backend does not follow.** `/flush_cache` resets
  SGLang's tree and pools (with `PICOKV_RADIX=host` the tree frees its host
  K/V) but not the backend's kept decode batch, which stays allocated until a
  later batch replaces it. Weight updates (`update_weights_from_*`) do not
  refit the offline projections, which were fit to the original weights. The
  KV usage metrics report 0 GB (the no-op pool's size). Parallel sampling
  (`n > 1`) without the radix cache pays SGLang's extra prefix prefill, whose
  result nothing reuses.
- **Unverified:** LoRA is accepted by the guard but untested with picokv.

Open: radix v2 (one shared, paged copy of a hit prefix in the GPU stage
buffers instead of one per row, with a cap: today 32 hits on a 30k prefix
need about 7.9 GB of stage buffers; caching decode tokens; requests aborted
while waiting keep their radix path protected until collected; a hit whose
prefix outgrew `PICOKV_RADIX_HOT_GB` still streams the rest over PCIe), the
high-concurrency short-prompt gap (0.91-0.92x flashinfer at 160 concurrent:
the compressed-path kernels per step and the rebuild + re-capture on a graph
bucket change), prefill without CUDA graphs (small extends) and the remaining
small-batch gap. Levers measured or estimated for it, not done: folding the
value projection (and the accumulator zeroing) into the tail's last block
(about 2-3 us per layer); a full decode graph instead of the breakable one
(GPU idle per step at 1 row is 0.40 ms vs flashinfer's 0.20 ms, so at most
about 2.5%; an earlier measurement with older kernels gave <= 1%, and every
in-place slot change would then re-capture);
FA3 with a forced split count for the dense window at 1-4 rows (8.16 -> 7.85
us per layer at 1 row, about 0.15% per step, not bitwise: rejected).

## Scripts

Every script takes the backend name (`picokv` or `flashinfer`) so both can be
compared with identical inputs. Run them from `<WS>`, with
`PYTHONPATH=<WS>/PiCoKV` (and `<WS>/sglang/python` unless sglang is installed).
Model dirs default to `llama31_8b_rand` (timing) / `llama31_8b_instruct`
(quality) relative to the cwd. Override with `BENCH_MODEL` / `NIAH_MODEL`.

- Speed: `run_itl.sh` (see its header) drives `scripts/bench_itl.py
  <backend> <plen> <batch> <gen> [reps] [memf]`.
- Smoke test: `scripts/niah_engine_check.py` runs an 8k haystack with two
  needles through `sgl.Engine` on real weights. Set `NIAH_BATCH=4` for a ragged
  batch and `BENCH_GRAPHS=1 BENCH_DECODE_GRAPH=breakable` for graph mode. Every
  output must print `needles found: PASS`.
- Accuracy (`scripts/eval/`). Each benchmark is split into a build/score half,
  which runs in the PiCoKV repo env (transformers 4.53 + lm_eval), and an infer
  half, which runs in the sglang env:
  - LongBench: `longbench_io.py --mode build` -> `longbench_infer.py --backend
    picokv --graph bcg` -> `longbench_io.py --mode score`.
    `ib_io.py` (InfiniteBench) and `lbv2_io.py` (LongBench-v2) reuse
    `longbench_infer.py`.
  - Datasets are read from `$PICOKV_EVAL_DATA/{longbench_data,ib_data,longbench_v2_data}`.
  - RULER: `ruler_io.py --mode capture` -> `ruler_infer.py` ->
    `ruler_io.py --mode score`.
  - NIAH grid: `build_niah.py <model> <records.jsonl>`, then generate with
    `longbench_infer.py --records`, then `score_niah.py name=tag` (reads
    `$NIAH_DIR/preds_<tag>.jsonl`).
- Serving behaviour (`scripts/serving/`), the measurements behind the
  Limitations section:
  - `bench_serve.py`: fixed batch, churn (requests finishing mid-decode), and
    poison (one short prompt in a long batch).
  - `bench_ttft.py`: TTFT.
  - `bench_capacity.py <backend> <plen> <n> [gen] [memf]`: how many long
    requests decode at once (the memory capacity table).
  - `bench_longgen.py`: long-generation ITL per bucket.
  - `probe_quality.py <backend> radix|chunk`: NIAH with radix on, or with
    small chunks.
  - `bench_prefix.py <pkrx|pk|firx|fi> [reps]`: TTFT with a cached shared
    prefix and on misses (run `pkrx` with `PICOKV_RADIX=host`).
  - `test_radix_cpu.py`: the picokv radix tree's host-slot bookkeeping
    (insert / split / evict / drop under pressure), CPU only.
  - `http_e2e.py <port> <out.json> [radix]`: end-to-end check of a running
    server (any attention backend) over HTTP, 23 checks (see Results); pass
    `radix` when the server keeps the radix cache, so the repeated long prompt
    must hit it. Prints `N/23 passed`.

Unit / kernel tests live in `picokv/`:
- `test_picokv_sglang_core.py`, `test_picokv_sglang_step23.py`: backend
  core vs reference.
- `test_append_equality.py`, `test_fused_append.py`: incremental append.
- `verify_tc_tail.py`: tensor-core tail vs popc, bitwise.
- `bench_tail_split.py`, `bench_window_attn.py`, `bench_kern_long.py`: kernel
  timing.

All of them need `PYTHONPATH=<WS>/PiCoKV` and the built extensions.
