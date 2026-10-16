# PiCoKV attention backend for SGLang

This directory adds `picokv` as an SGLang attention backend, so PiCoKV's
compressed KV cache can be served through `sgl.Engine` / `sglang.launch_server`
with continuous batching and decode CUDA graphs.

- Prefill: exact dense attention (FlashInfer ragged), then the prompt K/V is
  packed into PiCoKV codes (320 bits per head-token by default) with **offline
  projections** and `eas=0`. One fused prepare kernel per side; no
  per-request online fitting.
- Decode: a dense bf16 window is attended exactly (FlashInfer paged decode, or
  a fused window kernel at small batch). The window holds the first 64 tokens
  (sink), the last 256 prompt tokens, and every token generated so far. The rest
  of the prompt is compressed and goes through the PiCoKV tail kernel, which runs
  the `m16n8k256` tensor-core path by default (`PICOKV_NO_TC=1` uses the popc
  path). The per-layer ops are replayed through CUDA graphs when batch membership
  is stable.
- The SGLang KV pool is replaced by a no-op pool. PiCoKV owns its memory, and
  the pool sizing is told the compressed cell size, so far more tokens fit:
  about 6.4x the dense ceiling.

Hard requirements of the decode kernel: bf16, GQA = 4, head_dim = 128 (for
example Llama-3.1-8B). `tp_size=1` only.

## Contents

| Path | What |
|---|---|
| `picokv_backend.py` | The backend (`python/sglang/srt/layers/attention/picokv_backend.py` in the SGLang tree) |
| `sglang_picokv.patch` | Changes to 6 SGLang files, against upstream `sgl-project/sglang@567d5925` |
| `apply.sh` | Install both into an SGLang checkout |
| `export_from_sglang.sh` | Maintainer helper: regenerate the patch and backend copy from a working checkout |
| `run_itl.sh`, `scripts/bench_itl.py` | Decode inter-token-latency benchmark (picokv vs flashinfer) |
| `scripts/niah_engine_check.py` | Quick real-weight needle check through the engine |
| `scripts/eval/` | LongBench / RULER / InfiniteBench / LongBench-v2 / NIAH-grid drivers |
| `scripts/serving/` | Serving-behaviour probes (throughput, TTFT, long generation, radix/chunk quality) |

What the patch changes in SGLang:

- `attention_registry.py`: registers backend `picokv`.
- `kv_cache_configurator.py`: uses a no-op MHA token pool for `picokv`.
- `pool_configurator.py`: sizes the pool by the compressed cell
  (`2 * PICOKV_KV_BITS / 8` bytes per head-token per layer).
- `forward_batch_info.py`, `decode_cuda_graph_runner.py`: carry
  `req_pool_indices_cpu`, so the backend reads request ids without a
  device-to-host sync.
- `breakable_cuda_graph_backend.py`: lets breakable graphs return
  dataclass outputs.

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
    dtype="bfloat16", tp_size=1,
    disable_radix_cache=True,              # required, see Limitations
    chunked_prefill_size=1 << 20,          # >= longest prompt: prefill in one chunk
    cuda_graph_backend_decode="breakable", # decode graph mode for picokv
    disable_prefill_cuda_graph=True,
    mem_fraction_static=0.70,
)
out = eng.generate(prompt, {"temperature": 0, "max_new_tokens": 256})
```

The same fields work as `sglang.launch_server` flags (`--attention-backend picokv
--disable-radix-cache --chunked-prefill-size 1048576
--cuda-graph-backend-decode breakable --disable-prefill-cuda-graph`). All numbers
below were measured through `sgl.Engine`.

**Check the offline projection is actually used.** On init the backend logs one
line containing `mode=offline (<path>)`. If `PICOKV_OFFLINE_PROJ` is unset, it
looks for `PiCoKV/offline_projection/projections/<model_dir_name>_k320_v320.pt`.
If that file is missing, it **silently falls back to online projections** (a
single `no offline projection at ...` line), and long-context retrieval quality
drops (RULER, passkey). Either set `PICOKV_OFFLINE_PROJ` or name the model
directory `Llama-3.1-8B-Instruct`. Models with random weights are fine for
timing only.

Use `cuda_graph_backend_decode="breakable"`. The `full` decode graph is not
supported (`NotImplementedError`), and a full graph would buy <= 1% anyway
because the backend already graphs its own per-layer ops.

## Environment knobs

All are read at backend init. The defaults are the tested serving
configuration.

| Variable | Default | Meaning |
|---|---|---|
| `PICOKV_OFFLINE_PROJ` | (auto) | Offline projection `.pt`; see above |
| `PICOKV_ONLINE` | `0` | `1` forces online projection fitting (A/B only) |
| `PICOKV_EAS` | `0.0` | Fraction of error-selected prompt tokens kept dense. `0` means the window is exactly sink + local. Offline + `eas=0` is the supported path (incremental chunk append needs it) |
| `PICOKV_EXP_BITS` | `4` | In-kernel quantization bits of attention weights (1/2/4/8) |
| `PICOKV_KV_BITS` | `320` | Code bits per head-token, used for pool sizing (must match the projection) |
| `PICOKV_KV_DEDUP` | `1` | Free the per-request copy once a layer's batched copy exists |
| `PICOKV_PREFILL` | `fi` | Prefill dense attention: `fi` = FlashInfer ragged, `sdpa` = torch SDPA fallback |
| `PICOKV_WINDOW_KERNEL` | `auto` | Fused dense-window kernel for small batches (`<= PICOKV_WINDOW_MAX_BS`, default 8); `0`/`1` force it off/on |
| `PICOKV_GRAPHS` | `1` | Per-layer CUDA-graph replay (`PICOKV_GRAPH_WARMUP=3`, `PICOKV_GRAPH_FORK=1`) |
| `PICOKV_RIDS_CPU`, `PICOKV_RIDS_PER_STEP`, `PICOKV_FUSED_APPEND`, `PICOKV_ENG_DIRECT_OUT` | `1` | Host-overhead fast paths; set `0` only to debug |
| `PICOKV_NO_TC` | unset | `1` uses the popc tail instead of the tensor-core tail |
| `PICOKV_DECPROF` | unset | Per-step decode profiling; it perturbs timings, so never set it for A/B runs |
| `PICOKV_PREBUILT_DIR` | `PiCoKV/build/picokv_extensions` | Where the built extensions are loaded from |

Kernel-tuning switches (`PICOKV_TC_BT`, `PICOKV_TAIL_BT1`, `PICOKV_WIN_CH`,
`PICOKV_NO_KBSPEC`, `PICOKV_SCORE_BMMA`) are for experiments. Leave them
unset.

## Results (Llama-3.1-8B-Instruct, one sm_90 80 GB GPU)

Decode inter-token latency, in ms/step (median, idle GPU, 128 generated
tokens). picokv uses the breakable decode graph. flashinfer uses its best mode,
the full decode graph.

| Prompt | Batch | picokv | flashinfer |
|---|---|---|---|
| 8k | 1 | 6.91 | 6.61 |
| 8k | 16 | 9.25 | 11.88 |
| 32k | 8 | 10.35 | 17.06 |
| 120k | 1 | 8.34 | 11.21 |
| 120k | 16 | 27.92 | does not fit (max 2 concurrent: 16.34) |

At 120k, peak decode throughput is 573 tok/s for picokv versus 122 tok/s for
flashinfer (4.7x). To reproduce:
`ITL_REPS=2 ITL_WARM_LEN=1024 ./run_itl.sh <gpu> pk_8k_b16 picokv 8000 16 128 bcg`
(and `flashinfer ... full`).

Quality (greedy decoding, offline projections, `eas=0`):

| Benchmark | flashinfer | picokv |
|---|---|---|
| LongBench, all 3750 samples | 49.62 | 49.33-49.35 |
| Needle-in-a-haystack, 160 cells (lengths x depths up to 128k) | 98.12 | 98.30 |
| RULER 4k / 16k / 64k | 1.000 / 1.000 / 0.991 | 1.000 / 1.000 / 0.981 |

## Limitations / known issues

- **Radix cache must be off.** A prefix hit skips the prefill that builds the
  compressed cache. With radix on, NIAH drops to about 18.
- **Prefill must be a single chunk** (`chunked_prefill_size >=` longest prompt).
  At SGLang's default 8192, 32k-NIAH drops to about 15.5. The backend prints a
  warning on the first continuation chunk.
- **Generated tokens are never compressed.** They stay in the dense window. With
  long outputs, decode cost and memory grow like dense attention:
  - 8x8k prompts generating 16k tokens: the last buckets run at 19.9 vs 13.8
    ms/step for flashinfer.
  - The dense window is not counted in SGLang's pool accounting (about 28 GB
    real vs about 4 GB counted in that run).
  - Keep `mem_fraction_static` around 0.70, and watch memory at high concurrency
    with long contexts.
- **Short prompts are slower than flashinfer.** Batches of short prompts with
  long generations run at 17-18 vs 7-9 ms/step for flashinfer, and TTFT is
  25-56% higher for short-prompt batches. One short request in a batch can slow
  the whole batch by up to 3.6x.
- **Batch-membership changes** (request join or finish) rebuild the batched
  state and re-capture the per-layer graphs: about 40-50 ms per change.
- `tp_size=1` only. No speculative decoding, no mixed-chunk prefill+decode.
  Supported models are bf16 with GQA 4 and head_dim 128.

Planned next steps: a guard that rejects or falls back on unsupported settings
(radix, small chunk), compression of generated tokens (sliding re-compress),
short-prompt batching, and correct memory accounting.

## Scripts

Every script takes the backend name (`picokv` or `flashinfer`) so both can be
compared with identical inputs. Run them from `<WS>`, with
`PYTHONPATH=<WS>/PiCoKV` (and `<WS>/sglang/python` unless sglang is installed).
Model dirs default to `llama31_8b_rand` (timing) / `llama31_8b_instruct`
(quality) relative to the cwd. Override with `BENCH_MODEL` / `NIAH_MODEL`.

- Speed: `run_itl.sh` (see its header) drives `scripts/bench_itl.py
  <backend> <plen> <batch> <gen> [reps] [memf]`.
- Smoke test: `scripts/niah_engine_check.py` runs an 8k haystack with two
  needles through `sgl.Engine` on real weights.
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
  - `bench_longgen.py`: long-generation ITL per bucket.
  - `probe_quality.py <backend> radix|chunk`: NIAH with radix on, or with
    small chunks.

Unit / kernel tests live in `picokv/`:
- `test_picokv_sglang_core.py`, `test_picokv_sglang_step23.py`: backend
  core vs reference.
- `test_append_equality.py`, `test_fused_append.py`: incremental append.
- `verify_tc_tail.py`: tensor-core tail vs popc, bitwise.
- `bench_tail_split.py`, `bench_window_attn.py`, `bench_kern_long.py`: kernel
  timing.

All of them need `PYTHONPATH=<WS>/PiCoKV` and the built extensions.
