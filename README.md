# QuanPC KV Cache Experiments

This workspace contains QuanPC-style quantized KV-cache attention code for
long-context decoding. The cleaned Python-facing API is intentionally small;
experiments and legacy model-patching code remain available for reproduction.

## Environment Setup

The project uses [uv](https://docs.astral.sh/uv/) for dependency management
(`pyproject.toml` + `uv.lock`, Python `>=3.12`). Create the environment with:

```bash
uv sync
```

This installs all pinned dependencies (torch 2.9.1, transformers 4.53.0,
flash-attn, lm-eval, etc.) into `.venv/`. Run commands via `uv run <cmd>` or
activate the venv with `source .venv/bin/activate`.

CUDA is required for the fast decode/prefill kernels; the CUDA extensions under
`quantpc/quanpc_cuda_extension/` and `quantpc/quanpc_prefill_cuda_extension/`
build on first use.

## Public API

Use `quantpc/quanpc_api.py` for new code. It exposes three operations:

- `learn_projection(full_states, bits, ...)`: learn one K or V binary projection.
- `prepare_kv_cache(full_key, full_value, key_projection, value_projection, ...)`:
  quantize with the learned projections, select EAS tokens, build dense retained
  K/V, and pack global 1-bit K/V codes.
- `decode_attention(query, cache, ...)`: run one single-token decode attention
  step over dense retained K/V plus packed QuanPC global K/V.

The lower-level `quantpc/quanpc.py`, `quantpc/quanpc_cuda_extension/`, and
`quantpc/quanpc_prefill_cuda_extension/` modules are implementation details and
experiment utilities.

## Using `quantpc.quanpc_api`

`quantpc/quanpc_api.py` is the recommended entry point for new integrations.  The
expected tensor layout follows Llama-style GQA:

- `full_key`, `full_value`: `[B, KVH, L, D]`, usually bf16 CUDA tensors from
  prefill.
- `query`: `[B, QH, 1, D]`, a single-token decode query.
- Fast CUDA decode is optimized for `QH / KVH == 4`, `D == 128`, bf16, and
  code dimensions in `{128, 192, 256, 320, 384, 448, 512}`.
- `exp_bits` can be `1`, `2`, `4`, or `8`; current benchmarks usually use
  `exp_bits=4`.

Minimal end-to-end example:

```python
import torch
from quantpc.quanpc_api import learn_projection, prepare_kv_cache, decode_attention

device = "cuda"
dtype = torch.bfloat16

B, KVH, QH, L, D = 1, 8, 32, 100000, 128
full_key = torch.randn(B, KVH, L, D, device=device, dtype=dtype).contiguous()
full_value = torch.randn_like(full_key)

# 1. Learn projections. This can be done offline or during prefill.
key_proj = learn_projection(
    full_key,
    bits=256,
    sink_size=64,
    local_size=1024,
    zerorope=20,
    apply_zerorope=True,
)
value_proj = learn_projection(
    full_value,
    bits=256,
    sink_size=64,
    local_size=1024,
)

# 2. Quantize and pack the prefill KV cache. This also builds the retained
# dense cache: sink + EAS tokens + local window.
cache = prepare_kv_cache(
    full_key,
    full_value,
    key_proj,
    value_proj,
    sink_size=64,
    local_size=1024,
    eas=0.01,
    zerorope=20,
    exp_bits=4,
)

# 3. Run one decode attention step.
query = torch.randn(B, QH, 1, D, device=device, dtype=dtype).contiguous()
out = decode_attention(
    query,
    cache,
    use_pruning=True,
    prune_mode=3,
    log_eps=-10.0,
)
print(out.shape)  # [B, QH, 1, D]
```

To include the current decode token's K/V in the dense retained cache, pass
`decode_key` and `decode_value`:

```python
decode_key = torch.randn(B, KVH, 1, D, device=device, dtype=dtype).contiguous()
decode_value = torch.randn_like(decode_key)

out = decode_attention(
    query,
    cache,
    decode_key=decode_key,
    decode_value=decode_value,
    update_cache=True,
)
```

`update_cache=True` appends the new K/V to `cache.dense_key` and
`cache.dense_value`, so later decode calls can attend to it.  The current API
does not yet quantize newly generated tokens into the packed global cache or
maintain a sliding local window; long streaming generation still needs that
policy layer on top.

Useful cache metadata:

```python
print(cache.dense_len, cache.global_len)
print(cache.key_bits, cache.value_bits)
print(cache.dense_key.shape, cache.packed_key.shape, cache.packed_value.shape)
```

## Layout

The core algorithm now lives in the `quantpc/` package (benchmark-agnostic);
benchmark runners and corpora stay at the repo root (bench side).

Core package (`quantpc/`):

- `quantpc/quanpc_api.py`: cleaned public API with the three entry points above.
- `quantpc/quanpc.py`: legacy/internal model patching, torch fallback, and
  compatibility cache classes used by existing NIAH runners.
- `quantpc/quanpc_torch.py`: torch-only reference implementation for algorithm checks.
- `quantpc/kernel.py`: binary-projection CUDA extension utilities.
- `quantpc/quanpc_cuda_extension/`: decode CUDA extension, correctness tests, and
  historical benchmark/profiling scripts.
- `quantpc/quanpc_prefill_cuda_extension/`: prefill-side CUDA extension for projection
  learning, projection quantization/packing, EAS selection, and dense gather.
- `quantpc/extension/`: original/reference CUDA extension files.
- `quantpc/test_quanpc_api.py`: public API smoke test.

Bench side (`benchmarks/Needle/`, the NIAH benchmark):

- `benchmarks/Needle/run_bin_v4.py`: NIAH experiment runner. Backend is chosen
  by `--cuda-v1-exp8`: with the flag it uses the CUDA backend (`quantpc.quanpc`),
  without it the torch-only reference backend (`quantpc.quanpc_torch`).
- `benchmarks/Needle/search_quanpc_torch_params.py`: torch-only NIAH parameter search.
- `benchmarks/Needle/needle.py`, `benchmarks/Needle/PaulGrahamEssays/`: NIAH prompt/evaluation utilities and corpus.
- `benchmarks/Needle/run_niah.sh`: NIAH sweep entry point (sets `PYTHONPATH` to the repo root so runners can `import quantpc`).
- `results/`, `logs/`, `backups/`, `archive/`: experiment outputs and snapshots.

New benchmarks go in a sibling directory under `benchmarks/` and reuse the
`quantpc` package without touching it.

## Common Checks

Syntax check:

```bash
python -m py_compile \
  quantpc/quanpc.py quantpc/quanpc_api.py quantpc/quanpc_torch.py \
  benchmarks/Needle/run_bin_v4.py \
  quantpc/quanpc_cuda_extension/quanpc_cuda.py \
  quantpc/quanpc_prefill_cuda_extension/quanpc_prefill_cuda.py
```

Public API smoke test:

```bash
CUDA_VISIBLE_DEVICES=4  \
python quantpc/test_quanpc_api.py
```

Prefill CUDA correctness:

```bash
CUDA_VISIBLE_DEVICES=4  \
python quantpc/quanpc_prefill_cuda_extension/test_projection_learning.py --dtypes bfloat16

CUDA_VISIBLE_DEVICES=4  \
python quantpc/quanpc_prefill_cuda_extension/test_prefill_correctness.py --dtypes bfloat16
```

Decode CUDA correctness:

```bash
CUDA_VISIBLE_DEVICES=4  \
python quantpc/quanpc_cuda_extension/test_full_decode_correctness.py \
  --batch-sizes 1 --gqa 4 --exp-bits 4 --dense-len 128 --global-len 512 \
  --atol 0.12 --rtol 0.12
```

NIAH CUDA runner example (run from the repo root so `import quantpc` resolves):

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. \
python benchmarks/Needle/run_bin_v4.py \
  --lengths 64000 --depths 50 --output results/niah/results_cuda.json \
  --k-bits 256 --v-bits 256 --projected-query-bits 8 \
  --exp-bits 4 --eas 0.01 --zerorope 20 --cuda-v1-exp8
```

Or drive the full sweep via the entry-point script (it sets `PYTHONPATH` itself):

```bash
bash benchmarks/Needle/run_niah.sh
```



## Baseline Alignment Notes

1. **`apply_chat_template` tokenization**: changed from
   `tokenize=False` (string) + `tokenizer([text], return_tensors="pt")` to
   `tokenize=True, return_tensors="pt", return_dict=False` to avoid a
   potential double-BOS token introduced by the two-step path.

2. **`max_new_tokens`**: increased from 30 to 50 to match the reference
   baseline and avoid response truncation on borderline cases.
