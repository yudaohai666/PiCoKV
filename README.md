# QuanPC KV Cache Experiments

This workspace contains QuanPC quantized KV-cache attention code for
long-context decoding. The implementation lives under `quantpc/` and supports
both online projection learning and fully offline projections, whose whitening,
encoder and decoder are folded into a single stored pair.

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

## Core package: `quantpc.quanpc_standalone`

Projections are learned **online** from the current prefill K/V (block-32
CholeskyQR / Householder subspace quantization with a ridge-regularized
decoder). There is no separate `learn_projection` step and no fixed-projection,
refit, torch-decode, block-16, or async-prepare path. Decode requires bf16 CUDA
and GQA=4 (`QH/KVH == 4`, `D == 128`), code bits in
`{128,192,256,320,384,448,512}`, and `exp_bits` in `{1,2,4,8}`.

See `quantpc/README.md` for the full package reference.

### Minimal end-to-end example

```python
import torch
from quantpc.quanpc_standalone import (
    StandaloneConfig,
    prepare_kv_cache,
    decode_attention,
)

device, dtype = "cuda", torch.bfloat16
B, KVH, QH, L, D = 1, 8, 32, 100000, 128
full_key = torch.randn(B, KVH, L, D, device=device, dtype=dtype)
full_value = torch.randn_like(full_key)

# Projections are learned online inside prepare_kv_cache; no projection args.
config = StandaloneConfig()  # k320/v256, sink 64, local 256, eas 0.01, zerorope 20
cache = prepare_kv_cache(full_key, full_value, **config.prepare_kwargs())

query = torch.randn(B, QH, 1, D, device=device, dtype=dtype)
out = decode_attention(query, cache, **config.decode_kwargs())
print(out.shape)  # [B, QH, 1, D]
```
Append the current decode token's K/V to the dense retained cache so later
steps can attend to it:

```python
decode_key = torch.randn(B, KVH, 1, D, device=device, dtype=dtype)
decode_value = torch.randn_like(decode_key)
out = decode_attention(
    query,
    cache,
    decode_key=decode_key,
    decode_value=decode_value,
    update_cache=True,
    **config.decode_kwargs(),
)
```

The compressed global cache stays static during decode; generated tokens are
appended to `cache.dense_key` / `cache.dense_value` but not re-quantized into
the packed global cache.

### Hugging Face Llama

```python
from quantpc.quanpc_standalone import StandaloneConfig
from quantpc.quanpc_standalone.transformers import (
    StandaloneLlamaCache,
    modify_llama_for_standalone,
)

modify_llama_for_standalone(model, mlp_chunk_size=8192)
cache = StandaloneLlamaCache(StandaloneConfig())
out = model.generate(
    **inputs,
    past_key_values=cache,
    do_sample=False,
    max_new_tokens=30,
)
```

Prompt attention stays exact SDPA; single-token generation routes through the
CUDA decode path.

## Layout

Active core (`quantpc/`):

- `quantpc/quanpc_standalone/`: online-projection prepare/decode API + HF Llama adapter.
- `quantpc/quanpc_cuda_extension/`: single-token decode CUDA kernels.
- `quantpc/quanpc_prefill_cuda_extension/`: cache-packing / gather CUDA kernels.
- `quantpc/README.md`, `quantpc/test_quanpc_standalone_online.py`
  (+ `needle.py`, `PaulGrahamEssays/` for the self-contained corpus).

Benchmarks (`benchmarks/`):
`benchmarks/common.py::load_quantpc_model(method="quantpc")` patches the model
via `modify_llama_for_standalone` and injects a fresh `StandaloneLlamaCache`
per call. The Efficiency / Needle / Longbench / Ruler runners all go through it.

## Common Checks

Configure the environment (repo root on `PYTHONPATH`, single GPU):

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
```

Standalone regression test:

```bash
python quantpc/test_quanpc_standalone_online.py
```

NIAH. `benchmarks/Needle/run_bin.py` is the shared entry point for every
method (quantpc / kivi / oscar / dense baselines) and writes the flat format
`show_image_paper.py` plots:

```bash
python benchmarks/Needle/run_bin.py --method quantpc \
  --k-bits 320 --v-bits 256 --local-size 256 --zerorope 20 \
  --lengths 64000,110000 --depths 50 \
  --output results/niah/Llama-3.1-8B-Instruct_online_k320_v256_local256_nowhiten/results.json
```

`quantpc/run_standalone_niah.py` runs the same sweep for quantpc alone but also
records per-cell prefill timing and the full config, which is where the speed
numbers come from:

```bash
python quantpc/run_standalone_niah.py \
  --k-bits 320 --v-bits 256 --local-size 256 \
  --lengths 64000,110000 --depths 50 \
  --output-json results/niah/timed_runs/online_k320_v256_local256_nowhiten.json
```

See `results/niah/README.md` for what each output holds.

Efficiency benchmark (quantpc vs dense/KIVI baselines):

```bash
python benchmarks/Efficiency/run_efficiency.py \
  --method quantpc --cuda-v1-exp8 --k-bits 256 --v-bits 256
```

> The torch reference backend, `refit_decoder`, and tunable
> `projected_query_bits` were removed with the standalone refactor.
> `benchmarks/common.py` now requires `--cuda-v1-exp8` for `--method quantpc`.

## Baseline Alignment Notes

1. **`apply_chat_template` tokenization**: changed from
   `tokenize=False` (string) + `tokenizer([text], return_tensors="pt")` to
   `tokenize=True, return_tensors="pt", return_dict=False` to avoid a
   potential double-BOS token introduced by the two-step path.

2. **`max_new_tokens`**: increased from 30 to 50 to match the reference
   baseline and avoid response truncation on borderline cases.
