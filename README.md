# PiCoKV KV Cache Experiments

This workspace contains PiCoKV quantized KV-cache attention code for
long-context decoding. The implementation lives under `picokv/` and supports
both online projection learning and fully offline projections, whose encoder and
decoder are folded into a single stored pair.

## Environment Setup

The project uses [uv](https://docs.astral.sh/uv/) for dependency management
(`pyproject.toml` + `uv.lock`, Python `>=3.12`).

### Quick start (existing environment)

A prebuilt environment already exists at `.venv/` (Python 3.12.3). Just activate
it:

```bash
source .venv/bin/activate
```

### Install from scratch

1. Install `uv` (skip if already available — check with `uv --version`):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Create the virtualenv and install all pinned dependencies into `.venv/`:

   ```bash
   uv sync
   ```

   This resolves from `uv.lock` and installs torch 2.6.0 (cu124),
   transformers 4.53.2, accelerate 1.14.0, datasets 5.0.1, and
   flash-attn 2.7.4.post1. `torch`/`torchaudio`/`torchvision` come from the
   `pytorch-cu124` index configured in `pyproject.toml`, so you need a working
   **CUDA 12.4** toolchain (`nvcc`, a compatible `gcc`) on `PATH` — flash-attn
   and the local CUDA extensions compile against it.

3. The Ruler and InfiniteBench runners additionally need `lm-eval`, which is
   **not** part of `uv sync`. Install it separately when running those:

   ```bash
   uv pip install lm-eval==0.4.12
   ```

4. Activate the environment (or prefix commands with `uv run`):

   ```bash
   source .venv/bin/activate
   ```

CUDA 12.4 is required for the fast decode/prefill kernels; the CUDA extensions
under `picokv/picokv_cuda_extension/` and
`picokv/picokv_prefill_cuda_extension/` build on first use.

### Key package install commands

If you are not using `uv sync`, install the important pinned packages directly:

```bash
uv pip install \
  torch==2.6.0+cu124 torchaudio==2.6.0+cu124 torchvision==0.21.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124

uv pip install \
  transformers==4.53.2 \
  flash-attn==2.7.4.post1 \
  lm-eval==0.4.12 \
  accelerate==1.14.0 \
  datasets==5.0.1 \
  huggingface-hub==0.36.2 \
  tokenizers==0.21.4 \
  numpy==2.5.2
```

Plain `pip install ...` works the same if you are inside an activated venv.

### Baseline method setup (kivi / oscar / polar)

The `picokv` and dense baselines (`flash_attention_2` / `sdpa` / `eager`) work
out of the box. The KV-quant baselines need their CUDA kernels built into the
same `.venv/` first (each script activates `.venv` and builds `--no-build-isolation`
against the pinned torch 2.6.0+cu124, so it will not downgrade torch/transformers):

- **kivi** and **polar** — build the KIVI `kivi_gemv` kernel and verify:

  ```bash
  bash src/KIVI/setup_env_verify.sh
  ```

- **oscar** — build the `oscar_cuda` extension in place:

  ```bash
  bash src/OScaR-KV-Quant/build_oscar_cuda.sh
  ```

Both require a CUDA 12.4 toolchain on `PATH` (`nvcc`, `CUDA_HOME`, e.g.
`/usr/local/cuda`). Skip these if you only run `picokv`.

## Core package: `picokv.picokv_standalone`

PiCoKV supports **two projection paths**, both using block-32 CholeskyQR /
Householder subspace quantization with a ridge-regularized decoder:

- **Online** (`StandaloneConfig()` defaults, no projection args): projections are
  fit from the current prefill K/V inside `prepare_kv_cache`.
- **Offline**: load a stored `.pt` whose encoder + decoder are folded
  into a single pair (`load_offline_projections`, `transformers.py:51`), then pass
  them into `prepare_kv_cache` (`prepare.py:190-241`). A stored projection is used
  whole: encoder and decoder always come from the same `.pt`. Enable it with
  `USE_OFFLINE=1`.

Decode requires bf16 CUDA and GQA=4 (`QH/KVH == 4`, `D == 128`), code bits in
`{128,192,256,320,384,448,512}`, and `exp_bits` in `{1,2,4,8}` (default 4).

See `picokv/README.md` for the full package reference.

### Minimal end-to-end example

```python
import torch
from picokv.picokv_standalone import (
    StandaloneConfig,
    prepare_kv_cache,
    decode_attention,
)

device, dtype = "cuda", torch.bfloat16
B, KVH, QH, L, D = 1, 8, 32, 100000, 128
full_key = torch.randn(B, KVH, L, D, device=device, dtype=dtype)
full_value = torch.randn_like(full_key)

# Projections are learned online inside prepare_kv_cache; no projection args.
config = StandaloneConfig()  # k320/v320, sink 64, local 256, eas 0.01, zerorope 20
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
from picokv.picokv_standalone import StandaloneConfig
from picokv.picokv_standalone.transformers import (
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

## Serving with SGLang

`sglang_integration/` adds `picokv` as an SGLang attention backend (continuous
batching, decode CUDA graphs, offline projections with `eas=0`). It ships as a
patch against upstream `sgl-project/sglang@567d5925` plus the backend file, and
needs its own torch 2.13 environment. See
[`sglang_integration/README.md`](sglang_integration/README.md) for setup,
required server args, measured speed and quality, and known limitations.

## Layout

Active core (`picokv/`):

- `picokv/picokv_standalone/`: online/offline projection prepare/decode API + HF Llama adapter.
- `picokv/picokv_cuda_extension/`: single-token decode CUDA kernels.
- `picokv/picokv_prefill_cuda_extension/`: cache-packing / gather CUDA kernels.
- `picokv/README.md`, `picokv/test_picokv_standalone_online.py`
  (+ `needle.py`, `PaulGrahamEssays/` for the self-contained corpus).

Offline projections (`offline_projection/`):

- `create_offline_projections.py` (+ `.sh`): builds the folded encoder +
  decoder `.pt` pairs from `data/` (proof-pile / pg19 / codeparrot).
- `projections/`: stored PTs, named `{model}_k{K}_v{V}.pt`
  (on disk: `k320_v320`, `k320_v256`, `k256_v256`).

Baselines (`src/`): `KIVI/`, `OScaR-KV-Quant/`, `PolarQuant/`, plus `lmeval.py`.

Benchmarks (`benchmarks/`):
`benchmarks/common.py::load_model(method=...)` builds the model for each method.
Supported methods: `picokv`, `kivi`, `oscar`, `polar`, and the dense baselines
`flash_attention_2` / `sdpa` / `eager`. For `picokv` it patches the model via
`modify_llama_for_standalone` and injects a fresh `StandaloneLlamaCache` per call.
The Efficiency / Needle / Longbench / Ruler / InfiniteBench runners all go
through it.

## Common Checks

Configure the environment (repo root on `PYTHONPATH`, single GPU):

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
```

Standalone regression test:

```bash
python picokv/test_picokv_standalone_online.py
```

NIAH. `benchmarks/Needle/run_bin.py` is the shared entry point for every
method (picokv / kivi / oscar / polar / dense baselines) and writes the flat
format `show_image_paper.py` plots:

```bash
python benchmarks/Needle/run_bin.py --method picokv \
  --k-bits 320 --v-bits 320 --local-size 256 --zerorope 20 \
  --lengths 64000,110000 --depths 50 \
  --output results/niah/Llama-3.1-8B-Instruct_online_k320_v320/results.json
```

`picokv/run_standalone_niah.py` runs the same sweep for picokv alone but also
records per-cell prefill timing and the full config, which is where the speed
numbers come from:

```bash
python picokv/run_standalone_niah.py \
  --k-bits 320 --v-bits 320 --local-size 256 \
  --lengths 64000,110000 --depths 50 \
  --output-json results/niah/timed_runs/online_k320_v320.json
```

See `results/niah/` for the stored outputs of these runs.

Efficiency benchmark (picokv vs dense/KIVI baselines):

```bash
python benchmarks/Efficiency/run_efficiency.py \
  --method picokv --k-bits 320 --v-bits 320 --zerorope 20
```

Ruler / InfiniteBench (require `lm-eval`, see Environment Setup). Prefer the
`.sh` launchers, which auto-name the output tag from the config:

```bash
bash benchmarks/Ruler/run_ruler.sh
bash benchmarks/InfiniteBench/run_infinitebench.sh
```

Each launcher exposes a `METHODS=(...)` list near the top (comment methods in
or out, or force one with `METHOD=picokv`) and shares the same tunables:
`K_BITS`/`V_BITS` (320), `SINK` (64), `LOCAL` (256), `EXP_BITS` (4),
`EAS` (0.01), `ZEROROPE` (20), `USE_OFFLINE` (0, except 1 in
`run_efficiency.sh`). The decoder ridge is fixed at
`constants.ONLINE_DECODER_RIDGE = 1e-3` for both the online solve and the
offline fits, and decode-time pruning is off everywhere by default
(`run_efficiency.sh` can re-enable it with `PRUNING=1`, which adds a `_prune`
tag suffix). The fused prefill cascade is on by default
(`PICOKV_DISABLE_FUSED_QUANT` is unset, i.e. read as `0`); no launcher sets it,
and `=1` forces the reference PyTorch loop for A/B runs. It only applies to
sides that carry a stored decoder, so it is a no-op unless `USE_OFFLINE=1`.
The resolved picokv tag is `picokv_{mode}_k{K}_v{V}`. Note the default
`MODEL` path differs per launcher (`/data/Llama-3.1-8B-Instruct` for
Ruler/InfiniteBench/Longbench/Needle vs the full share-storage path for
Efficiency) — override `MODEL=...` as needed.

For PiCoKV, the standard configuration is K/V `320/320`, `sink_size=64`,
`local_size=256`, `eas=0.01`, and `zerorope=20`. Offline projections already
contain their fitted encoder/decoder pair; set `USE_OFFLINE=1` to load
`offline_projection/projections/{model}_k320_v320.pt` instead of fitting online.
A missing default PT is reported as an explicit `FileNotFoundError`.

LongBench 的数据目录不设机器相关默认值。直接运行 `pred.py` 时必须显式传入
`--data-dir /path/to/longbench/data`；使用 `run_longbench.sh` 时通过环境变量
`DATA_DIR=/path/to/longbench/data` 传入。

## Baseline Alignment Notes

1. **`apply_chat_template` tokenization**: changed from
   `tokenize=False` (string) + `tokenizer([text], return_tensors="pt")` to
   `tokenize=True, return_tensors="pt", return_dict=False` to avoid a
   potential double-BOS token introduced by the two-step path.

2. **`max_new_tokens`**: increased from 30 to 50 to match the reference
   baseline and avoid response truncation on borderline cases.
