# QuanPC Standalone

`quanpc_standalone` is the production online-projection path for compressed
KV-cache preparation and single-token CUDA decode attention.

> Repository integration: this package lives under `quantpc/`, together with
> its two JIT CUDA extensions, and is imported as
> `quantpc.quanpc_standalone`. Put the repository root on `PYTHONPATH`
> (`export PYTHONPATH="$PWD:$PYTHONPATH"`) and lock a single GPU
> (`export CUDA_VISIBLE_DEVICES=0`) before running any check below.
>
> ```
> quantpc/
>   quanpc_standalone/            # this package
>   quanpc_cuda_extension/        # decode CUDA kernels
>   quanpc_prefill_cuda_extension/  # cache-packing / gather CUDA kernels
> ```

## Scope

The package intentionally contains one implementation:

- projection directions are learned online from the current prefill K/V;
- cache preparation requires bf16 CUDA tensors;
- quantization uses block 32, one power iteration, batched CholeskyQR
  initialization, Householder QR after covariance iteration, bf16 matmuls,
  decoder ridge `1e-3`, and residual error for EAS;
- preparation is serial with the surrounding transformer work, while K and V
  learning overlap internally on separate CUDA streams;
- decode requires GQA 4 and uses the packed CUDA kernels.

The low-level ``prepare_kv_cache`` API is the online path. The Hugging Face
integration additionally accepts stored offline encoder/decoder pairs.
Torch-only decode, async prepare, block-16, and alternate dtype/error-mode
paths are not supported.

## Package layout

| File | Responsibility |
|---|---|
| `api.py` | Stable public import facade only |
| `cache.py` | `PreparedKVCache` data and decode workspace ownership |
| `constants.py` | Measured production invariants shared across stages |
| `online.py` | Block-32 online projection learning and binary quantization |
| `prepare.py` | K/V overlap, EAS selection, packing, and cache construction |
| `decode.py` | Dense attention, query projection, workspace reuse, CUDA tail |
| `config.py` | Separation of prepare and decode user options |
| `transformers.py` | Explicit Hugging Face Llama integration |

## API

```python
from quantpc.quanpc_standalone import (
    StandaloneConfig,
    decode_attention,
    prepare_kv_cache,
)

config = StandaloneConfig()
cache = prepare_kv_cache(
    full_key,                   # bf16 CUDA [B, KVH, L, D]
    full_value,                 # bf16 CUDA [B, KVH, L, D]
    **config.prepare_kwargs(),
)

output = decode_attention(
    query,                      # bf16 CUDA [B, 4*KVH, 1, D]
    cache,
    **config.decode_kwargs(),
)
```

Public prepare options are limited to values that remain useful across runs:

| Option | Default |
|---|---:|
| K/V bits | `320:320` |
| direction sample | `8192` |
| sink/local | `64 / 256` |
| EAS | `0.01` |
| zero-RoPE | `20` |
| exponent bits | `4` |
| seed | `0` |

Decode exposes pruning mode/log-epsilon, an optional attention scale, and
optional current-token K/V. Generated K/V can be appended to the retained
dense cache:

```python
output = decode_attention(
    query,
    cache,
    decode_key=current_key,
    decode_value=current_value,
    update_cache=True,
    **config.decode_kwargs(),
)
```

The compressed global cache remains static during decode. Generated tokens are
not yet evicted or re-quantized into the global cache.

## Hugging Face Llama

```python
from quantpc.quanpc_standalone import StandaloneConfig
from quantpc.quanpc_standalone.transformers import (
    StandaloneLlamaCache,
    modify_llama_for_standalone,
)

modify_llama_for_standalone(model, mlp_chunk_size=8192)
cache = StandaloneLlamaCache(StandaloneConfig())
generated = model.generate(
    **inputs,
    past_key_values=cache,
    do_sample=False,
    max_new_tokens=30,
)
```

Prompt attention stays exact SDPA. The adapter currently targets bf16 CUDA
Llama models with greedy, unpadded generation and a static compressed global
cache.

## Checks

Run from the repository root with the environment configured as above:

```bash
python quantpc/test_quanpc_standalone_online.py

python benchmarks/Needle/run_bin.py --method quantpc \
  --k-bits 320 --v-bits 320 --local-size 256 --zerorope 20 \
  --lengths 64000,110000 --depths 50
```

For offline evaluation, pass ``--offline`` or an explicit
``--offline-proj-path``. The default path is
``offline_projection/projections/{model}_k320_v320.pt`` when the
model basename is ``Llama-3.1-8B-Instruct``.
