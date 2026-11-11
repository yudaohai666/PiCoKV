# Rebuttal experiments

All experimental implementations and maintenance changes are confined to this directory. Production code in `picokv/`, `src/BPC/`, and `benchmarks/` is unchanged.

Start with the [retained results](RESULTS.md): four attention implementations, GPC/BPC model efficiency, single-layer prefill/KV preparation, and full LongBench evaluation.

## Directory guide

| Experiment | Entry points | Results and raw data |
|---|---|---|
| Packed / unpacked codes / reconstructed K/V | `run_efficiency.py`, `measure_attention.py`, `cuda/` | [32K UV evaluation](results/uv_cuda_32k_20261110/summary.md) |
| GPC / BPC model efficiency | `BPC/run.py` | [16K/32K/64K](BPC/results/uv_retest/full_model/RESULTS.md) |
| GPC / BPC single-layer efficiency | `BPC/single_layer_prefill.py` | [16K-256K, layer 0](BPC/results/uv_retest/single_layer/RESULTS.md) |
| Full LongBench | Original `benchmarks/Longbench/pred.py` and `eval.py` through the experiment adapter | [16 tasks, 3750 samples per method](BPC/results/uv_retest/longbench_full/RESULTS.md) |
| Current BPC source-copy validation | `BPC/source_encoding.py`, `BPC/check_source_copy.py`, `BPC/validate_model.py` | [Source parity and 2K model generation](BPC/results/source_copy_v3/RESULTS.md) |
| Layer-0 CD reconstruction | `BPC/reconstruction_comparison/` | [Configurations and results](BPC/reconstruction_comparison/README.md) |

The retained BPC efficiency numbers and LongBench score of 49.67 belong to the legacy BPC+HFM variant. The current default BPC disables HFM, uses the original global RNG, and retains temporary fitting codes. Historical numbers are unchanged and do not represent new measurements of the current version. See the [BPC configuration guide](BPC/README.md).

## Environment

Use the existing uv environment from the project root:

```bash
export UV_PROJECT_ENVIRONMENT=/data/zxh/PiCoKV/.venv
export UV_CACHE_DIR=/data/zxh/.cache/uv-rebuttal
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=/usr/local/cuda-12.4/bin:$PATH
export PICOKV_DISABLE_PREBUILT=1
export TORCH_EXTENSIONS_DIR=/data/zxh/.cache/picokv_bpc_uv_extensions
export TRITON_CACHE_DIR=/data/zxh/.cache/picokv_bpc_uv_triton
export TORCH_CUDA_ARCH_LIST=8.6
export MAX_JOBS=2 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
```

Select an idle GPU and set `CUDA_VISIBLE_DEVICES` before running GPU experiments. Use a fresh output directory for each configuration.

## Attention controls

`run_efficiency.py` forwards the original efficiency arguments and adds `--decode-backend`. The retained controls are `packed`, `unpacked_fused` (unpack without reconstructing K/V), and `reconstruct_cuda` (unpack and reconstruct K/V). Dense uses the original runner with `--method flash_attention_2`.

```bash
CUDA_VISIBLE_DEVICES=0 uv run --project . --no-sync python rebuttal/run_efficiency.py \
  --decode-backend unpacked_fused \
  --model /data/zxh/models/Llama-3.1-8B-Instruct --method picokv \
  --k-bits 320 --v-bits 320 --eas 0.01 --no-pruning \
  --lengths 32768 --decode-tokens 64 --warmup 2 --repeats 5 \
  --output rebuttal/results/new_unpacked_32k.json
```

- `cuda/attention.cu`, `attention_ptx.cuh`: handwritten CUDA/Tensor Core controls.
- `reconstruct_attention.py`: explicit K/V reconstruction reference, also used by BPC validation.
- `unpacked_attention.py`, `*_triton.py`: correctness references and implementation dependencies.
- `test_*.py`, `check_cuda_output_error.py`: correctness checks.
- `latency/`: additional prefill stage timing tools; previous Conda measurements were removed and are excluded from the retained tables.

Measurements and source-copy validation were performed against project revision `4a16afa` on `quantpc_v3`. Historical run manifests record exact source hashes; reproduce those measurements from that revision rather than assuming later production changes are equivalent.

Raw JSON/JSONL, logs, configurations, and measured source snapshots retain their original locations. The [cleanup manifest](cleanup_manifest.json) lists removed files and hashes of retained raw artifacts.
