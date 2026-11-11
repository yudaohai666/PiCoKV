# GPC and BPC coding experiments

[Retained result tables](../RESULTS.md) and [legacy UV measurements](results/uv_retest/RESULTS.md).

## Configurations

| Configuration | HFM | RNG / temporary fitting codes | Status |
|---|---|---|---|
| Legacy BPC+HFM | zerorope=20 | Private generator / omitted | Retained efficiency and full LongBench; average 49.67 |
| Intermediate BPC without HFM | Disabled | Private generator / omitted | Layer-0 CD reconstruction only |
| Current BPC source copy | Disabled | Original global RNG / retained | Source parity and model validation passed; full efficiency/LongBench pending |

The historical tables describe the first row; the current default entry point uses the third row.

## Current implementation

- `source_encoding.py`: `find_binary_projection` and `binary_project_pytorch` are copied verbatim from `src/BPC/bpc/bpc/bpc_pytorch.py`. `binary_project_unpacked` preserves all replay computations and returns codes immediately before the original 64-bit packing step.
- `encoding.py`: original global RNG, four alternating updates per bit, and `quan_A` allocation and writes. The original input is re-encoded after fitting. Each K/V vector uses 320 bits, without HFM, QR, ridge, or the GPC 8192-token sample.
- EAS, sink/local windows, packing, cache layout, query/weight quantization, and packed decode use PiCoKV. This experiment replaces coding; it does not reproduce the complete original sparse-attention BPC system.
- `run.py`: invokes the original efficiency or LongBench runner with effective BPC zerorope=0. The current version is `bpc320_source_global_rng_v3`; it cannot resume legacy outputs.
- `single_layer_prefill.py`: real layer-0 prefill/prepare timing, using the current BPC implementation for new runs.

## Validation and reconstruction

- [Current validation](results/source_copy_v3/RESULTS.md): 64/320-bit temporary fitting codes, decoder, replay codes, and RNG states match the source exactly. HFM-free integration, packing, decode, and 2K model generation passed.
- [Reconstruction index](reconstruction_comparison/README.md): historical configurations remain distinct; the current script writes to a fresh `source_global_rng/` directory.
- [Legacy BPC+HFM coding backup](reconstruction_comparison/legacy_shared_hfm_encoding.py).

## Running

Set the uv/CUDA environment in the [top-level README](../README.md) and choose an idle GPU. Current BPC model-efficiency example:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --project . --no-sync python rebuttal/BPC/run.py \
  --experiment efficiency --encoding binarypc \
  --model /data/zxh/models/Llama-3.1-8B-Instruct --method picokv \
  --k-bits 320 --v-bits 320 --eas 0.01 --zerorope 0 --no-pruning \
  --lengths 16384 --decode-tokens 64 --warmup 1 --repeats 3 \
  --output rebuttal/BPC/results/new_source_bpc_16k.json
```

Use `--encoding gpc --zerorope 20` and a separate output for GPC. For full LongBench, use `run.py --experiment longbench`, forward the original `benchmarks/Longbench/pred.py` arguments, and score a fresh output directory with the original `eval.py`.

```bash
CUDA_VISIBLE_DEVICES=0 uv run --project . --no-sync python rebuttal/BPC/check_source_copy.py \
  --device cuda --output rebuttal/BPC/results/new_source_check.json
```

## Retained historical measurements

- `results/uv_retest/full_model/`: successful 16K/32K/64K runs, memory metrics, commands, and logs. Formal 64K outputs are `*_65536_gc08.json`.
- `results/uv_retest/single_layer/`: individual measurements and summaries at 16K/32K/64K/128K/256K.
- `results/uv_retest/longbench_full/`: 3750 predictions per method for Dense/GPC/BPC, per-task scores, run and resume records.
- `results/uv_retest/environment.json`: historical environment and source hashes. Current experiment wrappers have since changed.
- `run_longbench_full_uv.py`, `resume_dense_auto_uv.py`, `resume_dense_auto_chunked_uv.py`: retained historical execution helpers with old paths/version checks, not current one-command resume entry points.

Conda results, mechanical group-32 BPC, the rejected fixed-code decoder experiment, small LongBench subsets, and obsolete diagnostic/tuning outputs were removed. Production code is unchanged.
