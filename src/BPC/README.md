# BPC

**Binary Principal Components** — sparse attention via 64-bit binary key hashing for long-context LLM inference. BPC compresses each cached key into a 64-bit hash, estimates Q-K similarity from hash codes during decode, and computes exact attention only over the top-k selected tokens.


## Installation

```bash
bash install.sh
```

The script creates a `bpc` conda env with Python 3.11 + PyTorch 2.6 + CUDA 11.8, then installs project dependencies and a prebuilt `flash-attn 2.7.1` wheel.

## Quick Start

```bash
conda activate bpc
./scripts/longbench/run_longbench-llama.sh
```

Each `run_*` script is a template — open it and toggle the `methods=()` and `models=()` arrays at the top to choose which compression method and model to evaluate.

## Project Layout

| Path                       | Purpose                                                     |
|----------------------------|-------------------------------------------------------------|
| [bpc/](bpc/)                       | Core package: BPC algorithm + all baseline modifiers        |
| [bpc/bpc/](bpc/bpc/)                   | BPC implementation (PyTorch + CUDA-kernel backends)         |
| [bpc/modifiers/](bpc/modifiers/)             | Unified `get_modifier(method)` registry                     |
| [benchmark/](benchmark/)                 | Per-benchmark drivers (longbench, ruler, needle, …)         |
| [config/](config/)                    | Per-model × per-method JSON configs (+ projections, scores) |
| [find_binary_projection/](find_binary_projection/)    | Offline calibration of BPC projection matrices              |
| [scripts/](scripts/)                   | Shell entry points for each benchmark                       |
| [ckp/](ckp/)                       | Checkpoints (e.g. learned projections)                      |


## Benchmarks

Each task lives in `benchmark/<task>/`; entry scripts in `scripts/<task>/`:

| Task            | Run                                                            | Eval                                                |
|-----------------|----------------------------------------------------------------|-----------------------------------------------------|
| general_tasks   | [run_general_tasks_bpc.sh](scripts/general_tasks/run_general_tasks_bpc.sh) | [compare_results.sh](scripts/general_tasks/compare_results.sh) |
| infinitebench   | [run_infinitebench.sh](scripts/infinitebench/run_infinitebench.sh)         | [compute_scores.sh](scripts/infinitebench/compute_scores.sh)   |
| longbench       | [run_longbench-llama.sh](scripts/longbench/run_longbench-llama.sh)         | [eval_longbench_eval.sh](scripts/longbench/eval_longbench_eval.sh) |
| longbench_v2    | [run_longbench_v2-llama3.sh](scripts/longbench_v2/run_longbench_v2-llama3.sh) | [eval_longbench_v2.sh](scripts/longbench_v2/eval_longbench_v2.sh) |
| needle          | [run_niah-llama.sh](scripts/needle/run_niah-llama.sh)                       | —                                                   |
| ruler_tasks     | [run_ruler_tasks_bpc.sh](scripts/ruler_tasks/run_ruler_tasks_bpc.sh)        | —                                                   |

## Configuration

Every run consumes a top-level model config:

```
config/<model>/<model>-<method>.json
```

It selects the `model_method` (e.g. `bpc`) and points at a `<model>-<method>.json` under `extra_configs/` for method-specific hyperparameters. Example for BPC on Llama-3.1-8B:

```jsonc
// config/llama3-1-8b-ins/llama3-1-8b-ins-bpc.json
{
    "model": {
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "model_method": "bpc",
        "config": "config/llama3-1-8b-ins/extra_configs/llama3-1-8b-ins-bpc.json",
        ...
    }
}
```

```jsonc
// config/llama3-1-8b-ins/extra_configs/llama3-1-8b-ins-bpc.json
{
    "enable": true,
    "fix_layers": [0, 1],
    "mask_out": 0.98,        // sparsity ratio (token_budget = ctx_len * (1 - mask_out))
    "min_remain": 128,
    "token_budget": null,    // if set, overrides mask_out
    "err_ratio": 0.1,        // fraction of token_budget reserved for high-error tokens
    "use_offline_proj": false,
    "offline_proj_path": null,
    "use_cuda_kernel": false
}
```

## Offline Projection Calibration (Optional)

BPC can learn projection matrices once on a calibration set instead of fitting them online during prefill. Calibrated projections live under `config/projections/`.

```bash
bash find_binary_projection/run_calibrate-llama3-mixdata.sh
```

To use them at inference, switch to a `*-bpc-offline.json` config (e.g. [llama3-1-8b-ins-bpc-offline.json](config/llama3-1-8b-ins/llama3-1-8b-ins-bpc-offline.json)), which sets `use_offline_proj: true` and points `offline_proj_path` at a `.pt` file.

## Citation

```bibtex
@inproceedings{
anonymous2026trainingfree,
title={Training-Free Hashing-Based Attention via Binary Principal Components},
author={Anonymous},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={https://openreview.net/forum?id=4spHlgHY9x}
}
```
