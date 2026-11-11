# Retained experiment results

User-selected uv measurements on Llama-3.1-8B-Instruct, RTX 3090 24GB, BF16, batch size 1.

> Historical BPC configuration: HFM (zerorope=20), private RNG, and omitted temporary fitting codes. Current source-copy BPC disables HFM, restores the original global RNG, and retains temporary codes. These historical numbers are not measurements of the current version.

## Packed and unpacked attention at 32K

Two warmups and five measured runs; 64 generated tokens. Attention time sums actual attention CUDA kernels across 32 layers and averages eight steady-state tokens in a separate instrumented run.

| Method | Prefill (s) | Decode (ms/token) | Throughput (tok/s) | Attention (ms/token) |
|---|---:|---:|---:|---:|
| Dense BF16 / FA2 | 11.427 | 41.992 | 23.81 | 6.117 |
| Original Packed | 12.796 | 27.658 | 36.16 | 4.125 |
| Unpack, no K/V reconstruction (CUDA) | 12.888 | 30.100 | 33.22 | 6.626 |
| Unpack and reconstruct K/V (CUDA) | 12.857 | 48.377 | 20.67 | 25.089 |

[Measurement details and raw data](results/uv_cuda_32k_20261110/summary.md).

## Full-model single-GPU efficiency

One warmup and three measured runs; medians, 64 generated tokens. Prefill is full-model time to first token in seconds. Each length uses one GPU.

| Input length | GPC prefill (s) | BPC prefill (s) | GPC decode (token/s) | BPC decode (token/s) |
|---|---:|---:|---:|---:|
| 16K | 5.40 | 21.94 | 37.13 | 36.77 |
| 32K | 12.76 | 46.62 | 35.17 | 34.97 |
| 64K | 34.39 | 95.76 | 32.72 | 32.70 |

[Details, memory, and raw data](BPC/results/uv_retest/full_model/RESULTS.md).

## Single-layer prefill and KV preparation

Real layer-0 attention; two warmups and five measured runs, reporting medians. KV preparation fits and encodes KV online, selects EAS tokens, and packs the decode cache. The 256K point measures single-layer efficiency only.

| Length | GPC single-layer prefill (ms) | BPC single-layer prefill (ms) | GPC single-layer KV prepare (ms) | BPC single-layer KV prepare (ms) |
| --- | ---: | ---: | ---: | ---: |
| 16K | 77.82 | 562.61 | 27.70 | 510.65 |
| 32K | 212.73 | 1172.45 | 39.53 | 995.75 |
| 64K | 703.66 | 2600.12 | 61.29 | 1950.24 |
| 128K | 2601.77 | 6413.27 | 102.43 | 3913.73 |
| 256K | 9998.26 | 17683.74 | 197.08 | 7848.16 |

[Single-layer measurement details](BPC/results/uv_retest/single_layer/RESULTS.md).

## Full LongBench

All 16 tasks, 3750 samples per method. All predictions, task scores, and run/resume records are retained.

| Method | LongBench average |
|---|---:|
| Dense BF16 | 49.65 |
| Original PiCoKV | 49.61 |
| PiCoKV + BPC coding | 49.67 |

[Per-task scores and predictions](BPC/results/uv_retest/longbench_full/RESULTS.md).

## Implementation and provenance

- [Current source-copy BPC validation](BPC/results/source_copy_v3/RESULTS.md): exact source parity at 64/320 bits and 2K full-model generation checks passed. Current-version full measurements are separate from the historical tables.
- [Layer-0 reconstruction experiments](BPC/reconstruction_comparison/README.md): shared-HFM and intermediate no-HFM configurations remain distinct.
- [Cleanup manifest](cleanup_manifest.json): removed artifacts and retained raw-data hashes. Conda results, mechanical group-32 BPC, and the rejected decoder experiment were removed.
