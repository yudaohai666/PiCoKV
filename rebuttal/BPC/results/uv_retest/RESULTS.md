# Retained uv measurements

> Historical BPC configuration: HFM (zerorope=20), private RNG, and omitted temporary fitting codes. Current source-copy BPC disables HFM, restores the original global RNG, and retains temporary codes. These historical numbers are not measurements of the current version.

Llama-3.1-8B-Instruct, RTX 3090 24GB, BF16, batch 1. Both methods encode online with 320 bits per K/V vector and share PiCoKV packed decode. BPC fits all global tokens with four alternating updates per bit. Measurements used `/data/zxh/PiCoKV/.venv`, Python 3.12.13, Torch 2.6.0+cu124, Transformers 4.53.2, FlashAttention 2.7.4.post1, with rebuilt CUDA extensions.

## Single-layer prefill and KV preparation

Real layer-0 attention on normalized Paul Graham embeddings; two warmups, five measured runs, medians. Single-layer prefill includes projections, RoPE, online preparation, causal FA2, and output projection. It excludes input construction, RMSNorm, RoPE-table precomputation, MLP, and other layers. KV preparation fits/encodes KV, selects EAS tokens, and packs the decode cache. Its production CUDA events preserve K/V stream overlap; full layer prefill uses synchronized wall-clock timing.

| Length | GPC single-layer prefill (ms) | BPC single-layer prefill (ms) | GPC single-layer KV prepare (ms) | BPC single-layer KV prepare (ms) |
| --- | ---: | ---: | ---: | ---: |
| 16K | 77.82 | 562.61 | 27.70 | 510.65 |
| 32K | 212.73 | 1172.45 | 39.53 | 995.75 |
| 64K | 703.66 | 2600.12 | 61.29 | 1950.24 |
| 128K | 2601.77 | 6413.27 | 102.43 | 3913.73 |
| 256K | 9998.26 | 17683.74 | 197.08 | 7848.16 |

All lengths ran on one GPU each. The 256K point measures efficiency only; both methods peaked at 11.221 GiB. [Single-layer report](single_layer/RESULTS.md), [raw summary](single_layer/comparison.json).

## Full-model single-GPU efficiency

Original efficiency runner; one warmup, three measured runs, medians, 64 decode tokens after two decode warmup steps. Prefill uses TTFT. 16K/32K/64K ran on GPU5/6/7 respectively, one GPU per task.

| Input length | GPC prefill (s) | BPC prefill (s) | GPC decode (token/s) | BPC decode (token/s) |
|---|---:|---:|---:|---:|
| 16K | 5.40 | 21.94 | 37.13 | 36.77 |
| 32K | 12.76 | 46.62 | 35.17 | 34.97 |
| 64K | 34.39 | 95.76 | 32.72 | 32.70 |

Both methods' prefill/decode KV memory was 0.437/0.445 GiB at 16K, 0.783/0.791 GiB at 32K, and 1.476/1.484 GiB at 64K. [Full-model report](full_model/RESULTS.md), [raw summary](full_model/comparison.json).

Both 64K methods required allocator `garbage_collection_threshold:0.8`; algorithm and precision were unchanged. Environment, hashes, and commands: [README](README.md), [environment.json](environment.json). Production sources are unchanged; removed Conda results are excluded.

## Full LongBench

All three methods completed and validated 3750 samples across 16 tasks. Dense 49.65, GPC 49.61, legacy BPC+HFM 49.67. Scored by the original eval.py. [Per-task report](longbench_full/RESULTS.md).
