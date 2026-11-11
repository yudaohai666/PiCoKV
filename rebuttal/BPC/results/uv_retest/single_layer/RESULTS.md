# Single-layer prefill and KV preparation in uv

> Historical BPC configuration: HFM (zerorope=20), private RNG, and omitted temporary fitting codes. Current source-copy BPC disables HFM, restores the original global RNG, and retains temporary codes. These historical numbers are not measurements of the current version.

All measurements used `/data/zxh/PiCoKV/.venv` through `UV_PROJECT_ENVIRONMENT=/data/zxh/PiCoKV/.venv uv run --project /data/zxh/PiCoKV2 --no-sync`. Python 3.12.13, Torch 2.6.0+cu124, Transformers 4.53.2, FlashAttention 2.7.4.post1 match uv.lock. CUDA extensions were rebuilt with Python 3.12/CUDA 12.4. Versions and hashes: [environment.json](../environment.json).

Real layer-0 weights of Llama-3.1-8B-Instruct and Paul Graham inputs; batch 1, 32 query heads, 8 KV heads, head_dim 128, BF16, online, K/V 320/320 bits, EAS 0.01, sink/local 64/256, zerorope 20, exp_bits 4, no pruning. Each length used one RTX 3090, alternating GPC/BPC, two warmups and five measured runs, reporting medians.

| Length | GPC single-layer prefill (ms) | BPC single-layer prefill (ms) | GPC single-layer KV prepare (ms) | BPC single-layer KV prepare (ms) |
| --- | ---: | ---: | ---: | ---: |
| 16K | 77.82 | 562.61 | 27.70 | 510.65 |
| 32K | 212.73 | 1172.45 | 39.53 | 995.75 |
| 64K | 703.66 | 2600.12 | 61.29 | 1950.24 |
| 128K | 2601.77 | 6413.27 | 102.43 | 3913.73 |
| 256K | 9998.26 | 17683.74 | 197.08 | 7848.16 |

Single-layer prefill uses synchronized wall-clock timing and includes QKV projections, RoPE application, online KV preparation, exact causal FlashAttention, and output projection. It excludes input construction, RMSNorm, precomputed RoPE cos/sin, other layers, MLP, and LM head. KV preparation is the production CUDA-event subinterval for online fitting/coding/error, EAS selection, packing, and gathering, preserving overlap between K/V streams.

| Length | GPC/BPC allocated peak (GiB) | GPC/BPC incrementalpeak (GiB) | Single-layer KV (GiB, same for both) |
| --- | ---: | ---: | ---: |
| 16K | 0.791 / 0.791 | 0.562 / 0.562 | 0.013657 |
| 32K | 1.487 / 1.487 | 1.125 / 1.125 | 0.024480 |
| 64K | 2.877 / 2.877 | 2.250 / 2.250 | 0.046128 |
| 128K | 5.658 / 5.658 | 4.500 / 4.500 | 0.089416 |
| 256K | 11.221 / 11.221 | 9.000 / 9.000 | 0.175979 |

All lengths completed without OOM. Checks verified finite outputs/projections, 320/320-bit caches, and sampled exact prefill-output equality. GPC retains 8192-token direction sampling; BPC fits all global tokens with four alternating updates per bit. GPU0/1/2/3/4 correspond to 16K/32K/64K/128K/256K.

The 256K point exceeds the model's 128K context and measures attention scaling only. This single-layer timing cannot be multiplied by 32 to obtain model TTFT. Individual samples, min/max, and source hashes are in `length_{tokens}.json`; summary: [comparison.json](comparison.json).
