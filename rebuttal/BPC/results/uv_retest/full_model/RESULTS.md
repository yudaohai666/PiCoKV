# Full-model single-GPU efficiency in uv

> Historical BPC configuration: HFM (zerorope=20), private RNG, and omitted temporary fitting codes. Current source-copy BPC disables HFM, restores the original global RNG, and retains temporary codes. These historical numbers are not measurements of the current version.

Llama-3.1-8B-Instruct, RTX 3090 24GB, BF16, batch 1; K/V 320 bits each, online, EAS 0.01, sink/local 64/256, exp_bits 4, zerorope 20, no pruning. uv Python 3.12, Torch 2.6.0+cu124, FlashAttention 2.7.4.post1.

Entry point: `rebuttal/BPC/run.py --experiment efficiency --encoding gpc|binarypc`, forwarding to the unchanged `benchmarks/Efficiency/run_efficiency.py`. Real Paul Graham text prefixes; 16K/32K/64K used GPU5/6/7 respectively. Each length ran GPC then BPC on one GPU, without model sharding or CPU offload.

One warmup and three measured runs, reporting medians. Each decode run follows an untimed prefill and two warmup decode steps, then times 64 steps. Prefill follows the original runner's `generate(max_new_tokens=1)` / TTFT definition: full-model forward, online preparation, and first-token generation, excluding model loading and tokenization.

| Input length | GPC prefill (s) | BPC prefill (s) | GPC decode (token/s) | BPC decode (token/s) |
|---|---:|---:|---:|---:|
| 16K | 5.40 | 21.94 | 37.13 | 36.77 |
| 32K | 12.76 | 46.62 | 35.17 | 34.97 |
| 64K | 34.39 | 95.76 | 32.72 | 32.70 |

Peak memory is CUDA max_memory_allocated, including resident model weights. KV memory counts unique persistent cache tensors, projection matrices, and retained dense KV. Runner fields ending in `*_gb` divide by 1024 cubed and represent GiB.

| Input length | Encoding | Prefill peak (GiB) | Decode peak (GiB) | Prefill KV (GiB) | Decode KV (GiB) |
|---|---|---:|---:|---:|---:|
| 16K | GPC | 16.647 | 15.433 | 0.437 | 0.445 |
| 16K | BPC | 16.646 | 15.431 | 0.437 | 0.445 |
| 32K | GPC | 17.630 | 15.774 | 0.783 | 0.791 |
| 32K | BPC | 17.631 | 15.775 | 0.783 | 0.791 |
| 64K | GPC | 20.187 | 16.468 | 1.476 | 1.484 |
| 64K | BPC | 20.187 | 16.468 | 1.476 | 1.484 |

Both 64K methods used `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8`; 16K/32K used `expandable_segments:True`. The initial 64K configuration ran out of memory; only allocator settings changed for the successful rerun. Formal 64K files are `*_65536_gc08.json`. Obsolete OOM/diagnostic artifacts were removed; successful measurements and run records remain.

Commands and source hashes are in each `.run.json`; validated inputs and summary are in [comparison.json](comparison.json). Shared environment: [environment.json](../environment.json).
