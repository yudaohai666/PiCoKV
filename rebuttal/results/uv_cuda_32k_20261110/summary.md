# Four-method 32K evaluation in uv

The user-selected `/home/zxh/zxh/PiCoKV/.venv` was launched with `uv run --active --no-project`, without dependency synchronization. Python 3.12.13, PyTorch 2.6.0+cu124, Triton 3.2.0, Transformers 4.53.2, FlashAttention 2.7.4.post1.

Llama-3.1-8B-Instruct, RTX 3090 24GB; input 32768, batch 1, 64 generated tokens, two warmups and five repeats. K/V bits 320/320, sink/local 64/256, EAS 0.01, exp_bits 4, zerorope 20, no pruning. Full efficiency ran sequentially on GPU2; attention timing ran sequentially on GPU3, with a fresh process per method. CUDA extensions were rebuilt in a separate directory for this Python/PyTorch environment, with prebuilt extensions disabled. All methods used the same efficiency runner with KV-memory statistics; source hashes remained unchanged during measurement.

| Method | Prefill (s) | Decode (ms/token) | Throughput (tok/s) | Attention (ms/token) |
|---|---:|---:|---:|---:|
| Dense BF16 / FA2 | 11.427 | 41.992 | 23.81 | 6.117 |
| Original Packed | 12.796 | 27.658 | 36.16 | 4.125 |
| Unpack, no K/V reconstruction (CUDA) | 12.888 | 30.100 | 33.22 | 6.626 |
| Unpack and reconstruct K/V (CUDA) | 12.857 | 48.377 | 20.67 | 25.089 |

Attention sums actual attention CUDA kernel times across all 32 layers for one generated token, averaged over eight steady-state tokens. It includes unpacking/reconstruction, necessary query handling, the retained dense branch, and output fusion; it excludes cache append, QKV projection, RoPE, o_proj, and MLP. Raw JSON also retains five rounds of 64-token CUDA-event timing. Throughput comes from a separate uninstrumented run and cannot be directly subtracted from attention time.

The efficiency runner traverses KV storage and modules after the last GPU synchronization but before reading the end timestamp. Decode therefore includes this CPU metadata traversal once per round, amortized over 64 tokens. It neither reads tensor contents nor copies/allocates GPU tensors or explicitly synchronizes the GPU. Separate attention measurements omit out_kv and this traversal. KV bytes exclude workspaces; total peak allocated memory includes actual workspace allocations.

The three compressed controls share prefill code. Dense uses FA2 and DynamicCache. Both controls use handwritten CUDA; Triton implementations remain correctness references.

All 30 same-cache/query CUDA/Triton boundary and bit-width correctness cases passed. Each method verified 256 profiler markers and 10240 event markers; compressed methods also verified 256 main-kernel and 256 output-fusion calls.

Commands, Python path, versions, environment, hashes, and exit statuses are in `efficiency_run_config.json` and `attention_run_config.json`. Raw measurements are `efficiency_*.json` and `attention_*.json`. Measured source provenance is defined by `source_snapshot/` and run-config hashes; later tile tuning means current experiment kernels can differ from the archived measurements.
