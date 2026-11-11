# Retained UV measurements

The [result overview](RESULTS.md) contains model efficiency, single-layer prefill/KV preparation, and full LongBench.

**BPC measurements here use the legacy BPC+HFM configuration: zerorope=20, a private RNG, and omitted temporary fitting codes. Current default BPC code is different; these records do not represent new-version measurements.** See the [legacy coding backup](../../reconstruction_comparison/legacy_shared_hfm_encoding.py) and [current BPC guide](../../README.md).

| Experiment | Report | Raw artifacts |
|---|---|---|
| Single-GPU model efficiency, 16K/32K/64K | [full_model/RESULTS.md](full_model/RESULTS.md) | JSON, run metadata, and logs in `full_model/` |
| Layer 0, 16K-256K | [single_layer/RESULTS.md](single_layer/RESULTS.md) | `single_layer/length_*.json` and logs |
| Full LongBench, 16 tasks, 3750 samples/method | [longbench_full/RESULTS.md](longbench_full/RESULTS.md) | All predictions, scores, manifests, and task logs |
| Environment and implementation | [environment.json](environment.json) | Package versions, source hashes, CUDA build log, and integration checks |

Environment: `/data/zxh/PiCoKV/.venv`, Python 3.12.13, Torch 2.6.0+cu124, Transformers 4.53.2, FlashAttention 2.7.4.post1, Triton 3.2.0, CUDA 12.4.

Single layer: one GPU per length, alternating GPC/BPC on the same card, two warmups and five measured repetitions, median reported. Full prefill covers one attention module; KV preparation is its internal CUDA-event interval.

Model efficiency: one GPU per length, one warmup and three measured repetitions, median reported, 64 decode tokens. At 16K/32K the allocator setting was `expandable_segments:True`; formal 64K runs used `expandable_segments:True,garbage_collection_threshold:0.8` and filenames `*_65536_gc08.json`. Failed/diagnostic attempts were removed; successful raw measurements are unchanged.

Dense LongBench recovery, placement, and commands are retained in the [historical execution notes](longbench_full/README.md) and original manifests. The tables use final complete scores.

Conda data and the superseded 16-task x 5-sample pilot were removed. New measurements must use the uv environment and a fresh output directory.
