# Full LongBench execution record

These are completed historical BPC+HFM measurements, 3750 samples per method. The current BPC default is a different configuration.

Llama-3.1-8B-Instruct; 16 tasks, 3750 samples per method, 11250 total. No earlier pilot predictions were reused. Original prompt templates, 127500-token cap, per-task generation lengths, seed 42, greedy decoding.

| Method | Initial GPUs | Configuration |
|---|---|---|
| Dense | 0,1 | Full BF16 DynamicCache, FlashAttention2 |
| PiCoKV | 2,3 | Original online GPC, K/V 320/320 bits, EAS 0.01 |
| PiCoKV+BPC coding | 4,5 | BinaryPC fitting and residual coding; PiCoKV cache/packed decode |

Initial scheduling assigned independent tasks to one complete model per GPU, two workers per method. Compressed methods used sink/local 64/256, exp_bits 4, zerorope 20, no pruning. uv/CUDA 12.4 extensions match the efficiency measurements.

Historical command from the project root:

```bash
UV_CACHE_DIR=/data/zxh/.cache/uv-rebuttal \
UV_PROJECT_ENVIRONMENT=/data/zxh/PiCoKV/.venv \
uv run --project /data/zxh/PiCoKV2 --no-sync python -u \
  rebuttal/BPC/run_longbench_full_uv.py --limit 0 \
  --dense-gpus 0,1 --gpc-gpus 2,3 --binarypc-gpus 4,5 \
  --output rebuttal/BPC/results/uv_retest/longbench_full
```

The retained scheduler has historical version checks; use fresh output directories for new configurations. [manifest.json](manifest.json) retains exact commands/progress; logs and task shards retain per-sample outputs. Completed predictions were verified against answers/all_classes/length and source order, merged, and scored with the original `benchmarks/Longbench/eval.py --eval_avg`. Failed tasks and incomplete outputs were never treated as complete scores. Machine-readable summaries remain in comparison.json; Markdown reports are static.

## Dense recovery

Single-GPU Dense ran out of memory in MLP allocations for gov_report, narrativeqa, and repobench-p. At the user's request, `CUDA_VISIBLE_DEVICES=0,1` enabled the original benchmark's `device_map="auto"`, distributing model layers across two GPUs. The first resume started after 56/4/236 predictions, covering 604 remaining samples. Other tasks and saved prefixes were preserved. Entry point: `rebuttal/BPC/resume_dense_auto_uv.py`, still calling original pred.py. Device maps, commands, starts, validation, and scores are in [dense_auto_resume.json](dense_auto_resume.json), with logs in dense_auto_resume.log. The initial manifest's failed records describe the historical single-GPU attempts.

Auto completed gov_report, but narrativeqa ran out of memory again after sample 7. `resume_dense_auto_chunked_uv.py` retained auto layer placement on GPU0/1 and chunked MLP execution into 8192-token pieces, matching PiCoKV's approach. It resumed narrativeqa after 7 and repobench-p after 236 predictions, covering 457 remaining samples. Full BF16 KV, attention, prompts, and generation parameters were retained. Token-wise MLP operations are mathematically equivalent, but BF16 GEMM shapes can change rounding and generated outputs; exact unchunked-output parity was not claimed. Saved prefixes were validated. Records: dense_auto_chunked_resume.json and dense_auto_chunked_resume.log.

## BPC worker expansion

After Dense and GPC finished, idle GPU0-3 joined the BPC workers on GPU4/5. A now-removed one-off helper took over scheduling without duplicate writes, preserving in-flight repobench-p and lcc workers. Commands and records remain in [expansion_manifest.json](expansion_manifest.json) and expansion.launch.log. All three methods ultimately completed and validated 3750 predictions: Dense 49.65, GPC 49.61, legacy BPC+HFM 49.67.
