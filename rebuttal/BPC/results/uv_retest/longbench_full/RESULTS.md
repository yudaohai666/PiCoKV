# Full LongBench: Dense, GPC, and legacy BPC coding

> Historical BPC configuration: HFM (zerorope=20), private RNG, and omitted temporary fitting codes. Current source-copy BPC disables HFM, restores the original global RNG, and retains temporary codes. These historical numbers are not measurements of the current version.

All 16 tasks, 3750 samples per method. Dense uses FA2 with full BF16 KV. Compressed methods use PiCoKV or its BPC coding replacement. Original prompts, 127500-token prompt cap, generation lengths, seed 42, and greedy decoding are retained. Online K/V 320/320 bits, EAS 0.01. The uv environment and rebuilt CUDA extensions match the efficiency evaluation.

| Task | Dense BF16 | PiCoKV (GPC) | PiCoKV+BPC | BPC − GPC |
|---|---:|---:|---:|---:|
| gov_report | 34.54 | 33.27 | 33.48 | +0.21 |
| repobench-p | 56.57 | 56.97 | 56.70 | -0.27 |
| triviaqa | 91.65 | 92.33 | 92.05 | -0.28 |
| lcc | 63.21 | 63.33 | 62.96 | -0.37 |
| qasper | 44.50 | 43.99 | 43.66 | -0.33 |
| multifieldqa_en | 55.56 | 55.86 | 56.18 | +0.32 |
| narrativeqa | 28.76 | 29.66 | 30.02 | +0.36 |
| hotpotqa | 57.86 | 58.11 | 57.61 | -0.50 |
| 2wikimqa | 49.16 | 48.89 | 50.28 | +1.39 |
| musique | 33.06 | 33.08 | 33.20 | +0.12 |
| qmsum | 25.53 | 24.80 | 25.32 | +0.52 |
| trec | 72.50 | 72.50 | 72.50 | +0.00 |
| samsum | 43.88 | 43.95 | 43.73 | -0.22 |
| passage_count | 11.10 | 10.25 | 10.35 | +0.10 |
| passage_retrieval_en | 99.50 | 100.00 | 100.00 | +0.00 |
| multi_news | 27.04 | 26.71 | 26.64 | -0.07 |
| avg | 49.65 | 49.61 | 49.67 | +0.06 |

The original eval.py scores each task; avg is the unweighted mean of 16 task scores. Predictions were checked against source order, answers, length, and all_classes. Initial scheduling and commands: [manifest.json](manifest.json). Final validation and completion: [expansion_manifest.json](expansion_manifest.json). [Raw summary](comparison.json).
