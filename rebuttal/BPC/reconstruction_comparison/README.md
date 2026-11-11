# Layer-0 CD reconstruction experiments

Each method learns its own codes C and decoder D. The metric is cumulative `||X-CD||F / ||X||F`. Single-layer, single-text reconstruction errors are not LongBench quality scores.

| Files | Configuration | Status |
|---|---|---|
| [RESULTS.md](RESULTS.md), [results.json](results.json) | Shared HFM for BPC/GPC; private BPC RNG; temporary fitting codes omitted | Historical measurement |
| [no_hfm_bpc/RESULTS.md](no_hfm_bpc/RESULTS.md), [results.json](no_hfm_bpc/results.json) | BPC without HFM, GPC with HFM; private BPC RNG; temporary fitting codes omitted | Intermediate configuration; common raw, unmasked KV target |
| `run.py` -> `source_global_rng/` | Current verbatim BPC copy, global RNG, temporary codes retained, no BPC HFM | Updated code; no new 16K measurement in this directory yet |

`legacy_shared_hfm_encoding.py` preserves the old coding implementation; `legacy_shared_hfm_run.py` is its historical runner. Do not overwrite retained measurements. Current 2K model integration validation is in [source_copy_v3](../results/source_copy_v3/RESULTS.md).

The historical shared-HFM K table is normalized against masked K. The HFM-free comparison uses original K for both methods and includes dimensions discarded by GPC HFM. K values from these tables are not directly comparable.
