# Additional online prefill stage timing tools

Only `run.py`, `instrument.py`, and `validate.py` are retained for future diagnostics. Previous Conda measurements, snapshots, and failed-attempt logs were removed and are excluded from retained tables.

`instrument.py` inserts CUDA events around the production online algorithm to distinguish direction/decoder fitting, coding/residual updates, EAS, packing, and gathering. K/V streams overlap, so stage intervals are not additive.

Use the uv/CUDA environment from the [top-level README](../README.md) and fresh output files. Experiment runners retain raw machine-readable output; static reports are maintained separately.

Retained model and single-layer measurements are in the [result overview](../RESULTS.md).
