# NIAH results

Two runners write here, and they are not redundant — one records more than the
other, and the extra fields are what the speed claims rest on.

## `timed_runs/<tag>.json` — full record

Written by `quantpc/run_standalone_niah.py` (quantpc only). Carries the run
config plus per-cell telemetry:

```
config                 k_bits / v_bits / local_size / whiten / zerorope / eas / …
rows[].prepare_total_ms      prefill cache-construction time, all 32 layers
rows[].prepare_per_layer_ms  the same, per layer
rows[].elapsed_sec, decode_calls, actual_tokens, found, hits, response
```

Every speedup number quoted for this method comes from `prepare_total_ms`.
Nothing else in the repo records it.

## `Llama-3.1-8B-Instruct_<tag>/results.json` — plot input

Written by `benchmarks/Needle/run_bin.py` (all methods: quantpc, kivi, oscar,
and the dense baselines), and read directly by `show_image_paper.py`. Flat
`{"<length>-<depth>": response}`, no timing, no config.

The responses in the two are identical for the same run; only the timing and
config are unique to `timed_runs/`.

## Tag format

```
{online|offline}_k{K}_v{V}_local{L}_{whiten|whitenK|nowhiten}
```

`whiten` on an offline tag means the whitening is folded into the stored
encoder/decoder pair, not applied at runtime — with a folded projection the
runtime `--whiten` switch is rejected, because the basis is already accounted
for. `local` is the dense local-window size, which the tag has to carry: a
sweep at 1024 and one at 256 are otherwise indistinguishable by name, and the
plot title is built from this tag.

`whitenK` marks the three historical offline runs whose folded artifacts used
the earlier K-only whitening fit. New rebuilt artifacts and their NIAH runs use
the plain `whiten` suffix and include the current K+V whitening fit.

## `vis/`

`show_image_paper.py` output, one PNG per tag.
