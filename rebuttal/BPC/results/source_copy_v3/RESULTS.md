# BPC source-copy validation

Configuration: original global RNG, original temporary fitting-code allocation and per-bit writes, 320-bit K/V, no BPC HFM.

- uv environment, RTX 3090 GPU0.
- 64/320-bit samples: temporary fitting codes, decoder, replay codes, and global RNG state exactly match the original `src/BPC` functions; prepare/packing/decode integration passed.
- Llama-3.1-8B-Instruct, 2048 real PG input tokens and 4 generated tokens: all 32 layers built 320-bit caches; caches and decode outputs were finite.
- 96 per-layer decode calls were checked against explicit K/V reconstruction using the same query and cache.
- Relative L2 error: median 0.035375, maximum 0.128944. This includes the original PiCoKV query/weight quantization; exact equality to ordinary BF16 attention is not expected.

This validates coding reproduction and model integration, not LongBench quality or efficiency. Full LongBench and efficiency have not been rerun for the current configuration.

Raw artifacts: `check_cuda.json`, `check_model_2k.json`, `check_model_2k.log`. Validation entry points: `rebuttal/BPC/check_source_copy.py` and `rebuttal/BPC/validate_model.py`.
