# Layer-0 BPC and GPC CD reconstruction

> Historical shared-HFM variant: both methods apply zerorope=20 to K. This is not the current HFM-free source-copy BPC configuration.

uv environment, Llama-3.1-8B-Instruct, first 16384 Paul Graham tokens, layer-0 K/V, 8 KV heads, head_dim 128. Sink/local 64/256, global segment 16064 tokens. Both methods share the same compressed input tensor; K uses zerorope=20 for both. Input hashes are retained in results.json.

Measurements precede EAS selection and exclude dense-retention error reduction.

- BPC learns one bit at a time, with four alternating updates and residual subtraction. After fitting 320 directions, it re-encodes the initial input sequentially. Source parity was checked exactly on a real-input subset.
- GPC uses production online block32 coding, 8192-token direction sampling, within-group QR, joint ridge decoder fitting (0.001), and residual updates between groups. Actual codes/decoders were captured from the attention forward.
- Each method generates its own C and D. Each point uses the first b bits of its full 320-bit fit; these are not independently trained b-bit quantizers.
- Stored int8 ±1 codes and BF16 decoders are evaluated in CPU FP64 as `||X - C[:b]D[:b]||_F / ||X||_F`, aggregated across 8 heads with the same target normalization. Internal BF16 iterative residuals are not the error metric.

| Cumulative bits | BPC K relative error | GPC K relative error | BPC V relative error | GPC V relative error |
|---:|---:|---:|---:|---:|
| 32 | 0.382635 | 0.427071 | 0.722112 | 0.753552 |
| 64 | 0.278368 | 0.297665 | 0.623612 | 0.637352 |
| 96 | 0.211635 | 0.222461 | 0.551404 | 0.558320 |
| 128 | 0.165327 | 0.172362 | 0.493797 | 0.494825 |
| 160 | 0.131301 | 0.136055 | 0.445359 | 0.441167 |
| 192 | 0.106030 | 0.108764 | 0.403612 | 0.394439 |
| 224 | 0.086502 | 0.088326 | 0.366990 | 0.353677 |
| 256 | 0.071501 | 0.072398 | 0.334148 | 0.318202 |
| 288 | 0.059676 | 0.059896 | 0.304921 | 0.286803 |
| 320 | 0.050290 | 0.050002 | 0.278567 | 0.258874 |

This compares local tensor reconstruction on one layer and text, rather than measuring LongBench quality. Environment, source/input/code/decoder hashes, per-head errors, and validation are retained in results.json. Reproduction entries are documented in the [experiment index](README.md). Production code is unchanged.
