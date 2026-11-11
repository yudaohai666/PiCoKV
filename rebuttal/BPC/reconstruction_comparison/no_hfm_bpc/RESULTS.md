# Layer-0 BPC and GPC CD reconstruction

> Intermediate historical variant: private RNG and omitted temporary fitting codes. Current source-copy BPC restores the original RNG and temporary-code writes; these measurements do not characterize its initialization or efficiency.

uv environment, Llama-3.1-8B-Instruct, first 16384 Paul Graham tokens, layer-0 K/V, 8 KV heads, head_dim 128. Sink/local 64/256, global segment 16064 tokens. Both methods start from the same raw KV. BPC encodes raw K without HFM; GPC applies HFM (zerorope=20). Neither applies HFM to V. Errors use the original unmasked KV target, so GPC K includes discarded-dimension error as well as quantization error. The HFM relative-error floor is 0.529206. Additional GPC errors against its masked coding target are retained in results.json.

Measurements precede EAS selection and exclude dense-retention error reduction.

- BPC learns one bit at a time, with four alternating updates and residual subtraction. After fitting 320 directions, it re-encodes the initial input sequentially. Source parity was checked exactly on a real-input subset.
- GPC uses production online block32 coding, 8192-token direction sampling, within-group QR, joint ridge decoder fitting (0.001), and residual updates between groups. Actual codes/decoders were captured from the attention forward.
- Each method generates its own C and D. Each point uses the first b bits of its full 320-bit fit; these are not independently trained b-bit quantizers.
- Stored int8 ±1 codes and BF16 decoders are evaluated in CPU FP64 as `||X - C[:b]D[:b]||_F / ||X||_F`, aggregated across 8 heads with the same target normalization. Internal BF16 iterative residuals are not the error metric.

| Cumulative bits | BPC K relative error | GPC K relative error | BPC V relative error | GPC V relative error |
|---:|---:|---:|---:|---:|
| 32 | 0.567332 | 0.641380 | 0.722112 | 0.753552 |
| 64 | 0.466565 | 0.586386 | 0.623612 | 0.637352 |
| 96 | 0.389310 | 0.561861 | 0.551404 | 0.558320 |
| 128 | 0.329332 | 0.549042 | 0.493797 | 0.494825 |
| 160 | 0.280113 | 0.541651 | 0.445359 | 0.441167 |
| 192 | 0.239660 | 0.537192 | 0.403612 | 0.394439 |
| 224 | 0.206332 | 0.534486 | 0.366990 | 0.353677 |
| 256 | 0.178269 | 0.532759 | 0.334148 | 0.318202 |
| 288 | 0.154585 | 0.531640 | 0.304921 | 0.286803 |
| 320 | 0.134651 | 0.530904 | 0.278567 | 0.258874 |

This compares local tensor reconstruction on one layer and text, rather than measuring LongBench quality. Environment, source/input/code/decoder hashes, per-head errors, and validation are retained in results.json. Reproduction entries are documented in the [experiment index](../README.md). Production code is unchanged.
