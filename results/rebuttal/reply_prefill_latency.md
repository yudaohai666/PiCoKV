# Reviewer reply — "report complete online prefill latency (fitting, GPC, EAS, packing)"

## Point 1 — the reported prefill latency is already complete

We thank the reviewer for pressing on this, and we first want to remove a possible
ambiguity in our presentation. The prefill numbers in the paper are **already the
complete end-to-end time-to-first-token of online mode**, not a cost that excludes
the compression work. In our implementation `prepare_kv_cache` is called
synchronously on the caller's stream from inside the attention forward of the
prefill pass, and the reported `prefill_ms` is measured as the wall time of
`generate(max_new_tokens=1)` with a device synchronization on both sides.
Projection fitting, GPC encoding, EAS selection and bit-packing are therefore all
inside the published figure; none of them is deferred, overlapped with a later
stage, or charged to a separate offline step.

What the paper was missing is not the total but its **decomposition**. We add it
below, obtained two independent ways that agree.

## Point 2 — stage decomposition (Table A)

Named `record_function` ranges inside the prefill path, Llama-3.1-8B-Instruct,
batch 1, online mode, k=v=320, EAS=0.01, one A800-80GB. Each entry is CUDA device
time summed over all 32 layers, with the K and V sides aggregated.

| stage (ms) | 16K | 32K | 64K | 128K | 256K |
|---|---|---|---|---|---|
| projection fitting (init + encoder + decoder) | 940.9 | 1083.8 | 1511.3 | 2064.7 | 3192.3 |
| GPC (sign encode + residual update) | 113.7 | 161.3 | 275.0 | 487.7 | 952.5 |
| EAS (error norm + selection) | 9.8 | 16.5 | 19.7 | 31.0 | 50.5 |
| packing + dense gather | 3.0 | 11.7 | 23.9 | 52.2 | 98.5 |
| slice + ZeroRoPE | 3.5 | 7.8 | 16.2 | 32.5 | 65.5 |
| **total compression work** | **1070.9** | **1281.1** | **1846.1** | **2668.1** | **4359.3** |
| **as % of online TTFT** | **83.0** | **50.9** | **30.2** | **14.1** | **6.6** |

Reference end-to-end TTFT on the same machine and session:

| TTFT (ms) | 16K | 32K | 64K | 128K | 256K |
|---|---|---|---|---|---|
| FlashAttention-2 (fp16, no compression) | 666.2 | 1718.9 | 5043.5 | 16839.7 | 61888.4 |
| PiCoKV online | 1291.0 | 2516.0 | 6107.6 | 18947.4 | 66012.0 |
| PiCoKV offline | 688.1 | 1782.1 | 5174.9 | 17273.7 | 63086.6 |
| online / FA2 | 1.94x | 1.46x | 1.21x | 1.13x | **1.07x** |
| offline / FA2 | 1.03x | 1.04x | 1.03x | 1.03x | **1.02x** |

Three facts we would highlight from Table A:

1. **EAS and packing are not cost items.** EAS is 9.8 -> 50.5 ms over a 16x length
   increase, i.e. <=0.76% of TTFT and 0.08% at 256K. Packing plus the dense gather
   is 98.5 ms at 256K, 0.15% of TTFT. Both are far below the measurement noise of
   a 66 s prefill.
2. **Projection fitting dominates, but it is the part that does not grow.** The
   power-iteration encoder fit operates on a fixed-size residual sample
   (`direction_sample_size = 8192` rows, subsampled by stride), so it is O(1) in
   sequence length: 695.8 -> 902.8 ms from 16K to 256K, a factor 1.3 for a 16x
   length increase. Only the per-group least-squares decoder solve is genuinely
   O(L) (225.9 -> 2278.0 ms, 10.1x), because its Gram matrix accumulates over all
   L tokens.
3. **Hence the overhead ratio decays monotonically**, 1.94x -> 1.46x -> 1.21x ->
   1.13x -> 1.07x. The absolute compression cost grows 4.1x while attention itself
   grows 93x, so the overhead is amortized away precisely in the long-context
   regime the method targets.

## Point 3 — end-to-end confirmation (Table B)

Because a profiler range reports device time and our K/V sides run on two CUDA
streams, Table A's per-stage times sum across the two sides and therefore cannot
be compared to wall time directly. To confirm the decomposition without the
profiler we also ran a four-arm end-to-end ablation, in which each stage is
switched off by substituting a fixed projection instead of fitting one:

| arm | encoders | decoders | fused kernel | TTFT 128K | TTFT 256K |
|---|---|---|---|---|---|
| 1 | fitted | fitted | no | 18613.9 | 65258.6 |
| 2 | fixed | fitted | no | 18095.1 | 64527.0 |
| 3 | fixed | fixed | no | 17633.9 | 63622.4 |
| 4 | fixed | fixed | yes | 17231.2 | 62883.9 |

| attributed cost (ms) | 128K | 256K |
|---|---|---|
| encoder fitting (arm1 - arm2) | 518.8 | 731.6 |
| decoder solve (arm2 - arm3) | 461.2 | 904.6 |
| kernel fusion gain (arm3 - arm4) | 402.7 | 738.5 |
| total online-fitting cost (arm1 - arm4) | 1382.7 | 2374.7 |

The ablation reproduces Table A's two qualitative claims: encoder fitting is
nearly flat in L (1.4x for a 2x length increase) while the decoder solve nearly
doubles (2.0x). The wall-clock differences are smaller than the profiler's device
sums, which is expected: the two sides overlap, so removing work from one side
only shortens the critical path by the unoverlapped part.

All four arms were run strictly serially on one GPU, one process at a time. We
mention this because our first pass ran the arms concurrently on three GPUs, and
on a shared node a 1% load fluctuation at a 66 s prefill is ~660 ms, i.e. the same
order as the quantity being measured; the concurrent pass produced a sign-flipped
decoder-solve estimate. The serial numbers above are the ones we report.

## Point 4 — and the overhead is optional

Offline mode reuses projections fitted once on a calibration set and pays none of
the fitting cost, at 1.02-1.04x FA2 prefill across all five lengths, which is the
configuration we recommend for deployment. The online path exists for the setting
where no calibration data is available; the tables above quantify exactly what
that convenience costs, and show it is 1.07x at 256K.

---

## Internal notes (do NOT paste into the rebuttal)

- Arms 2 and 3 are **timing-only** configurations. Arm 2 discards the decoder that
  was fitted to the very encoders it keeps, and arm 3 forces the PyTorch reference
  loop. Neither is deployable and **neither may ever be quoted for accuracy**.
- Table A rows are K+V device-time aggregates over 32 layers. They do not sum to
  wall time (two-stream overlap). Internal check: at 256K the stage sum is 4359 ms
  vs key_side 2055 + value_side 2066 = 4121 ms.
- Sources: `results/rebuttal/prof/picokv_online_k320_v320_prefill_{16384..262144}.txt`
  (Table A), `results/rebuttal/serial/b1/*.json` (Table B),
  `results/rebuttal/b1/Llama-3.1-8B-Instruct_{picokv_online_k320_v320,flash_attention_2,picokv_offline_k320_v320}.json`
  (TTFT reference).
- Table B currently covers 128K/256K only. If the reviewer asks for the full
  five-length end-to-end decomposition, the three shorter lengths need a serial
  rerun (the concurrent numbers at those lengths are noise-dominated).
