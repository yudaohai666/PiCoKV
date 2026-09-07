"""Validated invariants shared by standalone prepare and decode.

These are deliberately constants rather than public tuning options.  They
describe the single production path whose latency and accuracy were measured.
"""

from __future__ import annotations

import torch


# Cache code widths supported by the CUDA packing/decode kernels.
SUPPORTED_CODE_BITS = (128, 192, 256, 320, 384, 448, 512)
SUPPORTED_EXP_BITS = (1, 2, 4, 8)

# Online projection learning invariants.
ONLINE_GROUP_BITS = 32
ONLINE_DECODER_RIDGE = 1e-3
ONLINE_MATMUL_DTYPE = torch.bfloat16

# NOT A FLAG, AND MUST NOT BECOME ONE AGAIN: the block-32 decoder gram is always
# accumulated in fp32 (:func:`..online._pm_one_gram`), even though the matmuls
# feeding it are bf16.  ``codes`` is strictly +-1, so ``codes^T codes`` is an
# integer matrix whose diagonal is the segment length (~31800 at 32k); bf16 holds
# integers exactly only to 256, and a bf16 *output* matmul therefore puts ~2e-3
# relative noise on a gram whose true smallest eigenvalue is ~3e-6.  Note the
# noise survives a trailing ``.float()``: ``matmul(bf16, bf16)`` returns bf16, so
# the rounding has already happened -- that cast was the original bug's disguise.
#
# Measured on Qwen3-8B layer 0 V (a repeated haystack leaves it with 37 distinct
# rows out of 32158, so it is genuinely rank-deficient): the bf16 gram has
# eig_min = -3.4e-3 in every group -- numerically *indefinite*, so any ridge
# below 3.4e-3 leaves an indefinite system, ``solve_ex`` returns garbage
# (max|D| = 1.9e3 vs 0.09), and the greedy residual subtraction diverges to
# 169x||V||.  fp32 restores eig_min = +3.4e-6 and makes ridge 1e-3 the best
# setting (0.0039) rather than the divergent one.  End-to-end on RULER, 48 runs
# over two models x 32k/64k with both decoders solved online, fp32+1e-3 beats
# bf16+1e-3 by +0.183 / +0.260 / +0.117 / +0.167 mean score and matches or beats
# the old bf16+1e-2 default; the worst single cell recovers 0.160 -> 0.990.
#
# The ablation knob that produced those numbers (``--bf16-gram`` /
# ``BF16_GRAM=1``) was removed once the result was accepted, because leaving a
# switch that restores a divergence mode is a liability, not an option.  The raw
# runs are archived under ``results/ruler/gram_fix_l20{,_online}/``.
#
# Only the gram is promoted.  ``rhs = codes^T residual`` has bf16 *inputs*, so
# there is no precision there to recover, and rhs noise cannot make the system
# indefinite; promoting it too would double prefill cost for no measured gain.
# Nor is the encoder power step (:func:`..online._pm_covariance_step`) promoted:
# measured fp32/tf32/bf16 agree to 4-5 significant figures there with no
# consistent sign.  The discriminator is whether a matmul's output is *inverted*
# (promote) or merely re-orthonormalized and refit downstream (leave it).

# Tried and removed: rescaling each group's decoder by the exact line-search step
# ``alpha = <R, CD> / ||CD||^2``.  It was added to make the greedy cascade
# monotone, but the regularized solve already is -- it minimizes
# ``||R - CD||^2 / L + ridge ||D||^2``, for which ``D = 0`` is feasible, so
# ``||R - CD|| <= ||R||`` whenever the gram is positive definite.  The pre-fix
# divergence was the gram being *indefinite*, i.e. the returned ``D`` not being
# the minimizer at all; alpha never addressed that (on a bf16 gram it reaches
# only 0.4788 and is itself non-monotone, because its denominator reads
# ``||CD||^2`` off the same broken gram).  Measured with the fp32 gram at ridge
# 1e-3, alpha changes the final residual by less than 0.1% in 9 of 10
# layer x side cells and zero monotonicity violations occur without it; its one
# real effect was 5.95x on the rank-deficient layer-0 V block, which is alpha
# partially undoing the ridge shrinkage -- the same thing as lowering the ridge.
# End-to-end on RULER it was 0.000.

# Decode projects each query to signed int8 before packing its eight bitplanes.
PROJECTED_QUERY_BITS = 8


__all__ = [
    "ONLINE_DECODER_RIDGE",
    "ONLINE_GROUP_BITS",
    "ONLINE_MATMUL_DTYPE",
    "PROJECTED_QUERY_BITS",
    "SUPPORTED_CODE_BITS",
    "SUPPORTED_EXP_BITS",
]
