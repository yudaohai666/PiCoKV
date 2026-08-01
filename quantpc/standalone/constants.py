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
