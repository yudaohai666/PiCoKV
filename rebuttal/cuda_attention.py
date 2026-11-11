"""Loader for the two handwritten fused attention bodies (existing backends)."""
from pathlib import Path

_EXTENSION = None


def load_attention_extension():
    global _EXTENSION
    if _EXTENSION is None:
        from torch.utils.cpp_extension import load
        root = Path(__file__).resolve().parent / "cuda"
        _EXTENSION = load(
            name="picokv_rebuttal_bf16_attention",
            sources=[str(root / "attention.cpp"), str(root / "attention.cu")],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--ptxas-options=-v"],
        )
    return _EXTENSION
