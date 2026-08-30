"""Build the QuantPC CUDA extensions for use at model-load time.

Run this once after installing the Python/CUDA environment:

    python quantpc/build_cuda_extensions.py

The generated shared objects are written below ``build/`` (which is ignored by
git).  Runtime loaders prefer these artifacts and only fall back to the legacy
torch.utils.cpp_extension.load JIT path when they are absent.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

import torch
from torch.utils.cpp_extension import load


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "build" / "quantpc_extensions"


def _set_arch_list() -> None:
    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA device is visible. Set TORCH_CUDA_ARCH_LIST explicitly, "
            "for example TORCH_CUDA_ARCH_LIST=8.6."
        )
    major, minor = torch.cuda.get_device_capability()
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    print(f"[build] TORCH_CUDA_ARCH_LIST={major}.{minor}", flush=True)


def _build(name: str, source_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = [source_dir / "cuda_kernel.cu", source_dir / "torch_extension.cpp"]
    print(f"[build] {name}", flush=True)
    started = time.perf_counter()
    load(
        name,
        [str(path) for path in sources],
        build_directory=str(output_dir),
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        extra_ldflags=["-O3"],
        verbose=True,
    )
    print(f"[build] {name} done in {(time.perf_counter() - started):.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    _set_arch_list()
    _build(
        "_quanpc_prefill_cuda",
        REPO_ROOT / "quantpc" / "quanpc_prefill_cuda_extension",
        args.output_dir,
    )
    _build(
        "_quanpc_cuda",
        REPO_ROOT / "quantpc" / "quanpc_cuda_extension",
        args.output_dir,
    )


if __name__ == "__main__":
    main()
