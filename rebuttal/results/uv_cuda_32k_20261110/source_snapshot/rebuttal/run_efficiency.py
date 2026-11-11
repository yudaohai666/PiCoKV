"""Reuse the original efficiency runner with an isolated decode replacement."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--decode-backend", choices=("packed", "reconstruct", "reconstruct_cuda", "unpacked", "unpacked_kernel", "unpacked_fused"), default="reconstruct",
        help="rebuttal decode implementation (default: reconstruct); other args go to the original runner",
    )
    options, forwarded = parser.parse_known_args()
    if "--help" in forwarded or "-h" in forwarded:
        parser.print_help()
        print("\nOriginal efficiency runner arguments:")

    from benchmarks.Efficiency import run_efficiency as original
    from rebuttal.reconstruct_attention import reconstructed_decode

    if not any(arg == "--output" or arg.startswith("--output=") for arg in forwarded):
        forwarded += ["--output", str(REPO_ROOT / "rebuttal" / "results" / f"picokv_{options.decode_backend}.json")]

    original_run, original_tag, original_argv = original.run, original.output_tag, sys.argv

    def output_tag(*args, **kwargs):
        tag = original_tag(*args, **kwargs)
        suffix = {"packed": "", "reconstruct": "_reconstruct_bf16", "reconstruct_cuda": "_reconstruct_cuda_bf16", "unpacked": "_unpacked_bf16", "unpacked_kernel": "_unpacked_kernel_bf16", "unpacked_fused": "_unpacked_fused_bf16"}
        return tag + suffix[options.decode_backend]

    def run(args):
        if args.method != "picokv":
            raise ValueError("this rebuttal entry point requires --method picokv")
        if args.use_pruning:
            raise ValueError("disable --pruning for the packed/reconstruct comparison")
        expected_tag = output_tag(
            args.method, args.k_bits, args.v_bits, args.num_bits,
            offline=(args.offline or args.offline_proj_path is not None),
        )
        path = Path(args.output)
        if path.exists():
            previous = json.loads(path.read_text())
            if previous.get("tag") != expected_tag:
                raise ValueError(f"{path} contains a different variant; choose another --output")
        context = reconstructed_decode(options.decode_backend) if options.decode_backend != "packed" else nullcontext()
        with context:
            original_run(args)

    original.run, original.output_tag = run, output_tag
    sys.argv = [str(Path(__file__).resolve()), *forwarded]
    try:
        original.main()
    finally:
        original.run, original.output_tag, sys.argv = original_run, original_tag, original_argv


if __name__ == "__main__":
    main()
