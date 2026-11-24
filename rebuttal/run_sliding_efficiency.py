"""Efficiency runner with PiCoKV offline incremental decode (periodic slide).

Wraps ``benchmarks.Efficiency.run_efficiency`` unchanged; ``--slide-tokens N``
only sets ``StandaloneConfig.slide_tokens`` for ``--method picokv`` and adds a
``_slide{N}`` suffix to its tag.  Every other argument is forwarded.

    python rebuttal/run_sliding_efficiency.py --slide-tokens 64 --method picokv --offline ...
"""

from __future__ import annotations

import argparse
import functools
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--slide-tokens", type=int, default=64,
                        help="picokv: slide every N decode steps (0 = static cache)")
    options, forwarded = parser.parse_known_args()

    import picokv.picokv_standalone as standalone
    from benchmarks.Efficiency import run_efficiency as original

    original_config, original_tag, original_argv = standalone.StandaloneConfig, original.output_tag, sys.argv

    def output_tag(method, *args, **kwargs):
        tag = original_tag(method, *args, **kwargs)
        return f"{tag}_slide{options.slide_tokens}" if method == "picokv" and options.slide_tokens else tag

    # load_model imports StandaloneConfig from the package at call time.
    standalone.StandaloneConfig = functools.partial(original_config, slide_tokens=options.slide_tokens)
    original.output_tag = output_tag
    sys.argv = [str(Path(__file__).resolve()), *forwarded]
    try:
        original.main()
    finally:
        standalone.StandaloneConfig, original.output_tag, sys.argv = original_config, original_tag, original_argv


if __name__ == "__main__":
    main()
