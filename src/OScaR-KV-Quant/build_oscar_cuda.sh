#!/bin/bash
# Build OScaR's `oscar_cuda` CUDA extension IN-PLACE for the QuantPC venv, so that
# `run_efficiency.py --method oscar` (via benchmarks/common._load_oscar_model) can
# `import oscar` / `import oscar_cuda` from this copied tree.
#
# oscar_interface.py does a hard `import oscar_cuda` (no JIT fallback) — the .so is
# an ahead-of-time CUDAExtension built by setup.py. `build_ext --inplace` compiles
# it and drops oscar_cuda.<pyabi>.so right here in this dir, which common.py already
# puts on sys.path. We use the venv's existing torch (no reinstall, no build
# isolation), so it stays pinned at torch 2.6.0+cu124.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$(cd "${SCRIPT_DIR}/../.." && pwd)/.venv"

if [[ ! -f "${VENV}/bin/activate" ]]; then
  echo "[build] ERROR: QuantPC venv not found at ${VENV}" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "[build] ERROR: nvcc not on PATH. Set CUDA_HOME/PATH (e.g. /usr/local/cuda) first." >&2
  exit 1
fi

cd "${SCRIPT_DIR}"
echo "[build] python : $(python -c 'import sys;print(sys.version.split()[0])')"
echo "[build] torch  : $(python -c 'import torch;print(torch.__version__)')"
echo "[build] nvcc   : $(nvcc --version | grep -i release || true)"

# Clean stale artifacts (any python-abi tag) so we don't import a mismatched .so.
rm -rf oscar.egg-info build/ dist/ oscar_cuda.*.so

# Compile the extension in-place (needs the env's torch -> --no build isolation).
python setup.py build_ext --inplace

python - <<'PY'
import oscar_cuda, oscar
print("[build] OK ->", oscar_cuda.__file__)
PY
echo "[build] done."
