#!/bin/bash
# Compare General Tasks results and produce a comparison plot.
# Usage: bash compare_results.sh [output_dir]
# Example: bash compare_results.sh outputs/general_tasks/llama-3-8b

source /root/miniconda3/etc/profile.d/conda.sh
conda activate bpc
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

OUTPUT_DIR="${1:-outputs/general_tasks/llama-3-8b}"
OUTPUT_IMAGE="${OUTPUT_DIR}/comparison.png"

python -m benchmark.general_tasks.compare_results \
    "${OUTPUT_DIR}" \
    "${OUTPUT_IMAGE}"

echo "对比图已保存至: ${OUTPUT_IMAGE}"
