#!/bin/bash

source /root/miniconda3/etc/profile.d/conda.sh
conda activate bpc
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

cd outputs/longbench_v2
ln -sf . results
python ../../benchmark/longbench_v2/result.py
rm -f results
