#!/bin/bash
# Download offline projection calibration datasets (namespace-Pt/long-llm-data)
#
# pg19
#   J. W. Rae, A. Potapenko, S. M. Jayakumar, C. Hillier, and T. P. Lillicrap.
#   Compressive transformers for long-range sequence modelling. In ICLR, 2020.
#
# CodeParrot
#   Z. Peitian. Huggingface dataset: namespace-pt/long-llm-data, 2024.
#
# ProofPile
#   Z. Azerbayev, E. Ayers, and B. Piotrowski. Github repository:
#   hoskison-center/proof-pile, 2022.
set -e
cd "$(dirname "$0")"
mkdir -p data

BASE="https://hf-mirror.com/datasets/namespace-Pt/long-llm-data/resolve/main/lm"

for name in codeparrot pg19 proof-pile; do
    wget -O "data/${name}.json" "${BASE}/${name}.json"
done
