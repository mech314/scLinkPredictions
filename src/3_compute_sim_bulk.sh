#!/bin/bash

SIM_THRESHOLD=(0.6 0.7 0.8 0.9 0.95 0.99)
INPUT_DGE=$1
OUTPUT=$2
METRIC=$3
TOP_K=$4

for x in "${SIM_THRESHOLD[@]}"; do
    python src/compute_sim.py \
    --dge_csv ${INPUT_DGE} \
    --output "out/${OUTPUT}_${METRIC}_${x}.csv" \
    --metric ${METRIC} \
    --min_sim ${x} \
    --top_k ${TOP_K}
done
