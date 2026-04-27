#!/bin/bash

SPLITS_BASE=$1
MODELS_BASE=$2
DATASET=$3
OUTPUT_BASE=$4

for threshold_dir in ${SPLITS_BASE}/scGPT_*/; do
    threshold=$(basename ${threshold_dir})
    models_dir="${MODELS_BASE}/transe_${threshold}/models"
    output_dir="${OUTPUT_BASE}/transe_${threshold}"

    echo "=== ${threshold} ==="
    bash src/rank_groups_bulk.sh \
        ${models_dir} \
        ${threshold_dir} \
        ${DATASET} \
        ${output_dir} &
done

wait
echo "All done"