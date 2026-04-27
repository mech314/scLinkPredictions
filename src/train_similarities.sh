#!/bin/bash

SPLITS_BASE=$1
DATASET=$2     
CONFIG=$3       
SAVE_BASE=$4    
DEVICE=$5       

for threshold_dir in ${SPLITS_BASE}/scGPT_*/; do
    threshold=$(basename ${threshold_dir})
    echo "=== ${threshold} ==="
    bash src/train_splits.sh \
        ${threshold_dir} \
        ${DATASET} \
        ${CONFIG} \
        ${SAVE_BASE}/transe_${threshold} \
        ${DEVICE}
done