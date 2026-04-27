#!/bin/bash

SPLITS_PATH=$1
DATASET=$2
CONFIG=$3
SAVE_PATH=$4
DEVICE=$5

mkdir -p ${SAVE_PATH}

for split_dir in ${SPLITS_PATH}/*/; do
    split_name=$(basename ${split_dir})
    train="${split_dir}/${DATASET}/train.txt"
    test="${split_dir}/${DATASET}/test.txt"
    valid="${split_dir}/${DATASET}/valid.txt"

    echo "Training ${split_name} ${DATASET}..."
    python src/train_model_fromConfig.py \
        --config ${CONFIG} \
        --train ${train} \
        --test ${test} \
        --valid ${valid} \
        --save_path ${SAVE_PATH}/models/${split_name}/${DATASET} \
        --device ${DEVICE} \
        --seed 314
done