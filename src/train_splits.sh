#!/bin/bash

SPLITS_PATH=$1
DATASET=$2
CONFIG=$3
SAVE_PATH=$4

mkdir -p ${SAVE_PATH}

for dir1 in ${SPLITS_PATH}/*/; do
    split_name=$(basename ${dir1})
    train="${dir1}/${DATASET}/train.txt"
    test="${dir1}/${DATASET}/test.txt"
    valid="${dir1}/${DATASET}/valid.txt"
    python src/train_model_fromConfig.py \
        --config ${CONFIG} \
        --train ${train} \
        --test ${test} \
        --valid ${valid} \
        --save_path ${SAVE_PATH}/${split_name}/${DATASET} \
        --device cuda:1 \
        --seed 314
done