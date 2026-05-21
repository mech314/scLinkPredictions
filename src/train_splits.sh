#!/bin/bash

SPLITS_PATH=$1
DATASET=$2
CONFIG=$3
SAVE_PATH=$4
DEVICE=$5

REPO_DIR=$(dirname $(dirname $(readlink -f $0)))

mkdir -p ${SAVE_PATH}

for split_dir in ${SPLITS_PATH}/*/; do
    split_name=$(basename ${split_dir})
    train="${split_dir}/train.txt"
    test="${split_dir}/test.txt"
    valid="${split_dir}/valid.txt"

    echo "Training ${split_name} ${DATASET}..."
    python ${REPO_DIR}/src/train_model_fromConfig.py \
        --config ${CONFIG} \
        --train ${train} \
        --test ${test} \
        --valid ${valid} \
        --save_path ${SAVE_PATH}/models/${split_name}/ \
        --device ${DEVICE} \
        --seed 314
done


# DEBUG Uncomment this and delete lower part for full operation

# count=0
# for split_dir in ${SPLITS_PATH}/*/; do
#     [ $count -ge 1 ] && break
#     split_name=$(basename ${split_dir})
#     train="${split_dir}/train.txt"
#     test="${split_dir}/test.txt"
#     valid="${split_dir}/valid.txt"

#     echo "Training ${split_name} ${DATASET}..."
#     python ${REPO_DIR}/src/train_model_fromConfig.py \
#         --config ${CONFIG} \
#         --train ${train} \
#         --test ${test} \
#         --valid ${valid} \
#         --save_path ${SAVE_PATH}/models/${split_name}/ \
#         --device ${DEVICE} \
#         --seed 314
#     count=$((count + 1))
# done