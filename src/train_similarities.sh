#!/bin/bash

SIM_BASE=$1
CONFIG=$2       
SAVE_BASE=$3    
DEVICE=$4
MODEL=$5      

REPO_DIR=$(dirname $(dirname $(readlink -f $0)))

for threshold_dir in ${SIM_BASE}/*/; do
    threshold=$(basename ${threshold_dir})
    echo "=== ${threshold} ==="
    bash ${REPO_DIR}/src/train_splits.sh \
        ${threshold_dir} \
        ${threshold} \
        ${CONFIG} \
        ${SAVE_BASE}/${MODEL}_${threshold} \
        ${DEVICE}
done

# DEBUG Uncomment this and delete lower part for full operation

# count=0
# for threshold_dir in ${SIM_BASE}/*/; do
#     [ $count -ge 2 ] && break
#     threshold=$(basename ${threshold_dir})
#     echo "=== ${threshold} ==="
#     bash ${REPO_DIR}/src/train_splits.sh \
#         ${threshold_dir} \
#         ${threshold} \
#         ${CONFIG} \
#         ${SAVE_BASE}/${MODEL}_${threshold} \
#         ${DEVICE}
#     count=$((count + 1))
# done