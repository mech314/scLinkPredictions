#!/bin/bash

seeds=(42 314 123 678 9484)
SIM_FOLDER=$1
FILTERED_KG=$2

SPLIT_FOLDER="out/KG_split"

if [[ ! -d ${SPLIT_FOLDER} ]]; then
    mkdir -p ${SPLIT_FOLDER}
fi


for f in ${SIM_FOLDER}/*.csv; do
    fname=$(basename ${f} .csv) 
    for seed in "${seeds[@]}"; do
        python src/make_splits.py \
        --filtered_kg ${FILTERED_KG} \
        --sim_csv ${f} \
        --output_dir ${SPLIT_FOLDER}/${fname}/split_seed${seed} \
        --seed ${seed}
    done
done
