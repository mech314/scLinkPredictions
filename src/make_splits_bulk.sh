#!/bin/bash

REPO_DIR=$(dirname $(dirname $(readlink -f $0)))

SIM_FOLDER=$1
INPUT=$2
SEEDS=$(echo $3 | tr ',' ' ')
OUTDIR=$4
SPLIT=$5

if [[ "${SPLIT}" == "baseline" ]]; then
    for seed in ${SEEDS}; do
        python ${REPO_DIR}/src/make_baseline_splits.py \
            --filtered_kg ${INPUT} \
            --output_dir ${OUTDIR}/split_seed${seed} \
            --seed ${seed}
    done
else
    for f in ${SIM_FOLDER}/*.csv; do
        fname=$(basename ${f} .csv)
        for seed in ${SEEDS}; do
            python ${REPO_DIR}/src/make_augmented_splits.py \
                --baseline_dir ${INPUT}/split_seed${seed} \
                --sim_csv ${f} \
                --output_dir ${OUTDIR}/${fname}/split_seed${seed}
        done
    done
fi