#!/bin/bash

LOGS='logs'
BASELINE_SPLIT=$1

if [[ ! -d ${LOGS} ]]; then
    mkdir -p logs
fi

CUDA_VISIBLE_DEVICES=0 \
    python src/optimize_and_train_model.py \
     --model rotate --id hpo_baseline \
     --test ${BASELINE_SPLIT}/test.txt \
     --train ${BASELINE_SPLIT}/train.txt \
     --valid ${BASELINE_SPLIT}/valid.txt > logs/rotate.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 \
    python src/optimize_and_train_model.py \
     --model transe --id hpo_baseline \
     --test ${BASELINE_SPLIT}/test.txt \
     --train ${BASELINE_SPLIT}/train.txt \
     --valid ${BASELINE_SPLIT}/valid.txt > logs/transe.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 \
    python src/optimize_and_train_model.py \
     --model complex --id hpo_baseline \
     --test ${BASELINE_SPLIT}/test.txt \
     --train ${BASELINE_SPLIT}/train.txt \
     --valid ${BASELINE_SPLIT}/valid.txt > logs/complex.log 2>&1 &

wait