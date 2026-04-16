#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python Scripts/train_monarch_augmented.py \
    --kg_dir ELs_for_Rotate/Monarch_KG_scGPT_k10_cos01 \
    --out_dir PyKeenOut/transe_k10_cos01 \
    --devices 1 \
    --device cuda \
    > logs_train_cos01.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 python Scripts/train_monarch_augmented.py \
    --kg_dir ELs_for_Rotate/Monarch_KG_scGPT_k10_cos02 \
    --out_dir PyKeenOut/transe_k10_cos02 \
    --devices 1 \
    --device cuda \
    > logs_train_cos02.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 python Scripts/train_monarch_augmented.py \
    --kg_dir ELs_for_Rotate/Monarch_KG_scGPT_k10_cos03 \
    --out_dir PyKeenOut/transe_k10_cos03 \
    --devices 1 \
    --device cuda \
    > logs_train_cos03.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 python Scripts/train_monarch_augmented.py \
    --kg_dir ELs_for_Rotate/Monarch_KG_scGPT_k10_cos04 \
    --out_dir PyKeenOut/transe_k10_cos04 \
    --devices 1 \
    --device cuda \
    > logs_train_cos04.txt 2>&1 &
wait

python Scripts/predict_all_novel_links.py \
    --model PyKeenOut/transe_k10_cos01/trained_model.pkl \
    --output Integration/preds_k10_cos01.tsv
python Scripts/predict_all_novel_links.py \
    --model PyKeenOut/transe_k10_cos02/trained_model.pkl \
    --output Integration/preds_k10_cos02.tsv
python Scripts/predict_all_novel_links.py \
    --model PyKeenOut/transe_k10_cos03/trained_model.pkl \
    --output Integration/preds_k10_cos03.tsv
python Scripts/predict_all_novel_links.py \
    --model PyKeenOut/transe_k10_cos04/trained_model.pkl \
    --output Integration/preds_k10_cos04.tsv