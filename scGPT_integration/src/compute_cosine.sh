#!/bin/bash

python Scripts/augment_monarch_train_with_scgpt.py \
    --dge_csv Integration/knockout_dge.csv \
    --monarch_dir ELs_for_Rotate/Monarch_KG \
    --output_dir ELs_for_Rotate/Monarch_KG_scGPT_k10_cos0 \
    --top_k_per_gene 10 \
    --min_cosine 0.5 \
    --skip_if_connected

python Scripts/augment_monarch_train_with_scgpt.py \
    --dge_csv Integration/knockout_dge.csv \
    --monarch_dir ELs_for_Rotate/Monarch_KG \
    --output_dir ELs_for_Rotate/Monarch_KG_scGPT_k10_cos06 \
    --top_k_per_gene 10 \
    --min_cosine 0.6 \
    --skip_if_connected

python Scripts/augment_monarch_train_with_scgpt.py \
    --dge_csv Integration/knockout_dge.csv \
    --monarch_dir ELs_for_Rotate/Monarch_KG \
    --output_dir ELs_for_Rotate/Monarch_KG_scGPT_k10_cos07 \
    --top_k_per_gene 10 \
    --min_cosine 0.7 \
    --skip_if_connected

python Scripts/augment_monarch_train_with_scgpt.py \
    --dge_csv Integration/knockout_dge.csv \
    --monarch_dir ELs_for_Rotate/Monarch_KG \
    --output_dir ELs_for_Rotate/Monarch_KG_scGPT_k10_cos08 \
    --top_k_per_gene 10 \
    --min_cosine 0.8 \
    --skip_if_connected