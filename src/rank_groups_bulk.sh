#!/bin/bash

MODELS_DIR=$1      
SPLITS_DIR=$2      
DATASET=$3         
OUTPUT_DIR=$4      
GROUPS_DIR="data/BOB_disease_groups"
RELATION="biolink:gene_associated_with_condition"

mkdir -p ${OUTPUT_DIR}

declare -A GROUP_CONFIG
GROUP_CONFIG["300_mondo_ids.txt"]="RareDisease:head:HGNC:"
GROUP_CONFIG["290_non_ultra_rare.mondo_ids.tsv"]="NonUltraRare:head:HGNC:"
GROUP_CONFIG["random_300_diseases.txt"]="RandomDiseases:head:HGNC:"
GROUP_CONFIG["cancer_genes.txt"]="Cancer:tail:MONDO:"
GROUP_CONFIG["pediatric_cancer_genes.txt"]="PedCancer:tail:MONDO:"
GROUP_CONFIG["female_differentially_expressed_genes.txt"]="Female:tail:MONDO:"
GROUP_CONFIG["male_differentially_expressed_genes.txt"]="Male:tail:MONDO:"
GROUP_CONFIG["random_500_genes.txt"]="RandomGenes:tail:MONDO:"

for split_dir in ${SPLITS_DIR}/*/; do
    split_name=$(basename ${split_dir})
    model="${MODELS_DIR}/${split_name}/${DATASET}/trained_model.pkl"
    train="${split_dir}/${DATASET}/train.txt"
    valid="${split_dir}/${DATASET}/valid.txt"
    test="${split_dir}/${DATASET}/test.txt"

    for group_file in "${!GROUP_CONFIG[@]}"; do
        IFS=':' read -r label target prefix <<< "${GROUP_CONFIG[$group_file]}"
        output="${OUTPUT_DIR}/${split_name}_${label}.tsv"

        python src/rank_groups.py \
            --a_terms ${GROUPS_DIR}/${group_file} \
            --a_label ${label} \
            --relation ${RELATION} \
            --prediction_target ${target} \
            --prediction_prefix ${prefix} \
            --train_triples ${train} \
            --validation_triples ${valid} \
            --test_triples ${test} \
            --model ${model} \
            --output ${output} \
            --device cuda:1
    done
done