#!/usr/bin/env nextflow

nextflow.enable.dsl=2

process HPO {
    publishDir "${params.pykeenOut}/${params.study_name}", mode: 'copy'

    output: 
    path "best_pipeline", emit: hpo_out
    path "logs", emit: hpo_logs

    script:
    """
    mkdir -p logs
    CUDA_VISIBLE_DEVICES=${params.device} \
    python ${projectDir}/src/optimize_and_train_model.py \
        --model ${params.model} --id ${params.graph} \
        --test ${params.baselineSplitsDir}/test.txt \
        --train ${params.baselineSplitsDir}/train.txt \
        --valid ${params.baselineSplitsDir}/valid.txt > logs/${params.study_name}.log 2>&1
    """
}

process TRAIN_MODELS {
    publishDir "${params.pykeenOut}/${params.study_name}", mode: 'copy'

    input:
    val hpo_out

    output:
    path "*/**", emit: train_out

    script:
    """
    bash ${projectDir}/src/train_similarities.sh \
        ${params.augmentedSplitOut} \
        ${params.pykeenOut}/${params.study_name}/best_pipeline/pipeline_config.json \
        . \
        cuda:${params.device} \
        ${params.model}
    """

}

workflow { 
    HPO()
    TRAIN_MODELS(HPO.out.hpo_out.collect())

}