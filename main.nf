#!/usr/bin/env nextflow

nextflow.enable.dsl=2

process FILTER_MONARCH {
    publishDir "${params.filterOut}", mode: 'copy'

    output:
    path "${params.graph}_KG.txt", emit: filtered_kg

    script:
    """
    python ${projectDir}/src/filter_monarch.py \
        --input ${params.monarch_kg} \
        --nodes ${params.nodes2filter} \
        --output ${params.graph}_KG.txt
    """

}

process MAKE_BASELINE_SPLITS {
    publishDir "${params.baselineSplitsOut}", mode: 'copy'

    input:
    path filtered_kg

    output:
    path "split_seed*", emit: baseline_out

    script:
    """
    bash ${projectDir}/src/make_splits_bulk.sh \
        ${params.sim_dir} \
        ${filtered_kg} \
        ${params.seeds} \
        . \
        baseline
    """
}

process MAKE_AUGMENTED_SPLITS {
    publishDir "${params.augmentedSplitOut}", mode: 'copy'

    input:
    val ready

    output:
    path "scGPT_cosine*/**", emit: augmented_out

    script:
    """
    bash ${projectDir}/src/make_splits_bulk.sh \
        ${params.sim_dir} \
        ${params.baselineSplitsOut} \
        ${params.seeds} \
        . \
        augmented
    """
}

process HPO {
    publishDir "${params.pykeenOut}/${params.study_name}", mode: 'copy'

    input:
    val baseline_splits
    val augmented_splits

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
    FILTER_MONARCH()
    MAKE_BASELINE_SPLITS(FILTER_MONARCH.out.filtered_kg)
    MAKE_AUGMENTED_SPLITS(MAKE_BASELINE_SPLITS.out.baseline_out.collect())
    HPO(
        MAKE_BASELINE_SPLITS.out.baseline_out,
        MAKE_AUGMENTED_SPLITS.out.augmented_out
    )
    TRAIN_MODELS(HPO.out.hpo_out.collect())

}