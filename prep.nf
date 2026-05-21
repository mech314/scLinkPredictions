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
    path "*_cosine*/**", emit: augmented_out

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

workflow { 
    FILTER_MONARCH()
    MAKE_BASELINE_SPLITS(FILTER_MONARCH.out.filtered_kg)
    MAKE_AUGMENTED_SPLITS(MAKE_BASELINE_SPLITS.out.baseline_out.collect())
}