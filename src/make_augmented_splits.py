"""
Augment baseline train split with scGPT perturbation-similarity edges.

Reads existing baseline train/valid/test splits and augments the training
set with similarity edges derived from scGPT cosine similarity scores.
Maps gene symbols to HGNC IDs and filters to entities present in train only.

Output structure:
    output_dir/
        augmented/
            train.txt
            valid.txt
            test.txt

Example:
    python src/make_augmented_splits.py \
        --baseline_dir out/KG_split/MONDO_HGNC/scGPT_cosine_0.6/split_seed42 \
        --sim_csv out/cosine_sim/scGPT_cosine_0.6.csv \
        --output_dir out/KG_split/MONDO_HGNC/scGPT_cosine_0.6/split_seed42 \
        --seed 42
"""

from __future__ import annotations

import argparse
import os
from typing import Dict

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HGNC_PATH = os.path.join(REPO_ROOT, "data/MonarchKG", "HGNC_to_symbol.tsv")


def get_args():
    parser = argparse.ArgumentParser(description="Augment baseline train split with scGPT similarity edges.")

    parser.add_argument("--baseline_dir", required=True, help="Path to directory containing baseline splits")
    parser.add_argument("--sim_csv", required=True, help="Path to csv file with similarity data")
    parser.add_argument("--output_dir", required=True, help="Directory to write augmented splits")
    parser.add_argument("--relation", default="biolink:correlated_perturbation", help="Relation label for scGPT-derived edges")

    return parser.parse_args()


def load_symbol_to_hgnc() -> Dict[str, str]:
    df = pd.read_csv(HGNC_PATH, sep="\t", dtype=str)
    df = df.rename(columns={"HGNC ID": "hgnc_id", "Approved symbol": "symbol"})
    out: Dict[str, str] = {}
    for _, row in df.iterrows():
        sym = str(row["symbol"]).strip()
        hid = str(row["hgnc_id"]).strip()
        if sym and hid:
            out[sym.upper()] = hid
    return out


def load_sim_edges(sim_csv, train_entities, relation, sym_map):
    df = pd.read_csv(sim_csv)

    df["subject"] = df["gene_a"].str.upper().map(sym_map)
    df["object"] = df["gene_b"].str.upper().map(sym_map)

    df = df.dropna(subset=["subject", "object"])
    df = df[df["subject"].isin(train_entities) & df["object"].isin(train_entities)]
    df["predicate"] = relation

    return df[["subject", "predicate", "object"]]


def main() -> None:
    args = get_args()
    sym_map = load_symbol_to_hgnc()

    # Load baseline splits
    train = pd.read_csv(f"{args.baseline_dir}/train.txt", sep="\t", header=None, names=["subject", "predicate", "object"])
    valid = pd.read_csv(f"{args.baseline_dir}/valid.txt", sep="\t", header=None, names=["subject", "predicate", "object"])
    test  = pd.read_csv(f"{args.baseline_dir}/test.txt",  sep="\t", header=None, names=["subject", "predicate", "object"])

    # Augment train
    train_entities = set(train["subject"]).union(set(train["object"]))
    sim_edges = load_sim_edges(args.sim_csv, train_entities, args.relation, sym_map)
    augmented_train = pd.concat([train, sim_edges], ignore_index=True)

    # Save
    os.makedirs(f"{args.output_dir}/augmented", exist_ok=True)
    augmented_train.to_csv(f"{args.output_dir}/train.txt", sep="\t", index=False, header=False)
    valid.to_csv(f"{args.output_dir}/valid.txt",           sep="\t", index=False, header=False)
    test.to_csv(f"{args.output_dir}/test.txt",             sep="\t", index=False, header=False)


if __name__ == "__main__":
    main()