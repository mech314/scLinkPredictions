"""
Split filtered Monarch KG into train/valid/test and augment training set with
scGPT perturbation-similarity edges.

- Reads filtered Monarch KG (HGNC, MONDO, HP nodes only).
- Splits into train/valid/test with a given random seed.
- Maps scGPT gene symbols to HGNC and filters to train entities only.
- Writes baseline and augmented splits to output directory.

Output structure:
    output_dir/
        baseline/   train.txt valid.txt test.txt
        augmented/  train.txt valid.txt test.txt

Example:
    python src/make_splits.py \
        --filtered_kg out/filtered_KG.txt \
        --sim_csv out/cosine_sim/scGPT_cosine_0.6.csv \
        --output_dir out/split_seed42 \
        --seed 42
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Iterable, Set, Tuple

from sklearn.model_selection import train_test_split
import random

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HGNC_PATH = os.path.join(REPO_ROOT, "data/MonarchKG", "HGNC_to_symbol.tsv")

def get_args():
    parser = argparse.ArgumentParser(description="Augment Monarch train with scGPT similarity edges and split data.")
    
    parser.add_argument(
        "--filtered_kg", 
        required=True, 
        help="Path to filtered KG file (.txt)"
    )

    parser.add_argument(
        "--sim_csv",
        required=True,
        help="Path to csv file with similarity data"
    )

    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Seed"
    )

    parser.add_argument(
        "--valid_ratio",
        default=0.10,
        help="Validation set ratio"
    )

    parser.add_argument(
        "--test_ratio",
        default=0.10,
        help="Test set ratio"
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="New directory (created): augmented train + valid/test",
    )

    parser.add_argument(
        "--relation",
        default="biolink:correlated_perturbation",
        help="Relation label for scGPT-derived edges",
    )

    return parser.parse_args()

def load_sim_edges(sim_csv, train_entities, relation, sym_map):
    df = pd.read_csv(sim_csv)
    
    # map symbols to HGNC
    df["subject"] = df["gene_a"].str.upper().map(sym_map)
    df["object"] = df["gene_b"].str.upper().map(sym_map)
    
    # drop unmapped
    df = df.dropna(subset=["subject", "object"])
    
    # keep only pairs where both are in train
    df = df[df["subject"].isin(train_entities) & df["object"].isin(train_entities)]
    
    df["predicate"] = relation
    
    return df[["subject", "predicate", "object"]]

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


def split_triples(df, valid_ratio, test_ratio, seed):
    train, temp = train_test_split(df, test_size=valid_ratio + test_ratio, random_state=seed)
    valid, test = train_test_split(temp, test_size=test_ratio / (valid_ratio + test_ratio), random_state=seed)
    return train, valid, test

def save_baseline_kg(train, valid, test, output):
    os.makedirs(f"{output}/baseline", exist_ok=True)
    train.to_csv(f"{output}/baseline/train.txt", sep="\t", index=False, header=False)
    valid.to_csv(f"{output}/baseline/valid.txt", sep="\t", index=False, header=False)
    test.to_csv(f"{output}/baseline/test.txt", sep="\t", index=False, header=False)

    return None

def save_aug_kg(train, valid, test, output):
    os.makedirs(f"{output}/augmented", exist_ok=True)
    train.to_csv(f"{output}/augmented/train.txt", sep="\t", index=False, header=False)
    valid.to_csv(f"{output}/augmented/valid.txt", sep="\t", index=False, header=False)
    test.to_csv(f"{output}/augmented/test.txt", sep="\t", index=False, header=False)

    return None


def main() -> None:

    args = get_args()
    sym_map = load_symbol_to_hgnc()

    # Process Monarch KG
    df = pd.read_csv(args.filtered_kg, sep="\t", header=None, names=["subject", "predicate", "object"])
    train, valid, test = split_triples(df, args.valid_ratio, args.test_ratio, args.seed)
    save_baseline_kg(train, valid, test, args.output_dir)

    # Augment scGPT and Monarch
    train_entities = set(train["subject"]).union(set(train["object"]))
    sim_edges = load_sim_edges(args.sim_csv, train_entities, args.relation, sym_map)
    augmented_train = pd.concat([train, sim_edges], ignore_index=True)
    save_aug_kg(augmented_train, valid, test, args.output_dir)

    
if __name__ == "__main__":
    main()
