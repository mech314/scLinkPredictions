"""
Split filtered Monarch KG into baseline train/valid/test sets.

Reads filtered Monarch KG and splits edges into train/valid/test
using a given random seed.

Output structure:
    output_dir/
        baseline/
            train.txt
            valid.txt
            test.txt

Example:
    python src/make_baseline_splits.py \
        --filtered_kg out/filtered_KG.txt \
        --output_dir out/KG_split/MONDO_HGNC/split_seed42 \
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

def get_args():
    parser = argparse.ArgumentParser(description="Augment Monarch train with scGPT similarity edges and split data.")
    
    parser.add_argument(
        "--filtered_kg", 
        required=True, 
        help="Path to filtered KG file (.txt)"
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

def split_triples(df, valid_ratio, test_ratio, seed):
    train, temp = train_test_split(df, test_size=valid_ratio + test_ratio, random_state=seed)
    valid, test = train_test_split(temp, test_size=test_ratio / (valid_ratio + test_ratio), random_state=seed)
    return train, valid, test

def save_baseline_kg(train, valid, test, output):
    os.makedirs(f"{output}", exist_ok=True)
    train.to_csv(f"{output}/train.txt", sep="\t", index=False, header=False)
    valid.to_csv(f"{output}/valid.txt", sep="\t", index=False, header=False)
    test.to_csv(f"{output}/test.txt", sep="\t", index=False, header=False)

    return None


def main() -> None:

    args = get_args()

    # Process Monarch KG
    df = pd.read_csv(args.filtered_kg, sep="\t", header=None, names=["subject", "predicate", "object"])
    train, valid, test = split_triples(df, args.valid_ratio, args.test_ratio, args.seed)
    save_baseline_kg(train, valid, test, args.output_dir)

    
if __name__ == "__main__":
    main()
