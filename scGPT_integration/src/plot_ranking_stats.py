"""
Plot raking histogram and compute 1-MNR
"""

import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser(description='Run the test-omatic experiment')
    parser.add_argument('--ranking_tsv', type=str, required=True, help='tsv with ranking data')
    parser.add_argument('--output', required=True, type=str, help='Folder to save plot')
    parser.add_argument('--name', required=True, help='Group name')

    return parser.parse_args()


def main():

    args = get_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.ranking_tsv, sep="\t")

    mnr = 1 - df["rank"].median()
    print(f"1-MNR: {mnr:.4f}  (n={len(df)})")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["rank"], bins=20, color="#4c72b0", edgecolor="none")
    ax.axvline(df["rank"].median(), color="red", lw=1.5, ls="--", label=f"median {df['rank'].median():.3f}")
    ax.set_xlabel("normalized rank (higher = better)")
    ax.set_ylabel("count")
    ax.set_title(f"RareDisease — 1-MNR = {mnr:.4f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{args.output}/{args.name}_ranking.png", dpi=150)

if __name__ == '__main__':
    main()