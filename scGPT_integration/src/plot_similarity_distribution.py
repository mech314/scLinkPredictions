#!/usr/bin/env python3
"""
Plot cosine similarity distribution from long-format CSV (gene_a, gene_b, cosine).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True, help="Long-format CSV with cosine column")
    p.add_argument("--out", type=Path, default=Path("cosine_dist.png"))
    p.add_argument("--bins", type=int, default=100)
    args = p.parse_args()

    df = pd.read_csv(args.csv, index_col=0)
    mat = df.values.astype(np.float64)
    idx = np.triu_indices(mat.shape[0], k=1)
    vals = mat[idx]
    print(f"NxN matrix {mat.shape[0]}x{mat.shape[0]}, upper triangle pairs: {len(vals):,}")
    print(f"Pairs loaded: {len(vals):,}")
    print(f"min={vals.min():.4f}  max={vals.max():.4f}  mean={vals.mean():.4f}  median={np.median(vals):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Full distribution
    ax = axes[0]
    ax.hist(vals, bins=args.bins, color="#4c72b0", edgecolor="none")
    ax.axvline(np.median(vals), color="red",    lw=1.2, ls="--", label=f"median {np.median(vals):.3f}")
    ax.axvline(vals.mean(),      color="orange", lw=1.2, ls="--", label=f"mean {vals.mean():.3f}")
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("pair count")
    ax.set_title("Full distribution")
    ax.legend(fontsize=9)

    # Tail zoom: top 10%
    ax2 = axes[1]
    thresh = np.percentile(vals, 90)
    tail = vals[vals >= thresh]
    ax2.hist(tail, bins=args.bins // 2, color="#dd8452", edgecolor="none")
    ax2.axvline(thresh, color="gray", lw=1, ls=":", label=f"p90 = {thresh:.3f}")
    ax2.set_xlabel("cosine similarity")
    ax2.set_title(f"Top 10% tail  (>= {thresh:.3f},  n={len(tail):,})")
    ax2.legend(fontsize=9)

    fig.suptitle(f"Cosine similarity distribution — {args.csv.name}", fontsize=12)
    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.out}")

    # Percentile table
    print("\nPercentiles:")
    for pct in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  p{pct:>2}: {np.percentile(vals, pct):.4f}")

if __name__ == "__main__":
    main()