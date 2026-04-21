#!/usr/bin/env python3
"""
Compute and save pairwise similarity matrix from a DGE CSV.
Supported metrics: cosine, pearson, spearman
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def compute_cosine(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    valid = norms.ravel() > 1e-12
    Xn = np.zeros_like(X)
    Xn[valid] = X[valid] / norms[valid]
    S = Xn @ Xn.T
    np.fill_diagonal(S, 1.0)
    return S


def compute_pearson(X: np.ndarray) -> np.ndarray:
    return np.corrcoef(X)


def compute_spearman(X: np.ndarray) -> np.ndarray:
    # Convert each row to ranks
    ranks = np.apply_along_axis(
        lambda r: np.argsort(np.argsort(r)).astype(np.float64), axis=1, arr=X
    )
    return np.corrcoef(ranks)


METRICS = {
    "cosine": compute_cosine,
    "pearson": compute_pearson,
    "spearman": compute_spearman,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dge_csv", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True,
                   help="Output path: .csv or .npy")
    p.add_argument("--metric", choices=list(METRICS.keys()), default="pearson",
                   help="Similarity metric (default: pearson)")
    p.add_argument("--top_k", type=int, default=None,
                   help="Save top-k per row as sparse long-format CSV")
    p.add_argument("--min_sim", type=float, default=None,
                   help="Additional threshold filter on top of top_k")
    args = p.parse_args()

    print(f"Loading DGE from {args.dge_csv}...")
    dge = pd.read_csv(args.dge_csv, index_col=0)
    dge.index = dge.index.astype(str)
    print(f"Shape: {dge.shape}")

    X = dge.values.astype(np.float64)
    labels = dge.index.tolist()
    n = len(labels)

    print(f"Computing {args.metric} similarity ({n}x{n})...")
    S = METRICS[args.metric](X)
    print(f"Done. Range: [{S.min():.4f}, {S.max():.4f}]  mean: {S.mean():.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.top_k is not None:
        print(f"Saving top-{args.top_k} per gene (long format)...")
        S_work = S.copy()
        np.fill_diagonal(S_work, -np.inf)
        k = min(args.top_k, n - 1)
        rows = []
        for i, gene_a in enumerate(labels):
            row = S_work[i]
            ix = np.argpartition(-row, k - 1)[:k]
            ix = ix[np.argsort(-row[ix])]
            for j in ix:
                sim = float(row[j])
                if args.min_sim is not None and sim < args.min_sim:
                    continue
                rows.append((gene_a, labels[j], sim))
        out_df = pd.DataFrame(rows, columns=["gene_a", "gene_b", args.metric])
        out_path = args.output.with_suffix(".csv")
        out_df.to_csv(out_path, index=False)
        print(f"Saved {len(out_df):,} pairs to {out_path}")

    elif args.output.suffix == ".npy":
        np.save(args.output, S)
        idx_path = args.output.with_name(args.output.stem + "_index.txt")
        idx_path.write_text("\n".join(labels))
        print(f"Saved matrix to {args.output}, index to {idx_path}")

    else:
        sim_df = pd.DataFrame(S, index=labels, columns=labels)
        sim_df.to_csv(args.output)
        print(f"Saved {sim_df.shape} matrix to {args.output}")


if __name__ == "__main__":
    main()