#!/usr/bin/env python3
"""
PCA of DGE matrix with explained variance and 2D scatter.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dge_csv", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, default=Path("pca_out"))
    p.add_argument("--n_components", type=int, default=50)
    p.add_argument("--top_label", type=int, default=15,
                   help="Label top-N most active perturbations on scatter")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    dge = pd.read_csv(args.dge_csv, index_col=0)
    labels = dge.index.astype(str).tolist()
    activity = dge.abs().mean(axis=1)

    X = StandardScaler().fit_transform(dge.values)
    pca = PCA(n_components=min(args.n_components, X.shape[0], X.shape[1]))
    X_pca = pca.fit_transform(X)

    evr = pca.explained_variance_ratio_
    print(f"PC1-5 explained variance: {evr[:5].round(3)}")
    print(f"Cumulative top-10: {evr[:10].sum():.3f}")
    print(f"Cumulative top-50: {evr.sum():.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # 1. Scree plot
    ax = axes[0]
    ax.bar(range(1, len(evr) + 1), evr * 100, color="#4c72b0", width=0.8)
    ax.plot(range(1, len(evr) + 1), np.cumsum(evr) * 100,
            color="red", lw=1.5, marker="o", markersize=3, label="cumulative")
    ax.axhline(80, color="gray", lw=1, ls="--", label="80%")
    ax.set_xlabel("PC")
    ax.set_ylabel("explained variance %")
    ax.set_title("Scree plot")
    ax.legend(fontsize=9)

    # 2. PC1 vs PC2 scatter, colored by activity
    ax2 = axes[1]
    sc = ax2.scatter(X_pca[:, 0], X_pca[:, 1],
                     c=activity.values, cmap="YlOrRd", s=20, alpha=0.8)
    plt.colorbar(sc, ax=ax2, label="mean |DGE|")
    top_idx = activity.nlargest(args.top_label).index
    for name in top_idx:
        i = labels.index(name)
        ax2.annotate(name, (X_pca[i, 0], X_pca[i, 1]),
                     fontsize=6, ha="center", va="bottom",
                     xytext=(0, 4), textcoords="offset points")
    ax2.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
    ax2.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    ax2.set_title("PC1 vs PC2")

    # 3. PC2 vs PC3
    ax3 = axes[2]
    sc2 = ax3.scatter(X_pca[:, 1], X_pca[:, 2],
                      c=activity.values, cmap="YlOrRd", s=20, alpha=0.8)
    plt.colorbar(sc2, ax=ax3, label="mean |DGE|")
    for name in top_idx:
        i = labels.index(name)
        ax3.annotate(name, (X_pca[i, 1], X_pca[i, 2]),
                     fontsize=6, ha="center", va="bottom",
                     xytext=(0, 4), textcoords="offset points")
    ax3.set_xlabel(f"PC2 ({evr[1]*100:.1f}%)")
    ax3.set_ylabel(f"PC3 ({evr[2]*100:.1f}%)")
    ax3.set_title("PC2 vs PC3")

    fig.suptitle(f"PCA of DGE — {args.dge_csv.name}", fontsize=12)
    plt.tight_layout()
    out = args.out_dir / "dge_pca.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")

    # Save PCA coords for downstream similarity
    pca_df = pd.DataFrame(X_pca, index=labels,
                          columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])
    pca_csv = args.out_dir / "dge_pca_coords.csv"
    pca_df.to_csv(pca_csv)
    print(f"PCA coords saved: {pca_csv}  shape: {pca_df.shape}")


if __name__ == "__main__":
    main()