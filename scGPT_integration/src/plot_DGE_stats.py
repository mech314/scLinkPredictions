#!/usr/bin/env python3
"""
Quick EDA for a DGE matrix CSV (perturbations x genes).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--top_n", type=int, default=20, help="Top N perturbed genes to show")
    p.add_argument("--out_dir", type=Path, default=Path("dge_eda"))
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading CSV...")
    dge = pd.read_csv(args.csv, index_col=0)
    print(f"Shape: {dge.shape}  ({dge.shape[0]} perturbations x {dge.shape[1]} genes)")

    vals = dge.values.ravel()

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig)

    # 1. Global DGE value distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(vals, bins=200, color="#4c72b0", edgecolor="none")
    ax1.axvline(0, color="red", lw=1, ls="--")
    ax1.set_title("Global DGE distribution")
    ax1.set_xlabel("predicted expr - ctrl mean")
    ax1.set_ylabel("count")

    # 2. Per-perturbation mean absolute DGE (how strong is each perturbation)
    ax2 = fig.add_subplot(gs[0, 1])
    mean_abs = dge.abs().mean(axis=1).sort_values(ascending=False)
    mean_abs.head(args.top_n).plot(kind="bar", ax=ax2, color="#dd8452")
    ax2.set_title(f"Top {args.top_n} perturbations by mean |DGE|")
    ax2.set_ylabel("mean |delta|")
    ax2.tick_params(axis="x", labelsize=7)

    # 3. Per-gene variance across perturbations (most variable genes)
    ax3 = fig.add_subplot(gs[0, 2])
    gene_var = dge.var(axis=0).sort_values(ascending=False)
    gene_var.head(args.top_n).plot(kind="bar", ax=ax3, color="#55a868")
    ax3.set_title(f"Top {args.top_n} most variable genes across perturbations")
    ax3.set_ylabel("variance")
    ax3.tick_params(axis="x", labelsize=7)

    # 4. Distribution of per-perturbation mean DGE (bias check — should be ~0)
    ax4 = fig.add_subplot(gs[1, 0])
    per_pert_mean = dge.mean(axis=1)
    ax4.hist(per_pert_mean, bins=50, color="#c44e52", edgecolor="none")
    ax4.axvline(0, color="black", lw=1, ls="--")
    ax4.set_title("Per-perturbation mean DGE\n(bias check, should center at 0)")
    ax4.set_xlabel("mean delta")

    # 5. % genes significantly changed per perturbation (|DGE| > threshold)
    ax5 = fig.add_subplot(gs[1, 1])
    thresh = np.percentile(np.abs(vals), 95)  # data-driven threshold
    frac_sig = (dge.abs() > thresh).mean(axis=1).sort_values(ascending=False)
    frac_sig.head(args.top_n).plot(kind="bar", ax=ax5, color="#8172b2")
    ax5.set_title(f"Top {args.top_n}: fraction genes |DGE| > p95 ({thresh:.3f})")
    ax5.set_ylabel("fraction of genes")
    ax5.tick_params(axis="x", labelsize=7)

    # 6. Heatmap of top perturbations x top variable genes
    ax6 = fig.add_subplot(gs[1, 2])
    top_perts = mean_abs.head(30).index
    top_genes = gene_var.head(30).index
    sub = dge.loc[top_perts, top_genes]
    vmax = np.percentile(np.abs(sub.values), 98)
    im = ax6.imshow(sub.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax6.set_title("Heatmap: top 30 perts x top 30 genes")
    ax6.set_xticks(range(len(top_genes)))
    ax6.set_xticklabels(top_genes, rotation=90, fontsize=5)
    ax6.set_yticks(range(len(top_perts)))
    ax6.set_yticklabels(top_perts, fontsize=6)
    plt.colorbar(im, ax=ax6, shrink=0.6)

    fig.suptitle(f"DGE EDA — {args.csv.name}", fontsize=13)
    plt.tight_layout()
    out = args.out_dir / "dge_eda.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")

    # Print summary stats to terminal
    print(f"\n--- Summary ---")
    print(f"DGE range: [{vals.min():.4f}, {vals.max():.4f}]")
    print(f"Global mean: {vals.mean():.4f},  std: {vals.std():.4f}")
    print(f"Most impactful perturbation: {mean_abs.index[0]}  (mean |DGE| = {mean_abs.iloc[0]:.4f})")
    print(f"Most variable gene: {gene_var.index[0]}  (var = {gene_var.iloc[0]:.4f})")

if __name__ == "__main__":
    main()