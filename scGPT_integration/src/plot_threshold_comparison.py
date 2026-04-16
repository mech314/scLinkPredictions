#!/usr/bin/env python3
"""
Plot model comparison across threshold runs from PyKEEN results.json files.

Example:
  python Integration/src/plot_threshold_comparison.py \
    --run cos0.1=PyKeenOut/transe_k10_cos01/results.json \
    --run cos0.2=PyKeenOut/transe_k10_cos02/results.json \
    --run cos0.3=PyKeenOut/transe_k10_cos03/results.json \
    --run cos0.4=PyKeenOut/transe_k10_cos04/results.json \
    --output_png Integration/model_comparison_k10_thresholds.png \
    --output_tsv Integration/model_comparison_k10_thresholds.tsv
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_run(item: str) -> Tuple[str, str]:
    if "=" not in item:
        raise ValueError(f"Invalid --run value {item!r}; expected label=path")
    label, path = item.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise ValueError(f"Invalid --run value {item!r}; expected label=path")
    return label, path


def load_metric_block(path: str, side: str, flavor: str) -> Dict[str, float]:
    with open(path) as f:
        data = json.load(f)
    return data["metrics"][side][flavor]


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare threshold runs from PyKEEN results.json.")
    ap.add_argument(
        "--run",
        action="append",
        required=True,
        help="Repeatable. Format: label=path_to_results.json",
    )
    ap.add_argument("--side", choices=("both", "head", "tail"), default="both")
    ap.add_argument("--flavor", choices=("realistic", "optimistic", "pessimistic"), default="realistic")
    ap.add_argument("--title", default="Monarch + scGPT threshold sweep")
    ap.add_argument("--output_png", default="Integration/model_comparison_k10_thresholds.png")
    ap.add_argument("--output_tsv", default="Integration/model_comparison_k10_thresholds.tsv")
    args = ap.parse_args()

    runs: List[Tuple[str, str]] = [parse_run(x) for x in args.run]
    hits_metrics = [
        ("hits_at_1", "Hits@1"),
        ("hits_at_3", "Hits@3"),
        ("hits_at_10", "Hits@10"),
    ]
    mrr_metric = ("inverse_harmonic_mean_rank", "MRR-like")
    amr_metric = ("arithmetic_mean_rank", "AMR")

    values: Dict[str, Dict[str, float]] = {}
    for label, path in runs:
        block = load_metric_block(path, args.side, args.flavor)
        values[label] = {
            key: float(block[key])
            for key in [k for k, _ in hits_metrics] + [mrr_metric[0], amr_metric[0]]
        }

    x = np.arange(len(hits_metrics))
    n = len(runs)
    width = 0.8 / max(n, 1)
    palette = ["#4a6572", "#5f7c89", "#00a896", "#4cc9b0", "#2b3a42", "#7fb3a3"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))

    # Panel 1: Hits@k grouped bars
    for i, (label, _) in enumerate(runs):
        offset = (i - (n - 1) / 2) * width
        y = [values[label][k] for k, _ in hits_metrics]
        axes[0].bar(
            x + offset,
            y,
            width,
            label=label,
            color=palette[i % len(palette)],
            edgecolor="white",
            linewidth=0.5,
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([name for _, name in hits_metrics])
    axes[0].set_ylabel("Score (higher is better)")
    axes[0].set_title("Hits@k (filtered)")
    axes[0].grid(axis="y", linestyle=":", alpha=0.35)
    axes[0].legend(frameon=False, fontsize=9)

    # Panel 2: MRR-like (inverse harmonic mean rank)
    mrr_vals = [values[label][mrr_metric[0]] for label, _ in runs]
    axes[1].bar(
        np.arange(n),
        mrr_vals,
        color=[palette[i % len(palette)] for i in range(n)],
        edgecolor="white",
        linewidth=0.5,
    )
    axes[1].set_xticks(np.arange(n))
    axes[1].set_xticklabels([label for label, _ in runs], rotation=20, ha="right")
    axes[1].set_ylabel("Score (higher is better)")
    axes[1].set_title("Inverse harmonic mean rank (MRR-style)")
    axes[1].grid(axis="y", linestyle=":", alpha=0.35)

    # Panel 3: Arithmetic mean rank
    amr_vals = [values[label][amr_metric[0]] for label, _ in runs]
    axes[2].bar(
        np.arange(n),
        amr_vals,
        color=[palette[i % len(palette)] for i in range(n)],
        edgecolor="white",
        linewidth=0.5,
    )
    axes[2].set_xticks(np.arange(n))
    axes[2].set_xticklabels([label for label, _ in runs], rotation=20, ha="right")
    axes[2].set_ylabel("Rank (lower is better)")
    axes[2].set_title("Arithmetic mean rank")
    axes[2].grid(axis="y", linestyle=":", alpha=0.35)
    axes[2].text(
        0.5,
        0.02,
        "Lower is better",
        transform=axes[2].transAxes,
        fontsize=9,
        color="#555",
        ha="center",
    )

    fig.suptitle(f"{args.title} ({args.side} / {args.flavor})", fontsize=12, y=1.02)

    out_png = os.path.abspath(args.output_png)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for label, _ in runs:
        row = {"run": label}
        for key, metric_name in hits_metrics:
            row[metric_name] = values[label][key]
        row[mrr_metric[1]] = values[label][mrr_metric[0]]
        row[amr_metric[1]] = values[label][amr_metric[0]]
        rows.append(row)
    out_df = pd.DataFrame(rows)
    out_tsv = os.path.abspath(args.output_tsv)
    os.makedirs(os.path.dirname(out_tsv) or ".", exist_ok=True)
    out_df.to_csv(out_tsv, sep="\t", index=False)

    print(f"Wrote plot: {out_png}")
    print(f"Wrote table: {out_tsv}")


if __name__ == "__main__":
    main()
