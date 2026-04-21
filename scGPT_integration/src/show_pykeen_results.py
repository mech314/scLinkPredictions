"""
Print link-prediction metrics from PyKEEN pipeline results.json (after training).

Fair comparison: same test triples; each model evaluated with its own training graph.

Examples:
  python Scripts/show_pykeen_results.py PyKeenOut/transe_monarch_scgpt/results.json
  python Scripts/show_pykeen_results.py --baseline path/to/baseline/results.json \\
      --new PyKeenOut/transe_monarch_scgpt/results.json
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Optional

KEYS = (
    "hits_at_1",
    "hits_at_3",
    "hits_at_10",
    "inverse_harmonic_mean_rank",
    "arithmetic_mean_rank",
    "count",
)


def get_block(data: Dict[str, Any], side: str, flavor: str) -> Optional[Dict[str, Any]]:
    m = data.get("metrics") or {}
    part = m.get(side)
    if not isinstance(part, dict):
        return None
    block = part.get(flavor)
    return block if isinstance(block, dict) else None


def print_block(label: str, block: Dict[str, Any]) -> None:
    print(f"\n=== {label} ===")
    for k in KEYS:
        if k in block:
            v = block[k]
            if isinstance(v, float):
                print(f"  {k}: {v:.6g}")
            else:
                print(f"  {k}: {v}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Show / compare PyKEEN results.json metrics.")
    ap.add_argument("results_json", nargs="?", help="Single results.json path")
    ap.add_argument("--baseline", help="First run, e.g. baseline without scGPT")
    ap.add_argument("--new", dest="new_path", help="Second run, e.g. scGPT-augmented training")
    ap.add_argument(
        "--side",
        choices=("both", "head", "tail"),
        default="both",
        help="both = aggregate of head and tail prediction (default)",
    )
    ap.add_argument(
        "--flavor",
        choices=("realistic", "optimistic", "pessimistic"),
        default="realistic",
    )
    args = ap.parse_args()

    def load_show(path: str, label: str) -> Optional[Dict[str, Any]]:
        with open(path) as f:
            data = json.load(f)
        block = get_block(data, args.side, args.flavor)
        if block is None:
            print(f"Missing metrics['{args.side}']['{args.flavor}'] in {path}")
            return None
        print_block(f"{label} ({os.path.basename(os.path.dirname(path))})", block)
        return block

    if args.baseline and args.new_path:
        b = load_show(args.baseline, "baseline")
        n = load_show(args.new_path, "new")
        if b and n:
            print(
                "\n=== delta (new - baseline) ===\n"
                "Better: higher hits@*, higher inverse_harmonic_mean_rank (MRR), lower arithmetic_mean_rank\n"
            )
            for k in KEYS:
                if k == "count":
                    continue
                if k in b and k in n and isinstance(b[k], (int, float)) and isinstance(n[k], (int, float)):
                    print(f"  {k}: {n[k] - b[k]:+.6g}")
    elif args.results_json:
        load_show(args.results_json, "run")
    else:
        ap.print_help()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
