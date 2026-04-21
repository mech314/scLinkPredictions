#!/usr/bin/env python3
"""Inspect multi_gpu_predictions.pkl structure (keys, vector shapes).

The pickle does *not* store which gene each vector dimension corresponds to.
Dict keys are perturbation (row) IDs only; column order matches adata.var
from the PertData run, not key order or alphabetical order.
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("predictions_pkl", type=Path)
    p.add_argument("--sample_keys", type=int, default=15)
    args = p.parse_args()

    pkl = args.predictions_pkl.resolve()
    stem = pkl.with_suffix("")

    print(
        "Note: column gene names are not inside the pickle; "
        "use a sidecar file or export from PertData (see Scripts/scgpt_pickle_to_dge.py).\n"
    )

    sidecar_txt = Path(str(stem) + "_genes.txt")
    sidecar_meta = pkl.with_suffix(".meta.json")
    for path, label in (
        (sidecar_txt, f"{pkl.name} sidecar ({sidecar_txt.name})"),
        (sidecar_meta, f"{pkl.name} metadata ({sidecar_meta.name})"),
    ):
        if path.is_file():
            if path.suffix == ".json":
                with open(path) as jf:
                    meta = json.load(jf)
                gn = meta.get("gene_names")
                if isinstance(gn, list):
                    print(f"Found {label}: {len(gn)} gene_names in JSON")
                else:
                    print(f"Found {label}: (no gene_names list)")
            else:
                nlines = sum(1 for _ in open(path) if _.strip())
                print(f"Found {label}: {nlines} lines")
        else:
            print(f"Missing {label}")

    with open(args.predictions_pkl, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        print(f"Top-level type: {type(data)} (expected dict)")
        return

    print(f"N perturbations: {len(data)}")
    keys = list(data.keys())
    print(f"Sample keys: {keys[: args.sample_keys]}")

    lengths = {}
    for k, v in data.items():
        n = int(np.asarray(v).size)
        lengths[n] = lengths.get(n, 0) + 1
    print("Vector length -> count:")
    for n, c in sorted(lengths.items()):
        print(f"  {n}: {c}")

    if keys:
        v0 = np.asarray(data[keys[0]]).ravel()
        print(f"First vector: min={v0.min():.4g} max={v0.max():.4g} mean={v0.mean():.4g}")


if __name__ == "__main__":
    main()
