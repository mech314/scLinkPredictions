"""
Build a DGE matrix CSV from multi_gpu_predictions.pkl without GEARS.

Requires:
- Gene names in the same order as each prediction vector (one per line).
- Control mean vector (same length), from numpy .npy or one-row CSV.

Export genes once from scGPT, e.g.:
  adata.var['gene_name'].to_csv('genes.txt', index=False, header=False)

Export control mean (Adamson ctrl mean over cells × genes), e.g.:
  np.save('control_mean.npy', control_mean)

Then run this script and use the DGE CSV with scgpt_monarch_integrate.py.

Sidecar convention (optional): next to ``multi_gpu_predictions.pkl`` use
``multi_gpu_predictions_genes.txt`` and ``multi_gpu_predictions_control_mean.npy``
— then you can omit --gene_names / --control_mean.
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions_pkl", type=Path, required=True)
    p.add_argument(
        "--gene_names",
        type=Path,
        default=None,
        help="One gene per line (column order). Default: <pkl_stem>_genes.txt if present.",
    )
    p.add_argument(
        "--control_mean",
        type=Path,
        default=None,
        help=".npy 1d or one-row .csv. Default: <pkl_stem>_control_mean.npy if present.",
    )
    p.add_argument("--output_csv", type=Path, required=True)
    args = p.parse_args()

    pkl = args.predictions_pkl.resolve()
    stem_path = Path(str(pkl.with_suffix("")) + "_genes.txt")
    ctrl_path = Path(str(pkl.with_suffix("")) + "_control_mean.npy")

    gene_path = args.gene_names or stem_path
    if not gene_path.is_file():
        raise SystemExit(
            f"Gene list not found: {gene_path}\n"
            "The pickle does not contain column gene names. Add sidecar "
            f"{stem_path.name} (see scgpt_dump_gene_order.py) or pass --gene_names."
        )

    ctrl_arg = args.control_mean
    if ctrl_arg is None and ctrl_path.is_file():
        ctrl_arg = ctrl_path
    if ctrl_arg is None or not ctrl_arg.is_file():
        raise SystemExit(
            f"Control mean not found. Pass --control_mean or create {ctrl_path.name} "
            "(same gene order as genes file)."
        )

    with open(args.predictions_pkl, "rb") as f:
        preds = pickle.load(f)
    if not isinstance(preds, dict) or not preds:
        raise SystemExit("Invalid or empty predictions dict")

    genes = [line.strip() for line in open(gene_path) if line.strip()]
    n = len(genes)

    if ctrl_arg.suffix.lower() == ".npy":
        ctrl = np.load(ctrl_arg).astype(np.float64).ravel()
    else:
        ctrl = pd.read_csv(ctrl_arg, header=None).values.astype(np.float64).ravel()
    if ctrl.shape[0] != n:
        raise SystemExit(f"control_mean length {ctrl.shape[0]} != n_genes {n}")

    rows, idx = [], []
    for k, v in preds.items():
        vec = np.asarray(v, dtype=np.float64).ravel()
        if vec.shape[0] != n:
            raise SystemExit(f"Key {k!r}: vector len {vec.shape[0]} != n_genes {n}")
        idx.append(k)
        rows.append(vec - ctrl)

    dge = pd.DataFrame(np.vstack(rows), index=idx, columns=genes)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    dge.to_csv(args.output_csv)
    print(f"Wrote {dge.shape} DGE matrix to {args.output_csv}")


if __name__ == "__main__":
    main()
