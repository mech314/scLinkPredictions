#!/usr/bin/env python
"""
Dump sidecar files for scgpt_pickle_to_dge.py:
  <pkl_stem>_genes.txt
  <pkl_stem>_control_mean.npy

Run in the same scGPT/GEARS env that produced multi_gpu_predictions.pkl.
Gene order matches what model.pred_perturb returns:
pert_data.adata.var["gene_name"], i.e. pert_data.gene_names.
"""
import argparse
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from gears import PertData


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions_pkl", type=Path, required=True,
                   help="Path to multi_gpu_predictions.pkl (used only for stem).")
    p.add_argument("--data_name", default="adamson")
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--ctrl_label", default="ctrl",
                   help="Value in adata.obs['condition'] that marks control cells.")
    args = p.parse_args()

    pert_data = PertData(args.data_dir)
    pert_data.load(data_name=args.data_name)
    adata = pert_data.adata

    genes = adata.var["gene_name"].tolist()
    n = len(genes)
    print(f"n_genes = {n}")

    ctrl_mask = adata.obs["condition"] == args.ctrl_label
    n_ctrl = int(ctrl_mask.sum())
    if n_ctrl == 0:
        raise SystemExit(
            f"No control cells found with condition == {args.ctrl_label!r}. "
            f"Available: {adata.obs['condition'].unique().tolist()[:10]}"
        )
    print(f"n_ctrl_cells = {n_ctrl}")

    X = adata[ctrl_mask].X
    if sp.issparse(X):
        ctrl_mean = np.asarray(X.mean(axis=0)).ravel()
    else:
        ctrl_mean = np.asarray(X).mean(axis=0).ravel()
    assert ctrl_mean.shape[0] == n, (ctrl_mean.shape, n)

    pkl = args.predictions_pkl.resolve()
    stem = pkl.with_suffix("")
    genes_path = Path(str(stem) + "_genes.txt")
    ctrl_path = Path(str(stem) + "_control_mean.npy")

    genes_path.write_text("\n".join(genes) + "\n")
    np.save(ctrl_path, ctrl_mean.astype(np.float64))
    print(f"wrote {genes_path}")
    print(f"wrote {ctrl_path}  (shape={ctrl_mean.shape}, dtype={ctrl_mean.dtype})")


if __name__ == "__main__":
    main()