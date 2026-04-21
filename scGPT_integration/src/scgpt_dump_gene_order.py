#!/usr/bin/env python3
"""
Write gene column order + control mean as sidecars next to multi_gpu_predictions.pkl.

Run inside the same environment you used for scGPT (needs gears / PertData).

Example:
  python Scripts/scgpt_dump_gene_order.py \\
    --predictions_pkl scGPT_data/multi_gpu_predictions.pkl \\
    --data_root ./data \\
    --data_name adamson \\
    --split simulation

Creates:
  scGPT_data/multi_gpu_predictions_genes.txt
  scGPT_data/multi_gpu_predictions_control_mean.npy

Then scgpt_pickle_to_dge.py can omit --gene_names and --control_mean.
"""
import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions_pkl", type=Path, required=True, help="Your multi_gpu_predictions.pkl path")
    ap.add_argument("--data_root", type=Path, default=Path("./data"))
    ap.add_argument("--data_name", default="adamson")
    ap.add_argument("--split", default="simulation")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    try:
        from gears import PertData
    except ImportError as e:
        raise SystemExit(
            "Install/run this in the scGPT environment (pip install gears or project env). " + str(e)
        )

    pkl = args.predictions_pkl.resolve()
    stem = pkl.with_suffix("")
    genes_out = Path(str(stem) + "_genes.txt")
    ctrl_out = Path(str(stem) + "_control_mean.npy")

    pert_data = PertData(str(args.data_root))
    pert_data.load(data_name=args.data_name)
    pert_data.prepare_split(split=args.split, seed=args.seed)

    adata = pert_data.adata
    genes = adata.var["gene_name"].astype(str).tolist()
    with open(genes_out, "w") as f:
        for g in genes:
            f.write(g + "\n")

    ctrl_adata = adata[adata.obs["condition"] == "ctrl"]
    x = ctrl_adata.X
    ctrl_expr = x.toarray() if hasattr(x, "toarray") else x
    control_mean = np.asarray(np.mean(ctrl_expr, axis=0), dtype=np.float64).ravel()
    if control_mean.shape[0] != len(genes):
        raise SystemExit(f"control_mean len {control_mean.shape[0]} != n_genes {len(genes)}")
    np.save(ctrl_out, control_mean)

    print(f"Wrote {len(genes)} genes -> {genes_out}")
    print(f"Wrote control mean shape {control_mean.shape} -> {ctrl_out}")
    print("\nNext:")
    print(
        f"  python Scripts/scgpt_pickle_to_dge.py "
        f"--predictions_pkl {pkl} --output_csv /path/to/dge.csv"
    )


if __name__ == "__main__":
    main()
