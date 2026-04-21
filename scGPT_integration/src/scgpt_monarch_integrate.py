#!/usr/bin/env python3
"""
Combine scGPT perturbation profiles with Monarch KG link-prediction scores.

1) Find genes whose knockout DGE profile is most similar to a query gene (cosine similarity),
   or process *all* genes at once (--all_queries): top-k neighbors per gene.
2) Score (query_gene, relation, neighbor_gene) triples with a trained PyKEEN model on Monarch.

Prerequisites:
- DGE matrix: rows = perturbation keys (usually gene symbols), columns = genes.

Examples:
  # One query
  python Scripts/scgpt_monarch_integrate.py --dge_csv Integration/knockout_dge.csv \\
    --query TP53 --top_k_neighbors 75 --output Integration/TP53.tsv

  # Every gene as query (long output: N * top_k rows)
  python Scripts/scgpt_monarch_integrate.py --dge_csv Integration/knockout_dge.csv \\
    --all_queries --top_k_neighbors 50 --max_queries 200 --output Integration/all_pairs.tsv
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from pykeen.datasets.base import PathDataset
from pykeen.predict import predict_triples

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from monarch_paths import monarch_triple_paths

TRAIN_PATH = monarch_triple_paths()[0]


class MonarchKG(PathDataset):
    def __init__(self, **kwargs):
        train, test, valid = monarch_triple_paths()
        super().__init__(
            training_path=train,
            testing_path=test,
            validation_path=valid,
            **kwargs,
        )


def load_train_hgnc_gene_triples(path: str) -> Set[Tuple[str, str, str]]:
    s: Set[Tuple[str, str, str]] = set()
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            h, r, t = parts[0], parts[1], parts[2]
            if h.startswith("HGNC:") and t.startswith("HGNC:"):
                s.add((h, r, t))
    return s


def load_train_gene_disease_triples(path: str) -> Set[Tuple[str, str, str]]:
    s: Set[Tuple[str, str, str]] = set()
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            h, r, t = parts[0], parts[1], parts[2]
            if h.startswith("HGNC:") and t.startswith("MONDO:"):
                s.add((h, r, t))
    return s


def load_symbol_to_hgnc() -> Dict[str, str]:
    p = os.path.join(REPO_ROOT, "Resources", "HGNC_to_symbol.tsv")
    df = pd.read_csv(p, sep="\t", dtype=str)
    df = df.rename(columns={"HGNC ID": "hgnc_id", "Approved symbol": "symbol"})
    out: Dict[str, str] = {}
    for _, row in df.iterrows():
        sym = str(row["symbol"]).strip()
        hid = str(row["hgnc_id"]).strip()
        if sym and hid:
            out[sym.upper()] = hid
    return out


def resolve_pert_key_to_symbol(pert_key: str) -> str:
    if "_" in pert_key:
        return pert_key.split("_")[0]
    return pert_key


def cosine_similarity_to_query(dge: pd.DataFrame, query_row: str) -> pd.Series:
    if query_row not in dge.index:
        raise ValueError(
            f"Query {query_row!r} not in DGE index ({len(dge)} rows). "
            f"Sample index: {list(dge.index[:5])}"
        )
    q = dge.loc[query_row].values.astype(np.float64)
    qn = np.linalg.norm(q)
    if qn < 1e-12:
        raise ValueError("Query DGE vector has near-zero norm; cannot compute cosine similarity.")

    M = dge.values.astype(np.float64)
    norms = np.linalg.norm(M, axis=1)
    dots = M @ q
    sims = dots / (norms * qn + 1e-12)
    return pd.Series(sims, index=dge.index)


def pairwise_cosine_topk(
    dge: pd.DataFrame, top_k: int, max_queries: int
) -> List[Tuple[str, Optional[str], str, str, Optional[str], float]]:
    """
    For each row (query), return top_k neighbors by cosine similarity.
    Returns list of (query_pert, query_hgnc|None, neighbor_pert, n_sym_raw, n_hgnc|None, cos_sim).
    """
    X = dge.values.astype(np.float64)
    n_rows, _ = X.shape
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    valid = norms.ravel() > 1e-12
    Xn = np.zeros_like(X)
    Xn[valid] = X[valid] / norms[valid]

    print(f"Computing full pairwise cosine similarity ({n_rows} x {n_rows})...")
    S = Xn @ Xn.T
    np.fill_diagonal(S, -np.inf)

    labels = dge.index.astype(str).tolist()
    sym_map = load_symbol_to_hgnc()
    k = min(top_k, n_rows - 1)
    nq = n_rows if max_queries <= 0 else min(max_queries, n_rows)

    out: List[Tuple[str, Optional[str], str, str, Optional[str], float]] = []
    for i in range(nq):
        row = S[i]
        ix = np.argpartition(-row, k - 1)[:k]
        ix = ix[np.argsort(-row[ix])]
        q_key = labels[i]
        q_sym = resolve_pert_key_to_symbol(q_key).upper()
        q_hgnc = sym_map.get(q_sym)
        for j in ix:
            sim = float(row[j])
            if not np.isfinite(sim):
                continue
            n_key = labels[j]
            n_sym_raw = resolve_pert_key_to_symbol(n_key)
            n_sym = n_sym_raw.upper()
            n_hgnc = sym_map.get(n_sym)
            out.append((q_key, q_hgnc, n_key, n_sym_raw, n_hgnc, sim))
    return out


def score_triple_batch(
    model,
    factory,
    triples: List[List[str]],
) -> Optional[pd.DataFrame]:
    if not triples:
        return None
    pack = predict_triples(model=model, triples=triples, triples_factory=factory)
    return pack.process(factory=factory).df


def score_triple_chunked(
    model,
    factory,
    triples: List[List[str]],
    chunk_size: int,
) -> List[Optional[float]]:
    """Return list of scores aligned with triples (None if chunk failed)."""
    scores: List[Optional[float]] = []
    for start in range(0, len(triples), chunk_size):
        chunk = triples[start : start + chunk_size]
        df = score_triple_batch(model, factory, chunk)
        if df is None or len(df) != len(chunk):
            scores.extend([None] * len(chunk))
            continue
        for j in range(len(chunk)):
            scores.append(_score_column(df, j))
    return scores


def _score_column(df: pd.DataFrame, row_index: int) -> Optional[float]:
    row = df.iloc[row_index]
    for col in ("score", "scores"):
        if col in row.index:
            v = row[col]
            if v is not None and pd.notna(v):
                return float(v)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Perturbation similarity + Monarch link scores.")
    ap.add_argument("--dge_csv", required=True, help="DGE matrix: index=pert keys, columns=gene symbols")
    ap.add_argument(
        "--query",
        default="",
        help="Single query row label (e.g. TP53). Ignored if --all_queries.",
    )
    ap.add_argument(
        "--all_queries",
        action="store_true",
        help="Run every row as query; keep top_k_neighbors per gene (large output).",
    )
    ap.add_argument(
        "--max_queries",
        type=int,
        default=0,
        help="With --all_queries, only first N rows as queries (0 = all).",
    )
    ap.add_argument("--top_k_neighbors", type=int, default=100)
    ap.add_argument(
        "--relations",
        default="biolink:interacts_with",
        help="Comma-separated Monarch relations between two genes (e.g. interacts_with).",
    )
    ap.add_argument(
        "--mondo",
        default="",
        help="If set, also score (neighbor, biolink:gene_associated_with_condition, mondo) per row.",
    )
    ap.add_argument(
        "--model",
        default=os.path.join(REPO_ROOT, "Models", "transe_monarch.trained_model.pkl"),
        help="Trained PyKEEN model pickle (must match Monarch_KG training).",
    )
    ap.add_argument("--output", required=True, help="Output TSV path")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0")
    ap.add_argument(
        "--monarch_batch_size",
        type=int,
        default=4096,
        help="Triples per predict_triples call (memory vs speed).",
    )
    args = ap.parse_args()

    relations = [r.strip() for r in args.relations.split(",") if r.strip()]
    if any("gene_associated_with_condition" in r for r in relations):
        raise SystemExit(
            "Do not pass gene_associated_with_condition in --relations; use --mondo MONDO:..."
        )

    if args.device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(args.device)

    print("Loading DGE matrix...")
    dge = pd.read_csv(args.dge_csv, index_col=0)
    dge.index = dge.index.astype(str)

    sym_map = load_symbol_to_hgnc()

    # Build neighbor list: (q_key, q_hgnc, n_key, n_sym_raw, n_hgnc, cos_sim)
    flat: List[Tuple[str, Optional[str], str, str, Optional[str], float]] = []
    if args.all_queries:
        flat = pairwise_cosine_topk(dge, args.top_k_neighbors, args.max_queries)
        print(f"Collected {len(flat)} query–neighbor pairs.")
    else:
        if not args.query.strip():
            raise SystemExit("Pass --query GENE or use --all_queries.")
        print("Computing cosine similarity for single query...")
        sims = cosine_similarity_to_query(dge, args.query)
        sims = sims.drop(index=args.query, errors="ignore")
        sims = sims.sort_values(ascending=False).head(args.top_k_neighbors)
        q_sym = resolve_pert_key_to_symbol(args.query).upper()
        q_hgnc = sym_map.get(q_sym)
        if not q_hgnc:
            raise SystemExit(
                f"Could not map query {args.query!r} (symbol {q_sym!r}) to HGNC. "
                "Check HGNC_to_symbol.tsv."
            )
        for pert_key, cos_sim in sims.items():
            n_sym_raw = resolve_pert_key_to_symbol(str(pert_key))
            n_sym = n_sym_raw.upper()
            n_hgnc = sym_map.get(n_sym)
            flat.append((args.query, q_hgnc, str(pert_key), n_sym_raw, n_hgnc, float(cos_sim)))

    print("Loading Monarch KG + model...")
    data = MonarchKG()
    model = torch.load(args.model, map_location=dev, weights_only=False)
    model.eval()

    train_gene_triples = load_train_hgnc_gene_triples(TRAIN_PATH)
    train_gd_triples = load_train_gene_disease_triples(TRAIN_PATH)
    g2p_rel = "biolink:gene_associated_with_condition"
    mondo_id = args.mondo.strip()

    n_pairs = len(flat)
    all_scores: Dict[str, List[Optional[float]]] = {rel: [None] * n_pairs for rel in relations}
    all_in_train: Dict[str, List[bool]] = {rel: [False] * n_pairs for rel in relations}

    for rel in relations:
        triples: List[List[str]] = []
        pos: List[int] = []
        for i, (qk, qh, _, _, nh, _) in enumerate(flat):
            if not qh or not nh:
                continue
            triples.append([qh, rel, nh])
            pos.append(i)
        if not triples:
            continue
        print(f"Scoring {len(triples)} triples for {rel}...")
        scored = score_triple_chunked(model, data.training, triples, args.monarch_batch_size)
        for j, i in enumerate(pos):
            all_scores[rel][i] = scored[j]
            qh, nh = flat[i][1], flat[i][4]
            assert qh is not None and nh is not None
            all_in_train[rel][i] = (qh, rel, nh) in train_gene_triples

    g2p_scores: List[Optional[float]] = [None] * n_pairs
    g2p_in_train: List[bool] = [False] * n_pairs
    if mondo_id:
        triples = []
        pos = []
        for i, (_, _, _, _, nh, _) in enumerate(flat):
            if not nh:
                continue
            triples.append([nh, g2p_rel, mondo_id])
            pos.append(i)
        if triples:
            print(f"Scoring {len(triples)} gene–disease triples...")
            scored = score_triple_chunked(model, data.training, triples, args.monarch_batch_size)
            for j, i in enumerate(pos):
                g2p_scores[i] = scored[j]
                nh = flat[i][4]
                assert nh is not None
                g2p_in_train[i] = (nh, g2p_rel, mondo_id) in train_gd_triples

    out_rows: List[dict] = []
    for i, (q_key, q_hgnc, n_key, n_sym_raw, n_hgnc, cos_sim) in enumerate(flat):
        row: dict = {
            "query_pert_key": q_key,
            "query_hgnc": q_hgnc or "",
            "neighbor_pert_key": n_key,
            "neighbor_gene_symbol": n_sym_raw,
            "neighbor_hgnc": n_hgnc or "",
            "cosine_similarity": cos_sim,
        }
        for rel in relations:
            safe = rel.replace(":", "_").replace("-", "_")
            row[f"monarch_score__{safe}"] = all_scores[rel][i]
            row[f"monarch_in_train__{safe}"] = all_in_train[rel][i]
        if mondo_id:
            row["mondo_tail"] = mondo_id
            row["monarch_score__gene_associated_with_condition"] = g2p_scores[i]
            row["monarch_in_train__gene_associated_with_condition"] = g2p_in_train[i]
        out_rows.append(row)

    out_df = pd.DataFrame(out_rows)
    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out_df.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote {len(out_df)} rows to {out_path}")


if __name__ == "__main__":
    main()
