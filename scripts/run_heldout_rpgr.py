#!/usr/bin/env python3
"""
run_heldout_rpgr.py
=====================
RPGR on the held-out TCGA benchmark (test partition, n=1578): class-macro
Recall@1/3/5/10, MRR@10 and mAP@10 for I2T and T2I. KEEP backbone, Morphology-only
reports (stored as `strict`), subtype (OncotreeCode) labels, exact pair included.
The AI4SKIN benchmark (five-fold protocol) requires non-public data and is not covered.

Usage:
  python3 run_heldout_rpgr.py --config ../configs/paths.yaml --backbone KEEP --scenario strict
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rpgr.reranking import global_similarity, prototype_similarity, rpgr_score  # noqa: E402
from rpgr.metrics import evaluate_direction, evaluate_direction_tie_neutral, class_level_table, macro_from_class_level  # noqa: E402


def l2_normalize_rows(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.clip(norms, 1e-12, None)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--backbone", default="KEEP")
    p.add_argument("--scenario", default="strict", choices=["original", "strict", "diagnosis_only"])
    p.add_argument("--granularity", default="OncotreeCode", choices=["OncotreeCode", "project_id"])
    p.add_argument("--split", default="test")
    p.add_argument("--tie-break", default="stable", choices=["stable", "neutral"],
                   help="stable: tied candidates keep gallery order; neutral: average over random orders of tied candidates")
    p.add_argument("--n-tie-seeds", type=int, default=50)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    bb_cfg = cfg["backbones"][args.backbone]

    split_df = pd.read_csv(cfg["cohort"]["split_csv"])
    eval_ids = split_df[split_df.split == args.split]["slide_id"].tolist()
    print(f"{args.split}: n={len(eval_ids)}")

    cache = np.load(bb_cfg["pool_cache_path"], allow_pickle=True)
    idx_of = {cid: i for i, cid in enumerate(cache["case_ids"])}
    mean_pool = l2_normalize_rows(cache["mean_pool"].astype(np.float64)[[idx_of[c] for c in eval_ids]])

    text_dir = Path(bb_cfg["text_embeddings"][args.scenario])
    first = np.load(text_dir / f"{eval_ids[0]}.npy").astype(np.float64).reshape(-1)
    text_mat = np.empty((len(eval_ids), first.shape[0]))
    for i, cid in enumerate(eval_ids):
        v = np.load(text_dir / f"{cid}.npy").astype(np.float64).reshape(-1)
        text_mat[i] = v / max(np.linalg.norm(v), 1e-12)

    proto_dir = Path(bb_cfg["prototypes_dir"])
    prototype_banks = [
        l2_normalize_rows(np.load(proto_dir / f"{cid}_prototype.npz")["prototypes"].astype(np.float64))
        for cid in eval_ids
    ]

    labels_df = pd.read_csv(cfg["cohort"]["labels_csv"]).set_index("slide_id")
    label_map = labels_df.loc[eval_ids, args.granularity].to_dict()

    sim_global = global_similarity(mean_pool, text_mat)
    sim_proto = prototype_similarity(prototype_banks, text_mat)

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for direction in ["I2T", "T2I"]:
        G, P = (sim_global, sim_proto) if direction == "I2T" else (sim_global.T, sim_proto.T)
        final_scores = rpgr_score(G, P)  # whole gallery re-ranked, no initial pool
        if args.tie_break == "neutral":
            df_q, summary = evaluate_direction_tie_neutral(final_scores, eval_ids, label_map, n_seeds=args.n_tie_seeds)
        else:
            df_q, summary = evaluate_direction(final_scores, eval_ids, label_map, exclude_exact_pair=False)
        macro = macro_from_class_level(class_level_table(df_q))
        row = {"tie_break": args.tie_break, "backbone": args.backbone, "scenario": args.scenario, "granularity": args.granularity,
               "direction": direction, "method": "RPGR", "n_cases": summary["n_eligible_queries"]}
        row.update({m.replace("Recall", "R"): v for m, v in macro.items()})
        rows.append(row)
        print(f"\n--- {direction} ---")
        for m, v in row.items():
            if m not in ("tie_break", "backbone", "scenario", "granularity", "direction", "method", "n_cases"):
                print(f"  {m:8s} {v:.4f}")

    out_path = out_dir / f"heldout_rpgr_{args.backbone}_{args.scenario}_{args.granularity}_{args.split}_{args.tie_break}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nwritten {out_path}")


if __name__ == "__main__":
    main()
