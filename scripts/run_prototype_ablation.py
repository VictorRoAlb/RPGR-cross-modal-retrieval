#!/usr/bin/env python3
"""
run_prototype_ablation.py
===========================
Reproduces Table 4 (contribution of local prototypes): for each backbone,
computes S_B (Eq. 26, reciprocal global calibration without prototype
evidence) and the complete RPGR score, and reports mAP@10 for both plus
their difference, I2T and T2I, on the VAL split.

This is a descriptive ablation only: point estimates (S_B, RPGR, delta).
No confidence intervals, no hypothesis testing, no bootstrap, no
permutation test, no Holm-Bonferroni correction, no p-values.

Reads only from the paths given in --config. Writes only under output_dir.
Uses exclusively rpgr.reranking and rpgr.metrics.

Usage:
  python3 run_prototype_ablation.py --config ../configs/paths.yaml --scenario strict
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rpgr.reranking import (  # noqa: E402
    global_similarity, prototype_similarity, rpgr_score, no_prototype_score, rerank_top_pool,
)
from rpgr.metrics import evaluate_direction, class_level_table, macro_from_class_level  # noqa: E402

DIRECTIONS = ["I2T", "T2I"]


def l2_normalize_rows(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.clip(norms, 1e-12, None)


def load_text(text_dir: Path, case_ids: list[str]) -> np.ndarray:
    first = np.load(text_dir / f"{case_ids[0]}.npy").astype(np.float64).reshape(-1)
    out = np.empty((len(case_ids), first.shape[0]))
    for i, cid in enumerate(case_ids):
        v = np.load(text_dir / f"{cid}.npy").astype(np.float64).reshape(-1)
        out[i] = v / max(np.linalg.norm(v), 1e-12)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--backbones", nargs="+", default=None)
    p.add_argument("--scenario", default="strict")
    p.add_argument("--granularity", default="OncotreeCode")
    p.add_argument("--split", default="val")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    backbones = args.backbones or list(cfg["backbones"].keys())

    split_df = pd.read_csv(cfg["cohort"]["split_csv"])
    eval_ids_ordered = split_df[split_df.split == args.split]["slide_id"].tolist()
    labels_df = pd.read_csv(cfg["cohort"]["labels_csv"]).set_index("slide_id")

    rows = []
    for bb in backbones:
        bb_cfg = cfg["backbones"][bb]
        cache = np.load(bb_cfg["pool_cache_path"], allow_pickle=True)
        idx_of = {cid: i for i, cid in enumerate(cache["case_ids"])}
        eval_ids = [c for c in eval_ids_ordered if c in idx_of]
        mean_pool = cache["mean_pool"].astype(np.float64)[[idx_of[c] for c in eval_ids]]
        print(f"\n=== {bb}: {args.split} n={len(eval_ids)} ===")

        text_dir = Path(bb_cfg["text_embeddings"][args.scenario])
        text_mat = load_text(text_dir, eval_ids)

        proto_dir = Path(bb_cfg["prototypes_dir"])
        prototype_banks = [
            l2_normalize_rows(np.load(proto_dir / f"{cid}_prototype.npz")["prototypes"].astype(np.float64))
            for cid in eval_ids
        ]

        label_map = labels_df.loc[eval_ids, args.granularity].to_dict()

        sim_global = global_similarity(mean_pool, text_mat)
        sim_proto = prototype_similarity(prototype_banks, text_mat)
        s_b = no_prototype_score(sim_global)
        s_rpgr = rpgr_score(sim_global, sim_proto)

        for direction in DIRECTIONS:
            # Both S_B and RPGR are evaluated ONLY within the Top-50 pool
            # discovered by the raw global similarity (same discovery matrix
            # for every variant) -- matching the paper's protocol exactly,
            # not just the RPGR-vs-BGAP comparison but this ablation too.
            pooled_b = rerank_top_pool(sim_global, s_b, direction, n_pool=50)
            pooled_rpgr = rerank_top_pool(sim_global, s_rpgr, direction, n_pool=50)

            df_b, _ = evaluate_direction(pooled_b, eval_ids, label_map, exclude_exact_pair=False)
            map_b = macro_from_class_level(class_level_table(df_b))["mAP@10"]

            df_r, _ = evaluate_direction(pooled_rpgr, eval_ids, label_map, exclude_exact_pair=False)
            map_r = macro_from_class_level(class_level_table(df_r))["mAP@10"]

            rows.append({
                "backbone": bb, "direction": direction,
                "S_B_mAP@10": map_b, "RPGR_mAP@10": map_r, "delta": map_r - map_b,
            })
            print(f"  {direction}: S_B={map_b:.4f}  RPGR={map_r:.4f}  delta={map_r - map_b:+.4f}")

    df = pd.DataFrame(rows)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"prototype_ablation_table4_{args.split}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nescrito {out_path}")


if __name__ == "__main__":
    main()
