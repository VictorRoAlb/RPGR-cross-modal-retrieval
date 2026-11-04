#!/usr/bin/env python3
"""
run_score_ablation.py
=====================
Score-term ablation of RPGR on the held-out TCGA benchmark (test partition, KEEP,
Morphology-only, subtype labels). Every variant ranks the same gallery with the same
embeddings; only the score changes:

  mean_pooling                S_G
  global_only                 R(S_G)
  second_calibrated_channel   R(S_G) + 0.5 (D(S_G) + R(S_G))   (local channel replaced by S_G)
  local_only                  0.5 (D(S_P) + R(S_P))
  global_plus_reciprocal_local R(S_G) + R(S_P)
  global_plus_direct_local    R(S_G) + D(S_P)
  rpgr                        R(S_G) + 0.5 (D(S_P) + R(S_P))

For each variant and direction: class-macro mAP@10 and Recall@1, and the paired
class-stratified bootstrap of (variant - RPGR) resampling clinical cases (all slides of a
case move together). Holm correction is applied within each variant over its directions.

Usage:
  python3 run_score_ablation.py --config ../configs/paths.yaml [--n-boot 20000]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rpgr.metrics import evaluate_direction  # noqa: E402
from rpgr.reranking import direct_term, global_similarity, prototype_similarity, recip_term  # noqa: E402
from rpgr.stats import class_macro, holm, paired_cluster_bootstrap  # noqa: E402


def l2(m: np.ndarray) -> np.ndarray:
    return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-12, None)


def variants(G: np.ndarray, P: np.ndarray) -> dict[str, np.ndarray]:
    R_G = recip_term(G)
    return {
        "mean_pooling": G,
        "global_only": R_G,
        "second_calibrated_channel": R_G + 0.5 * (direct_term(G) + recip_term(G)),
        "local_only": 0.5 * (direct_term(P) + recip_term(P)),
        "global_plus_reciprocal_local": R_G + recip_term(P),
        "global_plus_direct_local": R_G + direct_term(P),
        "rpgr": R_G + 0.5 * (direct_term(P) + recip_term(P)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--backbone", default="KEEP")
    p.add_argument("--scenario", default="strict", choices=["original", "strict", "diagnosis_only"])
    p.add_argument("--split", default="test")
    p.add_argument("--n-boot", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    bb = cfg["backbones"][args.backbone]

    split = pd.read_csv(cfg["cohort"]["split_csv"])
    split = split[split.split == args.split]
    ids = split.slide_id.tolist()
    case_of = dict(zip(split.slide_id, split.case_id))
    labels = pd.read_csv(cfg["cohort"]["labels_csv"]).set_index("slide_id").loc[ids, "OncotreeCode"].to_dict()

    cache = np.load(bb["pool_cache_path"], allow_pickle=True)
    pos = {c: i for i, c in enumerate(cache["case_ids"])}
    V = l2(cache["mean_pool"].astype(np.float64)[[pos[c] for c in ids]])
    tdir = Path(bb["text_embeddings"][args.scenario])
    T = l2(np.stack([np.load(tdir / f"{c}.npy").astype(np.float64).reshape(-1) for c in ids]))
    pdir = Path(bb["prototypes_dir"])
    banks = [l2(np.load(pdir / f"{c}_prototype.npz")["prototypes"].astype(np.float64)) for c in ids]
    SG, SP = global_similarity(V, T), prototype_similarity(banks, T)

    rows = []
    for direction in ["I2T", "T2I"]:
        G, P = (SG, SP) if direction == "I2T" else (SG.T, SP.T)
        per_query = {}
        for name, S in variants(G, P).items():
            df, _ = evaluate_direction(S, ids, labels, exclude_exact_pair=False)
            per_query[name] = df
        ref = per_query["rpgr"]
        units = np.array([case_of[c] for c in ref.query_id])
        for name, df in per_query.items():
            row = {"direction": direction, "variant": name}
            for metric in ["mAP@10", "Recall@1"]:
                row[metric] = class_macro(df[metric].to_numpy(), df.label.to_numpy())
                if name != "rpgr":
                    r = paired_cluster_bootstrap(ref[metric].to_numpy(), df[metric].to_numpy(), df.label.to_numpy(), units,
                                                 B=args.n_boot, seed=args.seed)
                    row.update({f"{metric}_diff_vs_rpgr": r["diff"], f"{metric}_ci_lo": r["ci_lo"],
                                f"{metric}_ci_hi": r["ci_hi"], f"{metric}_p": r["p"]})
            rows.append(row)
    out = pd.DataFrame(rows)
    for metric in ["mAP@10", "Recall@1"]:
        out[f"{metric}_holm"] = False
        for name, g in out[out.variant != "rpgr"].groupby("variant"):
            out.loc[g.index, f"{metric}_holm"] = holm(g[f"{metric}_p"].to_numpy())
    out_dir = Path(cfg["output_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"score_ablation_{args.backbone}_{args.scenario}_{args.split}.csv"
    out.to_csv(path, index=False)
    print(out[["direction", "variant", "mAP@10", "mAP@10_diff_vs_rpgr", "mAP@10_p"]].round(4).to_string(index=False))
    print(f"written {path}")


if __name__ == "__main__":
    main()
