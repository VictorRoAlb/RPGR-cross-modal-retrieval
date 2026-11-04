#!/usr/bin/env python3
"""
run_rpgr_grid.py
================
RPGR versus mean pooling across backbones (KEEP, CONCH, MUSK, PATHO-CLIP), report
conditions (Original, Morphology-only, Diagnosis-only), semantic levels (organ, subtype)
and directions (I2T, T2I): 48 TCGA conditions.

Queries = gallery = the training + test partitions of the subtype split. The validation
partition is excluded because it was used for method development and baseline tuning.
Writes class-macro mAP@10 of both methods per condition. The AI4SKIN conditions require
the non-public AI4SKIN data and are not covered.

Usage:
  python3 run_rpgr_grid.py --config ../configs/paths.yaml [--backbones KEEP CONCH]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rpgr.reranking import global_similarity, prototype_similarity, rpgr_score  # noqa: E402
from rpgr.metrics import evaluate_direction, class_level_table, macro_from_class_level  # noqa: E402

SCENARIOS = ["original", "strict", "diagnosis_only"]
GRANULARITIES = ["project_id", "OncotreeCode"]
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
    p.add_argument("--backbones", nargs="+", default=None, help="Subset of backbones to run (default: all in config)")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    backbones = args.backbones or list(cfg["backbones"].keys())

    labels_df = pd.read_csv(cfg["cohort"]["labels_csv"]).set_index("slide_id")

    rows = []
    for bb in backbones:
        bb_cfg = cfg["backbones"][bb]
        t0 = time.time()
        cache = np.load(bb_cfg["pool_cache_path"], allow_pickle=True)
        split_df = pd.read_csv(cfg["cohort"]["split_csv"])
        keep = set(split_df[split_df.split.isin(["train", "test"])].slide_id)
        idx = [i for i, c in enumerate(cache["case_ids"]) if c in keep]
        case_ids = [cache["case_ids"][i] for i in idx]
        mean_pool = l2_normalize_rows(cache["mean_pool"][idx].astype(np.float64))
        n = len(case_ids)
        print(f"\n=== {bb}: {n} cases ===")

        proto_dir = Path(bb_cfg["prototypes_dir"])
        prototype_banks = [
            l2_normalize_rows(np.load(proto_dir / f"{cid}_prototype.npz")["prototypes"].astype(np.float64))
            for cid in case_ids
        ]
        print(f"  prototypes loaded ({time.time() - t0:.1f}s)")

        label_maps = {g: labels_df.loc[case_ids, g].to_dict() for g in GRANULARITIES}

        for scenario in SCENARIOS:
            t0 = time.time()
            text_dir = Path(bb_cfg["text_embeddings"][scenario])
            text_mat = load_text(text_dir, case_ids)

            sim_global = global_similarity(mean_pool, text_mat)
            sim_proto = prototype_similarity(prototype_banks, text_mat)
            print(f"  [{scenario}] scores ready ({time.time() - t0:.1f}s)")

            for direction in DIRECTIONS:
                G, P = (sim_global, sim_proto) if direction == "I2T" else (sim_global.T, sim_proto.T)
                final_scores = rpgr_score(G, P)
                bgap_scores = sim_global if direction == "I2T" else sim_global.T

                for granularity in GRANULARITIES:
                    lbl_map = label_maps[granularity]

                    df_q_bgap, _ = evaluate_direction(bgap_scores, case_ids, lbl_map, exclude_exact_pair=False)
                    macro_bgap = macro_from_class_level(class_level_table(df_q_bgap))

                    df_q_rpgr, _ = evaluate_direction(final_scores, case_ids, lbl_map, exclude_exact_pair=False)
                    macro_rpgr = macro_from_class_level(class_level_table(df_q_rpgr))

                    rows.append({
                        "backbone": bb, "scenario": scenario, "granularity": granularity, "direction": direction,
                        "bgap_mAP@10": macro_bgap["mAP@10"], "rpgr_mAP@10": macro_rpgr["mAP@10"],
                        "delta_pct": 100.0 * (macro_rpgr["mAP@10"] - macro_bgap["mAP@10"]) / macro_bgap["mAP@10"],
                    })
            print(f"  {scenario}: OK")

    df = pd.DataFrame(rows)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "rpgr_grid_tcga.csv"
    df.to_csv(out_path, index=False)
    print(f"\n{len(df)} rows (48 expected with the 4 backbones) -> {out_path}")


if __name__ == "__main__":
    main()
