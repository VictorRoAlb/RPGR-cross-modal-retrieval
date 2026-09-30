#!/usr/bin/env python3
"""
run_rpgr_grid.py
==================
Reproduces Table 2 (effect of RPGR across the complete 48-condition
evaluation grid): for every combination of backbone (4), textual scenario
(3), semantic granularity (2), and retrieval direction (2), computes both
the BGAP baseline mAP@10 and the RPGR mAP@10 on the full retrieval cohort
(all cases present in each backbone's pool cache -- N=5291 in the paper,
"C_paper"). Also reproduces Table 1 alone if you only need BGAP: see
run_stress_test.py, which shares the same loop but skips the RPGR side.

Reads only from the paths given in --config. Writes only under output_dir.
Uses exclusively rpgr.reranking and rpgr.metrics.

Usage:
  python3 run_rpgr_grid.py --config ../configs/paths.yaml
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
from rpgr.reranking import global_similarity, prototype_similarity, rpgr_score, rerank_top_pool  # noqa: E402
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
        case_ids = list(cache["case_ids"])
        mean_pool = cache["mean_pool"].astype(np.float64)
        n = len(case_ids)
        print(f"\n=== {bb}: {n} casos ===")

        proto_dir = Path(bb_cfg["prototypes_dir"])
        prototype_banks = [
            l2_normalize_rows(np.load(proto_dir / f"{cid}_prototype.npz")["prototypes"].astype(np.float64))
            for cid in case_ids
        ]
        print(f"  prototipos listos ({time.time() - t0:.1f}s)")

        label_maps = {g: labels_df.loc[case_ids, g].to_dict() for g in GRANULARITIES}

        for scenario in SCENARIOS:
            t0 = time.time()
            text_dir = Path(bb_cfg["text_embeddings"][scenario])
            text_mat = load_text(text_dir, case_ids)

            sim_global = global_similarity(mean_pool, text_mat)
            sim_proto = prototype_similarity(prototype_banks, text_mat)
            score = rpgr_score(sim_global, sim_proto)
            print(f"  [{scenario}] scores listos ({time.time() - t0:.1f}s)")

            for direction in DIRECTIONS:
                final_scores = rerank_top_pool(sim_global, score, direction, n_pool=50)
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
    out_path = out_dir / "rpgr_grid_table2.csv"
    df.to_csv(out_path, index=False)
    print(f"\n{len(df)} filas (esperado 48 con los 4 backbones) -> {out_path}")


if __name__ == "__main__":
    main()
