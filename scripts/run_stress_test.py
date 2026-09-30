#!/usr/bin/env python3
"""
run_stress_test.py
====================
Reproduces Table 1 (stress-testing of frozen VL-PFMs using BGAP, no
re-ranking): class-macro mAP@10 for every combination of backbone (4),
textual scenario (3), semantic granularity (2), and retrieval direction
(2), on the full retrieval cohort (all cases in each backbone's pool
cache -- N=5291, "C_paper").

Reads only from the paths given in --config. Writes only under output_dir.
Uses exclusively rpgr.reranking.global_similarity and rpgr.metrics -- no
prototype construction needed, since this is the BGAP-only baseline.

Usage:
  python3 run_stress_test.py --config ../configs/paths.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rpgr.reranking import global_similarity  # noqa: E402
from rpgr.metrics import evaluate_direction, class_level_table, macro_from_class_level  # noqa: E402

SCENARIOS = ["original", "strict", "diagnosis_only"]
GRANULARITIES = ["project_id", "OncotreeCode"]
DIRECTIONS = ["I2T", "T2I"]


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
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    backbones = args.backbones or list(cfg["backbones"].keys())
    labels_df = pd.read_csv(cfg["cohort"]["labels_csv"]).set_index("slide_id")

    rows = []
    for bb in backbones:
        bb_cfg = cfg["backbones"][bb]
        cache = np.load(bb_cfg["pool_cache_path"], allow_pickle=True)
        case_ids = list(cache["case_ids"])
        mean_pool = cache["mean_pool"].astype(np.float64)
        print(f"\n=== {bb}: {len(case_ids)} casos ===")
        label_maps = {g: labels_df.loc[case_ids, g].to_dict() for g in GRANULARITIES}

        for scenario in SCENARIOS:
            text_dir = Path(bb_cfg["text_embeddings"][scenario])
            text_mat = load_text(text_dir, case_ids)
            sim_global = global_similarity(mean_pool, text_mat)

            for direction in DIRECTIONS:
                sim_dir = sim_global if direction == "I2T" else sim_global.T
                for granularity in GRANULARITIES:
                    df_q, _ = evaluate_direction(sim_dir, case_ids, label_maps[granularity], exclude_exact_pair=False)
                    macro = macro_from_class_level(class_level_table(df_q))
                    rows.append({
                        "backbone": bb, "scenario": scenario, "granularity": granularity,
                        "direction": direction, "mAP@10": macro["mAP@10"],
                    })
            print(f"  {scenario}: OK")

    df = pd.DataFrame(rows)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "stress_test_table1.csv"
    df.to_csv(out_path, index=False)
    print(f"\n{len(df)} filas (esperado 48) -> {out_path}")


if __name__ == "__main__":
    main()
