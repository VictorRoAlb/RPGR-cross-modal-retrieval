"""Helpers shared by the experiment scripts."""
from __future__ import annotations

import numpy as np
import pandas as pd

from rpgr.metrics import class_level_table, evaluate_direction, macro_from_class_level

DIRECTIONS = ["I2T", "T2I"]


def macro_map(scores: np.ndarray, ids: list[str], label_map: dict[str, str]) -> float:
    """Class-macro mAP@10 of a square score matrix (stable order, paired item kept)."""
    df, _ = evaluate_direction(scores, ids, label_map, exclude_exact_pair=False)
    return float(macro_from_class_level(class_level_table(df))["mAP@10"])


def per_query(scores: np.ndarray, ids: list[str], label_map: dict[str, str]) -> pd.DataFrame:
    """Eligible per-query rows (query_id, label, metrics) with a stable order."""
    df, _ = evaluate_direction(scores, ids, label_map, exclude_exact_pair=False)
    return df[df.eligible].reset_index(drop=True)


def native(matrix: np.ndarray, direction: str) -> np.ndarray:
    """WSI x report matrix in the query-row orientation of ``direction``."""
    return matrix if direction == "I2T" else matrix.T


def best_by_validation(grid: pd.DataFrame, keys: list[str], value: str) -> pd.DataFrame:
    """Row with the highest ``value`` within each group of ``keys`` (pandas default sort, as in
    the paper; rows are generated in the same order as the original selection runs)."""
    rows = []
    for key, sub in grid.groupby(keys):
        best = sub.sort_values(value, ascending=False).iloc[0]
        rows.append({**dict(zip(keys, key if isinstance(key, tuple) else (key,))),
                     "best_param": best["param"], "best_regime": best["regime"], value: best[value]})
    return pd.DataFrame(rows)
