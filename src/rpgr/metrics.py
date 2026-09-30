"""
metrics.py
==========
Class-macro retrieval metrics (Recall@1/3/5/10, MRR@10, mAP@10) and the
exact-pair inclusion/exclusion policy used to rank each query's candidates.

Pure computation: given an in-memory (n_queries, n_candidates) similarity
matrix, the case ids ordering it, and a case_id -> label mapping, this
module computes per-query metrics and aggregates them class-macro (every
class counts equally, regardless of size). No file I/O, no dataset-specific
paths, and no statistical inference of any kind -- no bootstrap, confidence
intervals, permutation tests, Holm-Bonferroni correction, or p-values.

Exact-pair policy: the paper's final benchmark (Table 3) and the RPGR-wide
evaluation (Tables 1-2) both use exact-pair INCLUDED -- a WSI's own paired
report (and vice versa) is a valid retrieval candidate, never masked out,
and every query is always eligible. This is the default here
(`exclude_exact_pair=False`). Exact-pair EXCLUDED (diagonal masked; a query
left with no positive after exclusion is marked ineligible and dropped from
the macro average) is also implemented, but only ever runs when requested
explicitly via `exclude_exact_pair=True` -- it never silently replaces the
paper's default protocol.

Implementation note on Recall@k and MRR@10: the original evaluator computed
these via torchmetrics (`retrieval_hit_rate`, `retrieval_reciprocal_rank`)
applied to a synthetic strictly-descending score vector, purely to reuse an
existing library call. Both reduce to simple boolean-array operations once
`rel` is already sorted by descending similarity (which it always is here),
so they are reimplemented directly in numpy below to avoid a torch/
torchmetrics dependency for what training-free re-ranking should not need.
Verified to reproduce the original torch-based values exactly on real data
(see tests/test_metrics_equivalence.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KS = (1, 3, 5, 10)
K_MAX = 10
METRIC_NAMES = ["Recall@1", "Recall@3", "Recall@5", "Recall@10", "MRR@10", "mAP@10"]


def hit_rate_at_k(rel: np.ndarray, k: int) -> float:
    """1.0 if at least one relevant item appears among the top k of `rel`
    (already sorted by descending similarity), else 0.0."""
    return float(rel[:k].any())


def reciprocal_rank(rel: np.ndarray) -> float:
    """1 / (rank of the first relevant item in `rel`), or 0.0 if none is
    present (rel is truncated to the top K_MAX candidates, so this is
    MRR@10, not an unbounded MRR)."""
    if not rel.any():
        return 0.0
    return float(1.0 / (int(np.argmax(rel)) + 1))


def average_precision(rel: np.ndarray, total_relevant: int) -> float:
    """Average Precision truncated to len(rel) (10 in this benchmark).

    The denominator is min(total_relevant, len(rel)): the number of
    relevant items that actually exist for this query, capped at the
    truncation length -- not just the number found within the top-10. This
    avoids a query with a single hit at rank 1 (but many more relevant
    items further down that were never reached) scoring a misleading
    perfect AP=1.0.
    """
    denom = min(int(total_relevant), len(rel))
    if denom == 0 or not rel.any():
        return 0.0
    cum_hits = np.cumsum(rel)
    precisions_at_hits = cum_hits[rel] / (np.flatnonzero(rel) + 1)
    return float(precisions_at_hits.sum() / denom)


def evaluate_direction(
    sim: np.ndarray, case_ids: list[str], label_map: dict[str, str],
    *, exclude_exact_pair: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Per-query metrics for one retrieval direction.

    Parameters
    ----------
    sim : (n, n) similarity/score matrix, query rows / candidate columns,
        the same `case_ids` order on both axes.
    case_ids : length-n list giving the row/column order of `sim`.
    label_map : case_id -> class label (e.g. OncotreeCode subtype or
        project_id organ/project).
    exclude_exact_pair : if True, a query's own paired case is masked out
        (diagonal set to -inf) before ranking, and a query left with no
        positive after exclusion is marked ineligible and excluded from
        the macro average's denominator. Default False: exact-pair
        INCLUDED, the paper's benchmark protocol -- every query is always
        eligible.

    Returns
    -------
    (df_query_level, summary), where summary has n_total_queries,
    n_eligible_queries, n_zero_positive_queries.
    """
    n = sim.shape[0]
    label_counts: dict[str, int] = {}
    for lbl in label_map.values():
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    s = sim.copy()
    if exclude_exact_pair:
        np.fill_diagonal(s, -np.inf)
    order = np.argsort(-s, axis=1)

    rows = []
    n_zero_positive = 0
    for qi, qid in enumerate(case_ids):
        lbl = label_map[qid]
        total_rel = label_counts[lbl] - 1 if exclude_exact_pair else label_counts[lbl]
        if total_rel <= 0:
            n_zero_positive += 1
            rows.append({"query_id": qid, "label": lbl, "eligible": False})
            continue
        top = order[qi, :K_MAX]
        rel = np.array([label_map[case_ids[p]] == lbl for p in top])
        row = {"query_id": qid, "label": lbl, "eligible": True}
        for k in KS:
            row[f"Recall@{k}"] = hit_rate_at_k(rel, k)
        row["MRR@10"] = reciprocal_rank(rel)
        row["mAP@10"] = average_precision(rel, total_relevant=total_rel)
        rows.append(row)

    df = pd.DataFrame(rows)
    summary = {
        "n_total_queries": n,
        "n_eligible_queries": int(df["eligible"].sum()) if len(df) else 0,
        "n_zero_positive_queries": n_zero_positive,
    }
    return df, summary


def class_level_table(df_query: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-query metrics to per-class means, over eligible
    queries only. A class left with zero eligible queries (e.g. a
    singleton dropped entirely under exact-pair exclusion) does not appear
    here and therefore does not enter the macro average."""
    elig = df_query[df_query["eligible"]]
    if len(elig) == 0:
        return pd.DataFrame(columns=["class", "n_class", *METRIC_NAMES])
    grp = elig.groupby("label")
    out = grp[METRIC_NAMES].mean()
    out.insert(0, "n_class", grp.size())
    return out.reset_index().rename(columns={"label": "class"})


def macro_from_class_level(df_class: pd.DataFrame) -> dict:
    """Class-macro average: unweighted mean across classes (every class
    counts equally regardless of how many cases it has)."""
    if len(df_class) == 0:
        return {m: float("nan") for m in METRIC_NAMES}
    return df_class[METRIC_NAMES].mean().to_dict()
