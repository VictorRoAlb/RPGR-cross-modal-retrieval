"""
metrics.py
==========
Class-macro retrieval metrics: Recall@1/3/5/10, MRR@10 and mAP@10.

Given a (n, n) score matrix (query rows, candidate columns, same case order on both
axes) and a case_id -> label mapping, computes per-query metrics. Class-macro
aggregation gives every class the same weight. A candidate is relevant if it has the
query's label. The query's own paired item is included in the gallery
(`exclude_exact_pair=False`, the protocol used throughout).

Ties: `evaluate_direction` ranks with a stable sort (tied candidates keep gallery order).
`evaluate_direction_tie_neutral` gives the expected metrics under a random order of exactly
tied candidates (gallery permutations, seeds 0..n_seeds-1); this is the protocol used for
the cross-method comparison, where some methods produce many exact ties.

mAP@10 divides by min(#relevant items for the query, 10).
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
    s = np.asarray(sim, dtype=float).copy()
    if exclude_exact_pair:
        np.fill_diagonal(s, -np.inf)
    df = _per_query(s, list(case_ids), list(case_ids), label_map, exclude_exact_pair)
    return df, _summary(df, s.shape[0])


def evaluate_direction_tie_neutral(
    sim: np.ndarray, case_ids: list[str], label_map: dict[str, str],
    *, exclude_exact_pair: bool = False, n_seeds: int = 50,
) -> tuple[pd.DataFrame, dict]:
    """Tie-neutral per-query metrics: expected value under a uniformly random order of
    exactly tied candidates.

    For seeds s = 0..n_seeds-1 the gallery (columns) is permuted with
    numpy.random.default_rng(s).permutation and ranked with a stable sort. Scores are
    never modified, so only candidates with exactly equal scores change relative order.
    Per-query metrics are averaged over the seeds; `mAP@10_tie_sd` is the per-query
    standard deviation across seeds (tie-break uncertainty). For score matrices without
    ties the result equals `evaluate_direction`.
    """
    s = np.asarray(sim, dtype=float).copy()
    if exclude_exact_pair:
        np.fill_diagonal(s, -np.inf)
    ids = list(case_ids)
    runs = []
    for seed in range(n_seeds):
        perm = np.random.default_rng(seed).permutation(len(ids))
        df = _per_query(s[:, perm], ids, [ids[i] for i in perm], label_map, exclude_exact_pair)
        runs.append(df[METRIC_NAMES].to_numpy(dtype=float))
    runs = np.stack(runs)                               # (n_seeds, n_queries, n_metrics)
    out = df[["query_id", "label", "eligible"]].copy()
    out[METRIC_NAMES] = runs.mean(axis=0)
    out["mAP@10_tie_sd"] = runs[:, :, METRIC_NAMES.index("mAP@10")].std(axis=0)
    return out, _summary(out, s.shape[0])


def _per_query(s: np.ndarray, query_ids: list[str], cand_ids: list[str],
               label_map: dict[str, str], exclude_exact_pair: bool) -> pd.DataFrame:
    label_counts: dict[str, int] = {}
    for lbl in label_map.values():
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    order = np.argsort(-s, axis=1, kind="stable")
    rows = []
    for qi, qid in enumerate(query_ids):
        lbl = label_map[qid]
        total_rel = label_counts[lbl] - 1 if exclude_exact_pair else label_counts[lbl]
        if total_rel <= 0:
            rows.append({"query_id": qid, "label": lbl, "eligible": False, **{m: np.nan for m in METRIC_NAMES}})
            continue
        top = order[qi, :K_MAX]
        rel = np.array([label_map[cand_ids[p]] == lbl for p in top])
        row = {"query_id": qid, "label": lbl, "eligible": True}
        for k in KS:
            row[f"Recall@{k}"] = hit_rate_at_k(rel, k)
        row["MRR@10"] = reciprocal_rank(rel)
        row["mAP@10"] = average_precision(rel, total_relevant=total_rel)
        rows.append(row)
    return pd.DataFrame(rows)


def _summary(df: pd.DataFrame, n: int) -> dict:
    n_elig = int(df["eligible"].sum()) if len(df) else 0
    return {"n_total_queries": n, "n_eligible_queries": n_elig, "n_zero_positive_queries": n - n_elig}


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
