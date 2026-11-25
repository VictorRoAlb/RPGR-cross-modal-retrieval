"""
Tie-neutral evaluation used in the cross-method benchmark (Fig. 3).

Some score matrices contain exactly tied candidates (for instance the binary codes of DTN,
or repeated report texts). To resolve ties in the same way for every method, each metric is
averaged over random permutations of the gallery followed by a stable sort: for seeds
s = 0, ..., n_seeds - 1 (default 50), the gallery columns are permuted with
``numpy.random.default_rng(s).permutation`` and ranked with a stable sort. Scores are never
modified, so only candidates with exactly equal scores change relative order. Rows without
ties give exactly the stable-order value.

Metric definitions are identical to `rpgr.metrics.evaluate_direction` (paired item kept in
the gallery; AP@10 divided by min(#relevant, 10)).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TIE_SEEDS = 50
NEUTRAL_METRICS = ["Recall@1", "Recall@3", "Recall@5", "Recall@10", "MRR@10"]


def fast_metrics(S: np.ndarray, query_labels: np.ndarray, gallery_labels: np.ndarray) -> dict[str, np.ndarray]:
    """Per-query Recall@1/3/5/10, MRR@10 and AP@10 with a stable sort, vectorised."""
    order = np.argsort(-S, axis=1, kind="stable")[:, :10]
    rel = gallery_labels[order] == query_labels[:, None]
    tot = pd.Series(gallery_labels).value_counts().reindex(query_labels).to_numpy()
    cum = np.cumsum(rel, axis=1)
    pos = np.arange(1, rel.shape[1] + 1)          # 1..10 (fewer for galleries below 10)
    ap = (rel * cum / pos).sum(1) / np.minimum(tot, 10)
    first = np.where(rel.any(1), rel.argmax(1) + 1, np.inf)
    return {"mAP@10": ap, "Recall@1": rel[:, :1].any(1) * 1.0, "Recall@3": rel[:, :3].any(1) * 1.0,
            "Recall@5": rel[:, :5].any(1) * 1.0, "Recall@10": rel.any(1) * 1.0,
            "MRR@10": np.where(np.isfinite(first), 1 / first, 0.0)}


def row_has_ties(S: np.ndarray) -> np.ndarray:
    """True for every query row that contains at least two exactly equal scores."""
    s = np.sort(S, axis=1)
    return (np.diff(s, axis=1) == 0).any(1)


def tie_neutral_per_query(S: np.ndarray, labels: np.ndarray, n_seeds: int = TIE_SEEDS,
                          always_permute: bool = False) -> pd.DataFrame:
    """Stable and tie-neutral per-query metrics for a square score matrix (same item order on
    both axes). Returns columns mAP@10_stable, mAP@10_neutral, mAP@10_tie_sd, <metric>_neutral
    for Recall@1/3/5/10 and MRR@10, and row_has_tie.

    By default the permutations are skipped when no row has ties (the result is then the
    stable value); ``always_permute=True`` averages over the permutations in every case."""
    y = np.asarray(labels)
    det = fast_metrics(S, y, y)
    ties = row_has_ties(S)
    acc = {k: np.zeros(len(y)) for k in det}
    sq = np.zeros(len(y))
    if ties.any() or always_permute:
        for s in range(n_seeds):
            p = np.random.default_rng(s).permutation(len(y))
            m = fast_metrics(S[:, p], y, y[p])
            for k in acc:
                acc[k] += m[k]
            sq += m["mAP@10"] ** 2
        neu = {k: v / n_seeds for k, v in acc.items()}
        sd = np.sqrt(np.clip(sq / n_seeds - neu["mAP@10"] ** 2, 0, None))
    else:
        neu, sd = det, np.zeros(len(y))
    return pd.DataFrame({"label": y, "mAP@10_stable": det["mAP@10"], "mAP@10_neutral": neu["mAP@10"],
                         "mAP@10_tie_sd": sd, **{f"{k}_neutral": neu[k] for k in NEUTRAL_METRICS},
                         "row_has_tie": ties})
