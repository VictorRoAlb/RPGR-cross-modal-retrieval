"""Re-ranking regime of the training-free comparators.

Every comparator can re-order either the whole gallery or only the ``n_pool`` best
mean-pooling candidates of each query (the regime is selected on validation data).
Candidates outside the pool receive -inf, so the top-10 metrics only see the pool."""
from __future__ import annotations

import numpy as np

REGIMES = {"pool50": 50, "full": None}


def rerank_candidate_pool(sim_query: np.ndarray, score_query: np.ndarray, n_pool: int | None) -> np.ndarray:
    """Keep ``score_query`` inside the top-``n_pool`` of ``sim_query`` (per row) and -inf elsewhere.

    Both matrices are in query-row orientation. ``n_pool=None`` (or >= gallery size) re-ranks
    the whole gallery."""
    n_q, n_c = sim_query.shape
    npool = n_c if (n_pool is None or n_pool >= n_c) else n_pool
    out = np.full((n_q, n_c), -np.inf)
    for qi in range(n_q):
        top_n = np.argsort(-sim_query[qi], kind="stable")[:npool]
        out[qi, top_n] = score_query[qi, top_n]
    return out
