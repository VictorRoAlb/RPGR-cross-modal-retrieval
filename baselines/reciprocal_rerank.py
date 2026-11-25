"""
Cross-modal re-ranking (RR; Wang et al., ACM Multimedia 2019), bidirectional reciprocal
nearest-neighbour module without the text-text branch (one report per WSI).

"Order" = candidate indices of each query from most to least similar (argsort(-S));
"rank" = inverse permutation. RR only re-orders the top-K prefix of each order by the
reverse rank; the tail is unchanged. Ties in the reverse rank keep the original order
(stable sort). The initial orders use numpy's default argsort, as in all reported results.
"""
from __future__ import annotations

import numpy as np


def build_rank_matrices(S: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """S: (N, N) WSI x report similarity. Returns (i2t_order, t2i_order, rank_i2t, rank_t2i):
    i2t_order[i, :] ranks reports for WSI i; t2i_order[:, j] ranks WSIs for report j;
    rank_* are the corresponding 0-indexed positions."""
    n_query, n_gallery = S.shape
    i2t_order = np.argsort(-S, axis=1)
    t2i_order = np.argsort(-S, axis=0)
    rank_i2t = np.empty_like(i2t_order)
    rank_i2t[np.arange(n_query)[:, None], i2t_order] = np.arange(n_gallery)[None, :]
    rank_t2i = np.empty_like(t2i_order)
    rank_t2i[t2i_order, np.arange(n_gallery)[None, :]] = np.arange(n_query)[:, None]
    return i2t_order, t2i_order, rank_i2t, rank_t2i


def _reorder_prefix(order_row: np.ndarray, reverse_ranks: np.ndarray, k: int) -> np.ndarray:
    perm = np.argsort(reverse_ranks, kind="stable")
    return np.concatenate([order_row[:k][perm], order_row[k:]])


def rerank_i2t(i2t_order: np.ndarray, rank_t2i: np.ndarray, k: int) -> np.ndarray:
    k = min(k, i2t_order.shape[1])
    out = np.empty_like(i2t_order)
    for i in range(i2t_order.shape[0]):
        cand = i2t_order[i, :k]
        out[i] = _reorder_prefix(i2t_order[i], rank_t2i[i, cand], k)
    return out


def rerank_t2i(t2i_order: np.ndarray, rank_i2t: np.ndarray, k: int) -> np.ndarray:
    k = min(k, t2i_order.shape[0])
    out = np.empty_like(t2i_order)
    for j in range(t2i_order.shape[1]):
        cand = t2i_order[:k, j]
        out[:, j] = _reorder_prefix(t2i_order[:, j], rank_i2t[cand, j], k)
    return out


def rerank_bidirectional(S: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Re-ranked full orders (i2t_order, t2i_order) with the top-k prefix of each re-ordered."""
    i2t_order, t2i_order, rank_i2t, rank_t2i = build_rank_matrices(S)
    return rerank_i2t(i2t_order, rank_t2i, k), rerank_t2i(t2i_order, rank_i2t, k)


def score_matrix_from_order(order: np.ndarray) -> np.ndarray:
    """Turn full orders (query rows) into scores n_c, ..., 1 so that ranking reproduces them."""
    n_q, n_c = order.shape
    score = np.empty((n_q, n_c))
    score[np.arange(n_q)[:, None], order] = np.arange(n_c, 0, -1)
    return score


def rr_scores(S: np.ndarray, k: int) -> dict[str, np.ndarray]:
    """{'I2T': WSI x report scores, 'T2I': report x WSI scores} reproducing the RR orders."""
    i2t, t2i = rerank_bidirectional(S, k)
    return {"I2T": score_matrix_from_order(i2t), "T2I": score_matrix_from_order(t2i.T)}
