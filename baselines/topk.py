"""Patch-level Top-k scoring (Lu et al., 2023) followed by a reciprocal normalisation.

S_topk(i, j) = mean of the k highest patch-report similarities of WSI i and report j; the
final score is the geometric mean of its row- and column-wise percentile ranks."""
from __future__ import annotations

import numpy as np


def row_percentile(matrix: np.ndarray) -> np.ndarray:
    """Each entry replaced by the fraction of entries of its row that are <= it, in (0, 1]."""
    n_candidates = matrix.shape[1]
    out = np.empty_like(matrix)
    for i in range(matrix.shape[0]):
        sorted_row = np.sort(matrix[i])
        out[i] = (1.0 + np.searchsorted(sorted_row, matrix[i], side="left")) / n_candidates
    return out


def column_percentile(matrix: np.ndarray) -> np.ndarray:
    """Column-wise counterpart of `row_percentile`."""
    return row_percentile(matrix.T).T


def topk_patch_text(patches_list: list[np.ndarray], text_vecs: np.ndarray, k: int) -> np.ndarray:
    """(n_wsi, n_text); patches_list[i] are the L2-normalised patches of WSI i."""
    out = np.empty((len(patches_list), text_vecs.shape[0]))
    for j, p in enumerate(patches_list):
        sims = p @ text_vecs.T
        kk = min(k, sims.shape[0])
        out[j, :] = np.sort(sims, axis=0)[-kk:, :].mean(axis=0)
    return out


def reciprocal_topk_score(S_topk: np.ndarray) -> np.ndarray:
    """sqrt(row_percentile * column_percentile) of a query-row matrix."""
    return np.sqrt(np.clip(row_percentile(S_topk) * column_percentile(S_topk), 0, None))
