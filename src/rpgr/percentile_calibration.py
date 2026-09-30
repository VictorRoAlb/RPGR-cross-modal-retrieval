"""
percentile_calibration.py
==========================
Row-wise percentile calibration (Eqs. 21-23 of the RPGR paper).

Pure computation: takes an in-memory similarity matrix and returns its
row-wise percentile transformation. No file I/O, no dataset-specific paths.
"""
from __future__ import annotations

import numpy as np


def rowpct(matrix: np.ndarray) -> np.ndarray:
    """Row-wise percentile transformation, mapped to (0, 1].

    For each row, every entry is replaced by its percentile rank within
    that row: entry (i, j) becomes the fraction of entries in row i that
    are <= matrix[i, j]. This is rowpct(.) in the paper.

    Parameters
    ----------
    matrix : (n_queries, n_candidates) raw similarity matrix.

    Returns
    -------
    Array of the same shape, each row independently mapped to (0, 1].
    """
    n_candidates = matrix.shape[1]
    percentiles = np.empty_like(matrix)
    for i in range(matrix.shape[0]):
        sorted_row = np.sort(matrix[i])
        percentiles[i] = (1.0 + np.searchsorted(sorted_row, matrix[i], side="left")) / n_candidates
    return percentiles


def forward_global(sim_global: np.ndarray) -> np.ndarray:
    """F_G = rowpct(S_G) -- Eq. 21: direct global relevance."""
    return rowpct(sim_global)


def reciprocal_global(sim_global: np.ndarray) -> np.ndarray:
    """R_G = rowpct(S_G^T)^T -- Eq. 22: reciprocal global relevance."""
    return rowpct(sim_global.T).T


def reciprocal_prototype(sim_proto: np.ndarray) -> np.ndarray:
    """R_P = rowpct(S_P^T)^T -- Eq. 23: reciprocal local prototype relevance."""
    return rowpct(sim_proto.T).T
