"""
Self-scaled log-softmax calibrations of a similarity matrix (Eq. 7 of the paper).

For a score matrix X with query rows and candidate columns:

    D(X)[q, c] = X[q, c] / sigma_q - logsumexp_c'( X[q, c'] / sigma_q ),   sigma_q = std(X[q, :])
    R(X)[q, c] = X[q, c] / sigma_c - logsumexp_q'( X[q', c] / sigma_c ),   sigma_c = std(X[:, c])

D normalises each query over the candidates (direct term); R normalises each candidate over
the queries (reciprocal term). Standard deviations are population standard deviations
(numpy default, ddof=0) clipped at 1e-8. There is no temperature: the scale of each row or
column is estimated from the matrix itself.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

STD_FLOOR = 1e-8


def direct_term(X: np.ndarray) -> np.ndarray:
    """D(X): log-softmax over candidates, each query scaled by its own standard deviation."""
    sigma_row = np.clip(X.std(axis=1, keepdims=True), STD_FLOOR, None)
    return X / sigma_row - logsumexp(X / sigma_row, axis=1, keepdims=True)


def reciprocal_term(X: np.ndarray) -> np.ndarray:
    """R(X): log-softmax over queries, each candidate scaled by its own standard deviation."""
    sigma_col = np.clip(X.std(axis=0, keepdims=True), STD_FLOOR, None)
    return X / sigma_col - logsumexp(X / sigma_col, axis=0, keepdims=True)
