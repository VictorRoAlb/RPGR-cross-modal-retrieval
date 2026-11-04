"""
reranking.py
============
RPGR score (final formulation used in the paper).

Given, for one retrieval direction, the global similarity matrix S_G (query rows x
gallery columns; cosine between mean-pooled WSI vectors and report embeddings) and the
local prototype similarity S_P (same orientation; for each WSI-report pair the cosine
of the report with the WSI's MOST similar prototype, max_h), RPGR scores every pair as

    S_RPGR = R(S_G) + 0.5 * (D(S_P) + R(S_P))

with self-scaled log-softmax calibrations (temperature c = 1):

    D(X)[q, c] = X[q, c] / std(X[q, :]) - logsumexp_c'( X[q, c'] / std(X[q, :]) )   (direct, per query)
    R(X)[q, c] = X[q, c] / std(X[:, c]) - logsumexp_q'( X[q', c] / std(X[:, c]) )   (reciprocal, per candidate)

std is clipped at 1e-8. The WHOLE gallery is re-ranked (no initial pool), and there are
no learned or cohort-tuned fusion weights. The reciprocal terms use the statistics of the
full query set (transductive), see the paper and docs/REPRODUCING.md for the cost of an
incremental query.

Numerically identical to the implementation used for every number reported in the paper.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

_EPS_STD = 1e-8


def global_similarity(wsi_vectors: np.ndarray, text_vectors: np.ndarray) -> np.ndarray:
    """S_G(i, j) = cos(E_i^V, E_j^T). Inputs must be L2-normalised (rows)."""
    return wsi_vectors @ text_vectors.T


def prototype_similarity(prototype_banks: list[np.ndarray], text_vectors: np.ndarray) -> np.ndarray:
    """S_P(i, j) = max_h cos(p_{i,h}, E_j^T)  -- (n_wsi, n_text).

    prototype_banks: list of (H*_i, d) L2-normalised prototype arrays, one per WSI, in the
    same order as the WSI rows of S_G. Vectorised (one matrix product over the stacked
    banks + segment max); identical to the per-WSI loop."""
    sizes = np.array([b.shape[0] for b in prototype_banks])
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    stacked = np.concatenate(prototype_banks, axis=0)
    return np.maximum.reduceat(stacked @ text_vectors.T, starts, axis=0)


def direct_term(X: np.ndarray, c: float = 1.0) -> np.ndarray:
    sigma_row = np.clip(X.std(axis=1, keepdims=True) * c, _EPS_STD, None)
    return X / sigma_row - logsumexp(X / sigma_row, axis=1, keepdims=True)


def recip_term(X: np.ndarray, c: float = 1.0) -> np.ndarray:
    sigma_col = np.clip(X.std(axis=0, keepdims=True) * c, _EPS_STD, None)
    return X / sigma_col - logsumexp(X / sigma_col, axis=0, keepdims=True)


def rpgr_score(sim_global: np.ndarray, sim_proto: np.ndarray) -> np.ndarray:
    """S_RPGR = R(S_G) + 0.5 * (D(S_P) + R(S_P)).

    Both inputs must already be in the orientation of the direction being scored
    (I2T: WSI rows / report columns; T2I: pass the transposes)."""
    return recip_term(sim_global) + 0.5 * (direct_term(sim_proto) + recip_term(sim_proto))


def global_only_score(sim_global: np.ndarray) -> np.ndarray:
    """R(S_G) alone -- the 'global only' ablation (A1)."""
    return recip_term(sim_global)
