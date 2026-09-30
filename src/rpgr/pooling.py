"""
pooling.py
==========
Batch Global Average Pooling (BGAP) for whole-slide image representations.

Implements Eq. 3 of the RPGR paper: patch-level visual embeddings are
L2-normalized, averaged, and the resulting vector is L2-normalized again to
obtain a single global WSI-level representation.

This module is pure computation. It takes an in-memory array of patch
embeddings and returns the pooled vector -- it performs no file I/O and
contains no dataset-specific paths. Loading patch embeddings from disk is
the caller's responsibility (see scripts/).
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def l2_normalize(vector: np.ndarray, eps: float = EPS) -> np.ndarray:
    """L2-normalize a single 1-D vector."""
    norm = np.linalg.norm(vector)
    return vector / max(norm, eps)


def l2_normalize_rows(matrix: np.ndarray, eps: float = EPS) -> np.ndarray:
    """L2-normalize each row of a 2-D matrix independently."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, eps, None)


def bgap(patch_embeddings: np.ndarray) -> np.ndarray:
    """Batch Global Average Pooling (Eq. 3).

    Parameters
    ----------
    patch_embeddings : (N_i, d) array of patch-level visual embeddings for
        one WSI (need not be pre-normalized).

    Returns
    -------
    (d,) L2-normalized global WSI representation E_i^V.
    """
    patch_embeddings = np.asarray(patch_embeddings, dtype=np.float32)
    normalized = l2_normalize_rows(patch_embeddings)
    return l2_normalize(normalized.mean(axis=0))


def max_pool(patch_embeddings: np.ndarray) -> np.ndarray:
    """Max-pooling baseline: element-wise max over L2-normalized patches,
    L2-normalized again. Parameter-free global-aggregation comparator used
    in Table 3 (Max pooling row) -- not part of RPGR itself, kept here
    because it shares the same normalization utilities and source script.
    """
    patch_embeddings = np.asarray(patch_embeddings, dtype=np.float32)
    normalized = l2_normalize_rows(patch_embeddings)
    return l2_normalize(normalized.max(axis=0))
