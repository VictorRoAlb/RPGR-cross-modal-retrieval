"""
Global and local similarities used by RPGR.

* Global WSI representation (Eq. 1): mean pooling of L2-normalised patch embeddings,
  L2-normalised again (``mean_pool``).
* Global similarity (Eq. 2) S_G(i, j) = cos(E_i^V, E_j^T) (``global_similarity``).
* Local similarity (Eq. 6) S_P(i, j) = max_h cos(p_{i,h}, E_j^T), the cosine between report j and
  the most compatible prototype of WSI i (``prototype_similarity``).

All functions operate on in-memory arrays; reading embeddings from disk is done in
``data/loaders.py``.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def l2_normalize(vector: np.ndarray, eps: float = EPS) -> np.ndarray:
    """L2-normalise a single 1-D vector."""
    return vector / max(np.linalg.norm(vector), eps)


def l2_normalize_rows(matrix: np.ndarray, eps: float = EPS) -> np.ndarray:
    """L2-normalise each row of a 2-D matrix."""
    return matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), eps, None)


def mean_pool(patch_embeddings: np.ndarray) -> np.ndarray:
    """Mean of the L2-normalised patch embeddings of one WSI, L2-normalised again (float32)."""
    patch_embeddings = np.asarray(patch_embeddings, dtype=np.float32)
    return l2_normalize(l2_normalize_rows(patch_embeddings).mean(axis=0))


def global_similarity(wsi_vectors: np.ndarray, text_vectors: np.ndarray) -> np.ndarray:
    """S_G (n_wsi, n_text); both inputs must have L2-normalised rows."""
    return wsi_vectors @ text_vectors.T


def prototype_similarity(prototype_banks: list[np.ndarray], text_vectors: np.ndarray) -> np.ndarray:
    """S_P (n_wsi, n_text): for each WSI i and report j, max_h cos(p_{i,h}, E_j^T).

    ``prototype_banks[i]`` is the (H*_i, d) bank of WSI i (rows L2-normalised), in the same
    order as the WSI rows of S_G. Computed one WSI at a time, which is the exact arithmetic
    used for every number in the paper."""
    out = np.empty((len(prototype_banks), text_vectors.shape[0]))
    for i, bank in enumerate(prototype_banks):
        out[i] = (bank @ text_vectors.T).max(axis=0)
    return out
