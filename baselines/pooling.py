"""Parameter-free global aggregation: mean pooling (re-exported from rpgr) and max pooling."""
from __future__ import annotations

import numpy as np

from rpgr.similarity import l2_normalize, l2_normalize_rows, mean_pool

__all__ = ["max_pool", "mean_pool"]


def max_pool(patch_embeddings: np.ndarray) -> np.ndarray:
    """Element-wise maximum of the L2-normalised patch embeddings, L2-normalised again (float32)."""
    patch_embeddings = np.asarray(patch_embeddings, dtype=np.float32)
    return l2_normalize(l2_normalize_rows(patch_embeddings).max(axis=0))
