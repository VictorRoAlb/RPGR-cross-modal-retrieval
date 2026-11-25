"""
Evaluation of the trained comparators (TransMIL, SETMIL, DTN, FGCR) and of the slide-level
vision-language models (TITAN, PRISM) from their exported embeddings.

The repository does not retrain these models: the score matrices are rebuilt from the
WSI-level and report-level embeddings that each model produced on the test sets (see the
README for the training configuration and the original implementations).

* TransMIL, SETMIL, FGCR, TITAN, PRISM: cosine similarity of the exported embeddings.
* DTN: binary hash codes (sign of the continuous code, sgn(0) = -1); the cosine of +-1 codes
  is an affine function of the Hamming distance.
"""
from __future__ import annotations

import numpy as np

from rpgr.similarity import l2_normalize_rows


def binary_code(z: np.ndarray) -> np.ndarray:
    """DTN hashing convention: +1 if z > 0 else -1."""
    return np.where(z > 0, 1.0, -1.0)


def embedding_scores(img: np.ndarray, txt: np.ndarray, direction: str, *, normalize: bool = True,
                     binary: bool = False) -> np.ndarray:
    """Query-row scores for one direction: img @ txt.T (I2T) or txt @ img.T (T2I).

    The T2I product is computed directly, not as a transpose: with nearly tied embeddings the
    last-bit differences between the two BLAS products can swap candidates."""
    img, txt = np.asarray(img, dtype=np.float64), np.asarray(txt, dtype=np.float64)
    if binary:
        img, txt = binary_code(img), binary_code(txt)
    if normalize:
        img, txt = l2_normalize_rows(img), l2_normalize_rows(txt)
    return img @ txt.T if direction == "I2T" else txt @ img.T
