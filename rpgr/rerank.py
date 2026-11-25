"""
RPGR score and gallery ranking.

    S_RPGR = R(S_G) + 1/2 * ( D(S_P) + R(S_P) )                          (Eq. 8)

Both matrices must already be in the orientation of the retrieval direction: WSI rows and
report columns for image-to-text (I2T); pass the transposes for text-to-image (T2I). The
whole gallery is re-ranked; there are no fusion weights and no temperature.
"""
from __future__ import annotations

import numpy as np

from .calibration import direct_term, reciprocal_term


def rpgr_score(sim_global: np.ndarray, sim_proto: np.ndarray) -> np.ndarray:
    """S_RPGR = R(S_G) + 0.5 * (D(S_P) + R(S_P)), query rows x candidate columns."""
    return reciprocal_term(sim_global) + 0.5 * (direct_term(sim_proto) + reciprocal_term(sim_proto))


def rpgr_scores_both_directions(sim_global: np.ndarray, sim_proto: np.ndarray) -> dict[str, np.ndarray]:
    """{'I2T': WSI x report scores, 'T2I': report x WSI scores} from WSI x report S_G and S_P."""
    return {"I2T": rpgr_score(sim_global, sim_proto), "T2I": rpgr_score(sim_global.T, sim_proto.T)}


def rank_gallery(scores: np.ndarray, top_k: int | None = None) -> np.ndarray:
    """Candidate indices of each query, best first, with a stable sort (ties keep gallery order)."""
    order = np.argsort(-scores, axis=1, kind="stable")
    return order if top_k is None else order[:, :top_k]
