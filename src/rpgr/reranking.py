"""
reranking.py
============
Reciprocal Prototype-Guided Re-ranking (RPGR): assembly of the final score
(Eq. 24) and the Top-50 pool re-ranking rule (Section 3.6).

Pure computation: operates on in-memory similarity matrices and prototype
banks already produced by pooling.py / prototypes.py and a frozen text
encoder. No file I/O, no dataset-specific paths.

q_proto (the per-case prototype confidence computed in prototypes.py) is
deliberately NOT used anywhere in this module -- the final RPGR score
combines direct global, reciprocal global, and reciprocal prototype
evidence purely through percentile calibration and an unweighted geometric
mean. There is no confidence-weighted mixing term in the final method.
"""
from __future__ import annotations

import numpy as np

from .percentile_calibration import forward_global, reciprocal_global, reciprocal_prototype

TOP_K_PROTO = 3   # M_i = min(3, H*_i), Eq. 19
N_POOL = 50       # size of the initial Global Top-N retrieval pool, Section 3.6


def global_similarity(wsi_vectors: np.ndarray, text_vectors: np.ndarray) -> np.ndarray:
    """S_G(i, j) = cos(E_i^V, E_j^T) -- Eq. 4.

    Both inputs are assumed already L2-normalized (as produced by
    pooling.bgap and the frozen text encoder), so the dot product equals
    the cosine similarity.
    """
    return wsi_vectors @ text_vectors.T


def prototype_similarity(prototype_banks: list[np.ndarray], text_vectors: np.ndarray,
                          top_k: int = TOP_K_PROTO) -> np.ndarray:
    """S_P(i, j) -- Eqs. 19-20: mean cosine similarity between text j and
    the top_k = min(3, H*_i) most report-compatible prototypes of WSI i.

    Parameters
    ----------
    prototype_banks : list of length n_wsis, each an (H*_i, d) L2-normalized
        prototype array for one WSI (as produced by
        prototypes.build_adaptive_prototypes_for_case).
    text_vectors : (n_texts, d) L2-normalized report embeddings.
    """
    n_wsis = len(prototype_banks)
    n_texts = text_vectors.shape[0]
    sim_proto = np.empty((n_wsis, n_texts), dtype=np.float64)
    for i, protos in enumerate(prototype_banks):
        sims = protos @ text_vectors.T
        k = min(top_k, sims.shape[0])
        sim_proto[i, :] = np.sort(sims, axis=0)[-k:, :].mean(axis=0)
    return sim_proto


def rpgr_score(sim_global: np.ndarray, sim_proto: np.ndarray) -> np.ndarray:
    """S_RPGR = (F_G . R_G . R_P)^(1/3) -- Eq. 24."""
    f_g = forward_global(sim_global)
    r_g = reciprocal_global(sim_global)
    r_p = reciprocal_prototype(sim_proto)
    return np.cbrt(np.clip(f_g * r_g * r_p, 0, None))


def no_prototype_score(sim_global: np.ndarray) -> np.ndarray:
    """S_B = sqrt(F_G . R_G) -- Eq. 26, the exact no-prototype counterpart
    used to isolate the incremental contribution of prototype evidence
    (Table 4)."""
    f_g = forward_global(sim_global)
    r_g = reciprocal_global(sim_global)
    return np.sqrt(np.clip(f_g * r_g, 0, None))


def rerank_top_pool(sim_global: np.ndarray, score: np.ndarray, direction: str,
                     n_pool: int = N_POOL) -> np.ndarray:
    """Re-order only the initial Global Top-`n_pool` candidates of each
    query according to `score`; candidates outside the pool are excluded
    from it entirely, never promoted in (Section 3.6).

    Parameters
    ----------
    sim_global : (n_wsis, n_texts) direct global similarity S_G, always in
        the WSI-rows / text-columns orientation regardless of `direction`.
    score : (n_wsis, n_texts) the score to re-rank by (rpgr_score or
        no_prototype_score output), same orientation as sim_global.
    direction : "I2T" (WSI queries, text candidates) or "T2I" (text
        queries, WSI candidates).
    n_pool : size of the initial candidate pool (50 in the paper).

    Returns
    -------
    (n_queries, n_candidates) matrix, query-rows, where every entry outside
    that query's initial Top-`n_pool` (by direct global similarity) is set
    to -inf.

    Note on out-of-pool ordering: the paper states that candidates outside
    the Top-50 "preserve their original global ordering." In this
    implementation they are all set to -inf rather than re-sorted by S_G,
    so their relative order among themselves is whatever argsort's tie
    break leaves it as (not necessarily S_G order). This has NO effect on
    any metric reported in the paper (Recall@1/3/5/10, MRR@10, mAP@10 all
    only ever consider the top 10 positions, well inside the Top-50 pool),
    but it means the literal sentence about out-of-pool ordering is not
    exactly what the code does. Flagged here rather than silently changed;
    left unchanged pending the authors' decision on the paper wording.
    """
    sim_query_view = sim_global if direction == "I2T" else sim_global.T
    n_queries, n_candidates = sim_query_view.shape
    final_scores = np.full((n_queries, n_candidates), -np.inf, dtype=np.float64)
    for qi in range(n_queries):
        top_n = np.argsort(-sim_query_view[qi])[:n_pool]
        if direction == "I2T":
            final_scores[qi, top_n] = score[qi, top_n]
        else:
            final_scores[qi, top_n] = score[top_n, qi]
    return final_scores
