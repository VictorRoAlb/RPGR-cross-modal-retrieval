"""
Training-free hubness corrections applied to the mean-pooling similarities.

All functions take a (n_query, n_gallery) matrix in query-row orientation.

* Inverted softmax (IS; Smith et al., 2017): transductive softmax over the query axis.
* CSLS (Conneau et al., 2018).
* QB-Norm with dynamic inverted softmax (Bogolin et al., 2022), following the official
  ``dynamic_inverted_softmax.py``: activation set = unique top-1 gallery index retrieved by
  each query-bank item; the normalisation is applied only to queries whose own top-1 is in
  that set.
* DBNorm / DualIS (Wang et al., 2023), Eq. 1 of the paper (Eq. 2 for the dynamic variant),
  implemented from the equations.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def inverted_softmax(S: np.ndarray, tau: float) -> np.ndarray:
    """Log-space softmax over the query axis for each gallery column (rank-preserving)."""
    logits = S / tau
    return logits - logsumexp(logits, axis=0, keepdims=True)


def csls(S: np.ndarray, k: int) -> np.ndarray:
    """2 S - r_q - r_c, with r_q / r_c the mean of the k largest similarities of each query / candidate."""
    k_q = min(k, S.shape[1])
    k_c = min(k, S.shape[0])
    r_q = np.sort(S, axis=1)[:, -k_q:].mean(axis=1, keepdims=True)
    r_c = np.sort(S, axis=0)[-k_c:, :].mean(axis=0, keepdims=True)
    return 2 * S - r_q - r_c


def qbnorm_dis(S_eval: np.ndarray, S_bank: np.ndarray, beta: float) -> tuple[np.ndarray, float]:
    """QB-Norm (DIS). S_bank: (n_bank_queries, n_gallery), query bank against the same gallery.
    Returns the corrected scores and the fraction of queries that were normalised."""
    activation_set = set(np.argmax(S_bank, axis=1).tolist())
    lse_bank = logsumexp(beta * S_bank, axis=0, keepdims=True)
    dis_scores = beta * S_eval - lse_bank
    out = S_eval.copy()
    mask = np.isin(np.argmax(S_eval, axis=1), list(activation_set))
    out[mask, :] = dis_scores[mask, :]
    return out, float(mask.mean())


def dbnorm(S_eval: np.ndarray, S_unimodal_gallery_bank: np.ndarray, S_crossmodal_gallery_bank: np.ndarray,
           beta1: float, beta2: float, dynamic: bool = False) -> np.ndarray:
    """DualIS (dynamic=False, used in the paper) or DualDIS.

    S_unimodal_gallery_bank: (n_gallery, n_gallery_bank), same modality as the gallery.
    S_crossmodal_gallery_bank: (n_gallery, n_query_bank), gallery items against the query bank."""
    lse_g1 = logsumexp(beta1 * S_unimodal_gallery_bank, axis=1)
    lse_g2 = logsumexp(beta2 * S_crossmodal_gallery_bank, axis=1)
    if not dynamic:
        return (beta1 + beta2) * S_eval - lse_g1[None, :] - lse_g2[None, :]
    term1 = beta1 * S_eval - lse_g1[None, :]
    term2 = beta2 * S_eval - lse_g2[None, :]
    act_g1 = set(np.argmax(S_unimodal_gallery_bank, axis=0).tolist())
    act_g2 = set(np.argmax(S_crossmodal_gallery_bank, axis=0).tolist())
    gcols = np.arange(S_eval.shape[1])
    use1, use2 = np.isin(gcols, list(act_g1)), np.isin(gcols, list(act_g2))
    out = S_eval.copy()
    out[:, use1 & use2] = term1[:, use1 & use2] + term2[:, use1 & use2]
    out[:, use1 & ~use2] = term1[:, use1 & ~use2]
    out[:, ~use1 & use2] = term2[:, ~use1 & use2]
    return out
