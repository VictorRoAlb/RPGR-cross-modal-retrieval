"""
prototypes.py
=============
Adaptive WSI-specific local prototype construction (Eqs. 6-18 of the RPGR
paper): for each WSI, k-means is evaluated over a discrete grid of cluster
counts H, a coverage/support utility is computed for each feasible H, the
smallest H reaching 97% of the best observed utility is selected (H*), and
the resulting centroids are refined into similarity-weighted prototypes.

This module is pure computation, extracted unchanged from the algorithmic
core of the original prototype-construction library (part 1 of
tcga_adaptive_lib.py). It performs no file I/O: it takes an in-memory patch
matrix and returns the prototype bank. The batch/glue logic that discovers
cases, reads .npy files from disk, parallelizes with joblib, and writes
.npz caches (part 2 of the original file) is intentionally NOT included
here -- that is dataset-specific orchestration, not part of the method,
and belongs in scripts/.

Note: this function also computes and returns `q_proto`, a per-case
"prototype confidence" score, kept for provenance/parity with the original
implementation. RPGR itself (reranking.py) does NOT use q_proto -- the
final formula combines global and prototype evidence through percentile
calibration and a geometric mean (Eq. 24), with no confidence-weighted
mixing term.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.cluster import MiniBatchKMeans


@dataclass(frozen=True)
class AdaptiveSettings:
    k_grid: tuple[int, ...] = (2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 200)
    min_support: int = 6       # tau_min (Eq. 7)
    good_support: int = 20     # tau_sup (Eq. 10)
    near_best_ratio: float = 0.97  # rho (Eq. 13)
    random_state: int = 42
    n_init: int = 5
    max_iter: int = 200
    batch_size: int = 1024
    softmax_tau: float = 0.05  # tau_sm (Eq. 15)


def _clip01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def _normalize(array: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=axis, keepdims=True)
    return (arr / np.clip(norms, eps, None)).astype(np.float32)


def _mean_prototype(patch_matrix: np.ndarray) -> np.ndarray:
    if patch_matrix.shape[0] == 0:
        return np.zeros((patch_matrix.shape[1],), dtype=np.float32)
    mean_vector = _normalize(patch_matrix, axis=1).mean(axis=0)
    return _normalize(mean_vector, axis=0)


def _weighted_centroid_refine(
    normalized_patches: np.ndarray, assignments: np.ndarray, plain_prototypes: np.ndarray, tau: float,
) -> np.ndarray:
    """Eqs. 14-16: recompute each cluster's centroid from its actually
    assigned patches (not the k-means centroid), then take the
    softmax-weighted mean of those patches around that recomputed centroid.
    """
    k_total, dim = plain_prototypes.shape
    refined = np.empty((k_total, dim), dtype=np.float32)
    for k in range(k_total):
        cluster_patches = normalized_patches[assignments == k]
        if cluster_patches.shape[0] == 0:
            refined[k] = plain_prototypes[k]
            continue
        cluster_centroid = _normalize(cluster_patches.mean(axis=0), axis=0)  # Eq. 14
        sims = cluster_patches @ cluster_centroid
        logits = (sims / tau).astype(np.float64)
        logits -= logits.max()  # numerical stability only, does not change the softmax
        weights = np.exp(logits)
        weights /= weights.sum()  # Eq. 15
        refined[k] = (weights[:, None] * cluster_patches).sum(axis=0).astype(np.float32)  # Eq. 16
    return _normalize(refined, axis=1)


def _fallback_result(case_id: str, patch_matrix: np.ndarray, settings: AdaptiveSettings, notes: str) -> dict[str, Any]:
    """Eq. 18: N_i < 2*tau_min -> single prototype = normalized mean patch."""
    n_patches = int(patch_matrix.shape[0])
    embedding_dim = int(patch_matrix.shape[1]) if patch_matrix.ndim == 2 else 0
    if n_patches > 0:
        prototype_vector = _mean_prototype(patch_matrix).reshape(1, -1)
    else:
        prototype_vector = np.zeros((1, embedding_dim), dtype=np.float32)
    support = _clip01(n_patches / max(settings.good_support, 1))
    return {
        "case_id": case_id,
        "K_star": 1,
        "q_proto": 0.0,
        "prototypes": prototype_vector.astype(np.float32),
        "n_patches": n_patches,
        "coverage": math.nan,
        "support": support,
        "utility": 0.0,
        "selected_by": "fallback",
        "notes": notes,
    }


def build_adaptive_prototypes_for_case(
    case_id: str, patch_matrix: np.ndarray, settings: AdaptiveSettings,
) -> dict[str, Any]:
    """Eqs. 6-18: adaptive WSI-specific prototype bank for one WSI.

    Deterministic given (patch_matrix, settings): same random_state always
    produces the same result.

    Parameters
    ----------
    case_id : identifier, carried through to the output only.
    patch_matrix : (N_i, d) patch-level visual embeddings for one WSI
        (need not be pre-normalized).
    settings : AdaptiveSettings.

    Returns
    -------
    dict with keys: case_id, K_star, prototypes ((K_star, d) L2-normalized),
    q_proto, n_patches, coverage, support, utility, selected_by, notes.
    """
    patch_matrix = np.asarray(patch_matrix, dtype=np.float32)
    if patch_matrix.ndim != 2:
        raise ValueError(f"{case_id}: expected a 2D patch matrix, got {patch_matrix.shape}")

    normalized_patches = _normalize(patch_matrix, axis=1)
    n_patches = int(normalized_patches.shape[0])
    if n_patches < 2:
        return _fallback_result(case_id, patch_matrix, settings, "fallback_low_patch_support")

    valid_k = sorted({
        int(k) for k in settings.k_grid
        if int(k) <= n_patches and (n_patches / int(k)) >= settings.min_support  # Eq. 7
    })
    if not valid_k:
        return _fallback_result(case_id, patch_matrix, settings, "fallback_low_patch_support")

    records: dict[int, dict[str, Any]] = {}
    for k_value in valid_k:
        batch_size = min(max(int(settings.batch_size), 1), max(n_patches, 1))
        try:
            kmeans = MiniBatchKMeans(
                n_clusters=int(k_value), random_state=int(settings.random_state),
                n_init=int(settings.n_init), max_iter=int(settings.max_iter), batch_size=batch_size,
            )
            assignments = kmeans.fit_predict(normalized_patches).astype(np.int16)
            prototypes = _normalize(np.asarray(kmeans.cluster_centers_, dtype=np.float32), axis=1)
            sim = normalized_patches @ prototypes.T
            coverage = float(np.max(sim, axis=1).mean()) if sim.size else math.nan  # Eq. 9
            support = _clip01((n_patches / k_value) / max(settings.good_support, 1))  # Eq. 10
            utility = float(coverage * support) if math.isfinite(coverage) else math.nan  # Eq. 11
            records[int(k_value)] = {
                "K": int(k_value), "prototypes": prototypes.astype(np.float32),
                "assignments": assignments, "coverage": coverage, "support": support, "utility": utility,
            }
        except Exception:  # noqa: BLE001 -- one failed K must not invalidate the others
            continue

    valid_results = [r for r in records.values() if math.isfinite(r["utility"])]
    if not valid_results:
        return _fallback_result(case_id, patch_matrix, settings, "fallback_kmeans_failure")

    best_utility = float(max(r["utility"] for r in valid_results))  # Eq. 12 (U*_i)
    threshold = float(settings.near_best_ratio * best_utility)
    near_best = [r for r in valid_results if r["utility"] >= threshold]
    selected = min(near_best, key=lambda r: r["K"]) if near_best else max(valid_results, key=lambda r: r["utility"])  # Eq. 13

    q_proto = 0.0 if (not math.isfinite(best_utility) or best_utility <= 0.0) else _clip01(float(selected["utility"]))

    refined_prototypes = _weighted_centroid_refine(
        normalized_patches, selected["assignments"], selected["prototypes"], tau=settings.softmax_tau,
    )

    return {
        "case_id": case_id,
        "K_star": int(selected["K"]),
        "q_proto": q_proto,
        "prototypes": refined_prototypes,
        "n_patches": n_patches,
        "coverage": float(selected["coverage"]),
        "support": float(selected["support"]),
        "utility": float(selected["utility"]),
        "selected_by": "near_best_utility",
        "notes": "ok",
    }
