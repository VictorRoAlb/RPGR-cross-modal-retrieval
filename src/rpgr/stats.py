"""
stats.py — paired, class-stratified bootstrap of a class-macro
difference with a configurable resampling UNIT (query row, slide, physical case).

Estimand: macro over classes of the mean per-row value, diff = macro(b) - macro(a),
rows pooled.
Resampling: within each class, draw n_units_c units with replacement; every row
of a drawn unit enters the replicate (cluster bootstrap). Same draws for a and b
(paired). p = max(2*min(P(d<=0), P(d>=0)), 1/B). Units must be nested in a single
class (asserted).
"""
from __future__ import annotations

import numpy as np


def class_macro(values, labels):
    classes = np.unique(labels)
    return float(np.mean([values[labels == c].mean() for c in classes]))


def paired_cluster_bootstrap(va, vb, labels, units, B=2000, seed=42, fold_ids=None):
    """If fold_ids is given, the estimand is the mean over folds of the per-fold
    class-macro (sensitivity variant); resampling is still by unit."""
    va, vb = np.asarray(va, float), np.asarray(vb, float)
    labels, units = np.asarray(labels), np.asarray(units)
    rng = np.random.default_rng(seed)
    u_cls = {}
    for u, c in zip(units, labels):
        assert u_cls.setdefault(u, c) == c, f"unit {u} spans >1 class"
    uniq_units = np.array(sorted(u_cls))
    uidx = {u: i for i, u in enumerate(uniq_units)}
    row_u = np.array([uidx[u] for u in units])
    classes = np.unique(labels)

    if fold_ids is None:
        strata = [None]
        fold_ids = np.zeros(len(va), dtype=int)
    else:
        fold_ids = np.asarray(fold_ids)
        strata = list(np.unique(fold_ids))
    # per (fold, unit) sums and counts for both methods
    nU = len(uniq_units)
    acc = {}
    for f in (strata if strata[0] is not None else [0]):
        m = fold_ids == f
        sa = np.bincount(row_u[m], weights=va[m], minlength=nU)
        sb = np.bincount(row_u[m], weights=vb[m], minlength=nU)
        cn = np.bincount(row_u[m], minlength=nU).astype(float)
        acc[f] = (sa, sb, cn)
    units_of_class = {c: np.array([uidx[u] for u in uniq_units if u_cls[u] == c]) for c in classes}

    def stat(weights_per_class):  # weights_per_class: {c: (B, nU_c) counts of draws per unit}
        out = []
        for f, (sa, sb, cn) in acc.items():
            diffs_c = []
            for c, w in weights_per_class.items():
                ids = units_of_class[c]
                n = w @ cn[ids]
                ok = n > 0
                d = np.full(w.shape[0], np.nan)
                d[ok] = (w[ok] @ sb[ids] - w[ok] @ sa[ids]) / n[ok]
                diffs_c.append(d)
            out.append(np.nanmean(np.vstack(diffs_c), axis=0))  # macro over classes present in fold
        return np.mean(np.vstack(out), axis=0)

    point = stat({c: np.ones((1, len(ids))) for c, ids in units_of_class.items()})[0]
    W = {}
    for c, ids in units_of_class.items():
        draws = rng.integers(0, len(ids), size=(B, len(ids)))
        w = np.zeros((B, len(ids)))
        np.add.at(w, (np.repeat(np.arange(B), len(ids)), draws.ravel()), 1.0)
        W[c] = w
    diffs = stat(W)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = max(2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean()), 1.0 / B)
    return dict(diff=float(point), ci_lo=float(lo), ci_hi=float(hi), p=float(min(p, 1.0)),
                n_rows=len(va), n_units=nU, sd=float(diffs.std(ddof=1)))


def holm(pvals, alpha=0.05):
    p = np.asarray(pvals, float)
    order = np.argsort(p, kind="stable")
    rej = np.zeros(len(p), bool)
    for r, i in enumerate(order):
        if p[i] <= alpha / (len(p) - r):
            rej[i] = True
        else:
            break
    return rej


def paired_cluster_bootstrap_multi(V, ref_col, labels, units, B=20000, seed=42, chunk=1000):
    """Same estimand/test as paired_cluster_bootstrap, for several comparators at once
    and large n (memory-safe chunks). V: (n_rows, n_methods); returns, for every column j,
    diff = macro(V[:, ref_col]) - macro(V[:, j]) with the SAME bootstrap draws for all j
    (draw counts via multinomial per class). Used for the 72-condition stress test (n=4475)."""
    V = np.asarray(V, float)
    labels, units = np.asarray(labels), np.asarray(units)
    u_cls = {}
    for u, c in zip(units, labels):
        assert u_cls.setdefault(u, c) == c, f"unit {u} spans >1 class"
    uniq = np.array(sorted(u_cls)); uidx = {u: i for i, u in enumerate(uniq)}
    row_u = np.array([uidx[u] for u in units]); nU = len(uniq)
    S = np.stack([np.bincount(row_u, weights=V[:, j], minlength=nU) for j in range(V.shape[1])], 1)  # (nU, m)
    Cn = np.bincount(row_u, minlength=nU).astype(float)
    classes = np.unique(labels)
    ids_c = {c: np.array([uidx[u] for u in uniq if u_cls[u] == c]) for c in classes}
    point_c = np.stack([S[ids_c[c]].sum(0) / Cn[ids_c[c]].sum() for c in classes])  # (nC, m)
    point = point_c.mean(0); point_diff = point[ref_col] - point
    rng = np.random.default_rng(seed)
    out = []
    for s in range(0, B, chunk):
        b = min(chunk, B - s); acc = np.zeros((b, V.shape[1]))
        for c in classes:
            ids = ids_c[c]
            W = rng.multinomial(len(ids), np.full(len(ids), 1.0 / len(ids)), size=b).astype(float)
            acc += (W @ S[ids]) / (W @ Cn[ids])[:, None]
        m = acc / len(classes)
        out.append(m[:, [ref_col]] - m)
    D = np.vstack(out)  # (B, m)
    res = []
    for j in range(V.shape[1]):
        d = D[:, j]
        p = max(2.0 * min((d <= 0).mean(), (d >= 0).mean()), 1.0 / B) if j != ref_col else 1.0
        lo, hi = np.percentile(d, [2.5, 97.5])
        res.append(dict(diff=float(point_diff[j]), ci_lo=float(lo), ci_hi=float(hi), p=float(min(p, 1.0)), n_units=nU))
    return res
