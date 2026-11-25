"""
Computational cost by stage (Table C.3): TCGA test, KEEP, Morphology-only reports, one CPU
thread, medians over repetitions (pin the process to one core, e.g. ``taskset -c 0``).

Stages:
  * aggregation of one WSI with its patches in memory (mean / max pooling; 50 WSIs);
  * prototype construction of one WSI (offline, once per gallery WSI);
  * batch, whole 1578 x 1578 set: S_G, S_P (max_h), calibration given the similarities
    (RPGR, IS, CSLS, QB-Norm, DBNorm, RR with K=30), ranking (stable sort of all rows);
  * incremental update of one query appended to the 1578-query reference set (RPGR, IS),
    computed exactly (checked against recomputing the stacked matrix).
Timings depend on the hardware; the paper used one AMD EPYC 7742 thread.

Run with: python -m experiments.compute_cost --config <yaml> [--n-prototype-wsis 5]
"""
from __future__ import annotations

import data.blas  # noqa: F401  (must precede numpy)

import time

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from baselines.hubness import csls, dbnorm, inverted_softmax, qbnorm_dis
from baselines.pooling import max_pool
from baselines.reciprocal_rerank import rr_scores
from data.config import config_argument_parser, load_config, output_dir
from data.loaders import PatchStore, TCGA
from rpgr.calibration import direct_term, reciprocal_term
from rpgr.prototypes import AdaptiveSettings, build_adaptive_prototypes_for_case
from rpgr.rerank import rpgr_score
from rpgr.similarity import mean_pool

SEED, WARM, REP_AGG, REP_ROW, REP_BATCH, N_WSI, N_QUERY = 42, 3, 30, 5, 10, 50, 50
TAU, CSLS_K, BETA, DB_BETAS, RR_K = 0.2, 10, 5, (0.5, 5), 30   # cost does not depend on these values


def timeit(fn, reps, warm=WARM) -> np.ndarray:
    for _ in range(warm):
        fn()
    ts = np.empty(reps)
    for i in range(reps):
        t0 = time.perf_counter_ns()
        fn()
        ts[i] = (time.perf_counter_ns() - t0) / 1e6
    return ts


def _std_append(s1, s2, n, x):
    m = (s1 + x) / (n + 1)
    return np.clip(np.sqrt(np.clip((s2 + x * x) / (n + 1) - m * m, 0, None)), 1e-8, None)


def rpgr_row_incremental(G, P, st, g, p):
    """Exact RPGR row of a query appended to the reference query set (G, P)."""
    sg, sp = _std_append(st["g1"], st["g2"], st["n"], g), _std_append(st["p1"], st["p2"], st["n"], p)
    lg = np.logaddexp(logsumexp(G / sg, axis=0), g / sg)
    lp = np.logaddexp(logsumexp(P / sp, axis=0), p / sp)
    dp = p / max(p.std(), 1e-8)
    dp = dp - logsumexp(dp)
    return (g / sg - lg) + 0.5 * (dp + (p / sp - lp))


def is_row_incremental(st, g):
    return g / TAU - np.logaddexp(st["is_lse"], g / TAU)


def run(cfg, n_proto: int) -> pd.DataFrame:
    tcga = TCGA(cfg)
    te, tr = tcga.ids("test"), tcga.ids("train")
    store = PatchStore(tcga.backbone("KEEP")["patch_dir"])
    V, Vtr = tcga.pooled("KEEP", te), tcga.pooled("KEEP", tr)
    T, Ttr = tcga.text("KEEP", "morphology_only", te, mode="rows"), tcga.text("KEEP", "morphology_only", tr, mode="rows")
    banks = tcga.prototype_banks("KEEP", te)
    stack = np.concatenate(banks)
    starts = np.concatenate([[0], np.cumsum([b.shape[0] for b in banks])[:-1]])
    rng = np.random.default_rng(SEED)
    rows = []

    def rec(stage, method, ts, unit="ms"):
        rows.append(dict(stage=stage, method=method, median_ms=float(np.median(ts)), p25_ms=float(np.percentile(ts, 25)),
                         p75_ms=float(np.percentile(ts, 75)), n=len(ts)))

    sel = sorted(rng.choice(len(te), size=N_WSI, replace=False).tolist())
    agg_mean, agg_max = [], []
    for j in sel:
        P = store.load(te[j])
        agg_mean.append(np.median(timeit(lambda: mean_pool(P), REP_AGG)))
        agg_max.append(np.median(timeit(lambda: max_pool(P), REP_AGG)))
    rec("aggregation per WSI (patches in memory)", "Mean pooling", np.array(agg_mean))
    rec("aggregation per WSI (patches in memory)", "Max pooling", np.array(agg_max))
    proto = []
    for j in sel[:n_proto]:
        P = store.load(te[j])
        t0 = time.perf_counter_ns()
        build_adaptive_prototypes_for_case(te[j], P, AdaptiveSettings())
        proto.append((time.perf_counter_ns() - t0) / 1e6)
    rec("prototype construction (offline, once per WSI)", "RPGR", np.array(proto))

    S_G = V @ T.T
    S_P = np.maximum.reduceat(stack @ T.T, starts, axis=0)
    bank, uni = Vtr @ T.T, T @ Ttr.T
    rec("batch: global similarity S_G", "All methods", timeit(lambda: V @ T.T, REP_BATCH))
    rec("batch: local similarity S_P (max_h)", "RPGR", timeit(lambda: np.maximum.reduceat(stack @ T.T, starts, axis=0), REP_BATCH))
    rec("batch: calibration given the similarities", "RPGR", timeit(lambda: rpgr_score(S_G, S_P), REP_BATCH))
    rec("batch: calibration given the similarities", "IS", timeit(lambda: inverted_softmax(S_G, TAU), REP_BATCH))
    rec("batch: calibration given the similarities", "CSLS", timeit(lambda: csls(S_G, CSLS_K), REP_BATCH))
    rec("batch: calibration given the similarities", "QB-Norm", timeit(lambda: qbnorm_dis(S_G, bank, BETA), REP_BATCH))
    rec("batch: calibration given the similarities", "DBNorm", timeit(lambda: dbnorm(S_G, uni, bank.T, *DB_BETAS), REP_BATCH))
    rec("batch: calibration given the similarities", "RR (K=30)", timeit(lambda: rr_scores(S_G, RR_K), REP_BATCH))
    rec("batch: ranking (sort of all rows)", "All methods", timeit(lambda: np.argsort(-S_G, axis=1, kind="stable"), REP_BATCH))

    st = dict(g1=S_G.sum(0), g2=(S_G ** 2).sum(0), p1=S_P.sum(0), p2=(S_P ** 2).sum(0), n=S_G.shape[0],
              is_lse=logsumexp(S_G / TAU, axis=0))
    q = sel[0]
    g, p = S_G[q].copy(), S_P[q].copy()
    Gs, Ps = np.vstack([S_G, g]), np.vstack([S_P, p])
    assert np.abs(rpgr_row_incremental(S_G, S_P, st, g, p) - (reciprocal_term(Gs) + 0.5 * (direct_term(Ps) + reciprocal_term(Ps)))[-1]).max() < 1e-9
    assert np.abs(is_row_incremental(st, g) - inverted_softmax(Gs, TAU)[-1]).max() < 1e-9
    qs = sorted(rng.choice(len(te), size=N_QUERY, replace=False).tolist())
    rec("incremental update of one query", "RPGR",
        np.array([np.median(timeit(lambda: rpgr_row_incremental(S_G, S_P, st, S_G[i], S_P[i]), REP_ROW)) for i in qs]))
    rec("incremental update of one query", "IS", np.array([np.median(timeit(lambda: is_row_incremental(st, S_G[i]), REP_ROW)) for i in qs]))
    return pd.DataFrame(rows)


def main() -> None:
    p = config_argument_parser(__doc__)
    p.add_argument("--n-prototype-wsis", type=int, default=5)
    args = p.parse_args()
    cfg = load_config(args.config)
    df = run(cfg, args.n_prototype_wsis)
    df.to_csv(output_dir(cfg, "compute_cost") / "table_c3_cost.csv", index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
