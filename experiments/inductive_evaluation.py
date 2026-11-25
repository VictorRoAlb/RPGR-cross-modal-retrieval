"""
Same-case exclusion and inductive evaluation on the TCGA test partition (Section 5.5):
KEEP, Morphology-only, subtype level, n=1578 WSIs from 1237 cases.

Settings:
  a  standard: whole gallery, statistics over the full query set;
  b  own case excluded from the gallery (all WSIs of the query's case are masked);
  c  incremental: each query is scored with reciprocal statistics computed over the gallery
     with its own case removed (the query itself included);
  d  strictly inductive: reciprocal statistics from a fixed reference population (the 2897
     training WSIs / reports), independent of the query.
In b-d, relevance counts only same-subtype candidates left in the gallery. Methods: mean
pooling, R_G alone and RPGR; paired bootstrap of RPGR minus each by clinical case
(B=20,000, seed 42).

Run with: python -m experiments.inductive_evaluation --config <yaml>
"""
from __future__ import annotations

import data.blas  # noqa: F401  (must precede numpy)

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from data.config import config_argument_parser, load_config, output_dir
from data.loaders import TCGA
from rpgr.calibration import direct_term, reciprocal_term
from rpgr.similarity import prototype_similarity
from rpgr.statistics import class_macro, paired_cluster_bootstrap

EPS = 1e-8
BACKBONE, CONDITION = "KEEP", "morphology_only"
SETTINGS = ["a_standard", "b_own_case_excluded", "c_incremental", "d_strict_inductive_training_reference"]


def run(cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    tcga = TCGA(cfg)
    ids, tr = tcga.ids("test"), tcga.ids("train")
    lab = tcga.labels()
    y, case = lab.loc[ids, "OncotreeCode"].to_numpy(), lab.loc[ids, "case_id"].to_numpy()
    V, T = tcga.pooled(BACKBONE, ids), tcga.text(BACKBONE, CONDITION, ids, mode="rows")
    banks = tcga.prototype_banks(BACKBONE, ids)
    SG, SP = V @ T.T, prototype_similarity(banks, T)
    Vtr, Ttr = tcga.pooled(BACKBONE, tr), tcga.text(BACKBONE, CONDITION, tr, mode="rows")
    banks_tr = tcga.prototype_banks(BACKBONE, tr)
    REF = {"I2T": (Vtr @ T.T, prototype_similarity(banks_tr, T)), "T2I": ((V @ Ttr.T).T, prototype_similarity(banks, Ttr).T)}
    EX = case[:, None] == case[None, :]
    n = len(ids)

    def metrics_rows(score, excl):
        s = np.where(excl, -np.inf, score)
        order = np.argsort(-s, axis=1, kind="stable")[:, :10]
        rel = y[order] == y[:, None]
        tot = ((y[None, :] == y[:, None]) & ~excl).sum(1)
        cum = np.cumsum(rel, 1)
        ap = np.where(tot > 0, (rel * cum / np.arange(1, 11)).sum(1) / np.maximum(np.minimum(tot, 10), 1), np.nan)
        return ap, rel[:, 0].astype(float), tot > 0

    def recip_row_inductive(X, q, ref):
        R = X[ref]
        sd = np.clip(R.std(axis=0), EPS, None)
        return X[q] / sd - logsumexp(R / sd, axis=0)

    def direct_row(x, keep):
        sd = max(x[keep].std(), EPS)
        out = np.full_like(x, -np.inf)
        out[keep] = x[keep] / sd - logsumexp(x[keep] / sd)
        return out

    rows, store = [], {}
    for d in ["I2T", "T2I"]:
        G, P = (SG, SP) if d == "I2T" else (SG.T, SP.T)
        noex = np.zeros_like(EX)
        full = {"mean_pooling": G, "global_only": reciprocal_term(G), "RPGR": reciprocal_term(G) + 0.5 * (direct_term(P) + reciprocal_term(P))}
        for setting in SETTINGS:
            excl = noex if setting == "a_standard" else EX
            scores = dict(full)
            if setting == "c_incremental":
                ind = {k: v.copy() for k, v in full.items() if k != "mean_pooling"}
                for q in range(n):
                    keep = ~EX[q]
                    ref = keep.copy()
                    ref[q] = True
                    rg, rp = recip_row_inductive(G, q, ref), recip_row_inductive(P, q, ref)
                    ind["global_only"][q] = rg
                    ind["RPGR"][q] = rg + 0.5 * (direct_row(P[q], keep) + rp)
                scores.update(ind)
            if setting == "d_strict_inductive_training_reference":
                GR, PR = REF[d]

                def recip_fixed(X, XR):
                    sd = np.clip(XR.std(axis=0), EPS, None)
                    return X / sd - logsumexp(XR / sd, axis=0)[None, :]
                rg, rp = recip_fixed(G, GR), recip_fixed(P, PR)
                dp = np.vstack([direct_row(P[q], ~EX[q]) for q in range(n)])
                scores = {"mean_pooling": G, "global_only": rg, "RPGR": rg + 0.5 * (dp + rp)}
            for m, S in scores.items():
                ap, r1, elig = metrics_rows(S, excl)
                store[(setting, d, m)] = (ap, r1, elig)
                rows.append(dict(setting=setting, direction=d, method=m, n_eligible=int(elig.sum()),
                                 mAP10=class_macro(ap[elig], y[elig]), R1=class_macro(r1[elig], y[elig])))
            print(d, setting, "done", flush=True)
    tests = []
    for (setting, d, m), (ap, r1, elig) in store.items():
        if m == "RPGR":
            continue
        rp = store[(setting, d, "RPGR")]
        e = elig & rp[2]
        for metric, a, b in [("mAP@10", ap, rp[0]), ("Recall@1", r1, rp[1])]:
            r = paired_cluster_bootstrap(a[e], b[e], y[e], case[e], B=20000, seed=42)
            tests.append(dict(setting=setting, direction=d, comparator=m, metric=metric, diff_RPGR_minus=r["diff"],
                              ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], p=r["p"], n_units=r["n_units"]))
    return pd.DataFrame(rows), pd.DataFrame(tests)


def main() -> None:
    cfg = load_config(config_argument_parser(__doc__).parse_args().config)
    out = output_dir(cfg, "inductive_evaluation")
    metrics, tests = run(cfg)
    metrics.to_csv(out / "metrics.csv", index=False)
    tests.to_csv(out / "tests.csv", index=False)
    print("written", out)


if __name__ == "__main__":
    main()
