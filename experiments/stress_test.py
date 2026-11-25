"""
Stress-test of the frozen VL-PFMs (Table 1) and RPGR versus mean pooling in 72 conditions (Fig. 2).

Steps (``--step``, default ``all``):

1. ``table1``     Mean pooling on the complete TCGA cohort (N=5291, organ and subtype),
                  4 VL-PFMs x 3 text conditions x 2 directions. The AI4SKIN columns of Table 1
                  are the mean-pooling values of step 2.
2. ``evaluate``   72 conditions (TCGA training + test partitions, n=4475, organ and subtype;
                  AI4SKIN complete cohort as a single gallery of 276, subtype): per-query
                  mAP@10 of RPGR and mean pooling.
3. ``bootstrap``  Paired bootstrap stratified by class, resampling clinical cases (B=20,000,
                  seed 42), Holm over the 72 conditions.

Outputs: <output_dir>/stress_test/. Run with: python -m experiments.stress_test --config <yaml>
"""
from __future__ import annotations

import data.blas  # noqa: F401  (must precede numpy)

import time

import numpy as np
import pandas as pd

from data.config import BACKBONES, TEXT_CONDITIONS, TEXT_LABEL, config_argument_parser, load_config, output_dir
from data.loaders import AI4SKIN, TCGA
from experiments.common import DIRECTIONS, macro_map, native, per_query
from rpgr.rerank import rpgr_score
from rpgr.similarity import global_similarity, mean_pool, prototype_similarity
from rpgr.statistics import class_macro, holm, paired_cluster_bootstrap_multi

METHODS = ["RPGR", "mean_pooling"]   # RPGR first: reference column
METRIC = "mAP@10"
BOOTSTRAP_B, BOOTSTRAP_SEED = 20000, 42


# ----------------------------------------------------------------------------- table 1
def table1(cfg) -> pd.DataFrame:
    tcga = TCGA(cfg)
    ids = tcga.complete_cohort_ids()
    rows = []
    for bb in BACKBONES:
        V = tcga.pooled(bb, ids)
        for cond in TEXT_CONDITIONS:
            S = global_similarity(V, tcga.text(bb, cond, ids, mode="rows"))
            for level in ["organ", "subtype"]:
                lm = tcga.label_map(ids, level)
                for d in DIRECTIONS:
                    rows.append(dict(cohort="TCGA", backbone=bb, text=TEXT_LABEL[cond], granularity=level, direction=d,
                                     n=len(ids), mAP10_mean_pooling=macro_map(native(S, d), ids, lm)))
        print(f"table 1: TCGA {bb} done", flush=True)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 72 conditions
def _collect(out, finals, ids, lm, meta):
    ref = None
    for m in METHODS:
        df = per_query(finals[m], ids, lm)
        if ref is None:
            ref = list(df.query_id)
        assert list(df.query_id) == ref, m
        out.append(pd.DataFrame({**meta, "method": m, "query_id": df.query_id, "label": df.label,
                                 METRIC: df[METRIC]}))


def evaluate_tcga(cfg) -> list[pd.DataFrame]:
    tcga = TCGA(cfg)
    tt = tcga.ids(["train", "test"])
    out = []
    for bb in BACKBONES:
        t0 = time.time()
        mp_tt = tcga.pooled(bb, tt)
        banks = tcga.prototype_banks(bb, tt)
        for cond in TEXT_CONDITIONS:
            tx_tt = tcga.text(bb, cond, tt)
            S_G = global_similarity(mp_tt, tx_tt)
            S_P = prototype_similarity(banks, tx_tt)
            for level in ["organ", "subtype"]:
                lm = tcga.label_map(tt, level)
                for d in DIRECTIONS:
                    sim, SP = native(S_G, d), native(S_P, d)
                    finals = {"RPGR": rpgr_score(sim, SP), "mean_pooling": sim}
                    _collect(out, finals, tt, lm, dict(cohort="TCGA", backbone=bb, text=TEXT_LABEL[cond], granularity=level, direction=d))
        print(f"evaluate: TCGA {bb} done ({time.time() - t0:.0f}s)", flush=True)
    return out


def evaluate_ai4skin(cfg) -> list[pd.DataFrame]:
    ai = AI4SKIN(cfg)
    ids, lm = ai.cohort_ids(), ai.cohort_label_map()
    out = []
    for bb in BACKBONES:
        store = ai.patch_store(bb)
        mp = np.stack([mean_pool(store.load(c)).astype(np.float64) for c in ids])
        banks = ai.prototype_banks(bb, ids, renormalize=True)
        for cond in TEXT_CONDITIONS:
            tx = ai.text(bb, cond, ids)
            S_G, S_P = global_similarity(mp, tx), prototype_similarity(banks, tx)
            for d in DIRECTIONS:
                sim, SP = native(S_G, d), native(S_P, d)
                finals = {"RPGR": rpgr_score(sim, SP), "mean_pooling": sim}
                _collect(out, finals, ids, lm, dict(cohort="AI4SKIN", backbone=bb, text=TEXT_LABEL[cond], granularity="subtype", direction=d))
        print(f"evaluate: AI4SKIN {bb} done", flush=True)
    return out


COND = ["cohort", "backbone", "text", "granularity", "direction"]


def conditions_table(pq: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, g in pq.groupby(COND, sort=False):
        r = dict(zip(COND, key), n_queries=int((g.method == "RPGR").sum()))
        for m in METHODS:
            x = g[g.method == m]
            r[f"mAP10_{m}"] = class_macro(x[METRIC].to_numpy(), x.label.to_numpy())
        rows.append(r)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- bootstrap
def bootstrap(cfg, pq: pd.DataFrame) -> pd.DataFrame:
    """RPGR minus mean pooling in each condition."""
    case_of = TCGA(cfg).labels()["case_id"]
    res = []
    for key, g in pq.groupby(COND, sort=False):
        w = {m: g[g.method == m].reset_index(drop=True) for m in METHODS}
        qid, lab = w["RPGR"].query_id.values, w["RPGR"].label.values
        units = case_of.loc[qid].values if key[0] == "TCGA" else qid
        V = np.column_stack([w[m][METRIC].values for m in METHODS])
        r = paired_cluster_bootstrap_multi(V, 0, lab, units, B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED)[1]
        res.append(dict(zip(COND, key), B=BOOTSTRAP_B, n_queries=len(qid), **r))
    res = pd.DataFrame(res)
    assert len(res) == 72
    res["holm72"] = holm(res.p.values)
    return res


def main() -> None:
    p = config_argument_parser(__doc__)
    p.add_argument("--step", default="all", choices=["all", "table1", "evaluate", "bootstrap"])
    args = p.parse_args()
    cfg = load_config(args.config)
    out = output_dir(cfg, "stress_test")
    if args.step in ("all", "table1"):
        table1(cfg).to_csv(out / "table1_mean_pooling_tcga.csv", index=False)
    if args.step in ("all", "evaluate"):
        pq = pd.concat(evaluate_tcga(cfg) + evaluate_ai4skin(cfg), ignore_index=True)
        pq.to_csv(out / "per_query.csv.gz", index=False, compression="gzip")
        conditions_table(pq).to_csv(out / "conditions.csv", index=False)
    if args.step in ("all", "bootstrap"):
        bootstrap(cfg, pd.read_csv(out / "per_query.csv.gz")).to_csv(out / "bootstrap.csv", index=False)
    print("written", out)


if __name__ == "__main__":
    main()
