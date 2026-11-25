"""
Cross-method benchmark on the held-out test sets (Fig. 3): KEEP, Morphology-only reports,
subtype level; TCGA test partition (n=1578) and AI4SKIN five-fold cross-validation (each test
query ranked within its fold, 276 out-of-fold queries pooled).

Steps (``--step``, default ``all``):

1. ``validation``  Hyperparameter and re-ranking regime of every training-free comparator,
                   selected on validation data: TCGA validation partition; AI4SKIN, the
                   validation set of each fold (one selection per fold). RR uses K=30 on TCGA
                   (fixed in advance) and, on AI4SKIN, the K* of each fold selected on its
                   validation set from {6, 7, 10, 15, 20, 30}.
2. ``evaluate``    Test score matrices of the 15 methods; per-query metrics with a stable order
                   and with the tie-neutral protocol (50 gallery permutations, seeds 0-49).
                   Trained comparators and slide-level models are evaluated from their exported
                   embeddings. TITAN and PRISM lack embeddings for 4 and 1 TCGA test WSIs; RPGR
                   is recomputed on their galleries for the paired comparison.
3. ``stats``       RPGR minus each comparator (tie-neutral mAP@10), paired bootstrap stratified
                   by class and resampling clinical cases (B=20,000, seed 42), Holm over the 14
                   comparators of each cohort and direction.

Outputs: <output_dir>/benchmark/. Run with: python -m experiments.benchmark --config <yaml>
"""
from __future__ import annotations

import data.blas  # noqa: F401  (must precede numpy)

import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from baselines.candidate_pool import REGIMES, rerank_candidate_pool
from baselines.hubness import csls, dbnorm, inverted_softmax, qbnorm_dis
from baselines.pooling import max_pool
from baselines.reciprocal_rerank import rerank_bidirectional, rr_scores, score_matrix_from_order
from baselines.topk import reciprocal_topk_score, topk_patch_text
from baselines.trained import embedding_scores
from data.config import config_argument_parser, load_config, output_dir
from data.loaders import AI4SKIN, TCGA
from experiments.common import DIRECTIONS, best_by_validation, macro_map, native
from experiments.validation_cells import BACKBONE, CONDITION, ai4skin_validation_cell, tcga_validation_cell
from rpgr.metrics import evaluate_direction
from rpgr.rerank import rpgr_score
from rpgr.similarity import global_similarity, l2_normalize_rows, mean_pool, prototype_similarity
from rpgr.statistics import class_macro, holm, paired_cluster_bootstrap
from rpgr.ties import TIE_SEEDS, tie_neutral_per_query

TAU_GRID = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
CSLS_GRID = [1, 5, 10, 20, 30, 50, 75, 100, 150]
BETA_GRID = [5, 10, 20, 50, 100]
BETA12_GRID = [0.5, 1, 3, 5, 10, 20]
TOPK_GRID = [1, 5, 10, 20, 50, 100]
RR_K_TCGA = 30
RR_K_GRID_AI4SKIN = [6, 7, 10, 15, 20, 30]
BOOTSTRAP_B, BOOTSTRAP_SEED = 20000, 42
# The published scores of the trained and slide-level comparators were computed with 8 BLAS
# threads; one AI4SKIN FGCR query has a near tie whose order depends on the summation order.
COMPARATOR_EMBEDDING_BLAS_THREADS = 8
COMPARATORS = ["mean_pooling", "max_pooling", "CSLS", "IS", "QBNorm", "DBNorm", "RR", "topk_reciprocal",
               "TransMIL", "SETMIL", "DTN", "FGCR", "TITAN", "PRISM"]


# ----------------------------------------------------------------------------- validation
def _sweep_cell(rows, cohort, cell, d):
    S = cell[d]["S_G"]
    ids, lm = cell["ids"], cell["label_map"]

    def rec(method, param, score):
        for rname, npool in REGIMES.items():
            rows.append(dict(cohort=cohort, direction=d, method=method, param=param, regime=rname,
                             mAP10_val=macro_map(rerank_candidate_pool(S, score, npool), ids, lm)))
    for tau in TAU_GRID:
        rec("IS", f"tau={tau}", inverted_softmax(S, tau))
    for k in CSLS_GRID:
        rec("CSLS", f"k={k}", csls(S, k))
    for beta in BETA_GRID:
        rec("QBNorm", f"beta={beta}", qbnorm_dis(S, cell[d]["S_bank_G"], beta)[0])
    for b1 in BETA12_GRID:
        for b2 in BETA12_GRID:
            rec("DBNorm", f"beta1={b1},beta2={b2}", dbnorm(S, cell[d]["S_uni_bank_G"], cell[d]["S_bank_G"].T, b1, b2))


def _sweep_rr_topk(rows, cohort, cell, rr_k):
    rr = rr_scores(cell["I2T"]["S_G"], rr_k)
    for d in DIRECTIONS:
        for rname, npool in REGIMES.items():
            rows.append(dict(cohort=cohort, direction=d, method="RR", param=f"K={rr_k}", regime=rname,
                             mAP10_val=macro_map(rerank_candidate_pool(cell[d]["S_G"], rr[d], npool), cell["ids"], cell["label_map"])))
    for kk in TOPK_GRID:
        S_topk = topk_patch_text(cell["patches_val"], cell["text_val"], kk)
        for d in DIRECTIONS:
            score = reciprocal_topk_score(native(S_topk, d))
            for rname, npool in REGIMES.items():
                rows.append(dict(cohort=cohort, direction=d, method="topk_reciprocal", param=f"k={kk}", regime=rname,
                                 mAP10_val=macro_map(rerank_candidate_pool(cell[d]["S_G"], score, npool), cell["ids"], cell["label_map"])))


def select_rr_k_ai4skin(cfg) -> pd.DataFrame:
    """K* of RR per AI4SKIN fold: argmax over the grid of 0.5 * (I2T + T2I) class-macro mAP@10
    on the fold's validation set (smallest K on ties), computed on the mean-pooling similarity."""
    ai = AI4SKIN(cfg)
    store = ai.patch_store(BACKBONE)
    rows = []
    for k in range(5):
        ids, lm = ai.fold_ids(k, "val"), ai.fold_label_map(k, "val")
        S = _ai4skin_mean_pooling_similarity(store, ai, ids)
        sel = {}
        for K in RR_K_GRID_AI4SKIN:
            i2t, t2i = rerank_bidirectional(S, K)
            sel[K] = 0.5 * (macro_map(score_matrix_from_order(i2t), ids, lm) + macro_map(score_matrix_from_order(t2i.T), ids, lm))
        best = max(RR_K_GRID_AI4SKIN, key=lambda K: (sel[K], -K))
        rows.append(dict(fold=k, K_star=best, **{f"val_K{K}": v for K, v in sel.items()}))
    return pd.DataFrame(rows)


def _ai4skin_mean_pooling_similarity(store, ai, ids):
    """Mean-pooling similarity of one AI4SKIN partition (reports normalised over rows)."""
    txt = np.stack([np.load(f"{ai.backbone(BACKBONE)['text_dirs'][CONDITION]}/{c}.npy").astype(np.float64).reshape(-1) for c in ids])
    txt /= np.linalg.norm(txt, axis=1, keepdims=True)
    img = np.stack([mean_pool(store.load(c)) for c in ids]).astype(np.float64)
    return global_similarity(img, txt)


def validation(cfg) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    tcga_cell = tcga_validation_cell(cfg)
    for d in DIRECTIONS:
        _sweep_cell(rows, "TCGA", tcga_cell, d)
    _sweep_rr_topk(rows, "TCGA", tcga_cell, RR_K_TCGA)
    print("validation: TCGA done", flush=True)
    rr_k = select_rr_k_ai4skin(cfg)
    for k in range(5):
        cell = ai4skin_validation_cell(cfg, k)
        for d in DIRECTIONS:
            _sweep_cell(rows, f"AI4SKIN_fold{k}", cell, d)
        _sweep_rr_topk(rows, f"AI4SKIN_fold{k}", cell, int(rr_k.K_star[k]))
        print(f"validation: AI4SKIN fold {k} done", flush=True)
    grid = pd.DataFrame(rows)
    best = best_by_validation(grid, ["cohort", "method", "direction"], "mAP10_val")
    return grid, best, rr_k


def _param(method, s):
    if method == "DBNorm":
        d = dict(kv.split("=") for kv in s.split(","))
        return float(d["beta1"]), float(d["beta2"])
    return int(s.split("=")[1]) if method in ("CSLS", "topk_reciprocal", "RR") else float(s.split("=")[1])


# ----------------------------------------------------------------------------- test evaluation
def _hubness(method, G, prm, bank, uni):
    if method == "IS":
        return inverted_softmax(G, prm)
    if method == "CSLS":
        return csls(G, prm)
    if method == "QBNorm":
        return qbnorm_dis(G, bank, prm)[0]
    if method == "DBNorm":
        return dbnorm(G, uni, bank.T, *prm)
    raise ValueError(method)


def _run(rows, cohort, method, d, S, ids, lab, fold=None):
    ids = list(ids)
    y = np.array([lab[c] for c in ids])
    ref, _ = evaluate_direction(S, ids, lab, exclude_exact_pair=False)
    df = tie_neutral_per_query(S, y, TIE_SEEDS)
    assert np.allclose(ref["mAP@10"].to_numpy(), df["mAP@10_stable"].to_numpy(), atol=1e-12)
    df.insert(0, "query_id", ids)
    for k, v in [("fold", fold), ("direction", d), ("method", method), ("cohort", cohort)]:
        df.insert(0, k, v)
    rows.append(df)
    print(f"{cohort} {method:16s} {d} fold={fold} ties={df.row_has_tie.mean():.3f}", flush=True)


def evaluate_tcga(cfg, best, rows):
    tcga = TCGA(cfg)
    te, tr = tcga.ids("test"), tcga.ids("train")
    lab = tcga.label_map(te, "subtype")
    V, Vtr, Vmax = tcga.pooled(BACKBONE, te), tcga.pooled(BACKBONE, tr), tcga.pooled(BACKBONE, te, "max_pool")
    T, Ttr = tcga.text(BACKBONE, CONDITION, te, mode="rows"), tcga.text(BACKBONE, CONDITION, tr, mode="rows")
    SG, SP = V @ T.T, prototype_similarity(tcga.prototype_banks(BACKBONE, te), T)
    bank = {"I2T": Vtr @ T.T, "T2I": (V @ Ttr.T).T}
    uni = {"I2T": T @ Ttr.T, "T2I": V @ Vtr.T}
    patches = tcga.patches(BACKBONE, te)
    rr = rr_scores(SG, RR_K_TCGA)
    sel = lambda m, d: best[(best.cohort == "TCGA") & (best.method == m) & (best.direction == d)].iloc[0]  # noqa: E731
    for d in DIRECTIONS:
        G, P = native(SG, d), native(SP, d)
        _run(rows, "TCGA", "mean_pooling", d, G, te, lab)
        _run(rows, "TCGA", "max_pooling", d, Vmax @ T.T if d == "I2T" else T @ Vmax.T, te, lab)
        _run(rows, "TCGA", "RPGR", d, rpgr_score(G, P), te, lab)
        for m in ["IS", "CSLS", "QBNorm", "DBNorm"]:
            r = sel(m, d)
            _run(rows, "TCGA", m, d, rerank_candidate_pool(G, _hubness(m, G, _param(m, r.best_param), bank[d], uni[d]), REGIMES[r.best_regime]), te, lab)
        r = sel("RR", d)
        _run(rows, "TCGA", "RR", d, rerank_candidate_pool(G, rr[d], REGIMES[r.best_regime]), te, lab)
        r = sel("topk_reciprocal", d)
        St = topk_patch_text(patches, T, _param("topk_reciprocal", r.best_param))
        _run(rows, "TCGA", "topk_reciprocal", d, rerank_candidate_pool(G, reciprocal_topk_score(native(St, d)), REGIMES[r.best_regime]), te, lab)
    del patches
    blas = threadpool_limits(limits=COMPARATOR_EMBEDDING_BLAS_THREADS, user_api="blas")
    dtn = np.load(tcga.trained("dtn_similarity"), allow_pickle=True)
    assert list(dtn["ids"]) == te
    mil = np.load(tcga.trained("mil_embeddings"), allow_pickle=True)
    fg = tcga.trained("fgcr_dir")
    FI = l2_normalize_rows(np.stack([np.load(f"{fg}/img_cls/{c}.npy") for c in te]).astype(np.float64))
    FT = l2_normalize_rows(np.stack([np.load(f"{fg}/text_cls/{c}.npy") for c in te]).astype(np.float64))
    for d in DIRECTIONS:
        _run(rows, "TCGA", "DTN", d, native(dtn["sim"], d), te, lab)
        _run(rows, "TCGA", "FGCR", d, FI @ FT.T if d == "I2T" else FT @ FI.T, te, lab)
        for m in ["TransMIL", "SETMIL"]:
            mi, mt = l2_normalize_rows(mil[f"{m}_img"]), l2_normalize_rows(mil[f"{m}_txt"])
            _run(rows, "TCGA", m, d, mi @ mt.T if d == "I2T" else mt @ mi.T, list(mil[f"{m}_ids"]), lab)
        for key, m in [("titan_embeddings", "TITAN"), ("prism_embeddings", "PRISM")]:
            z = np.load(tcga.trained(key), allow_pickle=True)
            _run(rows, "TCGA", m, d, embedding_scores(z["z_img"], z["z_txt"], d, normalize=False), list(z["ids"]), lab)
    blas.restore_original_limits()


def evaluate_ai4skin(cfg, best, rr_k, rows):
    ai = AI4SKIN(cfg)
    store = ai.patch_store(BACKBONE)
    for k in range(5):
        tid, trid = ai.fold_ids(k, "test"), ai.fold_ids(k, "train")
        lab = ai.fold_label_map(k, "test")
        G0 = _ai4skin_mean_pooling_similarity(store, ai, tid)
        Tt, Ttr = ai.text(BACKBONE, CONDITION, tid, mode="rows"), ai.text(BACKBONE, CONDITION, trid, mode="rows")
        trmp = np.stack([mean_pool(store.load(c).astype(np.float64)) for c in trid])
        pt = [l2_normalize_rows(store.load(c).astype(np.float64)) for c in tid]
        temp = np.stack([mean_pool(p) for p in pt])
        tmax = l2_normalize_rows(np.stack([max_pool(p).astype(np.float64) for p in pt]))
        bank = {"I2T": global_similarity(trmp, Tt), "T2I": global_similarity(temp, Ttr).T}
        uni = {"I2T": global_similarity(Tt, Ttr), "T2I": global_similarity(temp, trmp)}
        SPf = prototype_similarity(ai.prototype_banks(BACKBONE, tid), Tt)
        rr = rr_scores(G0, int(rr_k.K_star[k]))
        sel = lambda m, d: best[(best.cohort == f"AI4SKIN_fold{k}") & (best.method == m) & (best.direction == d)].iloc[0]  # noqa: E731
        for d in DIRECTIONS:
            G, P = native(G0, d), native(SPf, d)
            _run(rows, "AI4SKIN", "mean_pooling", d, G, tid, lab, k)
            _run(rows, "AI4SKIN", "max_pooling", d, tmax @ Tt.T if d == "I2T" else Tt @ tmax.T, tid, lab, k)
            _run(rows, "AI4SKIN", "RPGR", d, rpgr_score(G, P), tid, lab, k)
            for m in ["IS", "CSLS", "QBNorm", "DBNorm"]:
                r = sel(m, d)
                _run(rows, "AI4SKIN", m, d, rerank_candidate_pool(G, _hubness(m, G, _param(m, r.best_param), bank[d], uni[d]), REGIMES[r.best_regime]), tid, lab, k)
            r = sel("RR", d)
            _run(rows, "AI4SKIN", "RR", d, rerank_candidate_pool(G, rr[d], REGIMES[r.best_regime]), tid, lab, k)
            r = sel("topk_reciprocal", d)
            St = topk_patch_text(pt, Tt, _param("topk_reciprocal", r.best_param))
            _run(rows, "AI4SKIN", "topk_reciprocal", d, rerank_candidate_pool(G, reciprocal_topk_score(native(St, d)), REGIMES[r.best_regime]), tid, lab, k)
            for m in ["DTN", "FGCR", "SETMIL", "TransMIL", "TITAN", "PRISM"]:
                z = np.load(ai.trained(m, k), allow_pickle=True)
                with threadpool_limits(limits=COMPARATOR_EMBEDDING_BLAS_THREADS, user_api="blas"):
                    S = embedding_scores(z["z_img"], z["z_txt"], d, binary=(m == "DTN"))
                _run(rows, "AI4SKIN", m, d, S, list(z["ids"]), lab, k)
        print(f"AI4SKIN fold {k} done", flush=True)


# ----------------------------------------------------------------------------- statistics
def rpgr_on_subset_galleries(cfg, nq: pd.DataFrame) -> dict:
    """RPGR recomputed (tie-neutral) on the TITAN and PRISM TCGA test galleries."""
    tcga = TCGA(cfg)
    te = tcga.ids("test")
    pos = {c: i for i, c in enumerate(te)}
    V, T = tcga.pooled(BACKBONE, te), tcga.text(BACKBONE, CONDITION, te, mode="rows")
    banks = dict(zip(te, tcga.prototype_banks(BACKBONE, te)))
    lab_all = tcga.labels()["OncotreeCode"]
    out = {}
    for m in ["TITAN", "PRISM"]:
        ids = nq[(nq.cohort == "TCGA") & (nq.method == m) & (nq.direction == "I2T")].query_id.tolist()
        p = [pos[c] for c in ids]
        y = lab_all.reindex(ids).to_numpy()
        SG = V[p] @ T[p].T
        SP = prototype_similarity([banks[c] for c in ids], T[p])
        for d in DIRECTIONS:
            df = tie_neutral_per_query(rpgr_score(native(SG, d), native(SP, d)), y, TIE_SEEDS, always_permute=True)
            df.insert(0, "query_id", ids)
            out[(m, d)] = df
    return out


def statistics(cfg, nq: pd.DataFrame) -> pd.DataFrame:
    case_of = TCGA(cfg).labels()["case_id"].to_dict()
    sub_ref = rpgr_on_subset_galleries(cfg, nq)
    rows = []
    for (cohort, d), g in nq.groupby(["cohort", "direction"]):
        ref_full = g[g.method == "RPGR"]
        for m in COMPARATORS:
            x = g[g.method == m].sort_values("query_id")
            ref = (sub_ref[(m, d)] if (cohort == "TCGA" and (m, d) in sub_ref) else ref_full).sort_values("query_id")
            assert (x.query_id.values == ref.query_id.values).all(), (cohort, d, m)
            units = x.query_id.map(case_of).values if cohort == "TCGA" else x.query_id.values
            r = paired_cluster_bootstrap(x["mAP@10_neutral"].values, ref["mAP@10_neutral"].values, x.label.values, units,
                                         B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED)
            rows.append(dict(cohort=cohort, direction=d, method=m, n=len(x),
                             mAP10_stable=class_macro(x["mAP@10_stable"].values, x.label.values),
                             mAP10_neutral=class_macro(x["mAP@10_neutral"].values, x.label.values),
                             rpgr_reference_neutral=class_macro(ref["mAP@10_neutral"].values, ref.label.values),
                             diff_RPGR_minus_method=r["diff"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], p=r["p"]))
    res = pd.DataFrame(rows)
    res["holm14"] = False
    for _, g in res.groupby(["cohort", "direction"]):
        res.loc[g.index, "holm14"] = holm(g.p.values)
    res["mark"] = np.where(~res.holm14, "", np.where(res.diff_RPGR_minus_method > 0, "*", "†"))
    rp = nq[nq.method == "RPGR"].groupby(["cohort", "direction"]).apply(
        lambda x: class_macro(x["mAP@10_neutral"].values, x.label.values), include_groups=False)
    res = pd.concat([pd.DataFrame([dict(cohort=c, direction=d, method="RPGR", n=int(((nq.cohort == c) & (nq.direction == d) & (nq.method == "RPGR")).sum()),
                                        mAP10_neutral=v, mark="") for (c, d), v in rp.items()]), res], ignore_index=True)
    return res


def main() -> None:
    p = config_argument_parser(__doc__)
    p.add_argument("--step", default="all", choices=["all", "validation", "evaluate", "stats"])
    args = p.parse_args()
    cfg = load_config(args.config)
    out = output_dir(cfg, "benchmark")
    if args.step in ("all", "validation"):
        grid, best, rr_k = validation(cfg)
        grid.to_csv(out / "validation_grid.csv", index=False)
        best.to_csv(out / "validation_selected.csv", index=False)
        rr_k.to_csv(out / "rr_k_star_ai4skin.csv", index=False)
    if args.step in ("all", "evaluate"):
        best = pd.read_csv(out / "validation_selected.csv")
        rr_k = pd.read_csv(out / "rr_k_star_ai4skin.csv")
        rows = []
        t0 = time.time()
        evaluate_tcga(cfg, best, rows)
        evaluate_ai4skin(cfg, best, rr_k, rows)
        pd.concat(rows, ignore_index=True).to_csv(out / "per_query.csv.gz", index=False, compression="gzip")
        print(f"evaluate done ({time.time() - t0:.0f}s)", flush=True)
    if args.step in ("all", "stats"):
        nq = pd.read_csv(out / "per_query.csv.gz")
        statistics(cfg, nq).to_csv(out / "comparison.csv", index=False)
    print("written", out)


if __name__ == "__main__":
    main()
