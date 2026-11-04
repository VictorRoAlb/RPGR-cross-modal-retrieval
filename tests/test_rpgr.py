"""Unit tests on synthetic data (no private data needed). Run: pytest -q tests/"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rpgr.reranking import global_similarity, prototype_similarity, rpgr_score, direct_term, recip_term  # noqa: E402
from rpgr.metrics import evaluate_direction, evaluate_direction_tie_neutral  # noqa: E402
from rpgr.stats import paired_cluster_bootstrap, holm  # noqa: E402

RNG = np.random.default_rng(0)


def _unit(X):
    return X / np.linalg.norm(X, axis=-1, keepdims=True)


def test_prototype_similarity_is_max_over_prototypes():
    banks = [_unit(RNG.normal(size=(RNG.integers(1, 9), 16))) for _ in range(12)]
    T = _unit(RNG.normal(size=(12, 16)))
    ref = np.stack([(b @ T.T).max(axis=0) for b in banks])
    assert np.allclose(prototype_similarity(banks, T), ref, rtol=0, atol=1e-12)  # BLAS blocking may differ at ~1e-16


def test_calibrations_are_log_softmax_with_self_scaling():
    X = RNG.normal(size=(7, 9))
    D, R = direct_term(X), recip_term(X)
    assert np.allclose(np.exp(D).sum(axis=1), 1.0)          # each row is a log-distribution over candidates
    assert np.allclose(np.exp(R).sum(axis=0), 1.0)          # each column over queries
    assert np.allclose(direct_term(3.0 * X + 2.0), direct_term(X))   # invariant to per-row affine rescaling
    assert np.allclose(recip_term(3.0 * X + 2.0), recip_term(X))


def test_rpgr_score_formula():
    G, P = RNG.normal(size=(6, 6)), RNG.normal(size=(6, 6))
    assert np.allclose(rpgr_score(G, P), recip_term(G) + 0.5 * (direct_term(P) + recip_term(P)))


def test_constant_rows_do_not_crash():
    X = np.ones((4, 5))
    assert np.isfinite(rpgr_score(X, X)).all()


def test_metrics_exact_pair_included_and_stable_ties():
    ids = ["a", "b", "c", "d"]
    labels = {"a": 0, "b": 0, "c": 1, "d": 1}
    S = np.eye(4)
    df, _ = evaluate_direction(S, ids, labels, exclude_exact_pair=False)
    assert (df["Recall@1"] == 1).all()
    S_tie = np.zeros((4, 4))  # all tied -> stable order = gallery order
    df2, _ = evaluate_direction(S_tie, ids, labels, exclude_exact_pair=False)
    assert df2.set_index("query_id").loc["a", "Recall@1"] == 1 and df2.set_index("query_id").loc["c", "Recall@1"] == 0


def test_tie_neutral_equals_stable_without_ties_and_averages_with_ties():
    ids = [f"c{i}" for i in range(6)]
    labels = {c: i % 2 for i, c in enumerate(ids)}
    S = RNG.normal(size=(6, 6))                              # continuous scores: no ties
    a, _ = evaluate_direction(S, ids, labels)
    b, _ = evaluate_direction_tie_neutral(S, ids, labels, n_seeds=10)
    assert np.allclose(a["mAP@10"], b["mAP@10"]) and (b["mAP@10_tie_sd"] < 1e-12).all()
    T = np.zeros((6, 6))                                     # all tied: stable order is gallery order
    st, _ = evaluate_direction(T, ids, labels)
    ne, _ = evaluate_direction_tie_neutral(T, ids, labels, n_seeds=200)
    assert st.set_index("query_id").loc["c1", "Recall@1"] == 0     # c0 (other class) always first under stable
    assert 0.35 < ne.set_index("query_id").loc["c1", "Recall@1"] < 0.65  # ~0.5 under random tie-break


def test_cluster_bootstrap_units_and_holm():
    n = 60
    lab = np.repeat([0, 1, 2], 20)
    units = np.repeat(np.arange(30), 2)                       # two rows per unit, nested in class
    a = RNG.random(n); b = a + 0.2
    r = paired_cluster_bootstrap(a, b, lab, units, B=500, seed=1)
    assert r["diff"] == pytest.approx(0.2) and r["p"] == pytest.approx(1 / 500) and r["n_units"] == 30
    assert holm([0.001, 0.02, 0.04]).tolist() == [True, True, True]
    assert holm([0.001, 0.03, 0.04]).tolist() == [True, False, False]


@pytest.mark.skipif("RPGR_CONFIG" not in os.environ, reason="needs the TCGA data layout (set RPGR_CONFIG=paths.yaml)")
def test_reproduces_reported_tcga_test_numbers(tmp_path):
    """End-to-end check against the reported held-out values (KEEP, Morphology-only, subtype)."""
    import subprocess
    import yaml
    cfg = yaml.safe_load(open(os.environ["RPGR_CONFIG"]))
    cfg["output_dir"] = str(tmp_path)
    p = tmp_path / "cfg.yaml"; p.write_text(yaml.safe_dump(cfg))
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_heldout_rpgr.py"
    subprocess.run([sys.executable, str(script), "--config", str(p), "--backbone", "KEEP", "--scenario", "strict"], check=True)
    out = pd.read_csv(next(tmp_path.glob("heldout_rpgr_*_stable.csv"))).set_index("direction")["mAP@10"]
    assert out["I2T"] == pytest.approx(0.1963670011017272, abs=1e-12)
    assert out["T2I"] == pytest.approx(0.15109014797251513, abs=1e-12)
    # tie-neutral protocol (50 seeds), as used for the cross-method comparison
    subprocess.run([sys.executable, str(script), "--config", str(p), "--backbone", "KEEP", "--scenario", "strict",
                    "--tie-break", "neutral"], check=True)
    neu = pd.read_csv(next(tmp_path.glob("heldout_rpgr_*_neutral.csv"))).set_index("direction")["mAP@10"]
    assert neu["I2T"] == pytest.approx(0.19634556938548833, abs=1e-12)
    assert neu["T2I"] == pytest.approx(0.1510901479725151, abs=1e-12)
