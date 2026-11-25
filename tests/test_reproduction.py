"""End-to-end check against the reported held-out RPGR values (TCGA test, KEEP, Morphology-only,
subtype, stable order). Needs the data: RPGR_CONFIG=configs/paths.yaml pytest tests/test_reproduction.py"""
import os

import pytest

pytestmark = pytest.mark.skipif("RPGR_CONFIG" not in os.environ, reason="needs the TCGA data (set RPGR_CONFIG)")


def test_reproduces_reported_tcga_test_values():
    import data.blas  # noqa: F401
    from data.config import load_config
    from data.loaders import TCGA
    from experiments.common import per_query
    from rpgr.rerank import rpgr_score
    from rpgr.similarity import prototype_similarity
    from rpgr.statistics import class_macro

    tcga = TCGA(load_config(os.environ["RPGR_CONFIG"]))
    ids = tcga.ids("test")
    T = tcga.text("KEEP", "morphology_only", ids, mode="rows")
    G = tcga.pooled("KEEP", ids) @ T.T
    P = prototype_similarity(tcga.prototype_banks("KEEP", ids), T)
    lm = tcga.label_map(ids, "subtype")
    got = {}
    for d, (g, p) in {"I2T": (G, P), "T2I": (G.T, P.T)}.items():
        df = per_query(rpgr_score(g, p), ids, lm)
        got[d] = class_macro(df["mAP@10"].to_numpy(), df.label.to_numpy())
    assert got["I2T"] == pytest.approx(0.1963670011017272, abs=1e-12)
    assert got["T2I"] == pytest.approx(0.15109014797251513, abs=1e-12)
