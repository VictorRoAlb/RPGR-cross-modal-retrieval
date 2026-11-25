"""
Contribution of each term of RPGR to its gain over mean pooling on the test sets (Table 2):
KEEP, Morphology-only, subtype level, class-macro mAP@10, stable order.

  global term         R_G                          minus mean pooling
  local term          RPGR                         minus R_G
    calibration       R_G + 1/2 (D_G + R_G)        minus R_G   (S_P replaced by S_G)
    prototype content RPGR                         minus the previous score

TCGA test partition (n=1578); AI4SKIN test set of each of the five folds (276 pooled queries).
Prototype content is tested with a paired bootstrap by clinical case (B=20,000, seed 42), Holm
over the four cells.

Output: <output_dir>/gain_decomposition/table2.csv
Run with: python -m experiments.gain_decomposition --config <yaml>
"""
from __future__ import annotations

import data.blas  # noqa: F401  (must precede numpy)

import numpy as np
import pandas as pd

from data.config import config_argument_parser, load_config, output_dir
from data.loaders import AI4SKIN, TCGA, load_text_vectors
from experiments.common import DIRECTIONS, native, per_query
from rpgr.calibration import reciprocal_term
from rpgr.rerank import rpgr_score
from rpgr.similarity import global_similarity, mean_pool, prototype_similarity
from rpgr.statistics import class_macro, holm, paired_cluster_bootstrap

BACKBONE, CONDITION = "KEEP", "morphology_only"
BOOTSTRAP_B, BOOTSTRAP_SEED = 20000, 42


def test_galleries(cfg):
    """(cohort, fold, S_G, S_P, ids, label map, clinical case of each id) of every test gallery."""
    tcga = TCGA(cfg)
    ids = tcga.ids("test")
    text = tcga.text(BACKBONE, CONDITION, ids)
    yield ("TCGA", -1, tcga.pooled(BACKBONE, ids) @ text.T, prototype_similarity(tcga.prototype_banks(BACKBONE, ids), text),
           ids, tcga.label_map(ids, "subtype"), dict(zip(ids, tcga.case_ids(ids))))
    ai = AI4SKIN(cfg)
    store = ai.patch_store(BACKBONE)
    for k in range(5):
        ids = ai.fold_ids(k, "test")
        text = ai.text(BACKBONE, CONDITION, ids)
        wsi = np.stack([mean_pool(store.load(c)) for c in ids]).astype(np.float64)
        text_rows = load_text_vectors(ai.backbone(BACKBONE)["text_dirs"][CONDITION], ids, "rows")
        yield ("AI4SKIN", k, global_similarity(wsi, text_rows), prototype_similarity(ai.prototype_banks(BACKBONE, ids), text),
               ids, ai.fold_label_map(k, "test"), {c: c for c in ids})


def evaluate(cfg) -> pd.DataFrame:
    """Per-query mAP@10 of the four scores."""
    out = []
    for cohort, fold, S_G, S_P, ids, labels, case_of in test_galleries(cfg):
        for d in DIRECTIONS:
            G, P = native(S_G, d), native(S_P, d)
            scores = {"mean_pooling": G, "global": reciprocal_term(G), "calibration": rpgr_score(G, G), "RPGR": rpgr_score(G, P)}
            df = None
            for name, S in scores.items():
                q = per_query(S, ids, labels)
                if df is None:
                    df = q[["query_id", "label"]].copy()
                df[name] = q["mAP@10"]
            df.insert(0, "cohort", cohort)
            df.insert(1, "direction", d)
            df.insert(2, "fold", fold)
            df["case"] = df.query_id.map(case_of)
            out.append(df)
    return pd.concat(out, ignore_index=True)


def table2(pq: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cohort, d), g in pq.groupby(["cohort", "direction"], sort=False):
        g = g.sort_values(["fold", "query_id"])
        y = g.label.to_numpy()
        mean, glob, cal, full = (class_macro(g[m].to_numpy(), y) for m in ["mean_pooling", "global", "calibration", "RPGR"])
        p = paired_cluster_bootstrap(g["RPGR"].to_numpy(), g["calibration"].to_numpy(), y, B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED,
                                     units=g["case"].to_numpy())["p"]
        rows.append(dict(cohort=cohort, direction=d, mean_pooling=mean, RPGR=full, global_term=glob - mean, local_term=full - glob,
                         calibration=cal - glob, prototype_content=full - cal, total=full - mean,
                         relative_gain_pct=100 * (full - mean) / mean, p_prototype_content=p))
    res = pd.DataFrame(rows)
    res["prototype_content_significant"] = holm(res.p_prototype_content.values)
    return res


def main() -> None:
    cfg = load_config(config_argument_parser(__doc__).parse_args().config)
    table = table2(evaluate(cfg))
    table.to_csv(output_dir(cfg, "gain_decomposition") / "table2.csv", index=False)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
