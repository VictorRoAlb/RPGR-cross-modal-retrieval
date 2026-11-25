"""
Qualitative T2I example (Fig. 4): the Morphology-only report of TCGA-77-7141 (LUSC) queried
against the TCGA test gallery (n=1578) with KEEP; top-10 WSIs of mean pooling and of RPGR,
with the rank of each WSI under the other method.

Run with: python -m experiments.qualitative_example --config <yaml> [--query TCGA-77-7141]
"""
from __future__ import annotations

import data.blas  # noqa: F401  (must precede numpy)

import numpy as np
import pandas as pd

from data.config import config_argument_parser, load_config, output_dir
from data.loaders import TCGA
from rpgr.rerank import rpgr_score
from rpgr.similarity import prototype_similarity

BACKBONE, CONDITION = "KEEP", "morphology_only"


def rankings(cfg, query_prefix: str, top: int = 10) -> pd.DataFrame:
    tcga = TCGA(cfg)
    ids = tcga.ids("test")
    lab = tcga.labels()["OncotreeCode"].reindex(ids).values
    V, T = tcga.pooled(BACKBONE, ids), tcga.text(BACKBONE, CONDITION, ids, mode="rows")
    G = (V @ T.T).T                                            # T2I: report rows, WSI columns
    P = prototype_similarity(tcga.prototype_banks(BACKBONE, ids), T).T
    E = rpgr_score(G, P)
    rows = []
    for qi, q in enumerate(ids):
        if not q.startswith(query_prefix):
            continue
        om, oe = np.argsort(-G[qi], kind="stable"), np.argsort(-E[qi], kind="stable")
        rm, re_ = np.empty(len(ids), int), np.empty(len(ids), int)
        rm[om] = np.arange(1, len(ids) + 1)
        re_[oe] = np.arange(1, len(ids) + 1)
        for which, order in [("mean_pooling", om), ("RPGR", oe)]:
            for k in range(top):
                j = order[k]
                rows.append(dict(query=q, query_label=lab[qi], list=which, rank=k + 1, slide=ids[j], label=lab[j],
                                 correct=lab[j] == lab[qi], rank_mean_pooling=int(rm[j]), rank_RPGR=int(re_[j])))
    return pd.DataFrame(rows)


def main() -> None:
    p = config_argument_parser(__doc__)
    p.add_argument("--query", default="TCGA-77-7141")
    args = p.parse_args()
    cfg = load_config(args.config)
    df = rankings(cfg, args.query)
    out = output_dir(cfg, "qualitative_example")
    df.to_csv(out / "rankings.csv", index=False)
    print(df[df["rank"] <= 5][["list", "rank", "slide", "label", "rank_mean_pooling", "rank_RPGR"]].to_string(index=False))


if __name__ == "__main__":
    main()
