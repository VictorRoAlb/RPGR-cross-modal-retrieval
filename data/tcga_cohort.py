"""
TCGA cohort, partitions and composition table (Section 4.1, Table A.1).

Input: the slide-level reports released with TITAN (``TCGA-Slide-Reports.csv.gz``, 10,108
WSI-report pairs) with the Morphology-only column added by
``text_conditions/generate_morphology_only.py`` (``tcga.reports_csv``, ';'-separated), and
the embedding directories of the four VL-PFMs.

Selection (each step is logged with its count in cohort_flow.csv):
  1. WSIs of the 15 TCGA projects whose patch embeddings were extracted (BLCA, BRCA, CESC,
     COAD, GBM, LGG, LUAD, LUSC, PCPG, PRAD, READ, SARC, SKCM, STAD, UCEC): 6733;
  2. with patch and report embeddings in the four backbones: 6727 (labels file);
  3. with a non-empty Morphology-only report: 5293;
  4. with a subtype name for the Diagnosis-only report: 5291 WSIs (complete cohort);
  5. subtypes (OncotreeCode) with at least 10 WSIs: 5227 WSIs, 35 subtypes (eligible);
  6. case-level split stratified by subtype (55 % / 15 % / 30 %, seed 42):
     2897 / 752 / 1578 WSIs.

Outputs (<output_dir>/cohort/): labels.csv, complete_cohort.csv, eligible.csv, split.csv,
cohort_flow.csv, table_a1_tcga_composition.csv. Point tcga.labels_csv, tcga.split_csv and
tcga.complete_cohort_csv of the config to these files.

Run with: python -m data.tcga_cohort --config <yaml>
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.config import BACKBONES, config_argument_parser, load_config, output_dir
from data.loaders import PatchStore

PROJECTS = ["TCGA-BLCA", "TCGA-BRCA", "TCGA-CESC", "TCGA-COAD", "TCGA-GBM", "TCGA-LGG", "TCGA-LUAD", "TCGA-LUSC",
            "TCGA-PCPG", "TCGA-PRAD", "TCGA-READ", "TCGA-SARC", "TCGA-SKCM", "TCGA-STAD", "TCGA-UCEC"]
MIN_WSIS_PER_SUBTYPE = 10
TRAIN_FRAC, VAL_FRAC, SPLIT_SEED = 0.55, 0.15, 42
LABEL_COLUMNS = ["slide_id", "case_id", "project_id", "cancer_type_detailed", "OncotreeCode"]


def case_level_split(cohort: pd.DataFrame, label_col: str = "OncotreeCode") -> pd.DataFrame:
    """Split by case (all WSIs of a case in the same partition), stratified by subtype."""
    if cohort.groupby("case_id")[label_col].nunique().max() > 1:
        raise SystemExit("a case has more than one subtype")
    cases = cohort.groupby("case_id")[label_col].first().reset_index()
    rng = np.random.default_rng(SPLIT_SEED)
    parts = []
    for _, group in cases.groupby(label_col, sort=True):
        idx = group.index.to_numpy()
        rng.shuffle(idx)
        n = len(idx)
        n_train, n_val = int(round(n * TRAIN_FRAC)), int(round(n * VAL_FRAC))
        if n >= 3:
            n_train = min(n_train, n - 2)
            n_val = max(n_val, 1)
            n_val = min(n_val, n - n_train - 1)
        for name, sel in [("train", idx[:n_train]), ("val", idx[n_train:n_train + n_val]), ("test", idx[n_train + n_val:])]:
            parts.append(pd.DataFrame({"case_id": cases.loc[sel, "case_id"], "split": name}))
    case_split = pd.concat(parts, ignore_index=True)
    assert case_split.groupby("case_id")["split"].nunique().max() == 1
    out = cohort.merge(case_split, on="case_id", how="left")[["slide_id", "case_id", label_col, "split"]]
    assert out["split"].notna().all()
    return out


def build(cfg) -> dict[str, pd.DataFrame]:
    tcga = cfg["tcga"]
    reports = pd.read_csv(tcga["reports_csv"], sep=";", low_memory=False)
    flow = [dict(step=0, description="TITAN slide-level reports", n=len(reports))]
    in_projects = reports[reports.project_id.isin(PROJECTS)]
    flow.append(dict(step=1, description="15 TCGA projects with extracted patch embeddings", n=len(in_projects)))
    available = None
    for bb in BACKBONES:
        b = tcga["backbones"][bb]
        ids = set(PatchStore(b["patch_dir"]).ids()) & set(PatchStore(b["text_dirs"]["original"]).ids())
        available = ids if available is None else available & ids
    labels = in_projects[in_projects.slide_id.isin(available)][LABEL_COLUMNS].sort_values("slide_id").reset_index(drop=True)
    flow.append(dict(step=2, description="patch and report embeddings in the four VL-PFMs", n=len(labels)))
    nonempty = reports.slide_reports_no_diagnosis_strict.fillna("").str.strip() != ""
    strict = reports[reports.slide_id.isin(labels.slide_id) & nonempty]
    flow.append(dict(step=3, description="non-empty Morphology-only report", n=len(strict)))
    complete = strict[strict.cancer_type_detailed.notna()][["slide_id"]]
    flow.append(dict(step=4, description="subtype name available (complete cohort)", n=len(complete)))
    sub = labels[labels.slide_id.isin(complete.slide_id)]
    counts = sub.OncotreeCode.value_counts()
    eligible = sub[sub.OncotreeCode.isin(counts[counts >= MIN_WSIS_PER_SUBTYPE].index)][["slide_id"]]
    flow.append(dict(step=5, description=f"subtypes with >= {MIN_WSIS_PER_SUBTYPE} WSIs (eligible)", n=len(eligible)))
    split = case_level_split(labels[labels.slide_id.isin(eligible.slide_id)].copy())
    for s in ["train", "val", "test"]:
        flow.append(dict(step=6, description=f"{s} partition", n=int((split.split == s).sum())))
    return dict(labels=labels, complete_cohort=complete, eligible=eligible, split=split, cohort_flow=pd.DataFrame(flow),
                table_a1_tcga_composition=composition(labels, complete, split))


def composition(labels: pd.DataFrame, complete: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    """Table A.1: WSIs per project and subtype in the complete cohort and per partition."""
    comp = labels[labels.slide_id.isin(complete.slide_id)]
    sp = split.set_index("slide_id")["split"]
    rows = []
    for proj, g in comp.groupby("project_id"):
        counts = g.OncotreeCode.value_counts()
        small = counts[counts < MIN_WSIS_PER_SUBTYPE]
        for code, n in counts[counts >= MIN_WSIS_PER_SUBTYPE].items():
            s = sp.reindex(g[g.OncotreeCode == code].slide_id)
            rows.append(dict(project=proj.replace("TCGA-", ""), project_complete=len(g), OncotreeCode=code,
                             subtype=g[g.OncotreeCode == code].cancer_type_detailed.iloc[0], complete=int(n), eligible=int(s.notna().sum()),
                             train=int((s == "train").sum()), val=int((s == "val").sum()), test=int((s == "test").sum())))
        if len(small):
            rows.append(dict(project=proj.replace("TCGA-", ""), project_complete=len(g), OncotreeCode="Others",
                             subtype=f"{len(small)} subtype{'s' if len(small) > 1 else ''} with <{MIN_WSIS_PER_SUBTYPE} samples", complete=int(small.sum()),
                             eligible=0, train=0, val=0, test=0))
    return pd.DataFrame(rows).sort_values(["project_complete", "project", "complete"], ascending=[False, True, False])


def main() -> None:
    cfg = load_config(config_argument_parser(__doc__).parse_args().config)
    out = output_dir(cfg, "cohort")
    for name, df in build(cfg).items():
        df.to_csv(out / f"{name}.csv", index=False)
    print(pd.read_csv(out / "cohort_flow.csv").to_string(index=False))


if __name__ == "__main__":
    main()
