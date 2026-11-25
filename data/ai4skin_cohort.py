"""
AI4SKIN cohort composition (Section 4.1, Table A.2).

The cohort has 276 cases with one WSI and one report each (283 cases minus 7 with a
discrepancy between the WSI and the class label) and seven diagnostic categories. The
five-fold protocol of the cross-method benchmark is given by ``ai4skin.folds_csv``
(columns case_id, class_name_en, outer_fold, role in {train, val, test}; stratified by class,
seed 42; 165 training, 55-56 validation and 55-56 test cases per fold). Every case is a
test case in exactly one fold. This script checks those properties and writes Table A.2.

Run with: python -m data.ai4skin_cohort --config <yaml>
"""
from __future__ import annotations

import pandas as pd

from data.config import config_argument_parser, load_config, output_dir
from data.loaders import AI4SKIN


def composition(ai: AI4SKIN) -> pd.DataFrame:
    folds = ai.folds()
    labels = ai.cohort_label_map()
    assert len(labels) == 276
    test = folds[folds.role == "test"]
    assert test.case_id.is_unique and set(test.case_id) == set(labels)
    for k in range(5):
        f = folds[folds.outer_fold == k]
        assert len(f) == 276 and f.case_id.is_unique
    rows = []
    complete = pd.Series(labels).value_counts()
    for cls, n in complete.items():
        r = dict(subtype=cls, complete=int(n))
        for role in ["train", "val", "test"]:
            per_fold = [int(((folds.outer_fold == k) & (folds.role == role) & (folds.class_name_en == cls)).sum()) for k in range(5)]
            r[role] = f"{min(per_fold)}-{max(per_fold)}" if min(per_fold) != max(per_fold) else str(min(per_fold))
        rows.append(r)
    tot = {"subtype": "Total", "complete": int(complete.sum())}
    for role in ["train", "val", "test"]:
        per_fold = [int(((folds.outer_fold == k) & (folds.role == role)).sum()) for k in range(5)]
        tot[role] = f"{min(per_fold)}-{max(per_fold)}" if min(per_fold) != max(per_fold) else str(min(per_fold))
    return pd.DataFrame(rows + [tot])


def main() -> None:
    cfg = load_config(config_argument_parser(__doc__).parse_args().config)
    table = composition(AI4SKIN(cfg))
    table.to_csv(output_dir(cfg, "cohort") / "table_a2_ai4skin_composition.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
