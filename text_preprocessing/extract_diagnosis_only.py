#!/usr/bin/env python3
"""
extract_diagnosis_only.py
============================
Builds a minimal slide_id -> text CSV holding only the tumour subtype name
(the `cancer_type_detailed` column of the master reports CSV), unchanged --
no API call, no rewriting. This is the opposite extreme from the Strict
condition: no morphology at all, only the diagnosis. The full subtype name
is used rather than its short OncoTree code (e.g. "IDC") because the text
encoders were trained on natural language, not acronyms.

Output is meant to be fed straight into the same text-embedding generation
script used for Original/Strict, via --text-field diagnosis_only_text --
no separate embedding code is needed.

Never writes to the source CSV.

Usage:
  python3 extract_diagnosis_only.py --csv reports.csv --ids cohort_ids.csv --out diagnosis_only.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, required=True, help="Master reports CSV (';'-separated).")
    p.add_argument("--ids", type=Path, default=None,
                   help="Optional slide_id list to restrict to (e.g. the same cohort used for Strict).")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    df = pd.read_csv(args.csv, sep=";", low_memory=False)
    if args.ids is not None:
        ids = pd.read_csv(args.ids)["slide_id"]
        df = df[df["slide_id"].isin(ids)]

    missing = df["cancer_type_detailed"].isna().sum()
    if missing:
        print(f"warning: {missing} rows without cancer_type_detailed, excluded")
        df = df[df["cancer_type_detailed"].notna()]

    out = df[["slide_id", "cancer_type_detailed"]].rename(columns={"cancer_type_detailed": "diagnosis_only_text"})
    out.to_csv(args.out, sep=";", index=False)
    print(f"wrote {args.out} ({len(out)} rows)")


if __name__ == "__main__":
    main()
