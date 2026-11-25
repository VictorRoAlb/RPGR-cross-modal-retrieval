#!/usr/bin/env python3
"""
Morphology-only text condition for TCGA (Appendix B).

Adds a new column, slide_reports_no_diagnosis_strict, to the master reports
CSV: a rewrite of slide_reports (the field used as the text-encoder input
for I2T/T2I retrieval) that describes ONLY histological morphology -- no
diagnosis, no grade/stage, no immunohistochemistry/biomarkers, no organ
used as an introductory frame ("The slide from the X shows..."), and no
other keyword that reveals the class. Uses GPT-4.1 with Structured Outputs
(strict JSON schema) so the model cannot deviate from the expected format,
plus an automatic regex check against label leakage with a stricter retry
prompt when leakage is detected.

Why this is needed: slide_reports gives away the literal diagnosis and
organ, which are the same labels used for classification (OncotreeCode /
cancer_type_detailed). Left in, they inflate I2T/T2I retrieval metrics
through keyword leakage rather than genuine image-text alignment.

Source note: this uses slide_reports only, not case_reports. slide_reports
is short (at most ~700 characters) and often mostly diagnosis -- for those
rows the result will legitimately be short once the diagnosis is removed.
That is a real property of the source data, not a script bug: morphology
that is not present in the original text must never be invented.

Anti-leakage check: beyond a list of universal forbidden patterns (grade/
stage, immunohistochemistry, interpretive phrasing, specimen margins/
extent -- valid for any cancer type), per-row dynamic terms are derived
from that row's own metadata columns (cancer_type_detailed,
site_of_resection_or_biopsy, OncotreeCode) -- necessary because TCGA is
pan-cancer (32 project types) and the organ/entity vocabulary cannot be
enumerated by hand.

This is a synchronous, row-by-row loop (no Batch API), with periodic
checkpointing to disk and resumability: re-running the script skips rows
that already have text in the new column, unless --overwrite is given.

Typical usage:
  # 1) small pilot, stratified by cancer type, to review quality before the
  #    full batch (never touches the master CSV; writes to a separate
  #    review CSV + JSON instead)
  python -m text_conditions.generate_morphology_only --pilot 40

  #    -> review the review CSV/JSON (what was removed per row, and
  #       whether leakage remains after retries) before continuing

  # 2) full batch over the target cohort: updates the master CSV (new
  #    column, atomic write) and writes an audit JSON alongside it with
  #    what was removed/changed per row
  python -m text_conditions.generate_morphology_only --csv reports.csv

  # if interrupted partway, re-running the same command resumes where it
  # left off

Requires: pip install openai pandas tqdm
Auth: OPENAI_API_KEY environment variable.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

try:
    from openai import OpenAI, RateLimitError, APIError, APITimeoutError
except ImportError:
    sys.exit("Missing OpenAI SDK. Install with: pip install openai")


# No default paths shipped with this release -- point --csv and
# --strict-common-ids at your own master reports CSV and cohort id list
# (formats in data/loaders.py).
DEFAULT_CSV = None
DEFAULT_STRICT_COMMON_IDS = None
SOURCE_COLUMN = "slide_reports"
NEW_COLUMN = "slide_reports_no_diagnosis_strict"
CSV_SEP = ";"
# OncotreeCode is deliberately not included here: it is a short code (e.g.
# "ASTR" for Astrocytoma) that adds no coverage beyond cancer_type_detailed
# (which carries the same information in full words), while its shortness
# made the dynamic term matcher treat it as a prefix that blocked legitimate
# words (e.g. "ASTR" blocking "astrocytic features", a real cell-lineage
# descriptor, in LGG rows). cancer_type_detailed alone, matched on whole
# words, does not have this problem.
METADATA_COLUMNS_FOR_DYNAMIC_TERMS = ["cancer_type_detailed", "site_of_resection_or_biopsy"]


# ---------------- TEXT UTILS ----------------

def one_paragraph(s) -> str:
    s = str(s or "").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = s.replace("\n", " ")
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


def ensure_period(s: str) -> str:
    s = (s or "").strip()
    if s and s[-1] not in ".!?":
        s += "."
    return s


def parse_retry_seconds(msg: str) -> Optional[int]:
    m = re.search(r"try again in (\d+)s", msg, re.I)
    return int(m.group(1)) if m else None


def has_useful_text(s) -> bool:
    if pd.isna(s):
        return False
    return len(one_paragraph(str(s))) >= 15


def safe_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, sep=CSV_SEP, index=False)
    tmp.replace(path)


def safe_write_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    tmp.replace(path)


# ---------------- ANTI-LEAKAGE: pan-cancer universal patterns ----------------

STATIC_FORBIDDEN_PATTERNS = [
    # diagnostic entity names (generic roots, covering most of the 32
    # project types without enumerating every combination)
    r"\bcarcinoma\b", r"\badenocarcinoma\b", r"\bsarcoma\b", r"\blymphoma\b",
    r"\bmelanoma\b", r"\bglioma\b", r"\bglioblastoma\b", r"\bastrocytoma\b",
    r"\boligodendroglioma\b", r"\bependymoma\b", r"\bmedulloblastoma\b",
    r"\bmesothelioma\b", r"\bthymoma\b", r"\bseminoma\b", r"\bchordoma\b",
    r"\bmeningioma\b", r"\bschwannoma\b", r"\bneurofibroma\b",
    r"\bpheochromocytoma\b", r"\bparaganglioma\b", r"\bteratoma\b",
    r"\bgerm cell (tumor|neoplasia)\b",
    # "ductal"/"lobular" are deliberately absent from this static list: they
    # name a diagnostic subtype only in breast cancer ("invasive ductal/
    # lobular carcinoma"), but elsewhere (and even within breast cancer)
    # they are legitimate architecture vocabulary ("tubulo-lobular
    # features", "ductal-pattern growth" -- the prompt itself encourages
    # the latter). Blocking them universally would fail correct cases: a
    # ductal tumor described with a "tubulo-lobular" pattern is a genuine
    # observation, not lobular-subtype leakage. They are instead handled
    # per row in dynamic_forbidden_terms(): if a row's own
    # cancer_type_detailed is "Breast Invasive Ductal Carcinoma", "ductal"
    # is forbidden only for that row; otherwise the word stays free to use
    # as architecture vocabulary.
    # "neuroendocrine", "melanocytic", "anaplastic" and similar terms are
    # also deliberately absent: they describe a visible cell type/lineage,
    # not the final entity label, and are kept even when they hint at the
    # diagnosis, as long as they describe what is actually observed (see
    # the prompt's label-vs-description distinction).

    # metastasis / origin
    r"\bmetastasis\b", r"\bmetastatic\b", r"\bsecondary\b",
    r"\bprimary tumor\b", r"\bsite of origin\b",

    # interpretive/conclusory language
    r"\bcompatible with\b", r"\bconsistent with\b", r"\bsuggestive of\b",
    r"\bin keeping with\b", r"\bfavor(s|ing)?\b", r"\bdiagnostic of\b",

    # meta-commentary about the rewriting/filtering process itself (the
    # model describing its own instructions instead of the tissue)
    r"\bdiagnostic leakage\b", r"\bmust be omitted\b", r"\bhas been (removed|omitted)\b",
    r"\bthese instructions\b", r"\bto (avoid|prevent) (revealing|disclosing|leaking)\b",

    # grade/stage as a FORMAL classification label (a number, a named
    # system, or "high/low-grade" used as a class) -- "well/moderately/
    # poorly differentiated" and "undifferentiated" are deliberately not
    # included here: differentiation is a real, observable morphological
    # property (cellular organization and maturity), distinct from a
    # formal grade number, and is kept, rephrased as an observation (see
    # the prompt), rather than blocked outright.
    r"\bwho grade\b", r"\bgleason\b", r"\bgrade\s*[ivx1-4]+\b",
    r"\bhigh[\s-]grade\b", r"\blow[\s-]grade\b",

    # staging
    r"\bp[tnm][0-4x]\b", r"\bstage [i1]+[iv]*\b", r"\btnm\b", r"\bconfined to\b",

    # specimen margins / extent
    r"\bmargin(s)?\b", r"\bresection margin\b", r"\bsurgical margin\b",
    r"\bentire specimen\b", r"\bwhole specimen\b", r"\bentire sample\b",

    # immunohistochemistry / biomarkers / molecular
    r"\bimmunohistochem\w*\b", r"\bmolecular\b", r"\bmutation\b",
    r"\bamplification\b", r"\btranslocation\b", r"\breceptor status\b",
    r"\bher2\b", r"\ber\b(?!\w)", r"\bpr\b(?!\w)", r"\bki[\s-]?67\b", r"\bmib[\s-]?1\b",
    r"\bpten\b", r"\begfr\b", r"\bp53\b", r"\bp63\b", r"\bvimentin\b",
    r"\bck\s*7\b", r"\bck\s*20\b", r"\bttf[\s-]?1\b", r"\bcdx2\b", r"\bgata3\b",
    r"\bpax8\b", r"\bs100\b", r"\bmelan[\s-]?a\b", r"\bcd\d{1,3}\b",
    r"\bsynaptophysin\b", r"\bchromogranin\b", r"\bdesmin\b", r"\bcalretinin\b",
    r"\bhbme1?\b",

    # gross/macroscopic specimen description -- these reports accompany a
    # microscopic WSI (a thin stained section), not a photograph of the
    # whole excised specimen. Any size/consistency/appearance of the
    # specimen to the naked eye or hand cannot be observed in a
    # microscopic image, so it is removed even when it is not diagnostic
    # by itself -- it is simply the wrong kind of information for this
    # image type.
    r"\b\d+(\.\d+)?\s*(cm|mm)\b",
    r"\bfleshy\b", r"\brubbery\b", r"\bfriable\b", r"\bspongy\b", r"\bgritty\b",
    r"\bfirm\b", r"\bcut surface(s)?\b", r"\bgross\b",
]

# Generic metadata words that are legitimate morphological vocabulary (they
# reveal nothing) and are therefore excluded from the per-row dynamic term
# list. Includes cytology/architecture adjectives that sometimes form part
# of an entity's formal name (e.g. "High-Grade Spindle Cell Sarcoma" in
# cancer_type_detailed) but are also legitimate morphology vocabulary
# outside that name (e.g. "highly infiltrative growth", "pleomorphic
# nuclei").
DYNAMIC_TERM_STOPWORDS = {
    "gland", "glands", "tissue", "site", "type", "cell", "cells", "duct",
    "ducts", "nos", "other", "unclassified", "specified", "otherwise", "not",
    "and", "with", "invasive",
    "high", "low", "grade", "spindle", "pleomorphic", "pleomorphism",
    "clear", "giant", "small", "large",
    # Same reasoning as "ductal"/"lobular" above: legitimate, universal
    # architecture/pattern vocabulary that can also appear inside a
    # subtype's formal name (e.g. "Papillary Serous Carcinoma" in UCEC).
    # Blocking them outright prevented describing real papillary/serous/
    # mucinous architecture that the prompt otherwise encourages keeping;
    # "soft"/"subcutaneous" similarly blocked real invasion-depth findings
    # in SARC ("invasion into deep soft tissue"), not just the organ of
    # origin.
    "papillary", "serous", "mucinous", "soft", "subcutaneous",
    # "malignant" is already absent from STATIC_FORBIDDEN_PATTERNS (TCGA is
    # 100% tumor, so it carries no discriminative information) but was
    # still being caught dynamically whenever cancer_type_detailed
    # contained it literally (e.g. "Malignant Peripheral Nerve Sheath
    # Tumor"). "nerve" names a real anatomical structure a tumor can arise
    # from or invade; describing it is a legitimate observation, not
    # leakage, even though it also appears in this entity's name.
    "malignant", "nerve",
}


def dynamic_forbidden_terms(row: pd.Series) -> set[str]:
    """Per-row forbidden terms derived from that row's own metadata
    (organ, detailed cancer type, Oncotree code) -- needed because TCGA is
    pan-cancer (32 types) and these terms vary by row, unlike a single-
    domain dataset."""
    terms: set[str] = set()
    for col in METADATA_COLUMNS_FOR_DYNAMIC_TERMS:
        val = str(row.get(col, "") or "")
        for tok in re.split(r"[^A-Za-z]+", val):
            tok_l = tok.lower()
            if len(tok_l) >= 4 and tok_l not in DYNAMIC_TERM_STOPWORDS:
                terms.add(tok_l)
    return terms


def has_forbidden_leakage(text: str, dynamic_terms: set[str]) -> list[str]:
    """Returns the list of patterns/terms that matched (empty if the text
    is clean); used both to decide whether to retry and for the audit JSON
    (what was removed / what still leaks)."""
    text = text or ""
    matched = [p for p in STATIC_FORBIDDEN_PATTERNS if re.search(p, text, re.I)]
    for term in dynamic_terms:
        if re.search(rf"\b{re.escape(term)}\w*\b", text, re.I):
            matched.append(term)
    return matched


# ---------------- PROMPTS (text_conditions/prompts/) ----------------
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
PROMPT_NO_DIAGNOSIS_STRICT = (PROMPT_DIR / "morphology_only_system.txt").read_text()
RETRY_SUFFIX_TEMPLATE = (PROMPT_DIR / "morphology_only_retry_suffix.txt").read_text()
USER_TEMPLATE = (PROMPT_DIR / "morphology_only_user.txt").read_text()
SCHEMA = json.loads((PROMPT_DIR / "morphology_only_output_schema.json").read_text())


def build_retry_prompt(matched_terms: list[str]) -> str:
    joined = ", ".join(sorted(set(matched_terms))[:20])
    return PROMPT_NO_DIAGNOSIS_STRICT + RETRY_SUFFIX_TEMPLATE.replace("{forbidden_terms}", joined)


def build_payload(source_text: str) -> str:
    return USER_TEMPLATE.replace("{source_excerpt}", one_paragraph(source_text))


# ---------------- LLM CALL ----------------

def call_llm(client: OpenAI, model: str, system_prompt: str, payload: str) -> dict:
    r = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": payload},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "tcga_no_diagnosis_strict_schema",
                "schema": SCHEMA,
                "strict": True,
            }
        },
        temperature=0.0,
        max_output_tokens=400,
        store=False,
    )
    return json.loads(r.output_text)


def call_llm_with_backoff(client: OpenAI, model: str, system_prompt: str, payload: str) -> dict:
    while True:
        try:
            return call_llm(client, model, system_prompt, payload)
        except RateLimitError as e:
            wait = parse_retry_seconds(str(e))
            time.sleep((wait + 2) if wait else 30)
        except (APITimeoutError, APIError):
            time.sleep(10)


# Safety net: even if has_sufficient_morphology comes back true, output
# shorter than this is still treated as insufficient, in case the model
# flags true but writes nothing usable in practice.
MIN_USEFUL_OUTPUT_CHARS = 20


def generate_validated_summary(
    client: OpenAI, model: str, row: pd.Series, max_attempts: int = 3
) -> dict:
    source_text = row[SOURCE_COLUMN]
    terms = dynamic_forbidden_terms(row)
    removed_from_source = sorted(set(has_forbidden_leakage(one_paragraph(source_text), terms)))

    base_result = {
        "slide_id": row["slide_id"],
        "case_id": row.get("case_id"),
        "project_id": row.get("project_id"),
        "original_slide_reports": row[SOURCE_COLUMN],
        "removed_terms": removed_from_source,
    }

    last_text = ""
    last_matched: list[str] = []
    attempts_used = 0
    for attempt in range(1, max_attempts + 1):
        attempts_used = attempt
        system_prompt = (
            build_retry_prompt(last_matched) if attempt > 1 else PROMPT_NO_DIAGNOSIS_STRICT
        )
        payload = build_payload(source_text)
        parsed = call_llm_with_backoff(client, model, system_prompt, payload)
        text = ensure_period(one_paragraph(parsed.get("slide_reports_no_diagnosis_strict", "")))
        has_morphology = bool(parsed.get("has_sufficient_morphology")) and len(text) >= MIN_USEFUL_OUTPUT_CHARS

        if not has_morphology:
            return {
                **base_result,
                "excluded_insufficient_source": True,
                "rewritten_text": "",
                "leakage_flag_after_retries": False,
                "remaining_matched_terms": [],
                "attempts_used": attempt,
            }

        last_text = text
        last_matched = has_forbidden_leakage(text, terms)
        if not last_matched:
            break

    return {
        **base_result,
        "excluded_insufficient_source": False,
        "rewritten_text": last_text,
        "leakage_flag_after_retries": bool(last_matched),
        "remaining_matched_terms": sorted(set(last_matched)),
        "attempts_used": attempts_used,
    }


# ---------------- CSV PLUMBING ----------------

def load_master(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=CSV_SEP, low_memory=False)
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed|^$")]
    return df


def select_universe(df: pd.DataFrame, ids_csv: Path | None) -> pd.DataFrame:
    if ids_csv is None:
        return df
    ids = pd.read_csv(ids_csv)["slide_id"]
    subset = df[df["slide_id"].isin(ids)].copy()
    missing = set(ids) - set(subset["slide_id"])
    if missing:
        print(f"[warning] {len(missing)} slide_id from {ids_csv} not found in the master CSV.", file=sys.stderr)
    return subset


def stratified_pilot(df: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    n = min(n, len(df))
    frac_per_group = n / len(df)
    sample = (
        df.groupby("project_id", group_keys=False)
        .apply(lambda g: g.sample(max(1, round(len(g) * frac_per_group)), random_state=seed),
               include_groups=False)
    )
    sample["project_id"] = df.loc[sample.index, "project_id"]
    if len(sample) > n:
        sample = sample.sample(n, random_state=seed)
    return sample.reset_index(drop=True)


# ---------------- MAIN ----------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV, required=DEFAULT_CSV is None,
                   help="Master CSV (';'-separated), see docs/DATA_FORMAT.md.")
    p.add_argument("--strict-common-ids", type=Path, default=DEFAULT_STRICT_COMMON_IDS,
                   help="Optional universe: slide_id of the cohort to process (omit = --all).")
    p.add_argument("--all", action="store_true",
                   help="Use every row in the master CSV instead of the strict-common subset.")
    p.add_argument("--pilot", type=int, default=0,
                   help="Sample N rows (stratified by project_id) to review quality before "
                        "launching the full batch.")
    p.add_argument("--pilot-review-out", type=Path, default=None,
                   help="With --pilot: instead of merging into the master CSV, write to this "
                        "review CSV (default /tmp/tcga_no_diagnosis_pilot_review.csv); --csv is "
                        "never touched.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default="gpt-4.1")
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--out", type=Path, default=None,
                   help="Defaults to overwriting --csv (atomic write). Use this to write "
                        "elsewhere and leave the original untouched.")
    p.add_argument("--audit-json", type=Path, default=None,
                   help="Defaults to the output CSV path with an _audit.json suffix. One object "
                        "per processed row with what was removed/changed and whether leakage "
                        "may remain.")
    p.add_argument("--overwrite", action="store_true",
                   help="Also regenerate rows that already have text in the new column "
                        "(default: skipped, so an interrupted batch can be resumed).")
    p.add_argument("--checkpoint-every", type=int, default=25)
    args = p.parse_args()

    df = load_master(args.csv)
    ids_csv = None if args.all else args.strict_common_ids
    universe = select_universe(df, ids_csv)

    is_pilot = bool(args.pilot)
    if is_pilot:
        universe = stratified_pilot(universe, args.pilot, seed=args.seed)
        print(f"Pilot: {len(universe)} rows stratified by project_id.")
    else:
        print(f"Full batch: {len(universe)} candidate rows (before applying resume logic).")

    universe = universe[universe[SOURCE_COLUMN].notna() & (universe[SOURCE_COLUMN].str.strip() != "")]

    if NEW_COLUMN not in df.columns:
        df[NEW_COLUMN] = pd.NA

    out_path = args.out or args.csv
    audit_path = args.audit_json or out_path.with_name(out_path.stem + "_audit.json")
    review_csv_path = args.pilot_review_out or Path("/tmp/tcga_no_diagnosis_pilot_review.csv")

    # Load any previous audit so resume logic also knows which rows were
    # already marked "insufficient morphology" -- those rows leave no text
    # in the CSV column, so without this they would be reprocessed on
    # every re-run.
    audit_rows: list[dict] = json.loads(audit_path.read_text()) if audit_path.exists() else []
    already_audited_ids = {r["slide_id"] for r in audit_rows}
    audit_by_id = {r["slide_id"]: r for r in audit_rows}

    if not args.overwrite:
        # Rows already audited in a previous run (e.g. a --pilot, which
        # never writes to the master CSV) may already have generated text
        # in the JSON but not yet in the CSV column -- backfill it here
        # before skipping, instead of losing (or re-paying for) that work.
        n_backfilled = 0
        for sid in already_audited_ids & set(universe["slide_id"]):
            r = audit_by_id[sid]
            if r["excluded_insufficient_source"]:
                continue
            mask = df["slide_id"] == sid
            if mask.any() and not has_useful_text(df.loc[mask, NEW_COLUMN].iloc[0]):
                df.loc[mask, NEW_COLUMN] = r["rewritten_text"]
                n_backfilled += 1
        if n_backfilled:
            print(f"Backfilled {n_backfilled} rows already generated in a previous run "
                  f"(from the audit JSON) -> CSV column filled without calling the API again.")

        already_done_mask = df["slide_id"].isin(universe["slide_id"]) & df[NEW_COLUMN].apply(has_useful_text)
        skip_ids = set(df.loc[already_done_mask, "slide_id"]) | (already_audited_ids & set(universe["slide_id"]))
        universe = universe[~universe["slide_id"].isin(skip_ids)]
        if skip_ids:
            print(f"Resuming: {len(skip_ids)} rows already processed (text present, or marked "
                  f"as insufficient morphology) -> skipped (use --overwrite to regenerate them).")
    else:
        audit_rows = [r for r in audit_rows if r["slide_id"] not in set(universe["slide_id"])]

    if universe.empty:
        print("Nothing to process (everything already generated, or the universe/filter selects no rows).")
        return

    client = OpenAI()
    n_leaked = 0
    n_excluded = 0

    for i, row in enumerate(tqdm(list(universe.itertuples(index=False, name="Row")),
                                  desc="tcga_no_diagnosis_strict"), start=1):
        row_series = pd.Series(row._asdict())
        result = generate_validated_summary(client, args.model, row_series, max_attempts=args.max_attempts)
        if result["excluded_insufficient_source"]:
            n_excluded += 1
        elif result["leakage_flag_after_retries"]:
            n_leaked += 1

        if not result["excluded_insufficient_source"]:
            df.loc[df["slide_id"] == result["slide_id"], NEW_COLUMN] = result["rewritten_text"]
        audit_rows.append(result)

        if args.checkpoint_every and i % args.checkpoint_every == 0:
            safe_write_json(audit_rows, audit_path)
            if not is_pilot:
                safe_write_csv(df, out_path)

    safe_write_json(audit_rows, audit_path)

    if is_pilot:
        review_df = pd.DataFrame([{
            "slide_id": r["slide_id"],
            "project_id": r["project_id"],
            SOURCE_COLUMN: r["original_slide_reports"],
            NEW_COLUMN: r["rewritten_text"],
            "excluded_insufficient_source": r["excluded_insufficient_source"],
            "leakage_flag_after_retries": r["leakage_flag_after_retries"],
            "attempts_used": r["attempts_used"],
        } for r in audit_rows if r["slide_id"] in set(universe["slide_id"])])
        review_df.to_csv(review_csv_path, index=False)
        print(f"\nPilot written for manual review to {review_csv_path} ({len(review_df)} rows) "
              f"and {audit_path} (full detail: original text, removed terms, etc). "
              f"{args.csv} was not touched.")
    else:
        safe_write_csv(df, out_path)
        print(f"\nColumn '{NEW_COLUMN}' updated for {len(universe) - n_excluded} rows -> {out_path}")

    print(f"Full audit at {audit_path}.")
    print(f"  {n_excluded}/{len(universe)} rows excluded for having no genuine morphology to "
          f"describe (left empty in '{NEW_COLUMN}', no filler text invented).")
    print(f"  {n_leaked}/{len(universe)} rows may still leak after {args.max_attempts} "
          f"attempts -- review these manually.")


if __name__ == "__main__":
    main()
