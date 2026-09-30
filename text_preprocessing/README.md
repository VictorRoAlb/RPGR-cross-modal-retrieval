# Text preprocessing

Three textual conditions are used throughout the paper: Original (the
report as-is, not reproduced here since it needs no processing), Strict,
and Diagnosis-only.

`generate_strict_reports.py` removes explicit diagnostic terminology from a
report while keeping its histological/morphological description, using
GPT-4.1 (OpenAI's Responses API, JSON-schema structured output, temperature
0) constrained by an explicit prompt plus a regex-based anti-leakage check
(a fixed pan-cancer term list, plus terms derived per-row from that case's
own `cancer_type_detailed`/`site_of_resection_or_biopsy`). A row that fails
the check is retried, up to 3 attempts by default; a row whose source has
no genuine morphology to extract is left empty rather than padded with
invented content -- this is logged, not silently hidden. See the prompt
itself (`PROMPT_NO_DIAGNOSIS_STRICT`) for the exact rules, including the
worked examples that distinguish a real morphological observation from the
diagnosis name translated into a synonym.

`extract_diagnosis_only.py` needs no model call: it copies the
`cancer_type_detailed` column verbatim for the same cohort.

Both scripts require an OpenAI-format master CSV with (at least) the
columns `slide_id`, `slide_reports`, `cancer_type_detailed`,
`site_of_resection_or_biopsy`; `generate_strict_reports.py` additionally
needs `OPENAI_API_KEY` set in the environment.
