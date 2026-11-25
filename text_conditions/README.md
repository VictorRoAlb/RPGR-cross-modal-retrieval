# Report conditions

Each report is used under three conditions (Section 4.2 and Appendix B of the paper).

* **Original** — the slide-level TCGA report released with TITAN; for AI4SKIN, the English
  translation of the original Spanish report.
* **Morphology-only** — the histological and morphological description without the diagnosis,
  grade, stage, immunohistochemistry or any other information not observable in the slide,
  generated with GPT-4.1 and the same prompt for both cohorts (`prompts/`; OpenAI Responses API,
  temperature 0, JSON-schema structured output). The AI4SKIN reports were translated into
  English before rewriting.
  For TCGA, `generate_morphology_only.py` additionally checks each output against a pan-cancer
  list of forbidden diagnostic terms and against case-specific terms from the metadata, and
  regenerates it with a stricter instruction when the check fails (up to three attempts in
  total). Reports without enough morphological description are left empty and excluded
  (1434 TCGA WSIs). This filter was applied to TCGA only.
  For AI4SKIN, the English translation and the Morphology-only text are read from the dataset
  export (fields `summary_en` and `summary_no_diagnosis_strict_en`).
* **Diagnosis-only** — the full subtype name of the case: for TCGA, `cancer_type_detailed`
  (extracted by `extract_diagnosis_only.py`); for AI4SKIN, the English name of the class:
  *Dermatofibroma*, *Melanoma*, *Atypical fibroxanthoma / pleomorphic dermal sarcoma*,
  *Angioleiomyoma / piloleiomyoma*, *Leiomyosarcoma*, *Dermatofibrosarcoma protuberans*,
  *Spindle cell / sarcomatoid squamous cell carcinoma*.

## Prompts (`prompts/`)

| File | Content |
|---|---|
| `morphology_only_system.txt` | system prompt for the Morphology-only rewriting (both cohorts) |
| `morphology_only_user.txt` | user message; `{source_excerpt}` is the original report (one paragraph) |
| `morphology_only_retry_suffix.txt` | TCGA filter: appended to the system prompt after a failed check; `{forbidden_terms}` lists the matched terms |
| `morphology_only_output_schema.json` | structured-output schema (`has_sufficient_morphology`, `slide_reports_no_diagnosis_strict`) |

`generate_morphology_only.py` reads these files; they are the exact texts sent to the model.
Install `requirements/text_conditions.txt` to run it. The openai version used for the paper
was not recorded; any version from 1.66 on provides the Responses API that the script uses.

## Report embeddings

`encode_reports.py` encodes any of the three conditions with the frozen text encoder of KEEP,
CONCH, MUSK or PATHO-CLIP. Install the matching `requirements/encoder_*.txt` in a separate
environment for each encoder.

## Text encoders

| VL-PFM | Hugging Face repository | Revision | Context |
|---|---|---|---|
| KEEP | `Astaxanthin/KEEP` | `28a25d95cc6ba27a7e6fab3f144e13dbafd8b21e` | 256 tokens |
| CONCH | `MahmoodLab/conch` (`conch_ViT-B-16`) | `f9ca9f877171a28ade80228fb195ac5d79003357` | 128 tokens |
| MUSK | `xiangjx/musk` (`musk_large_patch16_384`) | `de1ffed28608c197d2903f6fa42b491a3fbf0fb8` | 230 tokens |
| PATHO-CLIP | `WenchuanZhang/Patho-CLIP-L` (open_clip ViT-L-14) | `8c9d18432daa80f211dbcfcfcb40af42b9d1b9b7` | 77 tokens |

`encode_reports.py` uses one environment per encoder (`requirements/encoder_*.txt`).
