# RPGR: Reciprocal Prototype-Guided Re-ranking

RPGR is a training-free, label-free post-hoc re-ranking method for WSI-report
cross-modal retrieval. Given frozen visual and textual representations from a
vision-language pathology foundation model, it improves retrieval quality
without touching the underlying encoders. Each whole-slide image is summarised
globally by mean pooling and locally by an adaptive bank of prototypes built
from the same patch embeddings. For every WSI-report pair, the local evidence is
the cosine of the report with the WSI's most similar prototype (max over the
bank). The final score combines a reciprocally calibrated global similarity with
the average of direct and reciprocal calibrations of the local similarity,

    S_RPGR = R(S_G) + 0.5 * (D(S_P) + R(S_P)),

where D and R are self-scaled log-softmax normalisations over candidates and over
queries, respectively. The whole gallery is re-ranked; there are no learned
weights and no cohort-tuned temperature. Prototype construction has its own
fixed parameters (`rpgr.prototypes.AdaptiveSettings`).

This repository contains RPGR and the evaluation code used to report it. The
comparison methods (hubness corrections, trained MIL and cross-modal models,
slide-level foundation models) are not included; they come from their public
implementations, cited in the paper.

## Install

```bash
pip install -r requirements.txt   # pinned, verified environment
pip install -e .                  # makes `import rpgr` available (scripts also work without it)
```

Python 3.12 (versions in `requirements.txt` are the verified environment). No GPU
required: RPGR runs on pre-extracted embeddings.

## What you need to bring

RPGR does not compute visual or textual embeddings, and does not extract
patches from whole-slide images. It expects those to already exist as
per-case `.npy` files. See `docs/DATA_FORMAT.md` for the exact shapes and
`configs/paths.example.yaml` for how to point the scripts at your data.

## Quickstart

```python
import numpy as np
from rpgr.pooling import bgap
from rpgr.prototypes import AdaptiveSettings, build_adaptive_prototypes_for_case
from rpgr.reranking import global_similarity, prototype_similarity, rpgr_score

# patch_embeddings: dict[case_id] -> (N_i, d) array ; text_matrix: (n, d) L2-normalised
wsi_matrix = np.stack([bgap(patch_embeddings[c]) for c in case_ids])
banks = [build_adaptive_prototypes_for_case(c, patch_embeddings[c], AdaptiveSettings())["prototypes"]
         for c in case_ids]

S_G = global_similarity(wsi_matrix, text_matrix)     # WSI rows x report columns
S_P = prototype_similarity(banks, text_matrix)       # max over each WSI's prototypes
scores_i2t = rpgr_score(S_G, S_P)                     # rank each row (WSI query) over all reports
scores_t2i = rpgr_score(S_G.T, S_P.T)                 # rank each row (report query) over all WSIs
```

Prototype banks are built once per gallery WSI (offline). The reciprocal terms use
statistics over the whole query set: scores are meant for a fixed query/gallery
collection evaluated in batch; adding a single new query exactly requires
recomputing the column statistics (O(#queries x #gallery)).

For a runnable, config-driven version of the full pipeline (including
evaluation), see `scripts/`.

## Reproducing the paper

`docs/REPRODUCING.md` lists what each script reproduces and what was checked.
`tests/` contains unit tests on synthetic data and an end-to-end test that
reproduces the reported held-out RPGR values (TCGA test, KEEP, Morphology-only)
exactly when the data layout is available (`RPGR_CONFIG=paths.yaml pytest tests/`).

## Repository layout

```
src/rpgr/             the method: pooling, prototypes, RPGR score, metrics,
                      paired case-level bootstrap and Holm correction (stats.py)
scripts/              general command-line entry points (caches, prototypes,
                      stress test, held-out benchmark, score ablation)
tests/                unit tests and an end-to-end reproduction test
text_preprocessing/   Morphology-only (`strict`) and Diagnosis-only report preprocessing
figures/              script for the fixed-case spatial figure of the paper
configs/              example data-layout configuration
docs/                 data format and reproduction instructions
```

## Citation

If you use this code, please cite the paper (full reference to be added on
publication).

## License

See `LICENSE`.
