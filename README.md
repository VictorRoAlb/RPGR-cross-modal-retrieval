# RPGR: Reciprocal Prototype-Guided Re-ranking

RPGR is a training-free, label-free post-hoc re-ranking method for WSI-report
cross-modal retrieval. Given frozen visual and textual representations from a
vision-language pathology foundation model, it improves retrieval quality
without touching the underlying encoders: it summarizes each whole-slide
image globally through mean pooling, builds an adaptive bank of local
prototypes from the same patch embeddings, and combines direct global,
reciprocal global, and reciprocal prototype evidence through percentile
calibration and a geometric mean. Only the initial Top-50 candidates (by
global similarity) are ever re-ordered.

This repository contains RPGR itself and the evaluation code used to report
it, matching the paper's own code-availability statement. It does not
include the trained baselines it is compared against (TransMIL, SETMIL, DTN,
FGCR) -- those were adapted from their respective public repositories, cited
in the paper.

## Install

```bash
pip install -r requirements.txt
```

Python 3.11+. No GPU required; RPGR runs entirely on pre-extracted
embeddings.

## What you need to bring

RPGR does not compute visual or textual embeddings, and does not extract
patches from whole-slide images. It expects those to already exist as
per-case `.npy` files. See `docs/DATA_FORMAT.md` for the exact shapes and
`configs/paths.example.yaml` for how to point the scripts at your data.

## Quickstart

```python
from rpgr.pooling import bgap
from rpgr.prototypes import AdaptiveSettings, build_adaptive_prototypes_for_case
from rpgr.reranking import global_similarity, prototype_similarity, rpgr_score, rerank_top_pool

# patch_embeddings: dict[case_id] -> (N_i, d) array
wsi_vectors = {cid: bgap(patches) for cid, patches in patch_embeddings.items()}
prototypes = {
    cid: build_adaptive_prototypes_for_case(cid, patches, AdaptiveSettings())["prototypes"]
    for cid, patches in patch_embeddings.items()
}

sim_global = global_similarity(wsi_matrix, text_matrix)
sim_proto = prototype_similarity(list(prototypes.values()), text_matrix)
score = rpgr_score(sim_global, sim_proto)
final_ranking = rerank_top_pool(sim_global, score, direction="I2T")
```

For a runnable, config-driven version of the full pipeline (including
evaluation), see `scripts/`.

## Reproducing the paper

`docs/REPRODUCING.md` maps every table to the exact command that produces
it. Every number it claims was checked against the published value at
4-decimal precision from this code alone -- see `METHOD_PROVENANCE.md` for
what was verified, how, and the two real bugs that verification caught
before this release was considered done.

## Repository layout

```
src/rpgr/            the method itself: pooling, prototypes, percentile
                      calibration, re-ranking, evaluation metrics
scripts/              config-driven CLI entry points that use src/rpgr/
                      to reproduce the paper's tables
text_preprocessing/   the Strict/Diagnosis-only report preprocessing
figures/              qualitative and illustrative figures from the paper
configs/              example data-layout configuration
docs/                 data format and reproduction instructions
```

## Citation

If you use this code, please cite the paper (full reference to be added on
publication).

## License

See `LICENSE`.
