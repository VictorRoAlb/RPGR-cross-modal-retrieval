# Data format

RPGR does not ship any data, and does not include code to produce visual
patch embeddings from whole-slide images. This document states exactly what
it expects to be given, so nothing about the preprocessing pipeline is
implied that isn't actually backed by code in this repository.

## Patch-level visual embeddings

One `.npy` file per case, named `<case_id>.npy`, holding a `(N_i, d)` float
array: `N_i` patches for that whole-slide image, each a `d`-dimensional
embedding from a frozen vision-language pathology foundation model. `N_i`
varies freely across cases; `d` is fixed per backbone (768 for KEEP, 512 for
CONCH, 1024 for MUSK, 768 for PATHO-CLIP in the paper). Rows need not be
pre-normalized -- `pooling.bgap` and `prototypes.build_adaptive_prototypes_for_case`
both L2-normalize internally.

How these patches were produced (patch size, magnification, tissue
segmentation, colour normalization) is described in the paper's Section 4.2.
That preprocessing is external to this repository: it produced the `.npy`
files above, but no script here performs it, and none is included, because
none was used to write this release.

## Pathology-report text embeddings

One `.npy` file per case, named `<case_id>.npy`, holding a `(d,)` float
vector from the same frozen model's text encoder, `d` matching the visual
side for that backbone. A separate directory per textual scenario
(Original / Strict / Diagnosis-only) is expected, since a case's report is
represented differently under each.

## Precomputed caches (produced by this repository)

- **Pool cache** (`scripts/build_pool_cache.py` output): one `.npz` per
  backbone with `case_ids` (N,), `mean_pool` (N, d), `max_pool` (N, d) --
  the BGAP/max-pool of every case's patches, computed once so downstream
  scripts never re-read raw patches.
- **Prototype bank** (`scripts/build_prototypes.py` output): one
  `<case_id>_prototype.npz` per case, holding `prototypes` (K*, d,
  L2-normalized), `K_star`, `coverage`, `support`, `utility`, `q_proto`,
  `n_patches`. `q_proto` is retained for parity with the original
  implementation but is not used anywhere in the RPGR score itself.

## Cohort and label files

- **Split CSV**: at minimum a `slide_id` column and a `split` column with
  values among `train`/`val`/`test`. The paper's split is case-level
  (every WSI of the same case in the same partition) and stratified by
  subtype with a fixed seed -- see the paper's Section 4.1 for the exact
  procedure; this repository consumes whatever split file it is given, it
  does not regenerate one.
- **Labels CSV**: a `slide_id` column plus one column per granularity used
  (`OncotreeCode` for subtype, `project_id` for organ/project in the
  paper).

## What is genuinely not included

Raw whole-slide images, raw patch extraction code, and the code that
produced the visual patch embeddings above are all outside this
repository's scope -- see Section 4.2 of the paper for what was used, and
treat any claim beyond that paragraph as unverified until you have checked
it against your own pipeline.
