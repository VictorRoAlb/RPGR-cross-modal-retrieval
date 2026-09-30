# Method provenance

This file traces every piece of code under `src/rpgr/` and `scripts/` back to
the internal script it was extracted from, what was actually changed, and how
each extraction was checked against the original before being trusted. It
exists so that a reviewer (or a future version of ourselves) can confirm that
nothing in this release quietly drifted from what produced the paper's
numbers.

The rule followed throughout: the scientific logic (formulas, constants,
default parameters, order of operations) was never modified. Only two kinds
of change were allowed, and both are called out explicitly below wherever
they occur: (1) removing hardcoded paths and dataset-specific orchestration
in favour of a config file or function arguments, and (2) replacing a
dependency with a plain-numpy equivalent when the original only used that
dependency to reuse an existing call, not because the computation needed it.

## src/rpgr/pooling.py

**Source:** `tcga_pgr/build_pool_cache.py` (functions `l2_normalize`,
`l2_normalize_rows`, `pool_one`).

**What changed:** the hardcoded backbone-to-path dictionary and the joblib
batch/caching orchestration were removed. `pool_one` was split into `bgap()`
and `max_pool()`, matching the paper's own naming; the arithmetic (normalize
each patch, aggregate, normalize the result) is untouched.

**Verified:** `bgap()` and `max_pool()` reproduce `pool_one()`'s output
bit-for-bit on 5 real KEEP cases (`np.array_equal`, exact).

## src/rpgr/prototypes.py

**Source:** `code/prueba/tcga_adaptive_lib.py`, part 1 only (`AdaptiveSettings`,
`_clip01`, `_normalize`, `_mean_prototype`, `_weighted_centroid_refine`,
`_fallback_result`, `build_adaptive_prototypes_for_case`).

**What changed:** part 2 of the original file (case discovery, joblib
parallelization, `.npz` resume/write logic) was dropped entirely -- it is
dataset orchestration, not part of the method, and now lives in
`scripts/build_prototypes.py` instead. `config_hash()`/`to_json()` (cache
bookkeeping) were also dropped. The algorithmic core is copied unchanged,
with comments added tying each block to its equation in the paper (Eqs.
6-18).

**Verified:** bit-for-bit identical `K*`, full prototype arrays, `utility`
and `q_proto` against the original on 5 real cases spanning different patch
counts (2358-5500 patches, K* selected: 16/32/32/24/16).

Separately: prototypes regenerated from raw patches via
`scripts/build_prototypes.py` are *not* bit-identical to the ones already
sitting on disk in this project's own cache -- see "Known limitation" below.
This does not affect the equivalence claim above, which compares the
extracted function against the original function run in the same
environment.

## src/rpgr/percentile_calibration.py

**Source:** `tcga_pgr/run_line2_reciprocal_local_included.py`, function
`percentile_rows`.

**What changed:** nothing algorithmic; split into three named wrappers
(`forward_global`, `reciprocal_global`, `reciprocal_prototype`) so each call
site states which of Eqs. 21-23 it computes instead of a bare
`percentile_rows(...)`/`percentile_rows(...T).T` pattern repeated three
times.

**Verified:** as part of the `reranking.py` test below (F_G/R_G/R_P are not
exposed as separate matrices in the original script, only used inline, so
they were verified through their downstream effect on S_RPGR and S_B).

## src/rpgr/reranking.py

**Source:** `tcga_pgr/run_line2_reciprocal_local_included.py` (the score
assembly and `build_final_scores` function).

**What changed:** nothing algorithmic. The inline computation was split into
`global_similarity`, `prototype_similarity`, `rpgr_score`, `no_prototype_score`
(Eq. 26, used by the ablation), and `rerank_top_pool` (the direction-aware
Top-50 pool restriction, renamed from `build_final_scores`).

**Verified:** on 120 real KEEP+Strict cases, `S_global`, `S_proto`, `S_RPGR`
and `S_B` are bit-for-bit identical to the original inline computation
(max abs diff = 0.0), the Top-50 pool is identical for all 120 queries, and
the final I2T/T2I rankings match exactly at Top-1/3/5/10 for all queries.
Additionally, running this module alone (with `metrics.py`) against the
full TEST cohort (n=1578) reproduces all 12 metrics of Table 3's RPGR row
(I2T and T2I) exactly.

**Known, harmless wording gap:** the paper states that candidates outside
the Top-50 pool "preserve their original global ordering." The
implementation (original and this release, identically) sets those
candidates to `-inf` rather than re-imposing S_G order among them. This has
no effect on any reported metric, since Recall@1/3/5/10, MRR@10 and mAP@10
never look past position 10, well inside the pool -- confirmed by the exact
match above. Flagged for the authors to decide whether to adjust the
sentence; not something this codebase silently "fixed."

## src/rpgr/metrics.py

**Source:** `tcga_pgr/eval_protocol.py` (`evaluate_direction_incl`,
`evaluate_direction_excl`, `class_level_table`, `macro_from_class_level`) and
`tcga_eval/cross_modal_eval_lib.py` (`hit_rate_at_k`, `reciprocal_rank`,
`average_precision`).

**What changed, beyond I/O:** the original computed `hit_rate_at_k` and
`reciprocal_rank` by routing a synthetic strictly-descending score vector
through `torch`/`torchmetrics`, purely to reuse an existing library call --
not because the computation needs a tensor framework. Both are simple
boolean-array operations once `rel` is sorted by descending similarity
(which it always is here), so they were reimplemented directly in numpy.
`average_precision` was already plain numpy and is copied unchanged. The
two `evaluate_direction_*` functions were merged into one
`evaluate_direction(..., exclude_exact_pair=bool)`, defaulting to `False`
(exact-pair INCLUDED, the paper's protocol) so the excluded variant can
never silently become the default.

**Verified:** 500 randomized trials of the three metric primitives against
the originals; `average_precision` and `hit_rate_at_k` matched exactly,
`reciprocal_rank` differed by ~5e-9, traced to torch's float32 internals
versus this module's float64 (e.g. `1/6` in float64 vs. its float32
rounding -- confirmed by direct computation, not assumed). Full
`evaluate_direction` (both INCLUDED and EXCLUDED) matched the original
macro output to better than 1e-12 on the real TEST cohort. Running the
complete pipeline (`reranking.py` + `metrics.py`, no other file) against
KEEP+Strict+TEST reproduces all 12 values of Table 3's RPGR row exactly at
the reported 4-decimal precision.

## scripts/build_pool_cache.py, scripts/build_prototypes.py

Thin CLI wrappers around `pooling.bgap`/`pooling.max_pool` and
`prototypes.build_adaptive_prototypes_for_case`, reading raw per-case patch
embeddings from a config-specified directory and writing only the
requested cache file. No algorithmic content of their own.

## scripts/run_heldout_rpgr.py, run_rpgr_grid.py, run_stress_test.py, run_prototype_ablation.py

**Source:** `tcga_pgr/run_line2_reciprocal_local_included.py` (grid loop
structure), `tcga_pgr/run_line1_textual_robustness_included.py` (BGAP-only
loop), and `code/prueba/final_reproducibility_audit/pgr_prototype_ablation_cross_backbone_val.py`
(the ablation protocol).

**Verified against the paper directly** (not just against the internal
scripts), using the real, already-existing caches on this project's disk:

| script | reproduces | result |
|---|---|---|
| `run_heldout_rpgr.py` | Table 3, RPGR row, TEST n=1578 | 12/12 exact |
| `run_rpgr_grid.py` | Tables 1 and 2, all 48 conditions, 4 backbones | 96/96 exact |
| `run_stress_test.py` | Table 1 independently | 48/48 exact |
| `run_prototype_ablation.py` | Table 4, 4 backbones | 16/16 exact |

**A real bug was found and fixed while building `run_prototype_ablation.py`:**
the first version evaluated `S_B` and `S_RPGR` directly on the full
similarity matrix, without restricting to the Top-50 pool discovered by
`S_global`. The original ablation script (`pgr_prototype_ablation_cross_backbone_val.py`)
applies that same pool restriction to every variant, including the
no-prototype one -- missing it produced results close to but not exactly
matching Table 4 (off by 0.0001-0.0029 depending on backbone/direction).
Fixed by calling `rerank_top_pool` on both `S_B` and `S_RPGR` before
evaluation, matching the original protocol; re-verified exact afterwards.

**A second bug surfaced during that same debugging pass:** building the
evaluation cohort via `set(slide_id_list)` instead of preserving the
CSV's row order changed `argsort`'s tie-breaking at the Top-50 boundary for
a handful of queries, moving a few backbone/direction combinations by
0.0001-0.0002. Every script in this release now takes case ids directly
from `.tolist()` on the filtered dataframe, never through a `set`.

## Known limitation: prototype regeneration is not bit-exact across environments

Running `build_prototypes.py` from raw patches in this environment (scikit-learn
1.9.0) produces prototypes that are extremely close to, but not
bit-identical with, the ones already cached on this project's disk: K*,
`n_patches`, and the overall shape of the result all match, and every
regenerated prototype has cosine similarity 0.998-1.000 with its cached
counterpart. `coverage`/`utility` differ in the 4th decimal. This is
consistent with `MiniBatchKMeans` not being perfectly bit-reproducible
across scikit-learn versions even with a fixed `random_state`; the exact
scikit-learn version used to build the original on-disk cache is not
recorded anywhere in this project. It does not affect any number already
verified above, since every verification in this document uses the
already-existing cache, not a freshly regenerated one. Anyone regenerating
prototypes from raw patches in a different environment should expect
headline metrics to reproduce to within roughly 3-4 decimal places, not
bit-for-bit.

## Statistical analysis

RPGR-BGS, the older linear-blend re-ranking formula referenced in some of
the scripts above (`pgr_bgs`, `q_proto`-weighted mixing), and any bootstrap /
permutation / Holm-Bonferroni machinery associated with it, are
deliberately **not** part of this release. Table 4 is reported as a
descriptive ablation only (`S_B`, RPGR, delta) -- no confidence intervals,
no hypothesis tests, no p-values, matching the final decision for both the
paper and this repository.
