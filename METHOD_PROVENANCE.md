# Method provenance

The code under `src/rpgr/` and `scripts/` is extracted from a larger internal
project rather than written from scratch for this release. This document
says where each piece came from, what was changed on the way out, and what
was checked before trusting the extraction.

Two kinds of change were made, and only these: removing hardcoded paths and
dataset-specific orchestration in favour of a config file or function
arguments, and replacing a dependency with a plain-numpy equivalent when the
original only used that dependency to reuse an existing call, not because
the computation needed it. The scientific logic itself -- formulas,
constants, default parameters, order of operations -- was not touched.

## Where each module comes from

`pooling.py` (`bgap`, `max_pool`) is extracted from `tcga_pgr/build_pool_cache.py`;
only the hardcoded path dictionary and batch/caching orchestration were
removed, the arithmetic is unchanged, and `bgap()`/`max_pool()` reproduce the
original `pool_one()` bit-for-bit on real cases.

`prototypes.py` (`build_adaptive_prototypes_for_case`) is the algorithmic
core of `code/prueba/tcga_adaptive_lib.py` (part 1 of that file); the case
discovery, joblib parallelization, and `.npz` cache read/write logic (part 2)
is dataset orchestration rather than part of the method and now lives in
`scripts/build_prototypes.py` instead. The extracted function reproduces the
original bit-for-bit (`K*`, the full prototype arrays, `utility`, `q_proto`)
on real cases spanning different patch counts. One caveat: prototypes
regenerated from raw patches via `scripts/build_prototypes.py` are close to
but not bit-identical with the ones already cached on disk in the source
project -- cosine similarity 0.998-1.000, `coverage`/`utility` matching to
the 4th decimal. This tracks `MiniBatchKMeans` not being perfectly
bit-reproducible across scikit-learn versions even with a fixed
`random_state`; the scikit-learn version used to build the original cache
isn't recorded anywhere in the source project. None of the verification
below is affected, since it was all run against the existing cache, not a
freshly regenerated one -- but anyone regenerating prototypes from raw
patches in a different environment should expect headline metrics to
reproduce to roughly 3-4 decimal places rather than bit-for-bit.

`percentile_calibration.py` (`forward_global`, `reciprocal_global`,
`reciprocal_prototype`) comes from a single `percentile_rows` helper in
`tcga_pgr/run_line2_reciprocal_local_included.py`, split into three named
wrappers so each call site states which of Eqs. 21-23 it computes instead of
a bare `percentile_rows(...)` / `percentile_rows(...T).T` pattern repeated
three times with no label. The underlying arithmetic is identical; it isn't
exposed as separate matrices in the original script, so it was checked
through its effect downstream on `S_RPGR` and `S_B` instead (see below).

`reranking.py` (`global_similarity`, `prototype_similarity`, `rpgr_score`,
`no_prototype_score`, `rerank_top_pool`) is the score-assembly logic of that
same script, split into named functions instead of one inline block.
`S_global`, `S_proto`, `S_RPGR`, and `S_B` are bit-for-bit identical to the
original inline computation on 120 real cases (max absolute difference 0.0),
the Top-50 pool matches for all 120 queries, and running this module alone
with `metrics.py` against the full TEST cohort (n=1578) reproduces all 12
metrics of Table 2's RPGR row exactly. One wording gap worth flagging: the
paper states that candidates outside the Top-50 pool "preserve their
original global ordering," but the implementation (original and this
release alike) sets them to `-inf` rather than re-imposing `S_G` order among
them. This has no effect on any reported metric -- Recall@1/3/5/10, MRR@10,
and mAP@10 never look past position 10 -- but the sentence doesn't exactly
describe what the code does; left for the authors to adjust if they choose.

`metrics.py` combines `evaluate_direction_incl`/`evaluate_direction_excl`
and `class_level_table`/`macro_from_class_level` from
`tcga_pgr/eval_protocol.py` with `hit_rate_at_k`/`reciprocal_rank` from
`tcga_eval/cross_modal_eval_lib.py`. The two `evaluate_direction_*` variants
were merged into one `evaluate_direction(..., exclude_exact_pair=bool)`,
defaulting to `False` (exact-pair included, the paper's protocol) so the
excluded variant can't silently become the default. `hit_rate_at_k` and
`reciprocal_rank` originally ran through torchmetrics on a synthetic
score vector purely to reuse an existing call; both reduce to simple
boolean-array operations once the relevance vector is sorted by descending
similarity (which it always is here), so they're reimplemented directly in
numpy to avoid a torch dependency for something training-free re-ranking
doesn't need. Across 500 randomized trials, `average_precision` and
`hit_rate_at_k` match the originals exactly and `reciprocal_rank` differs by
about 5e-9, traced to torch's float32 internals versus this module's
float64. The full `evaluate_direction` pipeline matches the original macro
output to better than 1e-12 on the real TEST cohort.

`scripts/build_pool_cache.py` and `scripts/build_prototypes.py` are thin CLI
wrappers around the functions above, with no algorithmic content of their
own. `scripts/run_stress_test.py`, `run_rpgr_grid.py`, `run_heldout_rpgr.py`,
and `run_prototype_ablation.py` come from the grid-loop structure of
`run_line2_reciprocal_local_included.py`, the BGAP-only loop of
`run_line1_textual_robustness_included.py`, and the ablation protocol of
`code/prueba/final_reproducibility_audit/pgr_prototype_ablation_cross_backbone_val.py`,
checked directly against the published tables using this project's existing
caches: `run_heldout_rpgr.py` reproduces Table 2's RPGR row (12/12 values
exact), `run_rpgr_grid.py` reproduces Table 1 and the TCGA half of Figure 2
(96/96 exact across the 48 conditions), `run_stress_test.py` reproduces
Table 1 on its own (48/48 exact), and `run_prototype_ablation.py` reproduces
Table 4 (16/16 exact across 4 backbones).

## Two bugs the verification caught

Building `run_prototype_ablation.py` surfaced a real one: the first version
scored `S_B` and `S_RPGR` on the full similarity matrix without restricting
to the Top-50 pool discovered from `S_global`. The original ablation script
applies that same restriction to every variant, including the no-prototype
one; skipping it gave results close to but not exactly matching Table 4 (off
by 0.0001-0.0029 depending on backbone and direction). Fixed by calling
`rerank_top_pool` on both scores before evaluation, matching the original
protocol, and re-verified exact afterward.

A second, smaller one surfaced in the same pass: building the evaluation
cohort through `set(slide_id_list)` instead of preserving the CSV's row
order changed `argsort`'s tie-breaking at the Top-50 boundary for a handful
of queries, shifting a few backbone/direction combinations by 0.0001-0.0002.
Every script here now takes case ids from `.tolist()` on the filtered
dataframe directly, never through a `set`.

## What this release doesn't include

RPGR-BGS, an earlier linear-blend re-ranking formula referenced in some of
the source scripts above, and any bootstrap, permutation, or
Holm-Bonferroni machinery associated with it, are deliberately not part of
this release. Table 4 is a descriptive ablation only (`S_B`, RPGR, delta) --
no confidence intervals, hypothesis tests, or p-values, matching the final
decision for both the paper and this repository.
