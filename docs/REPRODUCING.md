# Reproducing the paper's results

All commands assume a `paths.yaml` filled in from `configs/paths.example.yaml`
(see `docs/DATA_FORMAT.md`), the pinned environment of `requirements.txt`, and that
you run from `scripts/`. Use a single BLAS thread for bit-exact results
(`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`): with multiple threads, near-ties can
change ranks in a few queries.

## 0. One-time precomputation, per backbone (offline)

```bash
python3 build_pool_cache.py --config ../configs/paths.yaml --backbone KEEP
python3 build_prototypes.py --config ../configs/paths.yaml --backbone KEEP
```

`build_prototypes.py` is the slow step: on one CPU thread it takes about 20 s per WSI
(adaptive k-means over the H grid). It is resumable and accepts `--n-jobs`.

## Baseline stress test (mean pooling only, no re-ranking)

```bash
python3 run_stress_test.py --config ../configs/paths.yaml
```

Gives class-macro mAP@10 for every backbone x text condition x granularity x
direction on the full TCGA retrieval cohort.

## RPGR across backbones and text conditions (TCGA half of the stress test)

```bash
python3 run_rpgr_grid.py --config ../configs/paths.yaml
```

Gives mean-pooling and RPGR mAP@10 for the 48 TCGA conditions on the train+test
cohort. The validation partition is excluded because it was used for method
development and baseline tuning (KEEP: n=4475). The 24 AI4SKIN conditions need the
non-public AI4SKIN data and are not covered.

## Held-out benchmark, RPGR row (TCGA test)

```bash
python3 run_heldout_rpgr.py --config ../configs/paths.yaml \
    --backbone KEEP --scenario strict --granularity OncotreeCode --split test
```

Gives R@1/3/5/10, MRR@10 and mAP@10 for I2T and T2I. The expected mAP@10 is
I2T 0.1964 and T2I 0.1511. This is checked automatically by
`RPGR_CONFIG=paths.yaml pytest tests/`.

### Tie handling

Some score matrices contain exactly tied candidates. In TCGA I2T, several reports have
identical text embeddings, and some comparison methods produce many ties. Two evaluation
modes are available:

- `--tie-break stable` (default): tied candidates keep gallery order.
- `--tie-break neutral`: expected metrics under a random order of tied candidates. The
  gallery is permuted with `numpy.random.default_rng(s)` for s = 0..49 (`--n-tie-seeds`),
  ranked with a stable sort, and per-query metrics are averaged. Scores are never modified.
  This is the protocol used for the cross-method comparison in the paper.

For RPGR both modes agree to four decimals (TCGA test mAP@10, neutral: I2T 0.196346,
T2I 0.151090). Both values are checked by the end-to-end test.

## Score-term ablation (held-out TCGA)

```bash
python3 run_score_ablation.py --config ../configs/paths.yaml --n-boot 20000
```

Compares RPGR with mean pooling, global calibration only (`R(S_G)`), a second calibrated
channel without prototypes (local channel replaced by `S_G`), local evidence only, and
RPGR with only one of the two local calibrations. It reports class-macro mAP@10 and
Recall@1, and paired case-level bootstrap tests against RPGR.

## Statistics

`rpgr.stats.paired_cluster_bootstrap` gives paired, class-stratified bootstrap CIs
and p-values for class-macro differences, resampling clinical cases.
`rpgr.stats.holm` applies Holm correction. The paper uses B=20000.

## Verification

On the TCGA data layout, with one BLAS thread and the pinned `requirements.txt`:

- `pytest tests/` passes (8 tests). With `RPGR_CONFIG` set, the end-to-end test reproduces the
  held-out RPGR mAP@10 to 1e-12, with stable (0.196367 / 0.151090) and tie-neutral
  (0.196346 / 0.151090) evaluation.
- `run_rpgr_grid.py --backbones KEEP` reproduces the 12 KEEP conditions of the stress
  test (mean pooling and RPGR) to 1e-15.
- `run_score_ablation.py` reproduces the reported ablation values and p-values.
- `build_prototypes.py` with scikit-learn 1.7.2 rebuilds stored prototype banks
  bit-for-bit. Other scikit-learn versions give the same H* but slightly different
  centroids.

## Figure scripts

`figures/prototype_overlay.py` reproduces the spatial view of the prototype matched to a
report for the fixed WSI-report pair shown in the paper. It needs the corresponding `.svs`
file and the patch coordinates, and it checks the expected patch count before drawing.
