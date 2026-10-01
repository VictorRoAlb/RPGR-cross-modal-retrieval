# Reproducing the paper's tables

All commands assume a `paths.yaml` filled in from `configs/paths.example.yaml`
(see `docs/DATA_FORMAT.md` for what each path should contain) and that you
are running from `scripts/`.

## 0. One-time precomputation, per backbone

```bash
python3 build_pool_cache.py  --config ../configs/paths.yaml --backbone KEEP
python3 build_prototypes.py  --config ../configs/paths.yaml --backbone KEEP
```

Repeat for CONCH, MUSK, PATHO_CLIP if you intend to reproduce Tables 1, 2 or
4 (which cover all four backbones); Table 3 only needs KEEP.
`build_prototypes.py` is the slow step (adaptive k-means per case); it is
resumable and safe to re-run, and accepts `--n-jobs` for parallelism.

Note: prototypes regenerated this way will be extremely close to, but not
bit-identical with, whatever cache produced the paper's own numbers --
see the "Known limitation" section of `METHOD_PROVENANCE.md`. Expect
agreement to 3-4 decimal places, not exact, unless you start from the
paper's original prototype cache rather than regenerating it.

## Table 1 — stress-testing frozen VL-PFMs (BGAP only)

```bash
python3 run_stress_test.py --config ../configs/paths.yaml
```

Writes `outputs/stress_test_table1.csv`: one row per (backbone, textual
scenario, granularity, direction), `mAP@10`.

## Table 2 — effect of RPGR across the full 48-condition grid

```bash
python3 run_rpgr_grid.py --config ../configs/paths.yaml
```

Writes `outputs/rpgr_grid_table2.csv`: same 48 rows as Table 1, plus
`rpgr_mAP@10` and the relative `delta_pct`.

## Table 3 — held-out benchmark, RPGR row only

This repository reproduces RPGR's own row; the other seven methods in
Table 3 (BGAP, Max pooling, RR, TransMIL, SETMIL, DTN, FGCR) are outside
its scope -- see the paper's Data and code availability statement.

```bash
python3 run_heldout_rpgr.py --config ../configs/paths.yaml \
    --backbone KEEP --scenario strict --granularity OncotreeCode --split test
```

Prints and writes the full metric set (R@1/3/5/10, MRR@10, mAP@10, I2T and
T2I) to `outputs/heldout_rpgr_KEEP_strict_OncotreeCode_test.csv`.

## Table 4 — contribution of local prototypes (descriptive ablation)

```bash
python3 run_prototype_ablation.py --config ../configs/paths.yaml \
    --scenario strict --granularity OncotreeCode --split val
```

Writes `outputs/prototype_ablation_table4_val.csv`: `S_B`, RPGR and their
difference per backbone and direction. No confidence intervals or
significance testing are computed here or anywhere in this repository --
see `METHOD_PROVENANCE.md`.

## Figures

- **Figure 2** (qualitative re-ranking example) is an illustrative,
  case-specific figure built from a single hand-picked example; it is not
  part of this release's reproducible pipeline in the same sense as the
  tables above.
- **Figure B.1** (prototype spatial overlay) can be regenerated for any
  case once you have its prototype bank and original WSI; see
  `figures/prototype_overlay.py`.

## What "reproduces" means here

Every number quoted above was checked against the actual published table
value at 4-decimal precision, from this release's code alone, run against
the real project data -- not against a re-derivation on paper. See
`METHOD_PROVENANCE.md` for the exact verification performed on each script.
