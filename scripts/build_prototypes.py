#!/usr/bin/env python3
"""
build_prototypes.py
=====================
Reads raw per-case patch embeddings and writes the adaptive WSI-specific
prototype bank for each case as a .npz file (one per case), using
rpgr.prototypes.build_adaptive_prototypes_for_case with the paper's default
AdaptiveSettings.

Read-only over the patch-embedding directory given in the config; writes
ONLY under `prototypes_dir`, never near the source data. Resumable: a case
whose .npz already exists is skipped unless --overwrite is given.

Usage:
  python3 build_prototypes.py --config ../configs/paths.yaml --backbone KEEP
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rpgr.prototypes import AdaptiveSettings, build_adaptive_prototypes_for_case  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--backbone", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--n-jobs", type=int, default=1, help="Parallel workers (joblib). 1 = sequential.")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    bb_cfg = cfg["backbones"][args.backbone]
    image_dir = Path(bb_cfg["image_embeddings_dir"])
    out_dir = Path(bb_cfg["prototypes_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    case_ids = sorted(f.stem for f in image_dir.glob("*.npy"))
    if not args.overwrite:
        case_ids = [cid for cid in case_ids if not (out_dir / f"{cid}_prototype.npz").exists()]
    print(f"[{args.backbone}] {len(case_ids)} casos por procesar (n_jobs={args.n_jobs})")

    settings = AdaptiveSettings()

    def process_one(cid: str) -> None:
        patches = np.load(image_dir / f"{cid}.npy").astype(np.float32)
        result = build_adaptive_prototypes_for_case(cid, patches, settings)
        tmp = out_dir / f".{cid}_prototype.npz.tmp"
        target = out_dir / f"{cid}_prototype.npz"
        with open(tmp, "wb") as fh:
            np.savez_compressed(
                fh,
                prototypes=result["prototypes"],
                K_star=np.asarray([result["K_star"]], dtype=np.int32),
                coverage=np.asarray([result["coverage"]], dtype=np.float32),
                support=np.asarray([result["support"]], dtype=np.float32),
                utility=np.asarray([result["utility"]], dtype=np.float32),
                q_proto=np.asarray([result["q_proto"]], dtype=np.float32),
                n_patches=np.asarray([result["n_patches"]], dtype=np.int32),
            )
        tmp.replace(target)  # atomic: never leaves a half-written .npz at the final name

    t0 = time.time()
    if args.n_jobs == 1:
        for i, cid in enumerate(case_ids):
            process_one(cid)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(case_ids)}...")
    else:
        from joblib import Parallel, delayed
        Parallel(n_jobs=args.n_jobs, backend="loky", verbose=10)(delayed(process_one)(cid) for cid in case_ids)

    print(f"listo en {time.time() - t0:.1f}s -> {out_dir}")


if __name__ == "__main__":
    main()
