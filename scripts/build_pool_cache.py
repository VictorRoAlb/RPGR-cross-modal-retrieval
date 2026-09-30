#!/usr/bin/env python3
"""
build_pool_cache.py
====================
Reads raw per-case patch embeddings ONCE and writes their BGAP mean-pool
and max-pool as a single .npz cache, so downstream table/figure scripts
never need to re-read raw patches.

Read-only over the patch-embedding directory given in the config; writes
ONLY the .npz cache file at `pool_cache_path`, never near the source data.

Usage:
  python3 build_pool_cache.py --config ../configs/paths.yaml --backbone KEEP
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rpgr.pooling import bgap, max_pool  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="Path to a paths.yaml (see configs/paths.example.yaml)")
    p.add_argument("--backbone", required=True, help="Backbone key under config['backbones'], e.g. KEEP")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    bb_cfg = cfg["backbones"][args.backbone]
    image_dir = Path(bb_cfg["image_embeddings_dir"])
    out_path = Path(bb_cfg["pool_cache_path"])

    if out_path.exists() and not args.overwrite:
        print(f"{out_path} ya existe, usa --overwrite para regenerarlo. Nada que hacer.")
        return

    case_ids = sorted(f.stem for f in image_dir.glob("*.npy"))
    print(f"[{args.backbone}] {len(case_ids)} casos en {image_dir}")

    t0 = time.time()
    mean_pool = np.empty((len(case_ids), bb_cfg["dim"]), dtype=np.float32)
    max_pool_arr = np.empty((len(case_ids), bb_cfg["dim"]), dtype=np.float32)
    for i, cid in enumerate(case_ids):
        patches = np.load(image_dir / f"{cid}.npy").astype(np.float32)
        mean_pool[i] = bgap(patches)
        max_pool_arr[i] = max_pool(patches)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(case_ids)}...")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, case_ids=np.array(case_ids), mean_pool=mean_pool, max_pool=max_pool_arr)
    print(f"escrito {out_path} ({time.time() - t0:.1f}s, {mean_pool.shape})")


if __name__ == "__main__":
    main()
