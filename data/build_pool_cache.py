"""
Mean- and max-pooled WSI vectors of one backbone, computed once from the patch embeddings.

Writes an .npz with ``case_ids``, ``mean_pool`` and ``max_pool`` (float32) to
``tcga.backbones.<BACKBONE>.pool_cache``; every TCGA experiment reads it.

Run with: python -m data.build_pool_cache --config <yaml> --backbone KEEP
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from baselines.pooling import max_pool
from data.config import config_argument_parser, load_config
from data.loaders import PatchStore
from rpgr.similarity import mean_pool


def main() -> None:
    p = config_argument_parser(__doc__)
    p.add_argument("--backbone", required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    bb = cfg["tcga"]["backbones"][args.backbone]
    store = PatchStore(bb["patch_dir"])
    ids = store.ids()
    rows_mean, rows_max = [], []
    for i, c in enumerate(ids):
        x = store.load(c).astype(np.float32)
        rows_mean.append(mean_pool(x))
        rows_max.append(max_pool(x))
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{len(ids)}", flush=True)
    mp = np.stack(rows_mean)
    out = Path(bb["pool_cache"])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, case_ids=np.array(ids), mean_pool=mp, max_pool=np.stack(rows_max))
    print("written", out, mp.shape)


if __name__ == "__main__":
    main()
