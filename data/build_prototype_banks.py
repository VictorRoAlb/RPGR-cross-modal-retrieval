"""
Adaptive prototype bank of every WSI of one backbone (rpgr.prototypes, default settings:
H grid {2,...,200}, tau_min=6, tau_sup=20, rho=0.97, tau_sm=0.05, mini-batch k-means with
5 initialisations, at most 200 iterations, batches of 1024 and random_state=42).

Writes ``<prototype_dir>/<slide_id>_prototype.npz`` (TCGA) or ``<case_id>.npz`` (AI4SKIN). Existing banks are skipped unless --overwrite.
The banks of the paper were built with scikit-learn 1.7.2 (TCGA) and 1.9.0 (AI4SKIN).

Run with: python -m data.build_prototype_banks --config <yaml> --cohort tcga --backbone KEEP [--n-jobs 16]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from data.config import config_argument_parser, load_config
from data.loaders import PatchStore
from rpgr.prototypes import AdaptiveSettings, build_adaptive_prototypes_for_case


def main() -> None:
    p = config_argument_parser(__doc__)
    p.add_argument("--cohort", choices=["tcga", "ai4skin"], required=True)
    p.add_argument("--backbone", required=True)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.config)
    bb = cfg[args.cohort]["backbones"][args.backbone]
    store, out_dir = PatchStore(bb["patch_dir"]), Path(bb["prototype_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    name = (lambda c: f"{c}_prototype.npz") if args.cohort == "tcga" else (lambda c: f"{c}.npz")
    ids = [c for c in store.ids() if args.overwrite or not (out_dir / name(c)).exists()]
    settings = AdaptiveSettings()

    def one(cid: str) -> None:
        r = build_adaptive_prototypes_for_case(cid, store.load(cid).astype(np.float32), settings)
        tmp = out_dir / f".{name(cid)}.tmp"
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, prototypes=r["prototypes"], K_star=np.asarray([r["K_star"]], dtype=np.int32),
                                coverage=np.asarray([r["coverage"]], dtype=np.float32), support=np.asarray([r["support"]], dtype=np.float32),
                                utility=np.asarray([r["utility"]], dtype=np.float32), q_proto=np.asarray([r["q_proto"]], dtype=np.float32),
                                n_patches=np.asarray([r["n_patches"]], dtype=np.int32))
        tmp.replace(out_dir / name(cid))

    print(f"{len(ids)} WSIs to process", flush=True)
    if args.n_jobs == 1:
        for c in ids:
            one(c)
    else:
        from joblib import Parallel, delayed
        Parallel(n_jobs=args.n_jobs, backend="loky", verbose=5)(delayed(one)(c) for c in ids)
    print("done", out_dir)


if __name__ == "__main__":
    main()
