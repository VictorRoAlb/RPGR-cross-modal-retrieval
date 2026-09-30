#!/usr/bin/env python3
"""
prototype_overlay.py
=======================
Reproduces Figure B.1: the three most report-compatible local prototypes
of a real WSI, projected back onto the slide as coloured patch overlays.

The query/target case pair (TCGA-E2-A56Z / TCGA-GM-A2DN) is hardcoded, not
chosen dynamically -- it is the exact case already verified in the paper.
The script re-derives the Top-3 prototype indices and patch assignments
from the actual prototype bank and patch embeddings (not from a cached
picture), reads the real patch-size footprint from the SVS's own metadata
instead of assuming one, and asserts that both match a previously recorded
count before drawing anything -- if either check fails, it stops rather
than silently drawing something wrong.

Requires a local copy of the query case's .svs file (this script does not
fetch it -- see figures/utils/remote_svs_thumb.py for a thumbnail-only
remote reader if you don't want the full file).

Usage:
  python3 prototype_overlay.py \\
      --svs-path /path/to/TCGA-E2-A56Z-...svs \\
      --patch-embeddings-dir /path/to/keep_image_embeddings \\
      --patch-coords-dir /path/to/patch_coords \\
      --prototypes-dir /path/to/prototypes/KEEP \\
      --text-embeddings-dir /path/to/keep_text_embeddings_strict \\
      --output-dir ../outputs
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
import numpy as np
import tifffile

QUERY_CASE = "TCGA-E2-A56Z-01Z-00-DX1.EC28E279-B869-4485-A19F-C732D4CCD374"
TARGET_CASE = "TCGA-GM-A2DN-01Z-00-DX1.593003B0-BA87-486B-B472-D6B85867D54D"
EXPECTED_TOP3_PATCH_COUNTS = [198, 210, 163]  # cross-checked against the candidate search, see METHOD_PROVENANCE.md

PYRAMID_LEVEL = 3           # 16x downsample -- sharp enough to show histology, small enough to render fast
COLORS = ["#4285F4", "#34A853", "#FB8C00"]
FILL_ALPHA = 0.16
EDGE_ALPHA = 0.95
TISSUE_BG_THRESHOLD = 235    # mean intensity above which a pixel counts as background
TISSUE_MIN_FRAC = 0.003      # minimum non-background pixel fraction for a row/col to count as tissue
CROP_MARGIN_FRAC = 0.015
DPI = 300


def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


def tissue_bbox(arr: np.ndarray) -> tuple[int, int, int, int]:
    gray = arr.mean(axis=2)
    nonwhite = gray < TISSUE_BG_THRESHOLD
    row_frac = nonwhite.mean(axis=1)
    col_frac = nonwhite.mean(axis=0)
    rows = np.where(row_frac > TISSUE_MIN_FRAC)[0]
    cols = np.where(col_frac > TISSUE_MIN_FRAC)[0]
    assert len(rows) > 0 and len(cols) > 0, "no tissue detected -- check TISSUE_BG_THRESHOLD"
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--svs-path", type=Path, required=True, help=f"Local path to {QUERY_CASE}.svs")
    p.add_argument("--patch-embeddings-dir", type=Path, required=True)
    p.add_argument("--patch-coords-dir", type=Path, required=True)
    p.add_argument("--prototypes-dir", type=Path, required=True)
    p.add_argument("--text-embeddings-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("loading prototypes + target report text...")
    d = np.load(args.prototypes_dir / f"{QUERY_CASE}_prototype.npz", allow_pickle=True)
    protos = l2n(d["prototypes"].astype(np.float64))
    text_vec = np.load(args.text_embeddings_dir / f"{TARGET_CASE}.npy").astype(np.float64).reshape(-1)
    text_vec = text_vec / max(np.linalg.norm(text_vec), 1e-12)
    sims = protos @ text_vec
    top3_idx = np.argsort(-sims)[:3]
    print(f"  Top-3 prototype indices: {top3_idx.tolist()}  sims={sims[top3_idx].round(3).tolist()}")

    print("loading raw patches + real coordinates, assigning nearest-centroid...")
    patches = l2n(np.load(args.patch_embeddings_dir / f"{QUERY_CASE}.npy").astype(np.float64))
    coords = np.load(args.patch_coords_dir / f"{QUERY_CASE}.npy")
    assign = np.argmax(patches @ protos.T, axis=1)
    counts = [int((assign == idx).sum()) for idx in top3_idx]
    print(f"  patches per Top-3 prototype: {counts}  (expected {EXPECTED_TOP3_PATCH_COUNTS})")
    assert counts == EXPECTED_TOP3_PATCH_COUNTS, "patch count mismatch vs the candidate search -- STOP"

    with tifffile.TiffFile(args.svs_path) as tf:
        desc = tf.pages[0].description
        level0_shape = tf.pages[0].shape
        arr = tf.pages[PYRAMID_LEVEL].asarray()
    appmag = float([kv.split("=")[1] for kv in desc.split("|") if kv.strip().startswith("AppMag")][0])
    mpp = float([kv.split("=")[1] for kv in desc.split("|") if kv.strip().startswith("MPP")][0])
    xs_sorted = np.sort(np.unique(coords[:, 0]))
    stride_l0 = int(np.median(np.diff(xs_sorted)[np.diff(xs_sorted) > 0]))
    patch_um = stride_l0 * mpp
    mpp_20x_equiv = mpp * (appmag / 20.0)
    px_20x_equiv = patch_um / mpp_20x_equiv
    print(f"  SVS: AppMag={appmag}x  MPP={mpp}  patch_coords stride={stride_l0}px (level 0)")
    print(f"  real patch footprint = {patch_um:.1f} um (~{px_20x_equiv:.0f}px at 20x-equivalent, "
          f"read from the file's own metadata, not assumed)")

    h0, w0 = level0_shape[0], level0_shape[1]
    h_lvl, w_lvl = arr.shape[0], arr.shape[1]
    scale_x = w0 / w_lvl
    scale_y = h0 / h_lvl
    patch_w_lvl = stride_l0 / scale_x
    patch_h_lvl = stride_l0 / scale_y

    print("cropping to the real tissue bounding box...")
    r0, r1, c0, c1 = tissue_bbox(arr)
    margin = int(CROP_MARGIN_FRAC * max(r1 - r0, c1 - c0))
    r0 = max(0, r0 - margin); r1 = min(arr.shape[0] - 1, r1 + margin)
    c0 = max(0, c0 - margin); c1 = min(arr.shape[1] - 1, c1 + margin)
    cropped = arr[r0:r1 + 1, c0:c1 + 1]

    px = coords[:, 0] / scale_x - c0
    py = coords[:, 1] / scale_y - r0

    mask_top3 = np.isin(assign, top3_idx)
    ok_bounds = ((px[mask_top3] >= -1) & (px[mask_top3] <= cropped.shape[1] + patch_w_lvl) &
                 (py[mask_top3] >= -1) & (py[mask_top3] <= cropped.shape[0] + patch_h_lvl)).all()
    assert ok_bounds, "a Top-3 patch falls outside the crop -- spatial misalignment, STOP"
    print("  verified: every Top-3 patch falls inside the crop, no spatial offset")

    def render(linewidth: float, tag: str) -> None:
        fig_w = cropped.shape[1] / DPI
        fig_h = cropped.shape[0] / DPI
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=DPI)
        ax.imshow(cropped, extent=(0, cropped.shape[1], cropped.shape[0], 0), interpolation="nearest")
        ax.set_xlim(0, cropped.shape[1])
        ax.set_ylim(cropped.shape[0], 0)
        ax.set_aspect("equal")
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        for k, proto_idx in enumerate(top3_idx):
            m = assign == proto_idx
            color = COLORS[k]
            for x, y in zip(px[m], py[m]):
                ax.add_patch(Rectangle((x, y), patch_w_lvl, patch_h_lvl,
                                        facecolor=color, alpha=FILL_ALPHA, edgecolor="none", zorder=2))
                ax.add_patch(Rectangle((x, y), patch_w_lvl, patch_h_lvl,
                                        facecolor="none", edgecolor=color, alpha=EDGE_ALPHA,
                                        linewidth=linewidth, zorder=3))

        legend_handles = [Patch(facecolor=COLORS[k], edgecolor=COLORS[k], alpha=0.85, label=f"Prototype {k + 1}")
                          for k in range(3)]
        leg = ax.legend(handles=legend_handles, loc="lower right", frameon=True, framealpha=0.95,
                         edgecolor="#333333", fontsize=13, handlelength=1.2, handleheight=1.2,
                         borderpad=0.7, labelspacing=0.5, bbox_to_anchor=(0.985, 0.02))
        leg.get_frame().set_linewidth(0.8)

        png_path = args.output_dir / f"prototype_overlay_{tag}.png"
        pdf_path = args.output_dir / f"prototype_overlay_{tag}.pdf"
        fig.savefig(png_path, dpi=DPI, facecolor="white")
        fig.savefig(pdf_path, facecolor="white")
        plt.close(fig)
        print(f"  saved {png_path.name} and {pdf_path.name}")

    render(linewidth=1.1, tag="thin_border")
    render(linewidth=2.2, tag="thick_border")
    print(f"\nwritten to: {args.output_dir}")


if __name__ == "__main__":
    main()
