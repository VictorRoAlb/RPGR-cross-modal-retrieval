"""
Fig. D.1: patches of an AI4SKIN test WSI that belong to the prototype that best matches its own
Morphology-only report (KEEP), drawn on the slide.

h* = argmax_h cos(p_h, report) is the prototype that enters S_P. The stored bank keeps no
per-patch assignment, so each patch is assigned to its nearest prototype (cosine). The case,
the pyramidal TIFF, the patch coordinates and the level of the TIFF they refer to (2, i.e. 10x,
in the public AI4SKIN slides) are given under ``ai4skin.figure_d1`` in the config; the patch
side is the coordinate stride.

Run with: python -m figures.figD1_prototypes --config <yaml>
Outputs: <output_dir>/figures/figD1_prototypes.{png,pdf}, figD1_prototype_evidence.csv
"""
from __future__ import annotations

import data.blas  # noqa: F401  (must precede numpy)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from scipy import ndimage

from data.config import config_argument_parser, load_config, output_dir
from data.loaders import AI4SKIN, PatchStore, load_text_vectors
from rpgr.similarity import l2_normalize_rows

BACKBONE, CONDITION = "KEEP", "morphology_only"
DRAW_DOWNSAMPLE_LEVELS = 2   # levels below the one of the coordinates (4x downsample)
COLOR, FILL_ALPHA, EDGE_ALPHA, EDGE_WIDTH = "#4285F4", 0.16, 0.95, 2.2
DARK_EDGE = 120              # mean intensity below which a pixel is slide edge, not tissue
MIN_COMPONENT_FRAC = 0.0005  # smallest tissue component kept (fraction of the image)
TISSUE_MIN_FRAC = 0.003      # minimum tissue fraction of a row / column inside the crop
CROP_MARGIN_FRAC = 0.015
DPI = 300


def prototype_evidence(ai: AI4SKIN, case_id: str) -> tuple[dict, np.ndarray]:
    """h*, its cosine and the runner-up, and the nearest prototype of every patch."""
    prototypes = l2_normalize_rows(ai.prototype_banks(BACKBONE, [case_id])[0])
    report = load_text_vectors(ai.backbone(BACKBONE)["text_dirs"][CONDITION], [case_id])[0]
    sims = prototypes @ report
    best, second = np.argsort(-sims, kind="stable")[:2]
    patches = l2_normalize_rows(ai.patch_store(BACKBONE).load(case_id).astype(np.float64))
    assignment = np.argmax(patches @ prototypes.T, axis=1)
    facts = dict(H_star=len(prototypes), h_star=int(best), cos_h_star=float(sims[best]), runner_up=int(second),
                 cos_runner_up=float(sims[second]), n_patches=len(patches), n_patches_h_star=int((assignment == best).sum()))
    return facts, assignment


def tissue_bbox(image: np.ndarray) -> tuple[int, int, int, int]:
    """Bounding box of the tissue: Otsu threshold on HSV saturation, without the dark slide edge,
    dust (5x5 opening, small components) and anything touching the image border."""
    saturation = np.asarray(Image.fromarray(image).convert("HSV"))[:, :, 1].astype(float)
    hist = np.histogram(saturation, bins=256, range=(0, 256))[0]
    w, m = np.cumsum(hist), np.cumsum(hist * np.arange(256))
    threshold = int(np.argmax((m[-1] * w / w[-1] - m) ** 2 / np.maximum(w * (w[-1] - w), 1)))
    tissue = (saturation > threshold) & (image.mean(axis=2) > DARK_EDGE)
    labels, n = ndimage.label(ndimage.binary_opening(tissue, structure=np.ones((5, 5))))
    sizes = ndimage.sum(labels > 0, labels, index=np.arange(1, n + 1))
    border = set(np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])))
    keep = [i for i, s in enumerate(sizes, start=1) if s >= MIN_COMPONENT_FRAC * labels.size and i not in border]
    tissue = np.isin(labels, keep)
    rows = np.where(tissue.mean(axis=1) > TISSUE_MIN_FRAC)[0]
    cols = np.where(tissue.mean(axis=0) > TISSUE_MIN_FRAC)[0]
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


def main() -> None:
    cfg = load_config(config_argument_parser(__doc__).parse_args().config)
    out = output_dir(cfg, "figures")
    ai = AI4SKIN(cfg)
    spec = ai.cfg["figure_d1"]
    facts, assignment = prototype_evidence(ai, spec["case_id"])
    pd.DataFrame([facts]).to_csv(out / "figD1_prototype_evidence.csv", index=False)
    print(facts)

    coords = PatchStore(spec["patch_coords_dir"]).load(spec["case_id"])
    assert len(coords) == len(assignment)
    with tifffile.TiffFile(spec["wsi"]) as tif:
        level = int(spec["coords_level"])
        height, width = tif.pages[level].shape[:2]
        image = tif.pages[level + DRAW_DOWNSAMPLE_LEVELS].asarray()
    scale_x, scale_y = width / image.shape[1], height / image.shape[0]
    steps = np.diff(np.unique(coords[:, 0]))
    side = int(np.median(steps[steps > 0]))
    patch_w, patch_h = side / scale_x, side / scale_y

    r0, r1, c0, c1 = tissue_bbox(image)
    margin = int(CROP_MARGIN_FRAC * max(r1 - r0, c1 - c0))
    r0, c0 = max(0, r0 - margin), max(0, c0 - margin)
    r1, c1 = min(image.shape[0] - 1, r1 + margin), min(image.shape[1] - 1, c1 + margin)
    crop = image[r0:r1 + 1, c0:c1 + 1]
    h, w = crop.shape[:2]
    selected = coords[assignment == facts["h_star"]]
    x, y = selected[:, 0] / scale_x - c0, selected[:, 1] / scale_y - r0
    assert ((x >= -1) & (x <= w) & (y >= -1) & (y <= h)).all(), "an h* patch falls outside the crop"

    fig, ax = plt.subplots(figsize=(w / DPI, h / DPI), dpi=DPI)
    ax.imshow(crop, extent=(0, w, h, 0), interpolation="nearest")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    for xi, yi in zip(x, y):
        ax.add_patch(Rectangle((xi, yi), patch_w, patch_h, facecolor=COLOR, alpha=FILL_ALPHA, edgecolor="none", zorder=2))
        ax.add_patch(Rectangle((xi, yi), patch_w, patch_h, facecolor="none", edgecolor=COLOR, alpha=EDGE_ALPHA,
                               linewidth=EDGE_WIDTH, zorder=3))
    legend = ax.legend(handles=[Patch(facecolor=COLOR, edgecolor=COLOR, alpha=0.85, label="Report-matched prototype")],
                       loc="lower right", frameon=True, framealpha=0.95, edgecolor="#333333", fontsize=13, handlelength=1.2,
                       handleheight=1.2, borderpad=0.7, labelspacing=0.5, bbox_to_anchor=(0.985, 0.02))
    legend.get_frame().set_linewidth(0.8)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"figD1_prototypes.{ext}", dpi=DPI, facecolor="white")
    plt.close(fig)
    print("written", out)


if __name__ == "__main__":
    main()
