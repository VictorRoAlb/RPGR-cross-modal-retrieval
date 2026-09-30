#!/usr/bin/env python3
"""
qualitative_reranking.py
===========================
Reproduces Figure 2: a real T2I re-ranking example (query TCGA-77-7141,
LUSC), comparing the initial BGAP Top-5 against the RPGR-reordered Top-5.
Green boxes mark candidates matching the query's ground-truth subtype, red
boxes mark other subtypes; the small caption under each thumbnail shows
where that candidate ranked under the other method.

This case is hardcoded, not chosen dynamically -- it is the exact example
independently re-verified from raw embeddings against the paper's
Section 4.9 and Figure 2 (see METHOD_PROVENANCE.md). Re-running this script
reproduces the same figure; it does not search for a new example.

Fetches the 9 real tissue thumbnails from GDC on first run (cached to
--cache-dir afterwards) using utils/remote_svs_thumb.py -- no full .svs
download needed. Requires network access to api.gdc.cancer.gov.

Usage:
  python3 qualitative_reranking.py --output-dir ../outputs --cache-dir ../outputs/thumbnails
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent / "utils"))
from remote_svs_thumb import resolve_file_id, get_thumbnail_array  # noqa: E402

QUERY_TEXT = ("The tissue displays a largely exophytic growth pattern with "
              "moderate differentiation, filling bronchial lumens and "
              "invading down to and between bronchial cartilages. "
              "Perineural infiltration is observed, while no lymphatic or "
              "vascular invasion is identified.")
QUERY_LABEL = "LUSC"

# tag -> source .svs filename on GDC
SVS_FILES = {
    "query_77-7141": "TCGA-77-7141-01Z-00-DX1.3263490a-cfba-43a8-8198-5139dbf00bf9.svs",
    "bgap1_55-6981": "TCGA-55-6981-01Z-00-DX1.5fecd134-42a0-40a2-be75-1e8bd14c5b30.svs",
    "bgap2_49-4494": "TCGA-49-4494-01Z-00-DX1.2bf10e85-465a-4b17-abd7-3565f0e538a5.svs",
    "bgap3_66-2786": "TCGA-66-2786-01Z-00-DX1.5fd5c25f-cb17-4650-b59e-122c6ddafe86.svs",
    "bgap4_55-A48X": "TCGA-55-A48X-01Z-00-DX1.A46C6373-8458-4D55-88C3-4C70A05F9F47.svs",
    "bgap5_C5-A7CG": "TCGA-C5-A7CG-01Z-00-DX1.42D2FD26-D8ED-49DA-AEF0-FCAD9DB6B4BD.svs",
    "rpgr3_66-2789": "TCGA-66-2789-01Z-00-DX1.93719e52-7dd4-4bbb-b908-c91db1d99b76.svs",
    "rpgr4_60-2720": "TCGA-60-2720-01Z-00-DX1.5ff16714-210d-44c3-856a-5877fd06e986.svs",
    "rpgr5_56-8624": "TCGA-56-8624-01Z-00-DX1.18DFDAB8-1701-4E15-9321-7D99CD15E4B0.svs",
}

# (thumbnail tag, ground-truth-matching label, correct?, rank under the OTHER method)
BGAP_TOP5 = [
    ("bgap1_55-6981", "LUAD", False, 20),
    ("bgap2_49-4494", "LUAD", False, 2),
    ("bgap3_66-2786", "LUSC", True, 1),
    ("bgap4_55-A48X", "LUAD", False, 32),
    ("bgap5_C5-A7CG", "CESC", False, 7),
]
RPGR_TOP5 = [
    ("bgap3_66-2786", "LUSC", True, 3),
    ("bgap2_49-4494", "LUAD", False, 2),
    ("rpgr3_66-2789", "LUSC", True, 14),
    ("rpgr4_60-2720", "LUSC", True, 17),
    ("rpgr5_56-8624", "LUSC", True, 13),
]

GREEN = "#1baf7a"
RED = "#e34948"


def fetch_thumbnails(cache_dir: Path) -> None:
    from PIL import Image
    cache_dir.mkdir(parents=True, exist_ok=True)
    for tag, fname in SVS_FILES.items():
        out_path = cache_dir / f"{tag}.png"
        if out_path.exists():
            continue
        print(f"resolving {tag} ({fname}) ...")
        file_id = resolve_file_id(fname)
        if file_id is None:
            raise RuntimeError(f"no GDC file_id found for {fname}")
        arr, w, h = get_thumbnail_array(file_id)
        Image.fromarray(arr).save(out_path)
        print(f"  saved {out_path} ({w}x{h})")


def draw_row(fig, gs, thumb_dir: Path, row_idx: int, items: list, title: str, rank_template: str) -> None:
    for ci, (tag, label, correct, other_rank) in enumerate(items):
        ax = fig.add_subplot(gs[row_idx, ci + 2])
        ax.axis("off")
        img = mpimg.imread(str(thumb_dir / f"{tag}.png"))
        ax.imshow(img)
        color = GREEN if correct else RED
        ax.add_patch(plt.Rectangle((0, 0), img.shape[1] - 1, img.shape[0] - 1,
                                    fill=False, edgecolor=color, linewidth=4.5))
        ax.text(0.0, 1.06, f"#{ci+1}", transform=ax.transAxes, fontsize=8, fontweight="bold",
                color="#26251f", ha="left", va="bottom")
        ax.text(1.0, 1.06, label, transform=ax.transAxes, fontsize=7, fontweight="bold",
                color=color, ha="right", va="bottom")
        if other_rank is not None:
            ax.text(0.5, -0.08, rank_template.format(other_rank), transform=ax.transAxes, fontsize=6.3,
                    color="#68675f", ha="center", va="top", style="italic")
    ax_title = fig.add_subplot(gs[row_idx, 2:])
    ax_title.axis("off")
    title_y = 1.55 if row_idx == 1 else 1.18
    ax_title.text(0.0, title_y, title, transform=ax_title.transAxes, fontsize=8.2,
                   fontweight="bold", ha="left", va="bottom")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="Where thumbnails are cached (default: <output-dir>/thumbnails).")
    args = p.parse_args()
    cache_dir = args.cache_dir or (args.output_dir / "thumbnails")

    fetch_thumbnails(cache_dir)

    fig = plt.figure(figsize=(7.0, 2.9))
    gs = fig.add_gridspec(2, 7, width_ratios=[1.55, 0.12, 1, 1, 1, 1, 1],
                           height_ratios=[1, 1], wspace=0.10, hspace=1.35,
                           left=0.02, right=0.985, top=0.80, bottom=0.06)

    ax_q = fig.add_subplot(gs[:, 0])
    ax_q.axis("off")
    ax_q.add_patch(FancyBboxPatch((0.03, 0.27), 0.94, 0.60, boxstyle="round,pad=0.02,rounding_size=0.03",
                                   linewidth=1.1, edgecolor="#3a3a38", facecolor="#fbfbf9",
                                   transform=ax_q.transAxes))
    ax_q.text(0.5, 1.0, "Text query (T2I)", transform=ax_q.transAxes, ha="center", va="top",
              fontsize=8.5, fontweight="bold")
    wrapped = "\n".join(textwrap.wrap(QUERY_TEXT, width=30))
    ax_q.text(0.5, 0.82, wrapped, transform=ax_q.transAxes, ha="center", va="top",
              fontsize=6.3, color="#26251f", linespacing=1.3)
    ax_q.text(0.5, 0.18, "Ground-truth subtype", transform=ax_q.transAxes, ha="center", va="top",
              fontsize=6.8, color="#68675f")
    ax_q.add_patch(FancyBboxPatch((0.28, 0.02), 0.44, 0.11, boxstyle="round,pad=0.015,rounding_size=0.05",
                                   linewidth=0, facecolor="#e7f5ef", transform=ax_q.transAxes))
    ax_q.text(0.5, 0.075, QUERY_LABEL, transform=ax_q.transAxes, ha="center", va="center",
              fontsize=9, fontweight="bold", color="#0d7a54")

    draw_row(fig, gs, cache_dir, 0, BGAP_TOP5, "Initial Top-5 (BGAP)", rank_template="now #{}")
    draw_row(fig, gs, cache_dir, 1, RPGR_TOP5, "Re-ranked Top-5 (RPGR)", rank_template="was #{}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = args.output_dir / "qualitative_reranking.pdf"
    out_png = args.output_dir / "qualitative_reranking.png"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"wrote {out_pdf} and {out_png}")


if __name__ == "__main__":
    main()
