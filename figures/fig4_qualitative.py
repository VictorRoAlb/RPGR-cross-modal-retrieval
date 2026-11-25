"""
Fig. 4: qualitative T2I example on the TCGA test set with KEEP. Query: Morphology-only report of
TCGA-77-7141 (LUSC); top-5 WSIs of mean pooling (with their rank after RPGR) and of RPGR (with
their rank under mean pooling); green = same subtype as the query.

Reads <output_dir>/qualitative_example/rankings.csv (experiments.qualitative_example), the
query text from tcga.reports_csv and thumbnails <short id>.png from tcga.thumbnail_dir
(figures.fetch_thumbnails).
Run with: python -m figures.fig4_qualitative --config <yaml>
"""
import textwrap
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch

from data.config import config_argument_parser, load_config, output_dir

cfg = load_config(config_argument_parser(__doc__).parse_args().config)
OUT = output_dir(cfg, "figures")
THUMBS = Path(cfg["tcga"]["thumbnail_dir"])
rk = pd.read_csv(output_dir(cfg, "qualitative_example") / "rankings.csv")
rep = pd.read_csv(cfg["tcga"]["reports_csv"], sep=";")
q = rep[rep.slide_id == rk["query"].iloc[0]].iloc[0]
QUERY_TEXT, QUERY_LABEL = q.slide_reports_no_diagnosis_strict, q.OncotreeCode
assert (rk.query_label == QUERY_LABEL).all()


def items(lst, other):
    g = rk[(rk.list == lst) & (rk["rank"] <= 5)].sort_values("rank")
    return [(THUMBS / f"{r.slide[5:12]}.png", r.label, bool(r.correct), int(r[other])) for _, r in g.iterrows()]


MP_TOP5 = items("mean_pooling", "rank_RPGR")
RPGR_TOP5 = items("RPGR", "rank_mean_pooling")
GREEN, RED = "#1baf7a", "#e34948"
plt.rcParams.update({"pdf.fonttype": 42})
fig = plt.figure(figsize=(7.0, 2.9))
gs = fig.add_gridspec(2, 7, width_ratios=[1.55, 0.12, 1, 1, 1, 1, 1], height_ratios=[1, 1], wspace=0.10, hspace=1.35,
                      left=0.02, right=0.985, top=0.74, bottom=0.06)
ax_q = fig.add_subplot(gs[:, 0]); ax_q.axis("off")
ax_q.add_patch(FancyBboxPatch((0.03, 0.27), 0.94, 0.60, boxstyle="round,pad=0.02,rounding_size=0.03", linewidth=1.1,
                              edgecolor="#3a3a38", facecolor="#fbfbf9", transform=ax_q.transAxes))
ax_q.text(0.5, 1.0, "Text query (T2I)", transform=ax_q.transAxes, ha="center", va="top", fontsize=8.5, fontweight="bold")
ax_q.text(0.5, 0.82, "\n".join(textwrap.wrap(QUERY_TEXT, width=30)), transform=ax_q.transAxes, ha="center", va="top",
          fontsize=6.3, color="#26251f", linespacing=1.3)
ax_q.text(0.5, 0.18, "Ground-truth subtype", transform=ax_q.transAxes, ha="center", va="top", fontsize=6.8, color="#68675f")
ax_q.add_patch(FancyBboxPatch((0.28, 0.02), 0.44, 0.11, boxstyle="round,pad=0.015,rounding_size=0.05", linewidth=0,
                              facecolor="#e7f5ef", transform=ax_q.transAxes))
ax_q.text(0.5, 0.075, QUERY_LABEL, transform=ax_q.transAxes, ha="center", va="center", fontsize=9, fontweight="bold", color="#0d7a54")


def draw_row(row_idx, its, title, tmpl):
    for ci, (path, label, correct, other) in enumerate(its):
        ax = fig.add_subplot(gs[row_idx, ci + 2]); ax.axis("off")
        img = mpimg.imread(path); ax.imshow(img); ax.set_adjustable("datalim")  # keep full cell width for labels, no distortion
        color = GREEN if correct else RED
        ax.add_patch(plt.Rectangle((0, 0), img.shape[1] - 1, img.shape[0] - 1, fill=False, edgecolor=color, linewidth=4.5))
        ax.text(0.0, 1.06, f"#{ci + 1}", transform=ax.transAxes, fontsize=8, fontweight="bold", color="#26251f", ha="left", va="bottom")
        ax.text(1.0, 1.06, label, transform=ax.transAxes, fontsize=7, fontweight="bold", color=color, ha="right", va="bottom")
        ax.text(0.5, -0.08, tmpl.format(other), transform=ax.transAxes, fontsize=6.3, color="#68675f", ha="center", va="top", style="italic")
    ax_t = fig.add_subplot(gs[row_idx, 2:]); ax_t.axis("off")
    ax_t.text(0.0, 1.55 if row_idx == 1 else 1.50, title, transform=ax_t.transAxes, fontsize=8.2, fontweight="bold", ha="left", va="bottom")


draw_row(0, MP_TOP5, "Initial top-5 (mean pooling)", "now #{}")
draw_row(1, RPGR_TOP5, "Re-ranked top-5 (RPGR)", "was #{}")
fig.savefig(OUT / "fig4_qualitative.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig4_qualitative.png", dpi=300, bbox_inches="tight")
print("written", OUT / "fig4_qualitative.pdf")
