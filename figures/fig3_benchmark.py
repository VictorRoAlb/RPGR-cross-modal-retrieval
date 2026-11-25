"""
Fig. 3: cross-method benchmark on the held-out test sets (KEEP, Morphology-only, subtype level).
(a) Class-macro mAP@10 under the tie-neutral protocol; boxes are the 2.5-97.5 % interval of a
case-stratified bootstrap of each method's class-macro value (B=2,000, seed 42); * / dagger:
RPGR significantly better / worse (paired bootstrap by case, B=20,000; Holm over the 14
comparators of each cohort and direction). (b) Recall@K.

Reads <output_dir>/benchmark/{per_query.csv.gz,comparison.csv} (experiments.benchmark).
Run with: python -m figures.fig3_benchmark --config <yaml>
Outputs: <output_dir>/figures/fig3_benchmark.{pdf,png}. The published figure was rendered
with matplotlib 3.10.7 (pinned in requirements.txt).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from data.config import config_argument_parser, load_config, output_dir
from data.loaders import TCGA

cfg = load_config(config_argument_parser(__doc__).parse_args().config)
BENCH = output_dir(cfg, "benchmark")
OUT = output_dir(cfg, "figures")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.linewidth": 0.5, "pdf.fonttype": 42,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 0, "ytick.major.size": 2})
INK = "#1d1d1b"; RULE = "#d9d7d2"; RED = "#d6453d"; BLUE = "#2a78d6"
FAMLAB = {"Global aggregation": "Global\npooling", "Hubness correction": "Hubness\ncorrection", "Patch-level": "Patch",
          "Trainable MIL": "Trainable\nMIL", "Trained cross-modal": "Cross-\nmodal",
          "Slide-level VL-FM": "Slide-level\nFM", "RPGR (ours)": "Ours"}
METHODS = [
    ("Global aggregation", "Mean pooling", "mean_pooling"), ("Global aggregation", "Max pooling", "max_pooling"),
    ("Hubness correction", "CSLS", "CSLS"), ("Hubness correction", "IS", "IS"), ("Hubness correction", "QB-Norm", "QBNorm"),
    ("Hubness correction", "DBNorm", "DBNorm"), ("Hubness correction", "RR", "RR"),
    ("Patch-level", "Top-k recip.", "topk_reciprocal"), ("Trainable MIL", "TransMIL", "TransMIL"), ("Trainable MIL", "SETMIL", "SETMIL"),
    ("Trained cross-modal", "DTN", "DTN"), ("Trained cross-modal", "FGCR", "FGCR"),
    ("Slide-level VL-FM", "TITAN", "TITAN"), ("Slide-level VL-FM", "PRISM", "PRISM"), ("RPGR (ours)", "RPGR (ours)", "RPGR")]
MC = {"Mean pooling": "#8f8d87", "Max pooling": "#c4c2bc", "CSLS": "#f6c3a8", "IS": "#f0a07a", "QB-Norm": "#e07f50",
      "DBNorm": "#c9663a", "RR": "#a85230", "Top-k recip.": "#62ad62", "TransMIL": "#e7837f", "SETMIL": "#b9504c",
      "DTN": "#e9c25e", "FGCR": "#b8902a", "TITAN": "#e4a3c4", "PRISM": "#b9739a", "RPGR (ours)": BLUE}
SET = [("TCGA", "I2T"), ("TCGA", "T2I"), ("AI4SKIN", "I2T"), ("AI4SKIN", "T2I")]

def boot_macro(values, labels, units, B=2000, seed=42):  # same procedure as fig3_final_panel_a_bootstrap.py
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"v": values, "l": labels, "u": units})
    lab_units = {l: g.u.unique() for l, g in df.groupby("l")}
    rows_of = {u: g.index.to_numpy() for u, g in df.groupby("u")}
    out = np.empty(B)
    for b in range(B):
        cls = []
        for l, us in lab_units.items():
            ch = rng.choice(us, size=len(us), replace=True)
            cls.append(np.concatenate([values[rows_of[u]] for u in ch]).mean())
        out[b] = np.mean(cls)
    return out


case_of = TCGA(cfg).labels()["case_id"].to_dict()
nq = pd.read_csv(BENCH / "per_query.csv.gz")
cmp_ = pd.read_csv(BENCH / "comparison.csv")
NREPL = {}
for (coh, d, m), g in nq.groupby(["cohort", "direction", "method"]):
    g = g.reset_index(drop=True)
    units = g.query_id.map(case_of).values if coh == "TCGA" else g.query_id.values
    NREPL[(coh, d, m)] = boot_macro(g["mAP@10_neutral"].values, g.label.values, units)


def point_sig(coh, d, key):
    if key == "RPGR":
        g = nq[(nq.cohort == coh) & (nq.direction == d) & (nq.method == "RPGR")]
        return float(g.groupby("label")["mAP@10_neutral"].mean().mean()), 0
    r = cmp_[(cmp_.cohort == coh) & (cmp_.direction == d) & (cmp_.method == key)].iloc[0]
    return float(r.mAP10_neutral), (0 if not r.holm14 else (1 if r.diff_RPGR_minus_method > 0 else -1))


def replicates(coh, d, key):
    return NREPL[(coh, d, key)]


RKCOLS = ["Recall@1", "Recall@3", "Recall@5", "Recall@10"]
RK = {}
for fam, disp, key in METHODS:
    RK[disp] = [nq[(nq.cohort == c) & (nq.direction == d) & (nq.method == key)].groupby("label")[[f"{k}_neutral" for k in RKCOLS]].mean().mean().to_numpy()
                for c, d in SET]

names = [d for _, d, _ in METHODS]; fams = [f for f, _, _ in METHODS]
xs, x = [], 0
for i, f in enumerate(fams):
    if i > 0 and f != fams[i - 1]:
        x += 0.75
    xs.append(x); x += 1
xs = np.array(xs)
groups = []
for i, f in enumerate(fams):
    if not groups or groups[-1][0] != f:
        groups.append([f, xs[i], xs[i]])
    else:
        groups[-1][2] = xs[i]


def draw_panel_a(fig, rects):
    for k, (coh, d) in enumerate(SET):
        ax = fig.add_axes(rects[k]); row = k // 2
        data, sigs, pts = [], [], []
        for fam, disp, key in METHODS:
            pt, sg = point_sig(coh, d, key)
            pts.append(pt); sigs.append(sg); data.append(replicates(coh, d, key))
        for f, a, b in groups:
            if f == "RPGR (ours)":
                ax.axvspan(a - 0.55, b + 0.55, color="#e9f1fb", lw=0, zorder=0)
        for (f, a, b), (f2, a2, b2) in zip(groups[:-1], groups[1:]):
            ax.axvline((b + a2) / 2, color=RULE, lw=0.6, zorder=0.5)
        bp = ax.boxplot(data, positions=xs, widths=0.64, whis=(2.5, 97.5), showfliers=False, patch_artist=True,
                        medianprops=dict(color="white", lw=1.0), whiskerprops=dict(color=INK, lw=0.55), capprops=dict(color=INK, lw=0.55))
        for p, n in zip(bp["boxes"], names):
            p.set_facecolor(MC[n]); p.set_edgecolor("none")
        ax.axhline(pts[0], color=INK, lw=0.6, ls=(0, (2.5, 2)), zorder=1)
        lo = min(np.percentile(dd, 2.5) for dd in data); hi = max(np.percentile(dd, 97.5) for dd in data); span = hi - lo
        ax.set_ylim(max(0, lo - span * 0.05), hi + span * 0.20)
        for xv, dd, pt, sg, n in zip(xs, data, pts, sigs, names):
            ytop = np.percentile(dd, 97.5) + span * 0.02
            if n == "RPGR (ours)":
                ax.text(xv, ytop, f"{pt:.3f}", ha="center", va="bottom", fontsize=6.8, color=BLUE, fontweight="bold")
            elif sg == 1:
                ax.text(xv, ytop - span * 0.01, "*", ha="center", va="bottom", fontsize=10, color=INK, fontweight="bold")
            elif sg == -1:
                ax.text(xv, ytop, "†", ha="center", va="bottom", fontsize=8.5, color=RED, fontweight="bold")
        ax.set_xlim(xs[0] - 0.8, xs[-1] + 1.1)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        for sp in ["left", "bottom"]:
            ax.spines[sp].set_color(RULE)
        ax.tick_params(axis="y", colors=RULE, labelcolor=INK, labelsize=6.8)
        ax.grid(axis="y", color="#efeeea", lw=0.5); ax.set_axisbelow(True)
        ax.text(0.0, 1.22 if row == 0 else 1.03, f"{coh} · {d}", transform=ax.transAxes, fontsize=8.2,
                fontweight="semibold", color=INK, va="bottom")
        if k % 2 == 0:
            ax.set_ylabel("mAP@10", fontsize=7.4, color=INK, labelpad=2)
        ax.set_xticks(xs)
        if row == 0:
            ax.set_xticklabels([])
            for f, a, b in groups:
                c = BLUE if f == "RPGR (ours)" else INK
                yb = 1.04
                ax.plot([a - 0.4, a - 0.4, b + 0.4, b + 0.4], [yb - 0.025, yb, yb, yb - 0.025], color=c, lw=0.7,
                        transform=ax.get_xaxis_transform(), clip_on=False)
                ax.text((a + b) / 2, yb + 0.02, FAMLAB[f], ha="center", va="bottom", fontsize=5.8, linespacing=0.9,
                        color=c, fontweight="semibold", transform=ax.get_xaxis_transform(), clip_on=False)
        else:
            ax.set_xticklabels(names, rotation=45, ha="right", rotation_mode="anchor", fontsize=7.0)
            for t, n in zip(ax.get_xticklabels(), names):
                t.set_color(BLUE if n == "RPGR (ours)" else INK); t.set_fontweight("bold" if n == "RPGR (ours)" else "regular")


def draw_panel_b(fig, rects):
    for k, (coh, d) in enumerate(SET):
        ax = fig.add_axes(rects[k])
        for disp, vals in RK.items():
            if vals[k] is None:
                continue
            ours = disp == "RPGR (ours)"
            ax.plot(range(4), vals[k], marker="o", color=MC[disp], lw=2.2 if ours else 1.0, ms=4.6 if ours else 2.8,
                    mec="white", mew=0.5, zorder=6 if ours else 3, ls="--" if disp in ("Mean pooling", "Max pooling") else "-")
        ax.set_xticks(range(4)); ax.set_xticklabels(["1", "3", "5", "10"], fontsize=6.8)
        ax.set_xlim(-0.2, 3.2); ax.set_ylim(0, 1.0 if k >= 2 else 0.8)
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.2))
        ax.tick_params(axis="both", colors=RULE, labelcolor=INK, labelsize=6.6, length=2)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        for sp in ["left", "bottom"]:
            ax.spines[sp].set_color(RULE)
        ax.grid(color="#efeeea", lw=0.5); ax.set_axisbelow(True)
        ax.set_title(f"{coh} · {d}", fontsize=7.6, color=INK, pad=3, fontweight="semibold", loc="left")
        ax.set_xlabel("K", fontsize=7, color=INK, labelpad=1)
        if k == 0:
            ax.set_ylabel("Recall@K", fontsize=7.4, color=INK, labelpad=2)


def panel_title(fig, x, y, letter, title):
    fig.text(x, y, letter, fontsize=11, fontweight="bold", ha="right", va="bottom")
    fig.text(x + 0.012, y + 0.002, title, fontsize=8.6, fontweight="semibold", va="bottom")


def draw_legend(fig, x0, y0, dy=0.022):
    LEG = [("Global aggregation", ["Mean pooling", "Max pooling"]), ("Hubness correction", ["CSLS", "IS", "QB-Norm", "DBNorm", "RR"]),
           ("Patch-level", ["Top-k recip."]), ("Trainable MIL", ["TransMIL", "SETMIL"]),
           ("Trained cross-modal", ["DTN", "FGCR"]), ("Slide-level VL-FM", ["TITAN", "PRISM"]), ("Ours", ["RPGR (ours)"])]
    R = fig.canvas.get_renderer(); tw = lambda t: t.get_window_extent(renderer=R).width / fig.bbox.width  # noqa: E731
    lx, ly = x0, y0
    for fam_, ml in LEG:
        if fam_ in ("Patch-level", "Slide-level VL-FM"):
            lx = x0; ly -= dy
        t = fig.text(lx, ly, fam_ + ":", fontsize=6.4, va="center", color=BLUE if fam_ == "Ours" else INK, fontweight="semibold")
        lx += tw(t) + 0.006
        for n in ml:
            ours = n == "RPGR (ours)"
            fig.patches.append(Rectangle((lx, ly - 0.005), 0.012, 0.009, transform=fig.transFigure, facecolor=MC[n], edgecolor="none"))
            t = fig.text(lx + 0.016, ly, "RPGR" if ours else n, fontsize=6.5, va="center", color=BLUE if ours else INK,
                         fontweight="bold" if ours else "regular")
            lx += 0.017 + tw(t) + 0.01
        lx += 0.014


fig = plt.figure(figsize=(7.16, 7.9))
panel_title(fig, 0.06, 0.965, "a", "Overall comparison across methods")
draw_legend(fig, 0.07, 0.948, dy=0.019)
L, R = 0.075, 0.985; colgap = 0.075; W = (R - L - colgap) / 2; H = 0.195; tops = [0.815, 0.575]
draw_panel_a(fig, [[L + (k % 2) * (W + colgap), tops[k // 2] - H, W, H] for k in range(4)])
panel_title(fig, 0.06, 0.255, "b", "Recall@K")
bx0 = 0.08; bgap = 0.05; bw = (0.985 - bx0 - 3 * bgap) / 4
draw_panel_b(fig, [[bx0 + k * (bw + bgap), 0.055, bw, 0.165] for k in range(4)])
fig.savefig(OUT / "fig3_benchmark.pdf")
fig.savefig(OUT / "fig3_benchmark.png", dpi=300)
print("written", OUT / "fig3_benchmark.pdf")
