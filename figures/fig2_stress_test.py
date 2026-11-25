"""
Fig. 2: RPGR versus mean pooling across the 72 stress-test conditions (class-macro mAP@10),
with an asterisk where RPGR differs significantly from mean pooling (paired bootstrap by
clinical case, B=20,000; Holm over the 72 conditions).

Reads <output_dir>/stress_test/{conditions,bootstrap}.csv (experiments.stress_test).
Run with: python -m figures.fig2_stress_test --config <yaml>
Outputs: <output_dir>/figures/fig2_stress_test.{pdf,png}. The published figure was rendered
with matplotlib 3.10.7 (pinned in requirements.txt).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import to_rgb
from matplotlib.legend_handler import HandlerBase

from data.config import config_argument_parser, load_config, output_dir

cfg = load_config(config_argument_parser(__doc__).parse_args().config)
SRC = output_dir(cfg, "stress_test")
OUT = output_dir(cfg, "figures")
_src = pd.read_csv(SRC / "conditions.csv")
sig_df = pd.read_csv(SRC / "bootstrap.csv")

_GRAN_TO_LEVEL = {("TCGA", "organ"): "TCGA organ", ("TCGA", "subtype"): "TCGA subtype",
                   ("AI4SKIN", "subtype"): "AI4SKIN subtype"}
_BB_TO_FIG = {"KEEP": "KEEP", "CONCH": "CONCH", "MUSK": "MUSK", "PATHO_CLIP": "PATHO-CLIP"}

SIG = {}
for _, r in sig_df.iterrows():
    SIG[(r["direction"], _BB_TO_FIG[r["backbone"]], r["text"], _GRAN_TO_LEVEL[(r["cohort"], r["granularity"])])] = bool(r["holm72"])

DATA = []
for _, r in _src.iterrows():
    DATA.append((r["direction"], _BB_TO_FIG[r["backbone"]], r["text"], _GRAN_TO_LEVEL[(r["cohort"], r["granularity"])],
                 float(r["mAP10_mean_pooling"]), float(r["mAP10_RPGR"])))
assert len(DATA) == 72 and len(SIG) == 72
print(f"significant conditions: {sum(SIG.values())}/72 (Holm over 72, RPGR vs mean pooling)")

DIRS = ['I2T', 'T2I']
BACKBONES = ['KEEP', 'CONCH', 'MUSK', 'PATHO-CLIP']
TEXTS = ['Original', 'Morphology-only', 'Diagnosis-only']
LEVELS = ['TCGA organ', 'TCGA subtype', 'AI4SKIN subtype']
TITLES = {'TCGA organ': 'TCGA · organ level', 'TCGA subtype': 'TCGA · subtype level', 'AI4SKIN subtype': 'AI4SKIN · subtype level'}
COLORS = dict(zip(BACKBONES, ['#3C5488', '#00A087', '#4DBBD5', '#E64B35']))

plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
    'font.size': 7.5, 'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'axes.titleweight': 'bold'})

def tint(c, a=0.62):
    r = np.array(to_rgb(c)); return tuple(r + (1 - r) * a)

def get(d, b, t, lev):
    return next(x for x in DATA if x[0] == d and x[1] == b and x[2] == t and x[3] == lev)

fig, axs = plt.subplots(2, 3, figsize=(7.16, 4.15), sharey=True)
w, step = 0.19, 0.215
for i, d in enumerate(DIRS):
    for j, lev in enumerate(LEVELS):
        ax = axs[i, j]; Bv, Rv = [], []
        for g, t in enumerate(TEXTS):
            for k, b in enumerate(BACKBONES):
                _, _, _, _, vb, vr = get(d, b, t, lev); x = g * 1.1 + (k - 1.5) * step
                ax.bar(x, vb, w, color=tint(COLORS[b]), lw=0, zorder=1)
                ax.bar(x, vr - vb, w, bottom=vb, color=COLORS[b], lw=0, zorder=2)
                if SIG[(d, b, t, lev)]:
                    ax.text(x, max(vr, vb) + 0.012, '*', ha='center', va='bottom', fontsize=7,
                            color='#222', zorder=4)
                Bv.append(vb); Rv.append(vr)
        gain = np.mean([(r - b) / b for b, r in zip(Bv, Rv)]) * 100
        ax.text(0.98, 0.97, f'Mean gain +{gain:.1f}%', transform=ax.transAxes, ha='right', va='top',
                fontsize=6.3, color='#333', bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#CCCCCC', lw=0.5))
        ax.set_xticks([0, 1.1, 2.2]); ax.set_xticklabels(['Original', 'Morphology\nonly', 'Diagnosis\nonly'], linespacing=1.0)
        ax.tick_params(axis='x', length=0, pad=3)
        ax.set_xlim(-0.5, 2.7); ax.set_ylim(0, 1.08); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
        ax.yaxis.grid(True, color='#EDEDED', lw=0.5); ax.set_axisbelow(True)
        ax.set_title(f'{TITLES[lev]} · {d}', fontsize=7.2, loc='left', pad=9)
        ax.text(-0.17 if j == 0 else -0.06, 1.08, 'abcdef'[i * 3 + j], transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='bottom')
        if j == 0: ax.set_ylabel('Class-macro mAP@10')

class TwoTone(HandlerBase):
    def create_artists(self, legend, orig, xd, yd, w, h, fs, trans):
        return [Rectangle((xd, yd), w, h * 0.6, color=tint('#555555'), transform=trans),
                Rectangle((xd, yd + h * 0.6), w, h * 0.4, color='#555555', transform=trans)]
class Key: pass
backbone_handles = [Patch(color=COLORS[b], label=b) for b in BACKBONES]
key_handle = Key()
fig.legend(handles=backbone_handles + [key_handle],
           labels=BACKBONES + ['light: mean pooling   dark: improvement with RPGR'],
           handler_map={Key: TwoTone()}, loc='upper center', ncol=5, frameon=False,
           bbox_to_anchor=(0.5, 1.0), handlelength=1.0, columnspacing=1.4)
fig.tight_layout(rect=(0, 0, 1, 0.945), h_pad=1.4, w_pad=0.5)
fig.savefig(OUT / "fig2_stress_test.pdf")
fig.savefig(OUT / "fig2_stress_test.png", dpi=300)
print("written", OUT / "fig2_stress_test.pdf")
