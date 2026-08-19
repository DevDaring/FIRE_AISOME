"""Paper figures. Palette validated with the dataviz skill's six checks:
Okabe-Ito blue/vermillion/green, all-pairs CVD dE 11.0 worst (floor 8),
normal-vision 18.7 (floor 15), contrast >=3:1 on white. Print- and CVD-safe."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HI, BN, THIRD = "#0072B2", "#D55E00", "#009E73"
INK, INK2, GRID = "#1a1a1a", "#52514e", "#d9d9d9"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.labelcolor": INK,
    "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

# ---------------------------------------------------------------- figure: results
# Form: magnitude comparison across many long-named systems -> horizontal bars,
# grouped by language. One axis. Two categorical series in fixed order.
systems = [
    ("DeBERTa-v3-L, EN pivot + calib.", 0.922, 0.915),
    ("DeBERTa-v3-L, EN-only pool",      0.886, 0.877),
    ("XLM-R-large",                     0.836, 0.812),
    ("MuRIL-large",                     0.807, 0.790),
    ("IndicBERTv2",                     0.804, 0.791),
    ("Argument-KG projection",          0.763, 0.722),
    ("MuRIL-base, distilled",           0.758, 0.781),
    ("Claim-conditioned NLI",           0.438, 0.404),
]
names = [s[0] for s in systems]
hi = np.array([s[1] for s in systems]); bn = np.array([s[2] for s in systems])
y = np.arange(len(names))[::-1]
h = 0.36

fig, ax = plt.subplots(figsize=(5.4, 3.15))
ax.barh(y + h/2 + 0.02, hi, h, color=HI, label="Hindi", zorder=3)
ax.barh(y - h/2 - 0.02, bn, h, color=BN, label="Bengali", zorder=3)
for yy, v in zip(y + h/2 + 0.02, hi):
    ax.text(v + 0.008, yy, f"{v:.3f}", va="center", ha="left", fontsize=7, color=INK2)
for yy, v in zip(y - h/2 - 0.02, bn):
    ax.text(v + 0.008, yy, f"{v:.3f}", va="center", ha="left", fontsize=7, color=INK2)
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
ax.set_xlabel("macro-F1 on the held-out development set")
ax.set_xlim(0, 1.06)
ax.xaxis.grid(True, color=GRID, linewidth=0.5, zorder=0)
ax.set_axisbelow(True)
for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.legend(frameon=False, loc="lower right", fontsize=8, handlelength=1.1)
fig.savefig("fig_results.pdf"); plt.close(fig)
print("  wrote fig_results.pdf")

# ------------------------------------------------- figure: class distribution shift
# Form: composition of a whole across three corpora -> 100% stacked horizontal bars.
corpora = [
    ("Permitted English pool\n(GWSD + SemEval-CC)", 367, 983, 1092),
    ("Taxonomy-conditioned\nsynthetic corpus",      500, 276,  264),
    ("Evaluation set\n(judge-panel labels)",        426, 338,  207),
]
labels = ["Against", "Favour", "None"]
cols = [BN, HI, THIRD]
fig, ax = plt.subplots(figsize=(5.4, 1.85))
ypos = np.arange(len(corpora))[::-1]
for i, (name, a, f, n) in enumerate(corpora):
    tot = a + f + n
    left = 0.0
    for val, c in zip((a, f, n), cols):
        frac = val / tot
        ax.barh(ypos[i], frac, 0.55, left=left, color=c, zorder=3,
                edgecolor="white", linewidth=1.2)   # 2px-equivalent surface gap
        if frac > 0.07:
            ax.text(left + frac/2, ypos[i], f"{frac*100:.0f}%", ha="center",
                    va="center", fontsize=7.5, color="white", fontweight="bold")
        left += frac
ax.set_yticks(ypos); ax.set_yticklabels([c[0] for c in corpora], fontsize=8)
ax.set_xlim(0, 1); ax.set_xticks([0, .25, .5, .75, 1])
ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
ax.set_xlabel("share of corpus")
for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
ax.tick_params(axis="y", length=0)
handles = [plt.Rectangle((0,0),1,1,color=c) for c in cols]
ax.legend(handles, labels, frameon=False, ncol=3, fontsize=8,
          loc="lower center", bbox_to_anchor=(0.5, -0.72), handlelength=1.1)
fig.savefig("fig_class_shift.pdf"); plt.close(fig)
print("  wrote fig_class_shift.pdf")
