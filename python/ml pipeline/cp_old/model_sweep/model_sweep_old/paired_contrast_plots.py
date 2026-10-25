"""
=======================================================================
PAIRED REPRESENTATION CONTRASTS UNDER TWO SPLIT REGIMES  —  figure
=======================================================================
Reads paired_contrast_long.csv (from paired_contrast_inflation.py).

Two rows of panels, one column per task:
  TOP    the paired gap itself (mean of the 50 within-split differences)
  BOTTOM the paired effect size dz for the same cells

In each panel, one row-block per model, three sub-rows per block (one per
contrast H1/H2/H3). A filled marker is the group-based arm, an open
marker the protein-level random arm, joined by a line.

The point of the figure is the comparison between the two rows: the top
row should show short lines (the gap is stable across split regimes)
while the bottom row shows long ones (leakage inflates confidence in
that gap, by shrinking the within-arm variance).

Style: font >= 12 throughout; contrast encoded by colour, arm by marker
fill. Split counts are reported as n+/n- in the analysis table, not here.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ======================================================================
# CONFIG  — edit here, then run the whole file in the interactive window.
# ======================================================================
CONFIG = {
    "LONG_CSV": Path("paired_contrast_long.csv"),
    "OUT":      Path("paired_contrast.png"),   # PDF written alongside
    "SHOW_CI":  True,    # faint band for the bootstrap CI on each change
}

plt.rcParams.update({
    "font.size":        13,
    "axes.titlesize":   15,
    "axes.labelsize":   13,
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
    "legend.fontsize":  12,
    "figure.titlesize": 17,
})

CONTRAST_ORDER = ["H1", "H2", "H3"]
CONTRAST_LABEL = {"H1": "Hypergraph - pairwise",
                  "H2": "HB-graph - hypergraph",
                  "H3": "HB-graph - pairwise"}
# Colour-blind-safe (Okabe-Ito subset), one per contrast.
CONTRAST_COLOR = {"H1": "#0072B2", "H2": "#009E73", "H3": "#CC79A7"}

MODEL_ORDER = [
    "LogisticRegression", "RBF-SVM", "MLP",
    "RandomForest", "XGBoost", "LightGBM", "TabPFN",
]
TASK_ORDER = ["ess", "hpa", "chembl"]
TASK_LABEL = {"ess": "Gene essentiality",
              "hpa": "Drug target (HPA)",
              "chembl": "Drug target (ChEMBL)"}


def _ordered(values, order):
    known = [v for v in order if v in set(values)]
    extra = sorted(set(values) - set(order))
    return known + extra


def plot(long_df: pd.DataFrame, out_path: Path, show_ci: bool):
    tasks  = _ordered(long_df["task"].unique(),  TASK_ORDER)
    models = _ordered(long_df["model"].unique(), MODEL_ORDER)

    n_con  = len(CONTRAST_ORDER)
    block  = n_con + 1
    n_rows = len(models) * block - 1

    def ypos(mi, ci):
        return n_rows - 1 - (mi * block + ci)

    # Row 0: the gap. Row 1: the effect size.
    panels = [
        ("mean_delta_group", "mean_delta_random", "ci_delta_lo", "ci_delta_hi",
         "Mean paired difference in PR-AUC"),
        ("dz_group", "dz_random", "ci_dz_lo", "ci_dz_hi",
         "Paired effect size $d_z$"),
    ]

    fig, axes = plt.subplots(
        2, len(tasks),
        figsize=(5.2 * len(tasks), max(9.0, 0.62 * n_rows + 2.0)),
        sharey=True, squeeze=False,
    )

    for row, (g_col, r_col, lo_col, hi_col, xlabel) in enumerate(panels):
        for col, task in enumerate(tasks):
            ax  = axes[row][col]
            sub = long_df[long_df["task"] == task]
            for mi, model in enumerate(models):
                for ci_, key in enumerate(CONTRAST_ORDER):
                    cell = sub[(sub["model"] == model) & (sub["contrast"] == key)]
                    if cell.empty:
                        continue
                    r = cell.iloc[0]
                    if not (np.isfinite(r[g_col]) and np.isfinite(r[r_col])):
                        continue
                    y = ypos(mi, ci_)
                    c = CONTRAST_COLOR[key]

                    ax.plot([r[g_col], r[r_col]], [y, y], color=c, lw=2.0,
                            alpha=0.55, zorder=1, solid_capstyle="round")
                    ax.scatter(r[g_col], y, s=46, color=c, zorder=3,
                               edgecolor="white", linewidth=0.6)
                    ax.scatter(r[r_col], y, s=46, facecolor="white",
                               edgecolor=c, linewidth=1.8, zorder=3)

                    if show_ci and np.isfinite(r.get(lo_col, np.nan)):
                        # CI is on the CHANGE; anchor it at the group value.
                        ax.plot([r[g_col] + r[lo_col], r[g_col] + r[hi_col]],
                                [y, y], color=c, lw=6, alpha=0.15, zorder=0)

            ax.axvline(0, color="0.5", lw=1.0, zorder=0)
            ax.grid(axis="x", linestyle=":", alpha=0.4)
            ax.set_axisbelow(True)
            ax.set_xlabel(xlabel)
            if row == 0:
                ax.set_title(TASK_LABEL.get(task, task))

            for mi in range(1, len(models)):
                ax.axhline(ypos(mi, 0) + 1.0, color="0.85", lw=0.8, zorder=0)

    # Model names on the middle contrast row of each block, both panel rows.
    yticks  = [ypos(mi, 1) for mi in range(len(models))]
    ylabels = list(models)
    for row in range(2):
        axes[row][0].set_yticks(yticks)
        axes[row][0].set_yticklabels(ylabels, fontweight="bold")
        axes[row][0].set_ylim(-0.8, n_rows - 0.2)

    arm_handles = [
        Line2D([0], [0], marker="o", color="0.3", markerfacecolor="0.3",
               markersize=8, lw=0, label="Group-based split (headline)"),
        Line2D([0], [0], marker="o", color="0.3", markerfacecolor="white",
               markeredgecolor="0.3", markeredgewidth=1.8, markersize=8, lw=0,
               label="Protein-level random split"),
    ]
    con_handles = [
        Line2D([0], [0], color=CONTRAST_COLOR[k], lw=3, label=CONTRAST_LABEL[k])
        for k in CONTRAST_ORDER
    ]
    leg1 = fig.legend(handles=arm_handles, loc="upper center",
                      bbox_to_anchor=(0.5, 0.055), ncol=2, frameon=False)
    fig.add_artist(leg1)
    fig.legend(handles=con_handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.028), ncol=3, frameon=False)

    fig.suptitle("The representation gap is stable across split regimes; "
                 "the paired effect size is not\n"
                 "(top: mean paired difference; bottom: $d_z$; "
                 "open marker = random split)", y=0.998)
    fig.subplots_adjust(left=0.13, bottom=0.11, top=0.92, right=0.98,
                        wspace=0.08, hspace=0.16)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Wrote {out_path}")
    pdf = out_path.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"Wrote {pdf}")


def main():
    C = CONFIG
    if not C["LONG_CSV"].exists():
        raise FileNotFoundError(
            f"{C['LONG_CSV']} not found — run paired_contrast_inflation.py first.")
    long_df = pd.read_csv(C["LONG_CSV"])
    plot(long_df, C["OUT"], show_ci=C["SHOW_CI"])


main()