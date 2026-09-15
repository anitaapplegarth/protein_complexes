"""
=======================================================================
GROUP-vs-RANDOM SPLIT INFLATION  —  figure
=======================================================================
Reads split_inflation_long.csv (from split_inflation_analysis.py) and
draws a dumbbell plot: for every (model, tier) cell, the group-based
split mean PR-AUC (filled marker) is joined to the protein-level random
split mean PR-AUC (open marker) by a line. A rightward line means the
random arm scored higher; the length of the line is the size of the
difference.

The random arm is the protein-level, label-stratified splitter in which
structural groups are ignored, so members of the same group fall on both
sides of the boundary. It is not the earlier group-atomic "unstrat" arm.

Layout: one COLUMN per task (essentiality / HPA / ChEMBL), one row-block
per model, three sub-rows per block (one per tier: pairwise / hypergraph
/ hb-graph). All three tiers shown, per the design decision.

Style: font >= 12 throughout; tier encoded by colour, arm by marker fill.
No W/L language anywhere.
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
    "LONG_CSV": Path("split_inflation_long.csv"),   # from split_inflation_analysis.py
    "OUT":      Path("split_inflation.png"),        # PDF written alongside
    "SHOW_CI":  True,   # overlay bootstrap CI of the difference as a faint band
}

# ----------------------------------------------------------------------
# Style — minimum font size 12, scaled up for a multi-panel figure.
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.size":        13,
    "axes.titlesize":   15,
    "axes.labelsize":   13,
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
    "legend.fontsize":  12,
    "figure.titlesize": 17,
})

TIER_ORDER = ["pairwise", "hypergraph", "hb_graph"]
TIER_LABEL = {"pairwise": "Pairwise", "hypergraph": "Hypergraph",
              "hb_graph": "HB-graph"}
# Colour-blind-safe (Okabe-Ito subset), one per tier.
TIER_COLOR = {"pairwise": "#0072B2", "hypergraph": "#009E73",
              "hb_graph": "#CC79A7"}

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

    # Vertical layout: each model gets a block of 3 tier-rows plus a gap.
    n_tier = len(TIER_ORDER)
    block  = n_tier + 1                       # +1 blank row between models
    n_rows = len(models) * block - 1          # drop trailing gap

    # y position: top model at the top.
    def ypos(mi, ti):
        return n_rows - 1 - (mi * block + ti)

    fig, axes = plt.subplots(
        1, len(tasks),
        figsize=(5.2 * len(tasks), max(4.5, 0.34 * n_rows + 1.2)),
        sharey=True, squeeze=False,
    )
    axes = axes[0]

    for ax, task in zip(axes, tasks):
        sub = long_df[long_df["task"] == task]
        for mi, model in enumerate(models):
            for ti, tier in enumerate(TIER_ORDER):
                cell = sub[(sub["model"] == model) & (sub["tier"] == tier)]
                if cell.empty:
                    continue
                r = cell.iloc[0]
                y = ypos(mi, ti)
                c = TIER_COLOR[tier]

                # connecting line: group -> random
                ax.plot([r["mean_group"], r["mean_random"]], [y, y],
                        color=c, lw=2.0, alpha=0.55, zorder=1,
                        solid_capstyle="round")
                # group = filled, random = open
                ax.scatter(r["mean_group"], y, s=46, color=c,
                           zorder=3, edgecolor="white", linewidth=0.6)
                ax.scatter(r["mean_random"], y, s=46, facecolor="white",
                           edgecolor=c, linewidth=1.8, zorder=3)

                if show_ci and np.isfinite(r.get("ci_lo", np.nan)):
                    # CI is on the DIFFERENCE; draw it as a light band anchored
                    # at the group mean to signal uncertainty in the gap.
                    lo = r["mean_group"] + r["ci_lo"]
                    hi = r["mean_group"] + r["ci_hi"]
                    ax.plot([lo, hi], [y, y], color=c, lw=6, alpha=0.15, zorder=0)

        ax.set_title(TASK_LABEL.get(task, task))
        ax.set_xlabel("Mean PR-AUC over 50 splits")
        ax.grid(axis="x", linestyle=":", alpha=0.4)
        ax.set_axisbelow(True)

    # y ticks: the MODEL name sits on the middle tier row of each block;
    # individual tier rows are left unlabelled (tier is encoded by colour and
    # given in the legend). This avoids model/tier label collisions and keeps
    # the axis readable at font size 12.
    yticks, ylabels = [], []
    for mi, model in enumerate(models):
        yticks.append(ypos(mi, 1))          # middle tier row
        ylabels.append(model)
    axes[0].set_yticks(yticks)
    axes[0].set_yticklabels(ylabels, fontweight="bold")
    axes[0].set_ylim(-0.8, n_rows - 0.2)

    # Faint horizontal separators between model blocks.
    for mi in range(1, len(models)):
        y_sep = ypos(mi, 0) + 1.0
        for ax in axes:
            ax.axhline(y_sep, color="0.85", lw=0.8, zorder=0)

    # Legend: arm (marker fill) + tier (colour).
    arm_handles = [
        Line2D([0], [0], marker="o", color="0.3", markerfacecolor="0.3",
               markersize=8, lw=0, label="Group-based split (headline)"),
        Line2D([0], [0], marker="o", color="0.3", markerfacecolor="white",
               markeredgecolor="0.3", markeredgewidth=1.8, markersize=8, lw=0,
               label="Protein-level random split"),
    ]
    tier_handles = [
        Line2D([0], [0], color=TIER_COLOR[t], lw=3, label=TIER_LABEL[t])
        for t in TIER_ORDER
    ]
    leg1 = fig.legend(handles=arm_handles, loc="upper center",
                      bbox_to_anchor=(0.5, 0.035), ncol=2, frameon=False)
    fig.add_artist(leg1)
    fig.legend(handles=tier_handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.005), ncol=3, frameon=False)

    fig.suptitle("Mean PR-AUC under group-based and protein-level random "
                 "splitting\n(open marker = random split; line length = "
                 "difference in means)",
                 y=0.995)
    # leave room on the left for model labels and at the bottom for legends
    fig.subplots_adjust(left=0.13, bottom=0.14, top=0.90,
                        right=0.98, wspace=0.08)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Wrote {out_path}")
    pdf = out_path.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"Wrote {pdf}")


def main():
    C = CONFIG
    if not C["LONG_CSV"].exists():
        raise FileNotFoundError(
            f"{C['LONG_CSV']} not found — run split_inflation_analysis.py first.")
    long_df = pd.read_csv(C["LONG_CSV"])
    plot(long_df, C["OUT"], show_ci=C["SHOW_CI"])


main()