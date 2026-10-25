"""
=======================================================================
OPTIMISM FROM RANDOM SPLITTING  —  dot plot
=======================================================================
Reads splitting_absolute.csv (written by cp_compare_splitting.py) and
draws, for each task, the mean PR-AUC difference between the two
splitting schemes with 95% CIs: one row per model, one marker per tier.

Positive = random splitting inflates performance for that model and
representation. A CI crossing zero means the optimism is not
distinguishable from noise.

The quantity sits on an axis rather than being a residual from a y=x
diagonal, and model names sit in the y-axis labels, so nothing has to
be nudged apart. This is the same idiom as the ladder script's panel B.

MODEL_ORDER:
  'match_ladder'  each panel sorted by that task's total effect
                  (hb-graph minus pairwise, structure-aware run),
                  descending. This reproduces the ordering in
                  plot_sweep_ladder.py, so the panels line up with the
                  ladder figures for direct comparison. Orders differ
                  between tasks, so the y-axis is not shared.
  'capacity'      a single fixed order, high-capacity models first,
                  with a rule between the groups. Makes the capacity
                  pattern obvious but no longer matches the ladder.

Model names in the y-axis labels are coloured with the ladder script's
MODEL_COLOURS, so a model can be followed across panels even though the
order changes. Set COLOUR_MODEL_LABELS = False for plain black.
=======================================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =======================================================================
# CONFIGURATION
# =======================================================================
CONFIG = {
    "INPUT":  Path("../figs/paper/splitting_absolute.csv"),
    "OUTPUT": Path("../figs/paper/fig_splitting_optimism.png"),

    # 'capacity' | 'match_ladder'   (see the docstring)
    "MODEL_ORDER": "capacity",

    "COLOUR_MODEL_LABELS": True,
}

TASKS = [("essentiality", "(a) Gene essentiality"),
         ("chembl",       "(b) Drug target (ChEMBL)"),
         ("hpa",          "(c) Drug target (HPA)")]

TIERS       = ["pairwise", "hypergraph", "hb_graph"]
TIER_COLOUR = {"pairwise": "#999999", "hypergraph": "#87CEEB",
               "hb_graph": "#4682B4"}
TIER_LABEL  = {"pairwise": "Pairwise", "hypergraph": "Hypergraph",
               "hb_graph": "Hb-graph"}

# Fixed colour per model, matching plot_sweep_ladder.py.
MODEL_COLOURS = {
    "TabPFN":             "#4C72B0",
    "RandomForest":       "#DD8452",
    "LightGBM":           "#55A868",
    "XGBoost":            "#C44E52",
    "RBF-SVM":            "#8172B2",
    "LogisticRegression": "#937860",
    "MLP":                "#DA8BC3",
}

# Used only when MODEL_ORDER == 'capacity'.
CAPACITY_ORDER = ["TabPFN", "RandomForest", "XGBoost", "LightGBM",
                  "MLP", "RBF-SVM", "LogisticRegression"]
CAPACITY_SPLIT_AFTER = 4

plt.rcParams.update({
    "font.size":        13,
    "axes.titlesize":   15,
    "axes.labelsize":   14,
    "xtick.labelsize":  13,
    "ytick.labelsize":  13,
    "legend.fontsize":  12,
    "figure.titlesize": 16,
})


def model_order(sub: pd.DataFrame) -> list:
    """Per-task ordering.

    'match_ladder' reproduces plot_sweep_ladder.py's sort key, which is
    the total effect (hb-graph minus pairwise) of the structure-aware
    run, descending. Computing it here from the tier means avoids having
    to read the summary files as well."""
    if CONFIG["MODEL_ORDER"] == "capacity":
        return [m for m in CAPACITY_ORDER if m in set(sub["model"])]
    p = sub.pivot_table(index="model", columns="tier",
                        values="structured_mean")
    total = p["hb_graph"] - p["pairwise"]
    return list(total.sort_values(ascending=False).index)


def main():
    d = pd.read_csv(CONFIG["INPUT"])
    by_ladder = CONFIG["MODEL_ORDER"] == "match_ladder"

    fig, axes = plt.subplots(1, 3, figsize=(17, 6.0),
                             sharey=not by_ladder, sharex=True)

    for ax, (task, lab) in zip(axes, TASKS):
        sub = d[d["task"] == task]
        models = model_order(sub)
        y = np.arange(len(models))[::-1]

        for off, tier in zip((+0.24, 0.0, -0.24), TIERS):
            m, lo, hi = [], [], []
            for mod in models:
                r = sub[(sub["model"] == mod) & (sub["tier"] == tier)].iloc[0]
                m.append(r["diff"])
                lo.append(r["diff"] - r["ci_lo"])
                hi.append(r["ci_hi"] - r["diff"])
            ax.errorbar(m, y + off, xerr=[lo, hi], fmt="o", ms=7, lw=2.0,
                        capsize=3, color=TIER_COLOUR[tier],
                        label=TIER_LABEL[tier], zorder=3)

        for yi in y[:-1]:
            ax.axhline(yi - 0.5, color="0.88", lw=0.8, zorder=1)
        if not by_ladder:
            ax.axhline(y[CAPACITY_SPLIT_AFTER - 1] - 0.5, color="0.55",
                       lw=1.4, ls=":", zorder=2)

        ax.axvline(0, color="black", lw=1.4, zorder=2)
        ax.set_yticks(y)
        ax.set_yticklabels(models)
        if CONFIG["COLOUR_MODEL_LABELS"]:
            for tick, mod in zip(ax.get_yticklabels(), models):
                tick.set_color(MODEL_COLOURS.get(mod, "black"))
        ax.set_ylim(-0.6, len(models) - 0.4)
        ax.set_title(lab, fontsize=15)
        ax.grid(axis="x", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Δ PR-AUC (random − structure-aware)")

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h[:3], l[:3], frameon=False, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.035), columnspacing=2.4,
               handletextpad=0.4)
    fig.suptitle("Optimism from random splitting, by model and "
                 "representation (mean, 95% CI)")
    fig.tight_layout(rect=[0, 0.03, 1, 1])

    out = CONFIG["OUTPUT"]
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()