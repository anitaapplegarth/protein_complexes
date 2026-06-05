"""
Distribution plots for the two key features from the logistic regression
analysis: stoich_MedianRatio and protein_MedComplexNodes.

Generates violin plots and density plots comparing drug targets vs
non-targets, confirming that the logistic regression results reflect
genuine distributional differences rather than artefacts.

Usage
-----
    Just hit Run in VS Code — paths are configured below.
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, mannwhitneyu
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration — edit these paths to match your local setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path("/Users/anitaapplegarth/github/dphil/protein_complexes")

FEATURES_FILE = PROJECT_ROOT / "data/lookup_tables/cp/hypergraph_features.csv"
LABELS_FILE   = PROJECT_ROOT / "data/lookup_tables/cp_drug_target_hpa.csv"
OUTPUT_DIR    = PROJECT_ROOT / "python/analysis/stoichiometry/regression"

PROTEIN_ID_COL = "ProteinId"
LABEL_COL = "target"

# The two features that survived the combined multivariate model
KEY_FEATURES = [
    "stoich_MedianRatio",
    "protein_MedComplexNodes",
]

KEY_LABELS = [
    "Median Stoichiometric Ratio",
    "Median Complex Size (unique proteins)",
]

mpl.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "figure.dpi": 150,
})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load and merge data
    feat_df = pd.read_csv(FEATURES_FILE)
    labels_df = pd.read_csv(LABELS_FILE)
    df = feat_df.merge(labels_df[[PROTEIN_ID_COL, LABEL_COL]],
                       on=PROTEIN_ID_COL, how="inner")

    print(f"Loaded {len(df)} proteins "
          f"({int(df[LABEL_COL].sum())} drug targets, "
          f"{int(len(df) - df[LABEL_COL].sum())} non-targets)\n")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for i, (feat, label) in enumerate(zip(KEY_FEATURES, KEY_LABELS)):
        pos = df.loc[df[LABEL_COL] == 1, feat].values
        neg = df.loc[df[LABEL_COL] == 0, feat].values

        # Wilcoxon rank-sum test
        stat, p = mannwhitneyu(pos, neg, alternative="two-sided")
        n1, n2 = len(pos), len(neg)
        r_rb = 1 - (2 * stat) / (n1 * n2)

        print(f"--- {feat} ---")
        print(f"  Drug targets:  median = {np.median(pos):.4f}, "
              f"mean = {np.mean(pos):.4f}")
        print(f"  Non-targets:   median = {np.median(neg):.4f}, "
              f"mean = {np.mean(neg):.4f}")
        print(f"  Mann-Whitney U p = {p:.2e}, "
              f"rank-biserial r = {abs(r_rb):.4f}\n")

        # --- Violin plot (left column) ---
        ax = axes[i, 0]
        parts = ax.violinplot(
            [neg, pos], positions=[0, 1],
            showmedians=True, showextrema=False,
        )
        for j, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(["#7fadcf", "#d45d5d"][j])
            pc.set_alpha(0.7)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(2)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Non-target", "Drug target"])
        ax.set_ylabel(label)
        ax.set_title(f"{label.split('(')[0].strip()} — Violin")

        # Annotate medians
        for data, xpos in [(neg, 0), (pos, 1)]:
            med = np.median(data)
            ax.annotate(
                f"med={med:.2f}", xy=(xpos, med),
                xytext=(15, 5), textcoords="offset points",
                fontsize=11, color="black",
            )

        # --- Density plot (right column) ---
        ax = axes[i, 1]

        for data, lbl, color in [(neg, "Non-target", "#7fadcf"),
                                  (pos, "Drug target", "#d45d5d")]:
            if len(np.unique(data)) > 10:
                try:
                    kde = gaussian_kde(data, bw_method=0.3)
                    x_range = np.linspace(
                        data.min(), np.percentile(data, 99), 300
                    )
                    ax.fill_between(x_range, kde(x_range),
                                    alpha=0.4, color=color, label=lbl)
                    ax.plot(x_range, kde(x_range), color=color, linewidth=2)
                except Exception:
                    ax.hist(data, bins=30, density=True, alpha=0.4,
                            color=color, label=lbl)
            else:
                ax.hist(data, bins=30, density=True, alpha=0.4,
                        color=color, label=lbl, edgecolor="white")

        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.set_title(f"{label.split('(')[0].strip()} — Density")
        ax.legend(fontsize=12)

    plt.tight_layout()
    outpath = outdir / "feature_distributions.png"
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {outpath}")


if __name__ == "__main__":
    main()