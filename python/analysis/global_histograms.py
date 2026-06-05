"""
Overlapping histograms for global (component + centrality) features
CP dataset, coloured by essentiality label.

Non-essential is plotted first (behind), Essential on top — both semi-transparent.
Counts are normalised to density so the unequal class sizes don't dominate.
Log x-axis applied automatically where the range spans >2 orders of magnitude
or where >50% of values are zero (log(x+1) transform in that case).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────
SPLITS_PATH = "../../data/lookup_tables/cp_ess_protein_splits.csv"
HG_PATH     = "../../data/lookup_tables/cp_hypergraph_features.csv"
PW_PATH     = "../../data/lookup_tables/cp_pairwise_features.csv"

# ── Feature definitions ────────────────────────────────────────────────────
PW_FEATURES = {
    "Component": [
        ("pair_ComponentSize",         "Component Size"),
    ],
    "Centrality": [
        ("pair_BetweennessCentrality", "Betweenness Centrality"),
        ("pair_EigenvectorCentrality", "Eigenvector Centrality"),
        ("pair_KatzCentrality",        "Katz Centrality"),
    ],
}

HG_FEATURES = {
    "Component": [
        ("base_ComponentSize",          "Component Size"),
        ("base_ComponentEdgeNodeRatio", "Edge-Node Ratio"),
    ],
    "Centrality": [
        ("base_BetweennessCentrality",  "Betweenness Centrality"),
        ("base_EigenvectorCentrality",  "Eigenvector Centrality"),
        ("base_KatzCentrality",         "Katz Centrality"),
    ],
}

# ── Style ──────────────────────────────────────────────────────────────────
C_ESS  = "#E63946"   # red   – Essential
C_NON  = "#457B9D"   # blue  – Non-essential
ALPHA  = 0.55
BINS   = 50

# ── Load & merge ───────────────────────────────────────────────────────────
splits = pd.read_csv(SPLITS_PATH)
hg     = pd.read_csv(HG_PATH)
pw     = pd.read_csv(PW_PATH)

labels = (splits[splits["label_mask"]]
          .drop_duplicates("UniProt_AC")[["UniProt_AC", "protein_label"]])

merged_hg = hg.merge(labels, left_on="ProteinId", right_on="UniProt_AC", how="inner")
merged_pw = pw.merge(labels, left_on="ProteinId", right_on="UniProt_AC", how="inner")

ess_hg  = merged_hg[merged_hg["protein_label"] == "Essential"]
non_hg  = merged_hg[merged_hg["protein_label"] == "Non-essential"]
ess_pw  = merged_pw[merged_pw["protein_label"] == "Essential"]
non_pw  = merged_pw[merged_pw["protein_label"] == "Non-essential"]


# ── Scaling helper ─────────────────────────────────────────────────────────
def decide_transform(vals_all):
    """
    Returns (transform_fn, xlabel_suffix, bin_edges).
    Rules:
      - If >50% zeros OR range > 3 orders of magnitude with zeros present
        → log10(x + 1)  [handles zero-inflation gracefully]
      - Elif range > 2 orders of magnitude, all positive
        → log10(x)
      - Else
        → identity
    Uses BINS uniform bins on the transformed scale.
    """
    pct_zero  = (vals_all == 0).mean()
    vmin, vmax = vals_all.min(), vals_all.max()
    span = vmax / (vmin + 1e-300) if vmin > 0 else np.inf

    if pct_zero > 0.50 or (pct_zero > 0 and span > 1e3):
        fn     = lambda x: np.log10(x + 1)
        suffix = " (log₁₀(x+1))"
    elif vmin > 0 and span > 100:
        fn     = np.log10
        suffix = " (log₁₀)"
    else:
        fn     = lambda x: x
        suffix = ""

    transformed = fn(vals_all)
    edges = np.linspace(transformed.min(), transformed.max(), BINS + 1)
    return fn, suffix, edges


# ── Per-axis plot ──────────────────────────────────────────────────────────
def hist_ax(ax, ess_vals, non_vals, xlabel):
    all_vals = pd.concat([ess_vals, non_vals]).dropna()
    fn, suffix, edges = decide_transform(all_vals)

    t_non = fn(non_vals.dropna())
    t_ess = fn(ess_vals.dropna())

    ax.hist(t_non, bins=edges, density=True,
            color=C_NON, alpha=ALPHA, label="Non-essential", linewidth=0)
    ax.hist(t_ess, bins=edges, density=True,
            color=C_ESS, alpha=ALPHA, label="Essential",     linewidth=0)

    # Median lines
    for vals, colour in [(t_non, C_NON), (t_ess, C_ESS)]:
        med = np.median(vals)
        ax.axvline(med, color=colour, lw=1.5, linestyle="--", alpha=0.9)

    ax.set_xlabel(xlabel + suffix, fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.tick_params(labelsize=12)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", color="0.88", lw=0.5)
    ax.set_axisbelow(True)


# ── Figure builder ─────────────────────────────────────────────────────────
def make_figure(feature_dict, ess_df, non_df, title, out_path):
    all_feats = [(g, col, lbl)
                 for g, feats in feature_dict.items()
                 for col, lbl in feats]
    n     = len(all_feats)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 5, nrows * 2.8),
                             constrained_layout=True)
    axes_flat = axes.flatten() if n > 1 else [axes]

    for i, (group, col, lbl) in enumerate(all_feats):
        hist_ax(axes_flat[i],
                ess_df[col], non_df[col],
                f"[{group}] {lbl}")

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # Shared legend
    from matplotlib.patches import Patch
    from matplotlib.lines  import Line2D
    handles = [
        Patch(facecolor=C_NON, alpha=0.7, label="Non-essential"),
        Patch(facecolor=C_ESS, alpha=0.7, label="Essential"),
        Line2D([0], [0], color="grey", lw=1.5, linestyle="--", label="Median"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=10, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


make_figure(
    PW_FEATURES, ess_pw, non_pw,
    "CP – Pairwise Graph: Global Features by Essentiality",
    "cp_pairwise_histograms.png",
)

make_figure(
    HG_FEATURES, ess_hg, non_hg,
    "CP – Hypergraph: Global Features by Essentiality",
    "cp_hypergraph_histograms.png",
)