"""
1-D strip plots for global (component + centrality) features
CP dataset, coloured by essentiality label.

Pairwise features (4):
  pair_ComponentSize, pair_TriangleCount,
  pair_BetweennessCentrality, pair_EigenvectorCentrality, pair_KatzCentrality
  → actually 5 pairwise; we use the 4 component+centrality ones per request:
    component : pair_ComponentSize, pair_TriangleCount
    centrality: pair_BetweennessCentrality, pair_EigenvectorCentrality, pair_KatzCentrality
  (We'll show all 5 pairwise and 5 hypergraph; adjustable via FEATURE dicts below.)

Hypergraph features (5):
  component : base_ComponentSize, base_ComponentEdgeNodeRatio,
              base_TriangleCount, base_UniquePartners
  centrality: base_BetweennessCentrality, base_EigenvectorCentrality, base_KatzCentrality
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Paths ──────────────────────────────────────────────────────────────────
SPLITS_PATH = "../../data/lookup_tables/cp_ess_protein_splits.csv"
HG_PATH     = "../../data/lookup_tables/cp_hypergraph_features.csv"
PW_PATH     = "../../data/lookup_tables/cp_pairwise_features.csv"
OUT_PATH     = "cp_global_features_1d.png"

# ── Feature definitions ────────────────────────────────────────────────────
PW_FEATURES = {
    "Component": [
        ("pair_ComponentSize",          "Component Size"),
        ("pair_TriangleCount",          "Triangle Count"),
    ],
    "Centrality": [
        ("pair_BetweennessCentrality",  "Betweenness Centrality"),
        ("pair_EigenvectorCentrality",  "Eigenvector Centrality"),
        ("pair_KatzCentrality",         "Katz Centrality"),
    ],
}

HG_FEATURES = {
    "Component": [
        ("base_ComponentSize",          "Component Size"),
        ("base_ComponentEdgeNodeRatio", "Edge-Node Ratio"),
        ("base_TriangleCount",          "Triangle Count"),
        ("base_UniquePartners",         "Unique Partners"),
    ],
    "Centrality": [
        ("base_BetweennessCentrality",  "Betweenness Centrality"),
        ("base_EigenvectorCentrality",  "Eigenvector Centrality"),
        ("base_KatzCentrality",         "Katz Centrality"),
    ],
}

# ── Colours ────────────────────────────────────────────────────────────────
COLOURS = {
    "Essential":     "#E63946",   # red
    "Non-essential": "#457B9D",   # steel blue
}
ALPHA     = 0.35
JITTER    = 0.25
RNG       = np.random.default_rng(42)
PT_SIZE   = 6


# ── Helpers ────────────────────────────────────────────────────────────────
def log_scale_if_needed(series, thresh=100):
    """Return (values, did_log_transform, x_label_suffix)."""
    s = series.dropna()
    if s.min() <= 0:
        # shift so we can log
        shift = -s.min() + 1e-9
        s_shifted = s + shift
    else:
        s_shifted = s
        shift = 0.0

    ratio = s_shifted.max() / (s_shifted.min() + 1e-300)
    if ratio > thresh or s_shifted.max() < 1e-3:
        transformed = np.log10(s_shifted + 1e-300)
        suffix = " (log₁₀)" if shift == 0 else f" (log₁₀, shifted +{shift:.1e})"
        return transformed, True, suffix
    return s, False, ""


def strip_plot_ax(ax, data_ess, data_non, label, log_thresh=100):
    """Draw a single 1-D strip plot on ax."""
    all_vals = pd.concat([data_ess, data_non])
    vals_plot, did_log, suffix = log_scale_if_needed(all_vals, thresh=log_thresh)

    # re-apply the same transform per group
    if did_log:
        if all_vals.min() <= 0:
            shift = -all_vals.min() + 1e-9
        else:
            shift = 0.0
        ess_vals  = np.log10(data_ess  + shift + 1e-300)
        non_vals  = np.log10(data_non  + shift + 1e-300)
    else:
        ess_vals  = data_ess
        non_vals  = data_non

    groups = [
        ("Essential",     ess_vals,  1),
        ("Non-essential", non_vals,  0),
    ]

    for (gname, gvals, y_base) in groups:
        n      = len(gvals)
        jitter = RNG.uniform(-JITTER, JITTER, size=n)
        y      = y_base + jitter
        ax.scatter(
            gvals, y,
            c=COLOURS[gname], alpha=ALPHA,
            s=PT_SIZE, linewidths=0,
            label=gname, rasterized=True,
        )
        # median line
        med = np.median(gvals)
        ax.plot(
            [med, med],
            [y_base - JITTER * 1.3, y_base + JITTER * 1.3],
            color="white", lw=2.5, zorder=5,
        )
        ax.plot(
            [med, med],
            [y_base - JITTER * 1.3, y_base + JITTER * 1.3],
            color=COLOURS[gname], lw=1.5, zorder=6,
        )

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Non-essential", "Essential"], fontsize=10)
    ax.set_xlabel(label + suffix, fontsize=10)
    ax.set_ylim(-0.6, 1.6)
    ax.tick_params(axis="x", labelsize=10)
    ax.grid(axis="x", color="0.85", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


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

print(f"Essential: {len(ess_hg)}  Non-essential: {len(non_hg)}")

# ── Layout ─────────────────────────────────────────────────────────────────
# Pairwise  : 4 component  (count top row) + … actually use dict totals
pw_n  = sum(len(v) for v in PW_FEATURES.values())   # 4
hg_n  = sum(len(v) for v in HG_FEATURES.values())   # 5

# We'll make two separate figures: one pairwise, one hypergraph
# Each figure: rows = features, 1 col per feature (horizontal strips)

def make_figure(feature_dict, ess_df, non_df, title, out_path):
    all_feats = []
    for group, feats in feature_dict.items():
        for col, label in feats:
            all_feats.append((group, col, label))

    n = len(all_feats)
    # 2 columns of strips
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 5.5, nrows * 2.0),
        constrained_layout=True,
    )
    axes_flat = axes.flatten() if n > 1 else [axes]

    for i, (group, col, label) in enumerate(all_feats):
        ax = axes_flat[i]
        d_ess = ess_df[col].dropna()
        d_non = non_df[col].dropna()
        strip_plot_ax(ax, d_ess, d_non, f"[{group}] {label}")

    # Hide unused axes
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # Legend (one shared)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOURS["Essential"],
               markersize=7, alpha=0.8, label="Essential"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOURS["Non-essential"],
               markersize=7, alpha=0.8, label="Non-essential"),
        Line2D([0], [0], color="grey", lw=1.5, label="Median"),
    ]
    fig.legend(handles=handles, loc="lower center",
               ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")

fig_pw = make_figure(
    PW_FEATURES, ess_pw, non_pw,
    "CP – Pairwise Graph: Global Features by Essentiality", "pairwise_1d.png"
)

fig_hg = make_figure(
    HG_FEATURES, ess_hg, non_hg,
    "CP – Hypergraph: Global Features by Essentiality", "hypergraph_1d.png"
)

