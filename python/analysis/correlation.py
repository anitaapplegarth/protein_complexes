"""
Spearman correlation heatmaps for all hypergraph and pairwise features.
- Features ordered by hierarchical clustering (Ward linkage on distance = 1 - |rho|)
- Annotated with rho values, masked upper triangle
- Separate figures for hypergraph and pairwise
- Binary essentiality label included as first row/col so you can see
  which features correlate most with the outcome
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import squareform

# ── Paths ──────────────────────────────────────────────────────────────────
SPLITS_PATH = "../../data/lookup_tables/cp_ess_protein_splits.csv"
HG_PATH     = "../../data/lookup_tables/cp_hypergraph_features.csv"
PW_PATH     = "../../data/lookup_tables/cp_pairwise_features.csv"

# ── Feature lists (all features, with clean labels) ────────────────────────
HG_COLS = {
    "base_Degree":                  "Degree",
    "base_LocalClustCoeff":         "Local Clust. Coeff.",
    "base_ComponentSize":           "Component Size",
    "base_ComponentEdgeNodeRatio":  "Edge-Node Ratio",
    "base_TriangleCount":           "Triangle Count",
    "stoich_WeightedTriangles":     "Weighted Triangles",
    "base_UniquePartners":          "Unique Partners",
    "stoich_MedianRatio":           "Stoich Median Ratio",
    "stoich_RangeRatio":            "Stoich Range Ratio",
    "stoich_MedComplexSize":        "Med Complex Size",
    "stoich_RangeComplexSize":      "Range Complex Size",
    "protein_MedComplexNodes":      "Med Complex Nodes",
    "protein_RangeComplexNodes":    "Range Complex Nodes",
    "base_BetweennessCentrality":   "Betweenness",
    "base_EigenvectorCentrality":   "Eigenvector",
    "base_KatzCentrality":          "Katz",
    "protein_RangeUniqueRatio":     "Range Unique Ratio",
    "protein_MedianUniqueRatio":    "Median Unique Ratio",
    "base_AvgNeighbourDegree":      "Avg Neighbour Degree",
    "stoich_AvgNeighbourDegreeStoich": "Avg Neighbour Degree (Stoich)",
}

PW_COLS = {
    "pair_Degree":                  "Degree",
    "pair_LocalClustCoeff":         "Local Clust. Coeff.",
    "pair_TriangleCount":           "Triangle Count",
    "pair_ComponentSize":           "Component Size",
    "pair_EigenvectorCentrality":   "Eigenvector",
    "pair_BetweennessCentrality":   "Betweenness",
    "pair_KatzCentrality":          "Katz",
    "pair_AvgNeighborDegree":       "Avg Neighbor Degree",
}

# ── Load & merge ───────────────────────────────────────────────────────────
splits = pd.read_csv(SPLITS_PATH)
hg     = pd.read_csv(HG_PATH)
pw     = pd.read_csv(PW_PATH)

labels = (splits[splits["label_mask"]]
          .drop_duplicates("UniProt_AC")[["UniProt_AC", "protein_label"]])
labels["essential"] = (labels["protein_label"] == "Essential").astype(int)

merged_hg = hg.merge(labels, left_on="ProteinId", right_on="UniProt_AC", how="inner")
merged_pw = pw.merge(labels, left_on="ProteinId", right_on="UniProt_AC", how="inner")


# ── Spearman matrix ────────────────────────────────────────────────────────
def spearman_matrix(df, cols):
    """Compute full Spearman rho matrix for given columns."""
    data = df[cols].values.astype(float)
    rho, _ = spearmanr(data)
    if data.shape[1] == 2:
        # spearmanr returns a scalar for 2 cols
        rho = np.array([[1.0, rho], [rho, 1.0]])
    return pd.DataFrame(rho, index=cols, columns=cols)


# ── Hierarchical clustering order ─────────────────────────────────────────
def cluster_order(rho_df):
    """Return reordered index using Ward linkage on 1 - |rho|."""
    dist = 1 - rho_df.abs().values
    np.fill_diagonal(dist, 0)
    # Ensure symmetry
    dist = (dist + dist.T) / 2
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="ward")
    order = leaves_list(Z)
    return rho_df.index[order].tolist()


# ── Heatmap plotter ────────────────────────────────────────────────────────
def plot_heatmap(rho_df, label_col, col_labels, title, out_path):
    """
    rho_df   : full square Spearman matrix (includes essentiality row/col)
    label_col: name of the essentiality column in rho_df
    col_labels: dict mapping col name → display label
    """
    # Reorder by clustering (exclude label col from clustering, prepend it)
    feat_cols = [c for c in rho_df.columns if c != label_col]
    order     = cluster_order(rho_df.loc[feat_cols, feat_cols])
    full_order = [label_col] + order

    rho_ordered = rho_df.loc[full_order, full_order]
    display_labels = [col_labels.get(c, c) for c in full_order]

    n = len(full_order)
    fig_size = max(10, n * 0.55)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.88))

    # Mask upper triangle (keep diagonal + lower)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    data_masked = np.where(mask, np.nan, rho_ordered.values)

    im = ax.imshow(data_masked, vmin=-1, vmax=1,
                   cmap="RdBu_r", aspect="auto")

    # Annotations — only lower triangle + diagonal
    for i in range(n):
        for j in range(n):
            if mask[i, j]:
                continue
            val = rho_ordered.values[i, j]
            txt_col = "white" if abs(val) > 0.65 else "black"
            fontsize = 10 if n > 20 else 12
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=fontsize, color=txt_col)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=12)
    ax.set_yticklabels(display_labels, fontsize=12)

    # Highlight the essentiality row/col with a box
    for spine in ax.spines.values():
        spine.set_visible(False)
    rect = plt.Rectangle((-0.5, -0.5), 1, n, linewidth=1.5,
                          edgecolor="#E63946", facecolor="none", zorder=5)
    ax.add_patch(rect)
    rect2 = plt.Rectangle((-0.5, -0.5), n, 1, linewidth=1.5,
                           edgecolor="#E63946", facecolor="none", zorder=5)
    ax.add_patch(rect2)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Spearman ρ", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── Build matrices & plot ──────────────────────────────────────────────────

# Hypergraph
hg_feat_cols = list(HG_COLS.keys())
hg_all_cols  = hg_feat_cols + ["essential"]
hg_labels    = {**HG_COLS, "essential": "★ Essential"}

rho_hg = spearman_matrix(merged_hg, hg_all_cols)
plot_heatmap(
    rho_hg, "essential", hg_labels,
    "Spearman ρ – Hypergraph Features (CP, Essentiality)",
    "cp_hg_correlation_heatmap.png",
)

# Pairwise
pw_feat_cols = list(PW_COLS.keys())
pw_all_cols  = pw_feat_cols + ["essential"]
pw_labels    = {**PW_COLS, "essential": "★ Essential"}

rho_pw = spearman_matrix(merged_pw, pw_all_cols)
plot_heatmap(
    rho_pw, "essential", pw_labels,
    "Spearman ρ – Pairwise Features (CP, Essentiality)",
    "cp_pw_correlation_heatmap.png",
)