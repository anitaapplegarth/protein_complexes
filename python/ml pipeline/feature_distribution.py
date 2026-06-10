import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =======================================================
# CONFIG
# =======================================================
DATABASES = {
    "Complex Portal": Path("/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/cp/"),
    "CORUM":          Path("/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/corum/"),
}

PROTEIN_FEATURES_FILE  = "hypergraph_features.csv"
PAIRWISE_FEATURES_FILE = "pairwise_features.csv"

FEATURE_GROUPS = {
    "Base (hypergraph)": [
        'base_Degree', 'base_LocalClustCoeff', 'base_TriangleCount',
        'base_UniquePartners', 'base_AvgNeighbourDegree',
    ],
    "Stoichiometry": [
        'stoich_WeightedTriangles', 'stoich_AvgNeighbourDegreeStoich',
        'stoich_RangeComplexSize', 'stoich_MedComplexSize',
        'stoich_MedianRatio', 'stoich_RangeRatio',
    ],
    "Protein-level": [
        'protein_MedianUniqueRatio', 'protein_RangeUniqueRatio',
        'protein_MedComplexNodes', 'protein_RangeComplexNodes',
    ],
    "Pairwise": [
        'pair_Degree', 'pair_LocalClustCoeff',
        'pair_TriangleCount', 'pair_AvgNeighborDegree',
    ],
}

OUTPUT_DIR = Path("./feature_distributions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =======================================================
# Plotting style
# =======================================================
plt.rcParams.update({
    'font.size':        12,
    'axes.titlesize':   14,
    'axes.labelsize':   12,
    'xtick.labelsize':  12,
    'ytick.labelsize':  12,
    'legend.fontsize':  12,
    'figure.titlesize': 16,
})

DB_COLOURS = {"Complex Portal": "#4C8BF5", "CORUM": "#E57A3A"}

# =======================================================
# Load data
# =======================================================
data = {}
for db_name, data_dir in DATABASES.items():
    hg_df   = pd.read_csv(data_dir / PROTEIN_FEATURES_FILE)
    pair_df = pd.read_csv(data_dir / PAIRWISE_FEATURES_FILE)
    combined = pd.merge(hg_df, pair_df, on='ProteinId', how='inner')
    data[db_name] = combined
    print(f"{db_name}: {len(combined)} proteins")

# =======================================================
# Summary stats: flag degenerate features
# =======================================================
print(f"\n{'='*80}")
print("  DEGENERACY CHECK — % of proteins at the single most common value")
print(f"{'='*80}")

all_feats = [f for feats in FEATURE_GROUPS.values() for f in feats]
degeneracy_rows = []

for db_name, df in data.items():
    print(f"\n  {db_name} ({len(df)} proteins)")
    print(f"  {'Feature':<42s} {'Mode':<12s} {'Count':<8s} {'%':<8s}")
    print(f"  {'-'*42} {'-'*12} {'-'*8} {'-'*8}")
    for f in all_feats:
        if f not in df.columns:
            continue
        mode_val   = df[f].mode().iloc[0]
        mode_count = (df[f] == mode_val).sum()
        mode_pct   = 100 * mode_count / len(df)
        flag = "  <--- degenerate" if mode_pct > 80 else ""
        print(f"  {f:<42s} {mode_val:<12.4f} {mode_count:<8d} {mode_pct:<7.1f}%{flag}")
        degeneracy_rows.append({
            'database':     db_name,
            'feature':      f,
            'mode_value':   round(mode_val, 6),
            'mode_count':   mode_count,
            'total':        len(df),
            'mode_pct':     round(mode_pct, 1),
            'degenerate':   mode_pct > 80,
        })

degeneracy_df = pd.DataFrame(degeneracy_rows)
degeneracy_df.to_csv(OUTPUT_DIR / 'feature_degeneracy.csv', index=False)
print(f"\nSaved: {OUTPUT_DIR / 'feature_degeneracy.csv'}")

# =======================================================
# Plot: one figure per feature group
# =======================================================
for group_name, feats in FEATURE_GROUPS.items():
    n_feats = len(feats)
    fig, axes = plt.subplots(n_feats, 2, figsize=(12, 3 * n_feats))
    if n_feats == 1:
        axes = axes.reshape(1, 2)

    for i, feat in enumerate(feats):
        for j, (db_name, df) in enumerate(data.items()):
            ax = axes[i, j]
            if feat not in df.columns:
                ax.set_visible(False)
                continue

            vals = df[feat].dropna()
            colour = DB_COLOURS[db_name]

            ax.hist(vals, bins=50, color=colour, alpha=0.7, edgecolor='white', linewidth=0.5)
            ax.set_title(f"{feat}" if j == 0 else f"{feat}", fontsize=12)
            ax.set_ylabel("Count" if j == 0 else "")

            # Annotate mode concentration
            mode_val   = vals.mode().iloc[0]
            mode_count = (vals == mode_val).sum()
            mode_pct   = 100 * mode_count / len(vals)
            if mode_pct > 30:
                ax.annotate(
                    f"{mode_pct:.0f}% at {mode_val:.2f}",
                    xy=(0.97, 0.92), xycoords='axes fraction',
                    ha='right', va='top', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8),
                )

            if i == 0:
                ax.set_title(f"{db_name}\n{feat}", fontsize=12)

    fig.suptitle(f"{group_name} features — CP vs CORUM", fontsize=16, y=1.01)
    fig.tight_layout()

    safe_name = group_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    out_path = OUTPUT_DIR / f"distributions_{safe_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out_path}")
    plt.close(fig)

print(f"\nAll plots saved to: {OUTPUT_DIR}")