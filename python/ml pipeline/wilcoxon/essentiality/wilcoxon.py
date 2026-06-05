"""
Wilcoxon Rank-Sum Bar Charts: Feature Significance for Gene Essentiality
=========================================================================
Produces separate horizontal bar charts for hypergraph and pairwise features,
for both CP and CORUM databases. Each bar shows the Wilcoxon rank-sum test
statistic (positive = higher in Essential, negative = higher in Non-essential).

Bars are coloured by direction: red-ish for positive, blue-ish for negative.
Stars indicate significance after Bonferroni correction.

Usage:
    python wilcoxon_barcharts.py

Requires: pandas, numpy, scipy, matplotlib

Inputs (edit DATA_DIR below if needed):
    - cp_hypergraph_features.csv
    - cp_pairwise_features.csv
    - corum_hypergraph_features.csv
    - corum_pairwise_features.csv
    - cp_ess_protein_splits.csv
    - corum_ess_protein_splits.csv

Outputs (in OUTPUT_DIR):
    - wilcoxon_cp_hypergraph.png
    - wilcoxon_cp_pairwise.png
    - wilcoxon_corum_hypergraph.png
    - wilcoxon_corum_pairwise.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ranksums
from pathlib import Path

# =======================================================
# CONFIGURATION
# =======================================================
DATA_DIR = Path("../../../data/lookup_tables")   # <-- adjust to your data location
OUTPUT_DIR = Path("./wilcoxon_charts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Plotting style (matches your pipeline conventions)
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.titlesize': 18,
})

# Pipeline features only — matches the "all" scope from the ML pipeline.
# CORUM excludes stoich_ features by design (~4% coverage).
PIPELINE_FEATURES = {
    'HYPERGRAPH': [
        # One-hop
        'base_Degree', 'base_UniquePartners',
        'stoich_RangeComplexSize', 'stoich_MedComplexSize',
        'stoich_MedianRatio', 'stoich_RangeRatio',
        'protein_MedianUniqueRatio', 'protein_RangeUniqueRatio',
        'protein_MedComplexNodes', 'protein_RangeComplexNodes',
        # Two-hop
        'base_LocalClustCoeff', 'base_TriangleCount', 'base_AvgNeighbourDegree',
        'stoich_WeightedTriangles', 'stoich_AvgNeighbourDegreeStoich',
        # Global
        'base_BetweennessCentrality', 'base_EigenvectorCentrality',
        'base_KatzCentrality', 'base_ComponentSize', 'base_ComponentEdgeNodeRatio',
    ],
    'PAIRWISE': [
        # One-hop
        'pair_Degree',
        # Two-hop
        'pair_LocalClustCoeff', 'pair_TriangleCount', 'pair_AvgNeighborDegree',
        # Global
        'pair_ComponentSize', 'pair_EigenvectorCentrality',
        'pair_BetweennessCentrality', 'pair_KatzCentrality',
    ],
}


# =======================================================
# DATA LOADING
# =======================================================
def load_labelled_features(db: str) -> pd.DataFrame:
    """
    Load features + essentiality labels for a given database.
    Uses split_index == 1 to get one consistent set of labels
    (labels are the same across all splits, we just need one).
    Returns DataFrame with only Essential / Non-essential proteins.
    """
    prefix = db  # 'cp' or 'corum'

    hg = pd.read_csv(DATA_DIR / f"{prefix}_hypergraph_features.csv")
    pw = pd.read_csv(DATA_DIR / f"{prefix}_pairwise_features.csv")
    splits = pd.read_csv(DATA_DIR / f"{prefix}_ess_protein_splits.csv")

    # Take one split for labels (they're consistent across splits)
    labels = splits[splits['split_index'] == 1][['UniProt_AC', 'protein_label']].copy()
    labels = labels.rename(columns={'UniProt_AC': 'ProteinId'})

    # Keep only Essential / Non-essential
    labels = labels[labels['protein_label'].isin(['Essential', 'Non-essential'])]

    # Merge features
    merged = labels.merge(hg, on='ProteinId', how='inner')
    merged = merged.merge(pw, on='ProteinId', how='inner')

    print(f"[{db.upper()}] Loaded {len(merged)} labelled proteins "
          f"(Essential: {(merged['protein_label']=='Essential').sum()}, "
          f"Non-essential: {(merged['protein_label']=='Non-essential').sum()})")

    return merged


# =======================================================
# WILCOXON TEST
# =======================================================
def run_wilcoxon_tests(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Run Wilcoxon rank-sum test for each feature: Essential vs Non-essential.
    Returns DataFrame with columns: feature, statistic, pvalue, significant.
    Positive statistic = higher values in Essential.
    """
    ess = df[df['protein_label'] == 'Essential']
    noness = df[df['protein_label'] == 'Non-essential']

    results = []
    for feat in features:
        x = ess[feat].dropna()
        y = noness[feat].dropna()

        if len(x) < 5 or len(y) < 5:
            continue

        stat, pval = ranksums(x, y)  # positive stat means x > y
        results.append({
            'feature': feat,
            'statistic': stat,
            'pvalue': pval,
        })

    results_df = pd.DataFrame(results)

    # Bonferroni correction
    n_tests = len(results_df)
    results_df['significant'] = results_df['pvalue'] < (0.05 / n_tests)
    results_df['pvalue_corrected'] = np.minimum(results_df['pvalue'] * n_tests, 1.0)

    # Sort by absolute statistic
    results_df = results_df.sort_values('statistic', key=abs, ascending=True)

    return results_df


# =======================================================
# PLOTTING
# =======================================================
def plot_wilcoxon_bars(results_df: pd.DataFrame, title: str, output_path: Path,
                       figsize_height: float = None):
    """
    Horizontal bar chart of Wilcoxon test statistics.
    Colour: positive (higher in Essential) = #C44E52 (red),
            negative (higher in Non-essential) = #4C72B0 (blue).
    Stars for Bonferroni-significant features.
    """
    n_features = len(results_df)
    if figsize_height is None:
        figsize_height = max(4, n_features * 0.4)

    fig, ax = plt.subplots(figsize=(10, figsize_height))

    colours = ['#C44E52' if s > 0 else '#4C72B0' for s in results_df['statistic']]

    bars = ax.barh(
        results_df['feature'],
        results_df['statistic'],
        color=colours,
        edgecolor='white',
        linewidth=0.5,
    )

    # Add significance stars
    for i, (_, row) in enumerate(results_df.iterrows()):
        if row['significant']:
            x_pos = row['statistic']
            offset = 0.3 if x_pos >= 0 else -0.3
            ha = 'left' if x_pos >= 0 else 'right'
            ax.text(x_pos + offset, i, '*', fontsize=14, fontweight='bold',
                    va='center', ha=ha, color='black')

    ax.set_xlabel('Wilcoxon Statistic (Higher in Essential →)')
    ax.set_title(title)
    ax.axvline(x=0, color='grey', linewidth=0.8, linestyle='-')

    # Clean up
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {output_path}")


# =======================================================
# MAIN
# =======================================================
def main():
    for db in ['cp', 'corum']:
        print(f"\n{'='*60}")
        print(f"Processing {db.upper()}")
        print(f"{'='*60}")

        df = load_labelled_features(db)

        # --- Determine feature lists ---
        hg_features = [f for f in PIPELINE_FEATURES['HYPERGRAPH'] if f in df.columns]
        pw_features = [f for f in PIPELINE_FEATURES['PAIRWISE'] if f in df.columns]

        # For CORUM, exclude stoichiometry features
        if db == 'corum':
            hg_features = [c for c in hg_features if not c.startswith('stoich_')]

        print(f"   Hypergraph features: {len(hg_features)}")
        print(f"   Pairwise features:   {len(pw_features)}")

        # --- Run tests ---
        hg_results = run_wilcoxon_tests(df, hg_features)
        pw_results = run_wilcoxon_tests(df, pw_features)

        # --- Print summary table ---
        print(f"\n   Hypergraph Wilcoxon results ({db.upper()}):")
        for _, row in hg_results.sort_values('statistic', ascending=False).iterrows():
            sig = '*' if row['significant'] else ' '
            print(f"   {sig} {row['feature']:40s}  stat={row['statistic']:+7.2f}  "
                  f"p={row['pvalue_corrected']:.2e}")

        print(f"\n   Pairwise Wilcoxon results ({db.upper()}):")
        for _, row in pw_results.sort_values('statistic', ascending=False).iterrows():
            sig = '*' if row['significant'] else ' '
            print(f"   {sig} {row['feature']:40s}  stat={row['statistic']:+7.2f}  "
                  f"p={row['pvalue_corrected']:.2e}")

        # --- Plot ---
        plot_wilcoxon_bars(
            hg_results,
            title=f'{db.upper()} Hypergraph Features: Essential vs Non-essential\n(Wilcoxon Rank-Sum Test)',
            output_path=OUTPUT_DIR / f'wilcoxon_{db}_hypergraph.png',
        )
        plot_wilcoxon_bars(
            pw_results,
            title=f'{db.upper()} Pairwise Features: Essential vs Non-essential\n(Wilcoxon Rank-Sum Test)',
            output_path=OUTPUT_DIR / f'wilcoxon_{db}_pairwise.png',
        )

        # --- Save results to CSV ---
        hg_results.to_csv(OUTPUT_DIR / f'wilcoxon_{db}_hypergraph.csv', index=False)
        pw_results.to_csv(OUTPUT_DIR / f'wilcoxon_{db}_pairwise.csv', index=False)

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()