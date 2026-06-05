"""
Analyse why some splits are harder than others.

Compares the test-set composition of the best and worst performing splits
to identify what makes a split easy or hard to predict.

Usage:
    python analyse_split_difficulty.py

Reads from:
    - CONFIG DATA_DIR / SPLITS_FILE  (split assignments)
    - CONFIG DATA_DIR / PROTEIN_FEATURES_FILE  (hypergraph features)
    - CONFIG BASE_OUTPUT_DIR / drug_target_family_splits / split_results.csv
    - CONFIG BASE_OUTPUT_DIR / drug_target_family_splits / hypergraph_predictions.csv

Outputs:
    - split_difficulty_analysis.txt   (text summary)
    - split_difficulty_analysis.png   (figure)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})

# =======================================================
# CONFIG — update these to match your main pipeline
# =======================================================
DATA_DIR    = Path("/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/cp/")
OUTPUT_DIR  = Path("./randomforest/hpa_two_excl_stoich_features/drug_target_family_splits")

SPLITS_FILE    = DATA_DIR / "hpa_protein_merged_splits.csv"
FEATURES_FILE  = DATA_DIR / "hypergraph_features.csv"
RESULTS_FILE   = OUTPUT_DIR / "split_results.csv"
PREDS_FILE     = OUTPUT_DIR / "hypergraph_predictions.csv"

# Top features from your permutation importance (used for profiling)
KEY_FEATURES = [
    'base_Degree',
    'protein_MedComplexNodes',
    'stoich_MedComplexSize',
    'protein_MedianUniqueRatio',
    'stoich_MedianRatio',
]

# =======================================================
# LOAD DATA
# =======================================================
print("Loading data...")
splits_df   = pd.read_csv(SPLITS_FILE)
splits_df   = splits_df.rename(columns={'UniProt_AC': 'ProteinId'})
features_df = pd.read_csv(FEATURES_FILE)
results_df  = pd.read_csv(RESULTS_FILE)
preds_df    = pd.read_csv(PREDS_FILE)

# Encode labels
label_map = {'Drug_target': 1, 'Non_target': 0}
splits_df['target'] = splits_df['protein_label'].map(label_map)

# Merge features onto splits for profiling
splits_feat = pd.merge(splits_df, features_df, on='ProteinId', how='inner')

# =======================================================
# PER-SPLIT TEST SET PROFILING
# =======================================================
print("Profiling test sets...\n")

split_indices = sorted(results_df['split_index'].values)
records = []

for idx in split_indices:
    mask = (splits_feat['split_index'] == idx) & (splits_feat['split'] == 'test')
    test = splits_feat[mask].copy()
    labelled = test[test['label_mask'] == True]

    pr_auc = results_df.loc[results_df['split_index'] == idx, 'hypergraph_pr_auc'].values[0]

    rec = {
        'split_index': idx,
        'hypergraph_pr_auc': pr_auc,
        'n_test': len(labelled),
        'n_positive': int(labelled['target'].sum()),
        'pos_rate': float(labelled['target'].mean()) if len(labelled) > 0 else 0,
    }

    # Structural family / group info (if available)
    if 'group_id' in test.columns:
        rec['n_families_test'] = test['group_id'].nunique()
        # Median family size in test
        fam_sizes = test.groupby('group_id').size()
        rec['median_family_size'] = float(fam_sizes.median())
        rec['max_family_size'] = int(fam_sizes.max())
        rec['n_singleton_families'] = int((fam_sizes == 1).sum())
        rec['frac_in_singletons'] = float((fam_sizes == 1).sum()) / len(fam_sizes) if len(fam_sizes) > 0 else 0
    elif 'foldseek_cluster' in test.columns:
        rec['n_families_test'] = test['foldseek_cluster'].nunique()
        fam_sizes = test.groupby('foldseek_cluster').size()
        rec['median_family_size'] = float(fam_sizes.median())
        rec['max_family_size'] = int(fam_sizes.max())
        rec['n_singleton_families'] = int((fam_sizes == 1).sum())
        rec['frac_in_singletons'] = float((fam_sizes == 1).sum()) / len(fam_sizes) if len(fam_sizes) > 0 else 0

    # Feature distributions in test set (labelled proteins only)
    available_feats = [f for f in KEY_FEATURES if f in labelled.columns]
    for feat in available_feats:
        rec[f'{feat}_median'] = float(labelled[feat].median())
        rec[f'{feat}_mean'] = float(labelled[feat].mean())
        rec[f'{feat}_std'] = float(labelled[feat].std())

    # Feature distributions for POSITIVE test proteins only
    positives = labelled[labelled['target'] == 1]
    if len(positives) > 0:
        for feat in available_feats:
            rec[f'{feat}_pos_median'] = float(positives[feat].median())

    records.append(rec)

profile_df = pd.DataFrame(records)

# =======================================================
# CORRELATION ANALYSIS
# =======================================================
print("Computing correlations with PR-AUC...\n")

corr_records = []
numeric_cols = [c for c in profile_df.columns
                if c not in ('split_index', 'hypergraph_pr_auc')
                and profile_df[c].dtype in ('float64', 'int64', 'float32', 'int32')]

for col in numeric_cols:
    valid = profile_df[[col, 'hypergraph_pr_auc']].dropna()
    if len(valid) >= 5:
        rho, p = spearmanr(valid[col], valid['hypergraph_pr_auc'])
        corr_records.append({'variable': col, 'spearman_rho': rho, 'p_value': p})

corr_df = pd.DataFrame(corr_records).sort_values('spearman_rho', key=abs, ascending=False)

# =======================================================
# BEST vs WORST COMPARISON
# =======================================================
best_idx  = profile_df.loc[profile_df['hypergraph_pr_auc'].idxmax(), 'split_index']
worst_idx = profile_df.loc[profile_df['hypergraph_pr_auc'].idxmin(), 'split_index']

best_row  = profile_df[profile_df['split_index'] == best_idx].iloc[0]
worst_row = profile_df[profile_df['split_index'] == worst_idx].iloc[0]

# =======================================================
# OUTPUT — TEXT SUMMARY
# =======================================================
out_path = OUTPUT_DIR / 'split_difficulty_analysis.txt'
with open(out_path, 'w') as f:
    f.write("SPLIT DIFFICULTY ANALYSIS\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"Best split:  {int(best_idx)}  (PR-AUC = {best_row['hypergraph_pr_auc']:.4f})\n")
    f.write(f"Worst split: {int(worst_idx)}  (PR-AUC = {worst_row['hypergraph_pr_auc']:.4f})\n\n")

    f.write(f"{'Property':<40} {'Best':>10} {'Worst':>10}\n")
    f.write("-" * 62 + "\n")
    compare_cols = [c for c in profile_df.columns
                    if c not in ('split_index', 'hypergraph_pr_auc')]
    for col in compare_cols:
        bv = best_row.get(col, float('nan'))
        wv = worst_row.get(col, float('nan'))
        if isinstance(bv, float):
            f.write(f"{col:<40} {bv:>10.4f} {wv:>10.4f}\n")
        else:
            f.write(f"{col:<40} {str(bv):>10} {str(wv):>10}\n")

    f.write("\n\nCORRELATIONS WITH PR-AUC (Spearman)\n")
    f.write("=" * 70 + "\n")
    f.write(f"{'Variable':<45} {'rho':>8} {'p':>10}\n")
    f.write("-" * 65 + "\n")
    for _, row in corr_df.iterrows():
        f.write(f"{row['variable']:<45} {row['spearman_rho']:>8.4f} {row['p_value']:>10.4f}\n")

    f.write("\n\nPER-SPLIT SUMMARY\n")
    f.write("=" * 70 + "\n")
    for _, row in profile_df.sort_values('hypergraph_pr_auc', ascending=False).iterrows():
        f.write(f"Split {int(row['split_index']):>2}  PR-AUC={row['hypergraph_pr_auc']:.4f}  "
                f"n_test={int(row['n_test'])}  pos_rate={row['pos_rate']:.3f}")
        if 'n_families_test' in row and not pd.isna(row.get('n_families_test', float('nan'))):
            f.write(f"  n_fam={int(row['n_families_test'])}  med_fam_size={row['median_family_size']:.0f}")
        f.write("\n")

print(f"Saved: {out_path}\n")

# =======================================================
# OUTPUT — CSVs
# =======================================================

# 1. Per-split summary (all profile columns, sorted by PR-AUC descending)
per_split_csv = OUTPUT_DIR / 'split_difficulty_per_split.csv'
profile_df.sort_values('hypergraph_pr_auc', ascending=False).to_csv(per_split_csv, index=False)
print(f"Saved: {per_split_csv}")

# 2. Best vs Worst comparison
compare_cols = [c for c in profile_df.columns
                if c not in ('split_index', 'hypergraph_pr_auc')]
best_vs_worst_records = []
for col in compare_cols:
    bv = best_row.get(col, float('nan'))
    wv = worst_row.get(col, float('nan'))
    best_vs_worst_records.append({'property': col, 'best_split': bv, 'worst_split': wv})
best_vs_worst_df = pd.DataFrame(best_vs_worst_records)
# Prepend a header row with the split indices and PR-AUC values
header_rows = pd.DataFrame([
    {'property': f'split_index',      'best_split': int(best_idx),                        'worst_split': int(worst_idx)},
    {'property': 'hypergraph_pr_auc', 'best_split': best_row['hypergraph_pr_auc'],         'worst_split': worst_row['hypergraph_pr_auc']},
])
best_vs_worst_df = pd.concat([header_rows, best_vs_worst_df], ignore_index=True)
best_vs_worst_csv = OUTPUT_DIR / 'split_difficulty_best_vs_worst.csv'
best_vs_worst_df.to_csv(best_vs_worst_csv, index=False)
print(f"Saved: {best_vs_worst_csv}")

# 3. Correlations with PR-AUC
corr_csv = OUTPUT_DIR / 'split_difficulty_correlations.csv'
corr_df.to_csv(corr_csv, index=False)
print(f"Saved: {corr_csv}\n")

# =======================================================
# OUTPUT — FIGURE
# =======================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Panel 1: PR-AUC per split (ordered)
ax = axes[0, 0]
ordered = profile_df.sort_values('hypergraph_pr_auc')
colours = ['#d73027' if i < 3 else '#4575b4' if i >= len(ordered) - 3 else '#999999'
           for i in range(len(ordered))]
ax.barh(range(len(ordered)), ordered['hypergraph_pr_auc'], color=colours)
ax.set_yticks(range(len(ordered)))
ax.set_yticklabels([f"Split {int(s)}" for s in ordered['split_index']])
ax.set_xlabel('PR-AUC')
ax.set_title('Splits Ranked by PR-AUC')

# Panel 2: PR-AUC vs positive rate
ax = axes[0, 1]
ax.scatter(profile_df['pos_rate'], profile_df['hypergraph_pr_auc'], s=80, zorder=3)
for _, row in profile_df.iterrows():
    ax.annotate(f"{int(row['split_index'])}", (row['pos_rate'], row['hypergraph_pr_auc']),
                fontsize=11, ha='center', va='bottom')
ax.set_xlabel('Positive rate in test set')
ax.set_ylabel('PR-AUC')
ax.set_title('PR-AUC vs Test Set Positive Rate')

# Panel 3: PR-AUC vs median base_Degree in test
if 'base_Degree_median' in profile_df.columns:
    ax = axes[0, 2]
    ax.scatter(profile_df['base_Degree_median'], profile_df['hypergraph_pr_auc'], s=80, zorder=3)
    for _, row in profile_df.iterrows():
        ax.annotate(f"{int(row['split_index'])}", (row['base_Degree_median'], row['hypergraph_pr_auc']),
                    fontsize=11, ha='center', va='bottom')
    rho = corr_df.loc[corr_df['variable'] == 'base_Degree_median', 'spearman_rho']
    if len(rho) > 0:
        ax.set_title(f'PR-AUC vs Median base_Degree\n(ρ={rho.values[0]:.3f})')
    else:
        ax.set_title('PR-AUC vs Median base_Degree')
    ax.set_xlabel('Median base_Degree (test set)')
    ax.set_ylabel('PR-AUC')

# Panel 4: PR-AUC vs median protein_MedComplexNodes
if 'protein_MedComplexNodes_median' in profile_df.columns:
    ax = axes[1, 0]
    ax.scatter(profile_df['protein_MedComplexNodes_median'], profile_df['hypergraph_pr_auc'],
               s=80, zorder=3)
    for _, row in profile_df.iterrows():
        ax.annotate(f"{int(row['split_index'])}",
                    (row['protein_MedComplexNodes_median'], row['hypergraph_pr_auc']),
                    fontsize=11, ha='center', va='bottom')
    rho = corr_df.loc[corr_df['variable'] == 'protein_MedComplexNodes_median', 'spearman_rho']
    if len(rho) > 0:
        ax.set_title(f'PR-AUC vs Median Complex Size\n(ρ={rho.values[0]:.3f})')
    else:
        ax.set_title('PR-AUC vs Median Complex Size')
    ax.set_xlabel('Median protein_MedComplexNodes (test set)')
    ax.set_ylabel('PR-AUC')

# Panel 5: PR-AUC vs number of test proteins
ax = axes[1, 1]
ax.scatter(profile_df['n_test'], profile_df['hypergraph_pr_auc'], s=80, zorder=3)
for _, row in profile_df.iterrows():
    ax.annotate(f"{int(row['split_index'])}", (row['n_test'], row['hypergraph_pr_auc']),
                fontsize=11, ha='center', va='bottom')
ax.set_xlabel('Number of labelled test proteins')
ax.set_ylabel('PR-AUC')
ax.set_title('PR-AUC vs Test Set Size')

# Panel 6: Top correlations bar chart
ax = axes[1, 2]
top_corrs = corr_df.head(10)
colors = ['#d73027' if v < 0 else '#4575b4' for v in top_corrs['spearman_rho']]
ax.barh(range(len(top_corrs)), top_corrs['spearman_rho'], color=colors)
ax.set_yticks(range(len(top_corrs)))
ax.set_yticklabels(top_corrs['variable'], fontsize=11)
ax.invert_yaxis()
ax.set_xlabel('Spearman ρ with PR-AUC')
ax.set_title('Top Correlates of Split Difficulty')
ax.axvline(0, color='gray', linestyle='--', linewidth=1)

plt.suptitle('What Makes a Split Easy or Hard?', fontsize=16, y=1.02)
plt.tight_layout()
fig_path = OUTPUT_DIR / 'split_difficulty_analysis.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {fig_path}")

# =======================================================
# QUICK CONSOLE SUMMARY
# =======================================================
print(f"\n{'='*60}")
print(f"Best split:  {int(best_idx)} (PR-AUC = {best_row['hypergraph_pr_auc']:.4f})")
print(f"Worst split: {int(worst_idx)} (PR-AUC = {worst_row['hypergraph_pr_auc']:.4f})")
print(f"\nTop 5 correlates of split difficulty:")
for _, row in corr_df.head(5).iterrows():
    print(f"  {row['variable']:<40} ρ = {row['spearman_rho']:+.4f}  (p={row['p_value']:.4f})")
print(f"{'='*60}")