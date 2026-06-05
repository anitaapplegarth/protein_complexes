"""
Split Difficulty Deep-Dive Analysis
====================================
Three rigorous analyses to explain why some splits are harder than others:

  1. Feature-space distance  — For each test drug target, compute the distance
                               to its nearest training drug target. Harder splits
                               should have higher mean nearest-neighbour distance.

  2. KS-test on positives    — Two-sample Kolmogorov-Smirnov test comparing the
                               feature distribution of training positives vs test
                               positives within each split. A significant KS stat
                               means the model is asked to extrapolate.

  3. Positive novelty        — What fraction of test drug targets come from a
                               structural family with NO drug target in training?
                               This is the most direct measure of extrapolation.

Outputs (written to OUTPUT_DIR):
  split_difficulty_distance.csv      — per-split mean/median NN distance
  split_difficulty_ks.csv            — per-split per-feature KS statistics
  split_difficulty_novelty.csv       — per-split positive novelty fractions
  split_difficulty_deep_dive.png     — summary figure (3 panels)
  split_difficulty_deep_dive.txt     — text summary with Spearman correlations

Usage:
    python split_difficulty_deep_dive.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import kstest, spearmanr
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
})

# =======================================================
# CONFIG — update to match your main pipeline
# =======================================================
DATA_DIR   = Path("/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/cp/")
OUTPUT_DIR = Path("./randomforest/hpa_two_excl_stoich_features/drug_target_family_splits")

SPLITS_FILE   = DATA_DIR / "hpa_protein_merged_splits.csv"
FEATURES_FILE = DATA_DIR / "hypergraph_features.csv"
RESULTS_FILE  = OUTPUT_DIR / "split_results.csv"

# Family column: try these in order, use whichever exists in splits_df
FAMILY_COL_CANDIDATES = ['foldseek_cluster', 'group_id', 'family_id']

# Features used for distance and KS analyses.
# Keep only those that are biologically meaningful for drug target prediction;
# drop features that are trivially identical across all splits (e.g. global stats).
ANALYSIS_FEATURES = [
    'base_Degree',
    'base_UniquePartners',
    'protein_MedComplexNodes',
    'protein_MedianUniqueRatio',
    'stoich_MedComplexSize',
    'stoich_MedianRatio',
    'stoich_RangeComplexSize',
    'stoich_RangeRatio',
    'protein_RangeComplexNodes',
    'protein_RangeUniqueRatio',
]

# =======================================================
# LOAD DATA
# =======================================================
print("Loading data...")
splits_df   = pd.read_csv(SPLITS_FILE).rename(columns={'UniProt_AC': 'ProteinId'})
features_df = pd.read_csv(FEATURES_FILE)
results_df  = pd.read_csv(RESULTS_FILE)

label_map = {'Drug_target': 1, 'Non_target': 0}
splits_df['target'] = splits_df['protein_label'].map(label_map)

# Detect family column
family_col = next((c for c in FAMILY_COL_CANDIDATES if c in splits_df.columns), None)
if family_col:
    print(f"   Family column detected: '{family_col}'")
else:
    print("   WARNING: No family column found — novelty analysis will be skipped.")

# Merge features onto splits
splits_feat = pd.merge(splits_df, features_df, on='ProteinId', how='inner')

# Restrict to features actually present
active_features = [f for f in ANALYSIS_FEATURES if f in splits_feat.columns]
missing = [f for f in ANALYSIS_FEATURES if f not in splits_feat.columns]
if missing:
    print(f"   WARNING: {len(missing)} features not found, skipping: {missing}")
print(f"   Active features ({len(active_features)}): {active_features}\n")

split_indices = sorted(results_df['split_index'].unique())

# =======================================================
# ANALYSIS 1: NEAREST-NEIGHBOUR DISTANCE
# =======================================================
# For each test drug target, find its nearest training drug target in
# standardised feature space (Euclidean). Summarise per split.
# Intuition: if test positives are far from any training positive, the model
# is extrapolating and will likely perform poorly.
print("Analysis 1: Nearest-neighbour distance (test positives → training positives)...")

distance_records = []

for split_idx in split_indices:
    mask = splits_feat['split_index'] == split_idx

    train_pos = splits_feat[mask & (splits_feat['split'] == 'train')
                            & (splits_feat['target'] == 1)
                            & (splits_feat['label_mask'] == True)][active_features].dropna()

    test_pos  = splits_feat[mask & (splits_feat['split'] == 'test')
                            & (splits_feat['target'] == 1)
                            & (splits_feat['label_mask'] == True)][active_features].dropna()

    pr_auc = results_df.loc[results_df['split_index'] == split_idx, 'hypergraph_pr_auc'].values[0]

    if len(train_pos) == 0 or len(test_pos) == 0:
        print(f"   Split {split_idx}: insufficient positives, skipping.")
        continue

    # Standardise using training positive statistics only (no leakage)
    scaler = StandardScaler().fit(train_pos)
    train_scaled = scaler.transform(train_pos)
    test_scaled  = scaler.transform(test_pos)

    # Pairwise distances: shape (n_test_pos, n_train_pos)
    dists = cdist(test_scaled, train_scaled, metric='euclidean')

    nn_distances = dists.min(axis=1)   # nearest neighbour for each test positive

    distance_records.append({
        'split_index':          split_idx,
        'hypergraph_pr_auc':    pr_auc,
        'n_train_pos':          len(train_pos),
        'n_test_pos':           len(test_pos),
        'nn_dist_mean':         float(nn_distances.mean()),
        'nn_dist_median':       float(np.median(nn_distances)),
        'nn_dist_std':          float(nn_distances.std()),
        'nn_dist_max':          float(nn_distances.max()),
        # Fraction of test positives that are "far" from any training positive
        # (threshold = 75th percentile of all NN distances across all splits —
        #  computed after the loop and appended below)
        '_nn_distances_raw':    nn_distances,
    })

dist_df = pd.DataFrame(distance_records)

# Compute global 75th-percentile threshold for "far" classification
all_nn = np.concatenate([r['_nn_distances_raw'] for r in distance_records])
far_threshold = np.percentile(all_nn, 75)
print(f"   'Far' threshold (75th pct of all NN distances): {far_threshold:.3f}")

dist_df['frac_far_positives'] = [
    float((r['_nn_distances_raw'] > far_threshold).mean())
    for r in distance_records
]
dist_df = dist_df.drop(columns=['_nn_distances_raw'])

rho_mean,   p_mean   = spearmanr(dist_df['nn_dist_mean'],   dist_df['hypergraph_pr_auc'])
rho_median, p_median = spearmanr(dist_df['nn_dist_median'], dist_df['hypergraph_pr_auc'])
rho_far,    p_far    = spearmanr(dist_df['frac_far_positives'], dist_df['hypergraph_pr_auc'])
print(f"   Spearman(nn_dist_mean,   PR-AUC): ρ={rho_mean:+.3f}  p={p_mean:.4f}")
print(f"   Spearman(nn_dist_median, PR-AUC): ρ={rho_median:+.3f}  p={p_median:.4f}")
print(f"   Spearman(frac_far_pos,   PR-AUC): ρ={rho_far:+.3f}  p={p_far:.4f}\n")

dist_csv = OUTPUT_DIR / 'split_difficulty_distance.csv'
dist_df.to_csv(dist_csv, index=False)
print(f"   Saved: {dist_csv}")

# =======================================================
# ANALYSIS 2: KS TEST — TRAIN vs TEST POSITIVE DISTRIBUTIONS
# =======================================================
# For each split and each feature, run a two-sample KS test comparing the
# distribution of that feature among training positives vs test positives.
# A large KS statistic means the test positives look different from training
# positives in that feature dimension — the model must extrapolate.
print("\nAnalysis 2: KS test — training positives vs test positives per feature...")

ks_records = []

for split_idx in split_indices:
    mask = splits_feat['split_index'] == split_idx
    pr_auc = results_df.loc[results_df['split_index'] == split_idx, 'hypergraph_pr_auc'].values[0]

    train_pos = splits_feat[mask & (splits_feat['split'] == 'train')
                            & (splits_feat['target'] == 1)
                            & (splits_feat['label_mask'] == True)]
    test_pos  = splits_feat[mask & (splits_feat['split'] == 'test')
                            & (splits_feat['target'] == 1)
                            & (splits_feat['label_mask'] == True)]

    rec = {'split_index': split_idx, 'hypergraph_pr_auc': pr_auc}

    for feat in active_features:
        a = train_pos[feat].dropna().values
        b = test_pos[feat].dropna().values
        if len(a) >= 3 and len(b) >= 3:
            # Two-sample KS test via kstest on empirical CDF
            from scipy.stats import ks_2samp
            stat, pval = ks_2samp(a, b)
            rec[f'{feat}_ks_stat'] = float(stat)
            rec[f'{feat}_ks_p']    = float(pval)
        else:
            rec[f'{feat}_ks_stat'] = float('nan')
            rec[f'{feat}_ks_p']    = float('nan')

    # Summary: mean KS stat across all features (higher = more distributional shift)
    ks_stats = [rec[f'{f}_ks_stat'] for f in active_features if not np.isnan(rec.get(f'{f}_ks_stat', float('nan')))]
    rec['mean_ks_stat'] = float(np.mean(ks_stats)) if ks_stats else float('nan')

    ks_records.append(rec)

ks_df = pd.DataFrame(ks_records)

rho_ks, p_ks = spearmanr(ks_df['mean_ks_stat'].dropna(), 
                          ks_df.loc[ks_df['mean_ks_stat'].notna(), 'hypergraph_pr_auc'])
print(f"   Spearman(mean_ks_stat, PR-AUC): ρ={rho_ks:+.3f}  p={p_ks:.4f}")

# Per-feature correlations with PR-AUC
print("   Per-feature KS stat correlations with PR-AUC:")
ks_feat_corrs = []
for feat in active_features:
    col = f'{feat}_ks_stat'
    if col in ks_df.columns:
        valid = ks_df[[col, 'hypergraph_pr_auc']].dropna()
        if len(valid) >= 5:
            rho, p = spearmanr(valid[col], valid['hypergraph_pr_auc'])
            ks_feat_corrs.append({'feature': feat, 'rho': rho, 'p': p})
            print(f"     {feat:<40} ρ={rho:+.3f}  p={p:.4f}")

ks_csv = OUTPUT_DIR / 'split_difficulty_ks.csv'
ks_df.to_csv(ks_csv, index=False)
print(f"   Saved: {ks_csv}")

# =======================================================
# ANALYSIS 3: POSITIVE NOVELTY
# =======================================================
# For each split, what fraction of test drug targets belong to a structural
# family that has NO drug target in the training set?
# This is the most direct measure of whether the model must generalise to
# entirely unseen positive-class families.
print("\nAnalysis 3: Positive novelty (test positives in families with no training positive)...")

novelty_records = []

if family_col:
    for split_idx in split_indices:
        mask = splits_feat['split_index'] == split_idx
        pr_auc = results_df.loc[results_df['split_index'] == split_idx, 'hypergraph_pr_auc'].values[0]

        train_pos = splits_feat[mask & (splits_feat['split'] == 'train')
                                & (splits_feat['target'] == 1)
                                & (splits_feat['label_mask'] == True)]
        test_pos  = splits_feat[mask & (splits_feat['split'] == 'test')
                                & (splits_feat['target'] == 1)
                                & (splits_feat['label_mask'] == True)]

        train_pos_families = set(train_pos[family_col].dropna().unique())
        test_pos_families  = test_pos[family_col].dropna()

        n_test_pos_total  = len(test_pos)
        n_novel           = int((~test_pos_families.isin(train_pos_families)).sum())
        frac_novel        = float(n_novel / n_test_pos_total) if n_test_pos_total > 0 else float('nan')

        # Also count families, not just proteins
        novel_families    = set(test_pos_families[~test_pos_families.isin(train_pos_families)].unique())
        n_novel_families  = len(novel_families)
        n_test_pos_families = len(set(test_pos_families.unique()))

        novelty_records.append({
            'split_index':           split_idx,
            'hypergraph_pr_auc':     pr_auc,
            'n_train_pos':           len(train_pos),
            'n_test_pos':            n_test_pos_total,
            'n_train_pos_families':  len(train_pos_families),
            'n_test_pos_families':   n_test_pos_families,
            'n_novel_pos_proteins':  n_novel,
            'n_novel_pos_families':  n_novel_families,
            'frac_novel_pos_proteins': frac_novel,
            'frac_novel_pos_families': float(n_novel_families / n_test_pos_families)
                                       if n_test_pos_families > 0 else float('nan'),
        })

    novelty_df = pd.DataFrame(novelty_records)

    rho_nov_prot, p_nov_prot = spearmanr(novelty_df['frac_novel_pos_proteins'], novelty_df['hypergraph_pr_auc'])
    rho_nov_fam,  p_nov_fam  = spearmanr(novelty_df['frac_novel_pos_families'], novelty_df['hypergraph_pr_auc'])
    print(f"   Spearman(frac_novel_proteins, PR-AUC): ρ={rho_nov_prot:+.3f}  p={p_nov_prot:.4f}")
    print(f"   Spearman(frac_novel_families, PR-AUC): ρ={rho_nov_fam:+.3f}  p={p_nov_fam:.4f}")

    novelty_csv = OUTPUT_DIR / 'split_difficulty_novelty.csv'
    novelty_df.to_csv(novelty_csv, index=False)
    print(f"   Saved: {novelty_csv}")
else:
    novelty_df = None
    print("   Skipped (no family column).")

# =======================================================
# FIGURE
# =======================================================
print("\nGenerating figure...")

n_panels  = 3 if novelty_df is not None else 2
fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6))

def _annotate(ax, x_col, y_col, df):
    for _, row in df.iterrows():
        ax.annotate(f"{int(row['split_index'])}",
                    (row[x_col], row[y_col]),
                    fontsize=10, ha='center', va='bottom')

# Panel 1: NN distance vs PR-AUC
ax = axes[0]
ax.scatter(dist_df['nn_dist_mean'], dist_df['hypergraph_pr_auc'], s=80, zorder=3, color='#4575b4')
_annotate(ax, 'nn_dist_mean', 'hypergraph_pr_auc', dist_df)
ax.set_xlabel('Mean NN distance\n(test pos → nearest train pos, standardised)')
ax.set_ylabel('PR-AUC (hypergraph)')
ax.set_title(f'Feature-Space Distance\nρ={rho_mean:+.3f}  p={p_mean:.4f}')
ax.grid(True, alpha=0.3)

# Panel 2: Mean KS stat vs PR-AUC
ax = axes[1]
ax.scatter(ks_df['mean_ks_stat'], ks_df['hypergraph_pr_auc'], s=80, zorder=3, color='#d73027')
_annotate(ax, 'mean_ks_stat', 'hypergraph_pr_auc', ks_df)
ax.set_xlabel('Mean KS statistic\n(train pos vs test pos, across features)')
ax.set_ylabel('PR-AUC (hypergraph)')
ax.set_title(f'Distributional Shift (KS test)\nρ={rho_ks:+.3f}  p={p_ks:.4f}')
ax.grid(True, alpha=0.3)

# Panel 3: Novelty vs PR-AUC (if available)
if novelty_df is not None:
    ax = axes[2]
    ax.scatter(novelty_df['frac_novel_pos_proteins'], novelty_df['hypergraph_pr_auc'],
               s=80, zorder=3, color='#1a9641')
    _annotate(ax, 'frac_novel_pos_proteins', 'hypergraph_pr_auc', novelty_df)
    ax.set_xlabel('Fraction of test drug targets\nin families with no training drug target')
    ax.set_ylabel('PR-AUC (hypergraph)')
    ax.set_title(f'Positive Novelty\nρ={rho_nov_prot:+.3f}  p={p_nov_prot:.4f}')
    ax.grid(True, alpha=0.3)

plt.suptitle('Why Are Some Splits Harder? Three Mechanistic Analyses', y=1.02)
plt.tight_layout()
fig_path = OUTPUT_DIR / 'split_difficulty_deep_dive.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {fig_path}")

# =======================================================
# TEXT SUMMARY
# =======================================================
txt_path = OUTPUT_DIR / 'split_difficulty_deep_dive.txt'
with open(txt_path, 'w') as f:

    f.write("SPLIT DIFFICULTY DEEP-DIVE\n")
    f.write("=" * 70 + "\n\n")
    f.write("Three analyses of WHY some splits are harder to predict.\n\n")

    # --- Analysis 1 ---
    f.write("ANALYSIS 1: NEAREST-NEIGHBOUR DISTANCE\n")
    f.write("-" * 70 + "\n")
    f.write("For each test drug target, distance to nearest training drug target\n")
    f.write("in standardised feature space. Higher = model must extrapolate further.\n\n")
    f.write(f"  Spearman(nn_dist_mean,   PR-AUC): ρ={rho_mean:+.6f}  p={p_mean:.6f}\n")
    f.write(f"  Spearman(nn_dist_median, PR-AUC): ρ={rho_median:+.6f}  p={p_median:.6f}\n")
    f.write(f"  Spearman(frac_far_pos,   PR-AUC): ρ={rho_far:+.6f}  p={p_far:.6f}\n\n")
    f.write(f"  {'Split':<8} {'PR-AUC':>8} {'n_train+':>10} {'n_test+':>9} "
            f"{'NN_mean':>9} {'NN_med':>8} {'frac_far':>9}\n")
    f.write("  " + "-" * 65 + "\n")
    for _, row in dist_df.sort_values('hypergraph_pr_auc', ascending=False).iterrows():
        f.write(f"  {int(row['split_index']):<8} {row['hypergraph_pr_auc']:>8.4f} "
                f"{int(row['n_train_pos']):>10} {int(row['n_test_pos']):>9} "
                f"{row['nn_dist_mean']:>9.4f} {row['nn_dist_median']:>8.4f} "
                f"{row['frac_far_positives']:>9.4f}\n")

    # --- Analysis 2 ---
    f.write("\n\nANALYSIS 2: KS TEST — TRAINING vs TEST POSITIVE DISTRIBUTIONS\n")
    f.write("-" * 70 + "\n")
    f.write("Two-sample KS statistic per feature between training and test positives.\n")
    f.write("Higher stat = greater distributional shift = harder extrapolation.\n\n")
    f.write(f"  Spearman(mean_ks_stat, PR-AUC): ρ={rho_ks:+.6f}  p={p_ks:.6f}\n\n")
    f.write("  Per-feature correlations (KS stat vs PR-AUC):\n")
    for rec in sorted(ks_feat_corrs, key=lambda x: abs(x['rho']), reverse=True):
        f.write(f"    {rec['feature']:<40} ρ={rec['rho']:+.4f}  p={rec['p']:.4f}\n")

    f.write(f"\n  {'Split':<8} {'PR-AUC':>8} {'mean_KS':>9}\n")
    f.write("  " + "-" * 28 + "\n")
    for _, row in ks_df.sort_values('hypergraph_pr_auc', ascending=False).iterrows():
        f.write(f"  {int(row['split_index']):<8} {row['hypergraph_pr_auc']:>8.4f} "
                f"{row['mean_ks_stat']:>9.4f}\n")

    # --- Analysis 3 ---
    if novelty_df is not None:
        f.write("\n\nANALYSIS 3: POSITIVE NOVELTY\n")
        f.write("-" * 70 + "\n")
        f.write("Fraction of test drug targets in families with NO training drug target.\n")
        f.write("These are proteins the model has never seen a positive example near.\n\n")
        f.write(f"  Spearman(frac_novel_proteins, PR-AUC): ρ={rho_nov_prot:+.6f}  p={p_nov_prot:.6f}\n")
        f.write(f"  Spearman(frac_novel_families, PR-AUC): ρ={rho_nov_fam:+.6f}  p={p_nov_fam:.6f}\n\n")
        f.write(f"  {'Split':<8} {'PR-AUC':>8} {'n_test+':>9} {'n_novel_prot':>13} "
                f"{'frac_novel':>11} {'frac_nov_fam':>13}\n")
        f.write("  " + "-" * 68 + "\n")
        for _, row in novelty_df.sort_values('hypergraph_pr_auc', ascending=False).iterrows():
            f.write(f"  {int(row['split_index']):<8} {row['hypergraph_pr_auc']:>8.4f} "
                    f"{int(row['n_test_pos']):>9} {int(row['n_novel_pos_proteins']):>13} "
                    f"{row['frac_novel_pos_proteins']:>11.4f} "
                    f"{row['frac_novel_pos_families']:>13.4f}\n")

    f.write("\n\n" + "=" * 70 + "\n")
    f.write("INTERPRETATION GUIDE\n")
    f.write("=" * 70 + "\n")
    f.write("""
If NN distance correlates negatively with PR-AUC:
  → Harder splits contain drug targets that are structurally unlike any
    drug target seen in training. The model is interpolating well within
    the training distribution but struggling to extrapolate.

If KS stat correlates negatively with PR-AUC:
  → The feature distributions of test drug targets diverge from training
    drug targets in harder splits. Specific features with high KS–PR-AUC
    correlation tell you WHICH dimensions drive the difficulty.

If novelty fraction correlates negatively with PR-AUC:
  → The primary driver is the family-level split design: harder splits
    put drug targets from structurally novel families in the test set,
    leaving no related training signal. This is the expected behaviour
    under your Foldseek-based splitting scheme and validates that your
    evaluation is genuinely testing generalisation.

If none of the above correlate significantly:
  → Split difficulty is likely driven by stochastic variation in which
    proteins (not families) end up in the test set, rather than any
    systematic structural property. With n=15 splits you are underpowered
    to detect modest effects (|ρ| < 0.5 requires n ≥ ~20 for p < 0.05).
""")

print(f"   Saved: {txt_path}")
print("\nDone.")