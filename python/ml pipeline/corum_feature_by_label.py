import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score

# =======================================================
# CONFIG  (paths and file names mirror corum_model_sweep.py;
# run from the same folder as that script)
# =======================================================
DATA_DIR = Path("../../data/lookup_tables/corum/")
PROTEIN_FEATURES_FILE  = "hypergraph_features.csv"
PAIRWISE_FEATURES_FILE = "pairwise_features.csv"

TASKS = {
    "Essentiality": dict(
        SPLITS_FILE="ess_protein_splits.csv",
        LABEL_MAP={'Essential': 1, 'Non-essential': 0},
    ),
    "Drug target (HPA)": dict(
        SPLITS_FILE="hpa_protein_splits.csv",
        LABEL_MAP={'Drug_target': 1, 'Non_target': 0},
    ),
    "Drug target (ChEMBL)": dict(
        SPLITS_FILE="chembl_protein_splits.csv",
        LABEL_MAP={'Drug_target': 1, 'Non_target': 0},
    ),
}

# Same features as the model tiers in corum_model_sweep.py
RECORDED_FLAG = 'stoich_Recorded'
FEATURE_GROUPS = {
    "Base (hypergraph)": [
        'base_Degree', 'base_LocalClustCoeff', 'base_TriangleCount',
        'base_UniquePartners', 'base_AvgNeighbourDegree',
    ],
    "Stoichiometry": [
        'stoich_Recorded',
        'stoich_WeightedTriangles', 'stoich_AvgNeighbourDegreeStoich',
        'stoich_RangeComplexSize', 'stoich_MedComplexSize',
        'stoich_MedianRatio', 'stoich_RangeRatio',
    ],
    "Protein-level": [
        'protein_RangeUniqueRatio', 'protein_MedComplexNodes',
        'protein_RangeComplexNodes',
    ],
    "Pairwise": [
        'pair_Degree', 'pair_LocalClustCoeff',
        'pair_TriangleCount', 'pair_AvgNeighborDegree',
    ],
}

# Stoichiometry value features are analysed two ways:
#   "recorded_only" : only proteins with stoich_Recorded == 1
#   "pooled"        : all proteins, unrecorded zeros included (what the
#                     models see)
# The flag itself is always analysed on all proteins.
STOICH_MODES = ["recorded_only", "pooled"]
STOICH_FEATURES = set(FEATURE_GROUPS["Stoichiometry"]) - {RECORDED_FLAG}

MODE_LABELS = {
    "all":           "all",
    "recorded_only": "recorded only",
    "pooled":        "pooled, zeros included",
}

N_BOOT         = 1000    # bootstrap resamples for AUROC CIs
RANDOM_STATE   = 42
REDUNDANCY_RHO = 0.95    # report feature pairs with |Spearman rho| >= this

TASK_COLOURS = ["#1B9E77", "#D95F02", "#7570B3"]
TASK_MARKERS = ["o", "s", "^"]

OUTPUT_DIR = Path("./feature_distributions/corum/by_label")
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

# =======================================================
# Load data
# =======================================================
hg   = pd.read_csv(DATA_DIR / PROTEIN_FEATURES_FILE)
pair = pd.read_csv(DATA_DIR / PAIRWISE_FEATURES_FILE)
features = pd.merge(hg, pair, on='ProteinId', how='inner')

all_feats = [f for fs in FEATURE_GROUPS.values() for f in fs]
missing = [f for f in all_feats if f not in features.columns]
if missing:
    raise KeyError(f"Features absent from the feature files: {missing}")
features[all_feats] = features[all_feats].fillna(0)   # as in the model script
print(f"Features merged: {features.shape}")


def load_labels(splits_file, label_map):
    df = pd.read_csv(DATA_DIR / splits_file).rename(
        columns={'UniProt_AC': 'ProteinId'})

    n_varying = (df.groupby('ProteinId')['label_mask'].nunique() > 1).sum()
    print(f"  {splits_file}: {n_varying} proteins with label_mask varying "
          f"across splits")

    df = df[df['label_mask']].drop_duplicates('ProteinId')
    df = df[['ProteinId', 'protein_label']].copy()
    df['target'] = df['protein_label'].map(label_map)
    n_unmapped = df['target'].isna().sum()
    if n_unmapped:
        print(f"  WARNING: {n_unmapped} labels not in LABEL_MAP; dropped. "
              f"Values: {df.loc[df['target'].isna(), 'protein_label'].unique()}")
        df = df.dropna(subset=['target'])
    df['target'] = df['target'].astype(int)
    return df


print("\nLoading labels")
task_data, class_names = {}, {}
for task_name, cfg in TASKS.items():
    labels = load_labels(cfg["SPLITS_FILE"], cfg["LABEL_MAP"])
    df = features.merge(labels, on='ProteinId', how='inner')
    task_data[task_name] = df
    inv = {v: k.replace('_', ' ') for k, v in cfg["LABEL_MAP"].items()}
    class_names[task_name] = (inv[1], inv[0])
    print(f"  {task_name}: n+ = {(df['target'] == 1).sum()}, "
          f"n- = {(df['target'] == 0).sum()}")

# =======================================================
# Redundancy check: near-identical features (rank correlation)
# =======================================================
rho = features[all_feats].corr(method='spearman')
print(f"\nFeature pairs with |Spearman rho| >= {REDUNDANCY_RHO} (all proteins):")
found = False
for i, a in enumerate(all_feats):
    for b in all_feats[i + 1:]:
        if abs(rho.loc[a, b]) >= REDUNDANCY_RHO:
            print(f"  {a:<34s} {b:<34s} rho = {rho.loc[a, b]:.3f}")
            found = True
if not found:
    print("  none")
rho.to_csv(OUTPUT_DIR / 'feature_spearman.csv')


# =======================================================
# Helpers
# =======================================================
def modes_for(feat):
    return STOICH_MODES if feat in STOICH_FEATURES else ["all"]


def analysis_subset(df, feat, mode):
    if feat in STOICH_FEATURES and mode == "recorded_only":
        return df[df[RECORDED_FLAG] == 1]
    return df


def unit_label(feat, mode):
    return feat if mode == "all" else f"{feat} ({MODE_LABELS[mode]})"


FEAT_GROUP = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}
UNITS = [(f, m) for f in all_feats for m in modes_for(f)]

rng = np.random.default_rng(RANDOM_STATE)


def boot_auroc_ci(y, x, n_boot):
    """Stratified bootstrap (resample each class separately) 95% CI."""
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan, np.nan
    aucs = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)),
                              rng.choice(neg, len(neg))])
        aucs[b] = roc_auc_score(y[idx], x[idx])
    return tuple(np.percentile(aucs, [2.5, 97.5]))


def fmt(v):
    return "–" if pd.isna(v) else f"{v:.3g}"


# =======================================================
# Summary statistics
# =======================================================
print(f"\nComputing AUROCs with {N_BOOT} bootstrap resamples...")
rows = []
for task_name, df in task_data.items():
    for feat, mode in UNITS:
        sub = analysis_subset(df, feat, mode)
        y, x = sub['target'].values, sub[feat].values.astype(float)
        row = {
            'task':    task_name,
            'group':   FEAT_GROUP[feat],
            'feature': feat,
            'subset':  MODE_LABELS[mode],
            'n_pos':   int((y == 1).sum()),
            'n_neg':   int((y == 0).sum()),
        }
        for target, tag in [(1, 'pos'), (0, 'neg')]:
            v = x[y == target]
            q25, q50, q75 = (np.percentile(v, [25, 50, 75]) if len(v)
                             else (np.nan,) * 3)
            row[f'median_{tag}']   = q50
            row[f'q25_{tag}']      = q25
            row[f'q75_{tag}']      = q75
            row[f'pct_zero_{tag}'] = 100 * np.mean(v == 0) if len(v) else np.nan
        row['auroc'] = (roc_auc_score(y, x) if len(np.unique(y)) == 2
                        else np.nan)
        row['auroc_lo'], row['auroc_hi'] = boot_auroc_ci(y, x, N_BOOT)
        rows.append(row)

stats_df = pd.DataFrame(rows)
stats_df.to_csv(OUTPUT_DIR / 'feature_by_label_summary.csv', index=False)
print(f"Saved: {OUTPUT_DIR / 'feature_by_label_summary.csv'}")

# Readable table: one row per (task, feature, subset)
table = pd.DataFrame({
    'Task':    stats_df['task'],
    'Feature': [unit_label(f, m) for f, m in
                zip(stats_df['feature'],
                    [{v: k for k, v in MODE_LABELS.items()}[s]
                     for s in stats_df['subset']])],
    'n+ / n-': stats_df['n_pos'].astype(str) + " / " + stats_df['n_neg'].astype(str),
    'Positive: median [IQR]': [f"{fmt(m)} [{fmt(a)}, {fmt(b)}]" for m, a, b in
                               zip(stats_df['median_pos'], stats_df['q25_pos'],
                                   stats_df['q75_pos'])],
    'Negative: median [IQR]': [f"{fmt(m)} [{fmt(a)}, {fmt(b)}]" for m, a, b in
                               zip(stats_df['median_neg'], stats_df['q25_neg'],
                                   stats_df['q75_neg'])],
    '% zero (pos / neg)': [f"{p:.0f} / {n:.0f}" for p, n in
                           zip(stats_df['pct_zero_pos'], stats_df['pct_zero_neg'])],
    'AUROC [95% CI]': [f"{a:.2f} [{lo:.2f}, {hi:.2f}]" for a, lo, hi in
                       zip(stats_df['auroc'], stats_df['auroc_lo'],
                           stats_df['auroc_hi'])],
})
table.to_csv(OUTPUT_DIR / 'feature_by_label_table.csv', index=False)
print(f"Saved: {OUTPUT_DIR / 'feature_by_label_table.csv'}")

with pd.option_context('display.max_rows', None, 'display.width', 250,
                       'display.max_colwidth', 60):
    for task_name in task_data:
        pos_name, neg_name = class_names[task_name]
        print(f"\n=== {task_name}: positive = {pos_name}, "
              f"negative = {neg_name} ===")
        print(table[table['Task'] == task_name].drop(columns='Task')
              .to_string(index=False))

# =======================================================
# Dot plot: AUROC with 95% CI, one row per feature, one marker per task
# =======================================================
task_names = list(task_data.keys())
lookup = stats_df.set_index(['task', 'feature', 'subset'])
ypos = np.arange(len(UNITS))
offsets = np.linspace(-0.22, 0.22, len(task_names))

fig, ax = plt.subplots(figsize=(10, 0.5 * len(UNITS) + 2))
for k, t in enumerate(task_names):
    a  = np.array([lookup.loc[(t, f, MODE_LABELS[m]), 'auroc']    for f, m in UNITS])
    lo = np.array([lookup.loc[(t, f, MODE_LABELS[m]), 'auroc_lo'] for f, m in UNITS])
    hi = np.array([lookup.loc[(t, f, MODE_LABELS[m]), 'auroc_hi'] for f, m in UNITS])
    ax.errorbar(a, ypos + offsets[k], xerr=[a - lo, hi - a],
                fmt=TASK_MARKERS[k], color=TASK_COLOURS[k], markersize=7,
                capsize=2, linewidth=1.2, label=t)

ax.axvline(0.5, color='black', linewidth=1, linestyle='--')

# Separators between feature groups
prev_group = None
for r, (f, m) in enumerate(UNITS):
    g = FEAT_GROUP[f]
    if prev_group is not None and g != prev_group:
        ax.axhline(r - 0.5, color='grey', linewidth=0.8)
    prev_group = g

ax.set_yticks(ypos)
ax.set_yticklabels([unit_label(f, m) for f, m in UNITS])
ax.invert_yaxis()
ax.set_xlabel("Single-feature AUROC (95% bootstrap CI)\n"
              "< 0.5: lower in positive class     > 0.5: higher in positive class")
ax.set_title("Single-feature AUROC by task, CORUM")
ax.legend(loc='lower right', bbox_to_anchor=(1.0, 1.04), ncol=len(task_names),
          frameon=False)
ax.grid(axis='x', alpha=0.3)
fig.tight_layout()
out = OUTPUT_DIR / "auroc_dotplot.png"
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved: {out}")

# =======================================================
# Heatmap of single-feature AUROC
# =======================================================
heat = np.array([[lookup.loc[(t, f, MODE_LABELS[m]), 'auroc']
                  for t in task_names] for f, m in UNITS])
d = np.nanmax(np.abs(heat - 0.5))
fig, ax = plt.subplots(figsize=(5 + 2.2 * len(task_names), 0.5 * len(UNITS) + 2))
im = ax.imshow(heat, cmap='RdBu_r', vmin=0.5 - d, vmax=0.5 + d, aspect='auto')
ax.set_xticks(range(len(task_names)))
ax.set_xticklabels(task_names, rotation=20, ha='right')
ax.set_yticks(range(len(UNITS)))
ax.set_yticklabels([unit_label(f, m) for f, m in UNITS])
for r in range(heat.shape[0]):
    for c in range(heat.shape[1]):
        if not np.isnan(heat[r, c]):
            ax.text(c, r, f"{heat[r, c]:.2f}", ha='center', va='center',
                    fontsize=12)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Single-feature AUROC\n(> 0.5: higher in positive class)")
ax.set_title("Single-feature AUROC by task, CORUM")
fig.tight_layout()
out = OUTPUT_DIR / "auroc_heatmap.png"
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {out}")