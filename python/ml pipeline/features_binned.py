import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =======================================================
# CONFIG  (paths and file names mirror cp_model_sweep.py;
# run from the same folder as that script)
# =======================================================
DATA_DIR = Path("../../data/lookup_tables/cp/")
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

# Legend names (drug target negatives are unlabelled, not confirmed negatives)
CLASS_DISPLAY = {
    'Essential':     'Essential',
    'Non-essential': 'Non-essential',
    'Drug_target':   'Drug target',
    'Non_target':    'Not annotated',
}

# Rows of the figure, top to bottom. Pairwise rows first, then the
# hypergraph rows that split pairwise degree into its two parts.
# Bins are right-closed: (low, high]. Half-values from medians (e.g. 2.5)
# fall into the higher bin. 'show' limits which bins are drawn.
FEATURE_BINS = {
    'pair_Degree': dict(
        title="Pairwise: number of partners",
        bins=[-1, 0, 2, 5, 10, 30, np.inf],
        labels=["0", "1–2", "3–5", "6–10", "11–30", "31+"],
    ),
    'pair_LocalClustCoeff': dict(
        title="Pairwise: clustering coefficient",
        bins=[-0.001, 0, 0.5, 0.999999, 1],
        labels=["0", ">0 to 0.5", ">0.5 to <1", "1"],
    ),
    'protein_MedComplexNodes': dict(
        title="Hypergraph: complex size (median proteins)",
        bins=[0, 2, 3, 5, 9, np.inf],
        labels=["1–2", "3", "4–5", "6–9", "10+"],
    ),
    'base_Degree': dict(
        title="Hypergraph: number of complexes",
        bins=[0, 1, 2, 4, np.inf],
        labels=["1", "2", "3–4", "5+"],
    ),
    'stoich_Recorded': dict(
        title="Hb-graph: stoichiometry recorded",
        bins=[-0.5, 0.5, 1.5],
        labels=["No", "Yes"],
        show=["Yes"],
    ),
}

POS_COLOUR = "#C0392B"
NEG_COLOUR = "#95A5A6"

OUTPUT_DIR = Path("./feature_distributions/by_label")
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
feats = list(FEATURE_BINS)
features[feats] = features[feats].fillna(0)   # as in the model script


def load_labels(splits_file, label_map):
    df = pd.read_csv(DATA_DIR / splits_file).rename(
        columns={'UniProt_AC': 'ProteinId'})
    df = df[df['label_mask']].drop_duplicates('ProteinId')
    df = df[['ProteinId', 'protein_label']].copy()
    df['target'] = df['protein_label'].map(label_map)
    df = df.dropna(subset=['target'])
    df['target'] = df['target'].astype(int)
    return df


task_data, class_names = {}, {}
for task_name, cfg in TASKS.items():
    labels = load_labels(cfg["SPLITS_FILE"], cfg["LABEL_MAP"])
    task_data[task_name] = features.merge(labels, on='ProteinId', how='inner')
    inv = {v: CLASS_DISPLAY.get(k, k) for k, v in cfg["LABEL_MAP"].items()}
    class_names[task_name] = (inv[1], inv[0])

# =======================================================
# Binned percentages and figure
# =======================================================
rows = []
n_feat, n_task = len(FEATURE_BINS), len(task_data)
fig, axes = plt.subplots(n_feat, n_task, figsize=(5 * n_task, 3.4 * n_feat),
                         squeeze=False)

for i, (feat, spec) in enumerate(FEATURE_BINS.items()):
    show = spec.get('show', spec['labels'])
    for j, (task_name, df) in enumerate(task_data.items()):
        ax = axes[i, j]
        cats = pd.cut(df[feat], bins=spec['bins'], labels=spec['labels'],
                      include_lowest=True)
        pct = (pd.crosstab(cats, df['target'], normalize='columns') * 100
               ).reindex(spec['labels']).fillna(0)
        n_pos = int((df['target'] == 1).sum())
        n_neg = int((df['target'] == 0).sum())
        pos_name, neg_name = class_names[task_name]

        x, w = np.arange(len(show)), 0.4
        ax.bar(x - w / 2, pct.loc[show, 1], w, color=POS_COLOUR,
               label=f"{pos_name} (n={n_pos})")
        ax.bar(x + w / 2, pct.loc[show, 0], w, color=NEG_COLOUR,
               label=f"{neg_name} (n={n_neg})")
        ax.set_xticks(x)
        ax.set_xticklabels(show)
        if 'show' in spec:
            ax.set_xlim(-1, len(show))
        ax.set_ylabel("% of class" if j == 0 else "")
        ax.set_title(f"{task_name}\n{spec['title']}" if i == 0
                     else spec['title'], fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        ax.set_axisbelow(True)
        if i == n_feat - 1:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
                      frameon=False, ncol=1)

        for lab in spec['labels']:
            rows.append({'task': task_name, 'feature': feat, 'bin': lab,
                         'pct_pos': round(pct.loc[lab, 1], 1),
                         'pct_neg': round(pct.loc[lab, 0], 1)})

fig.suptitle("Protein properties by label, CP", y=1.01)
fig.tight_layout()
out = OUTPUT_DIR / "binned_by_label.png"
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {out}")

binned_df = pd.DataFrame(rows)
binned_df.to_csv(OUTPUT_DIR / 'binned_by_label.csv', index=False)
print(f"Saved: {OUTPUT_DIR / 'binned_by_label.csv'}\n")
with pd.option_context('display.max_rows', None, 'display.width', 200):
    print(binned_df.to_string(index=False))

# Numbers for the text: median (or % recorded) per class
print("\nSummary for the text (positive vs negative class):")
for task_name, df in task_data.items():
    pos_name, neg_name = class_names[task_name]
    print(f"\n  {task_name}: {pos_name} vs {neg_name}")
    for feat, spec in FEATURE_BINS.items():
        pos = df.loc[df['target'] == 1, feat]
        neg = df.loc[df['target'] == 0, feat]
        if feat == 'stoich_Recorded':
            print(f"    {spec['title']:<45s} {100 * pos.mean():.0f}% vs "
                  f"{100 * neg.mean():.0f}%")
        else:
            print(f"    {spec['title']:<45s} median {pos.median():g} vs "
                  f"{neg.median():g}")