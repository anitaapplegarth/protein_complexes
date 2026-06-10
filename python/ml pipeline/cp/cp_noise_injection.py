import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import List, Dict

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score, f1_score
from sklearn.inspection import permutation_importance

# =======================================================
# Plotting style
# =======================================================
plt.rcParams.update({
    'font.size':        16,
    'axes.titlesize':   18,
    'axes.labelsize':   16,
    'xtick.labelsize':  14,
    'ytick.labelsize':  14,
    'legend.fontsize':  14,
    'figure.titlesize': 20,
})

# =======================================================
# TASK METADATA
# =======================================================
# Central lookup: add a new entry here to support a new task.
# SPLITS_FILE   — filename in DATA_DIR containing the split assignments
# LABEL_COL     — value of protein_label column used for positive class
# DISPLAY_NAME  — human-readable task name for reports and plot titles
TASK_META = {
    "ess": {
        "SPLITS_FILE":  "ess_protein_merged_splits.csv",
        "DISPLAY_NAME": "Gene Essentiality",
    },
    "chembl": {
        "SPLITS_FILE":  "chembl_protein_merged_splits.csv",
        "DISPLAY_NAME": "Drug Target Prediction (ChEMBL)",
    },
    "hpa": {
        "SPLITS_FILE":  "hpa_protein_merged_splits.csv",
        "DISPLAY_NAME": "Drug Target Prediction (HPA)",
    },
}

# =======================================================
# CONFIGURATION
# =======================================================
CONFIG = {
    # --- Task ---
    # Set to one of: "ess", "chembl", "hpa"
    "TASK": "chembl",

    # --- Paths ---
    "DATA_DIR":        Path("/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/cp/"),
    "BASE_OUTPUT_DIR": Path("./randomforest"),

    # --- Files ---
    "PROTEIN_FEATURES_FILE":  "hypergraph_features.csv",
    "PAIRWISE_FEATURES_FILE": "pairwise_features.csv",

    # --- Model ---
    "MODEL_TYPE":   "RandomForest",   # "RandomForest" | "LightGBM" | "XGBoost"
    "RANDOM_STATE": 42,
    "N_SPLITS_CV":  5,

    # --- Noise injection ---
    "N_NOISE":  5,
    "N_SEEDS":  10,

    # --- Hyperparameter grids ---
    "PARAM_GRIDS": {
        "RandomForest": {
            'n_estimators':      [80, 100, 200],
            'max_depth':         [None, 5, 10],
            'min_samples_split': [2, 5, 10],
            'class_weight':      ['balanced'],
        },
        "LightGBM": {
            'n_estimators':  [80, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth':     [None, 5, 10],
            'num_leaves':    [30, 50, 100],
            'class_weight':  ['balanced'],
        },
        "XGBoost": {
            'n_estimators':  [80, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth':     [None, 5, 10],
            'subsample':     [0.75, 0.8, 1.0],
        },
    },

    # --- Feature lists ---
    "FEATURES": {
        "HYPERGRAPH": [
            'base_Degree',
            'base_LocalClustCoeff',
            'base_TriangleCount',
            'base_UniquePartners',
            'base_AvgNeighbourDegree',
            'stoich_WeightedTriangles',
            'stoich_AvgNeighbourDegreeStoich',
            'stoich_RangeComplexSize',
            'stoich_MedComplexSize',
            'stoich_MedianRatio',
            'stoich_RangeRatio',
            'protein_MedianUniqueRatio',
            'protein_RangeUniqueRatio',
            'protein_MedComplexNodes',
            'protein_RangeComplexNodes',
        ],
        "PAIRWISE": [
            'pair_Degree',
            'pair_LocalClustCoeff',
            'pair_TriangleCount',
            'pair_AvgNeighborDegree',
        ],
    },
}


# =======================================================
# HELPERS — resolve task-specific values from CONFIG
# =======================================================

def get_task_meta() -> Dict:
    task = CONFIG["TASK"]
    if task not in TASK_META:
        raise ValueError(f"Unknown TASK {task!r}. Choose from: {list(TASK_META)}")
    return TASK_META[task]


def get_output_dir() -> Path:
    task = CONFIG["TASK"]
    model = CONFIG["MODEL_TYPE"].lower()
    d = CONFIG["BASE_OUTPUT_DIR"] / f"{model}_{task}_noise_injection"
    d.mkdir(parents=True, exist_ok=True)
    return d


# =======================================================
# DATA LOADING
# =======================================================

def load_all_features() -> pd.DataFrame:
    print("1. Loading feature data...")
    hg_df   = pd.read_csv(CONFIG["DATA_DIR"] / CONFIG["PROTEIN_FEATURES_FILE"])
    pair_df = pd.read_csv(CONFIG["DATA_DIR"] / CONFIG["PAIRWISE_FEATURES_FILE"])
    combined = pd.merge(hg_df, pair_df, on='ProteinId', how='inner')
    print(f"   Hypergraph features shape : {hg_df.shape}")
    print(f"   Pairwise features shape   : {pair_df.shape}")
    print(f"   Combined shape            : {combined.shape}")
    return combined


def load_splits() -> pd.DataFrame:
    meta = get_task_meta()
    print(f"2. Loading splits ({meta['SPLITS_FILE']})...")
    splits_df = pd.read_csv(CONFIG["DATA_DIR"] / meta["SPLITS_FILE"])
    splits_df = splits_df.rename(columns={'UniProt_AC': 'ProteinId'})

    label_map = {'Essential': 1, 'Non-essential': 0,
                 'Drug_target': 1, 'Non_target': 0, 'Unknown': 0}
    splits_df['target'] = splits_df['protein_label'].map(label_map)

    n_splits = splits_df['split_index'].nunique()
    print(f"   Rows: {len(splits_df)}  |  Proteins: {splits_df['ProteinId'].nunique()}")
    print(f"   Splits: {n_splits}")
    labelled = splits_df[splits_df['label_mask']].drop_duplicates('ProteinId')
    n_pos = (labelled['target'] == 1).sum()
    print(f"   Labelled: {len(labelled)}  ({100*n_pos/len(labelled):.1f}% positive)")
    return splits_df


# =======================================================
# MODEL HELPERS
# =======================================================

def tune_and_train_model(X_train: pd.DataFrame, y_train: pd.Series):
    model_type = CONFIG["MODEL_TYPE"]
    if model_type == "RandomForest":
        base = RandomForestClassifier(random_state=CONFIG["RANDOM_STATE"])
        grid = CONFIG["PARAM_GRIDS"]["RandomForest"]
    elif model_type == "LightGBM":
        base = LGBMClassifier(random_state=CONFIG["RANDOM_STATE"], n_jobs=1, verbose=-1)
        grid = CONFIG["PARAM_GRIDS"]["LightGBM"]
    elif model_type == "XGBoost":
        pos = int((y_train == 1).sum())
        neg = int((y_train == 0).sum())
        spw = float(neg) / float(pos) if pos > 0 else 1.0
        base = XGBClassifier(
            random_state=CONFIG["RANDOM_STATE"], n_jobs=-1,
            verbosity=0, eval_metric='logloss', scale_pos_weight=spw,
        )
        grid = CONFIG["PARAM_GRIDS"]["XGBoost"]
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {model_type!r}")

    gs = GridSearchCV(base, grid, scoring='average_precision',
                      cv=CONFIG["N_SPLITS_CV"], n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)
    return gs.best_estimator_


def perm_importance(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Permutation importance (PR-AUC drop), averaged over 10 repeats."""
    result = permutation_importance(
        model, X_test, y_test,
        scoring='average_precision',
        n_repeats=10,
        random_state=CONFIG["RANDOM_STATE"],
        n_jobs=-1,
    )
    return dict(zip(X_test.columns, result.importances_mean))


def make_noise_cols(n_rows: int, n_noise: int, seed: int, prefix: str = 'noise') -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_rows, n_noise))
    cols = [f'{prefix}_{j}' for j in range(n_noise)]
    return pd.DataFrame(data, columns=cols)


def score_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Return PR-AUC and F1@0.5 for a fitted model."""
    proba = model.predict_proba(X_test)[:, 1]
    pred  = (proba >= 0.5).astype(int)
    return {
        'pr_auc': float(average_precision_score(y_test, proba)),
        'f1':     float(f1_score(y_test, pred, zero_division=0)),
    }


# =======================================================
# PER-SPLIT RUNNER
# =======================================================

def run_split_noise(
    split_idx: int,
    merged_df: pd.DataFrame,
    splits_df: pd.DataFrame,
    hypergraph_features: List[str],
    pairwise_features: List[str],
    n_noise: int,
    n_seeds: int,
) -> List[Dict]:
    """
    For a single pre-assigned split, run all three noise-injection models
    across n_seeds independent random seeds.

    Returns a list of records, one per (split × seed), each containing
    PR-AUC, F1@0.5, and feature importances for every real and noise feature.
    """
    split_info = splits_df[splits_df['split_index'] == split_idx][
        ['ProteinId', 'split', 'target', 'label_mask']
    ].copy()

    df = pd.merge(merged_df, split_info, on='ProteinId', how='inner')
    labelled = df[df['label_mask']].copy()
    train_df = labelled[labelled['split'] == 'train']
    test_df  = labelled[labelled['split'] == 'test']

    y_train = train_df['target'].astype(int)
    y_test  = test_df['target'].astype(int)

    records = []

    for seed_idx in range(n_seeds):
        noise_seed = CONFIG["RANDOM_STATE"] + split_idx * 1000 + seed_idx

        # ── 1. PURE RANDOM model ───────────────────────────────────────────
        noise_train = make_noise_cols(len(train_df), n_noise, noise_seed, 'rand')
        noise_test  = make_noise_cols(len(test_df),  n_noise, noise_seed + 500, 'rand')
        noise_train.index = train_df.index
        noise_test.index  = test_df.index

        rand_model  = tune_and_train_model(noise_train, y_train)
        rand_imp    = perm_importance(rand_model, noise_test, y_test)
        rand_scores = score_model(rand_model, noise_test, y_test)

        # ── 2. PAIRWISE + noise model ──────────────────────────────────────
        pair_noise_train = pd.concat(
            [train_df[pairwise_features].reset_index(drop=True),
             noise_train.reset_index(drop=True)], axis=1
        )
        pair_noise_test = pd.concat(
            [test_df[pairwise_features].reset_index(drop=True),
             noise_test.reset_index(drop=True)], axis=1
        )
        pair_noise_train.index = train_df.index
        pair_noise_test.index  = test_df.index

        pair_model  = tune_and_train_model(pair_noise_train, y_train)
        pair_imp    = perm_importance(pair_model, pair_noise_test, y_test)
        pair_scores = score_model(pair_model, pair_noise_test, y_test)

        # ── 3. HYPERGRAPH + noise model ────────────────────────────────────
        hyper_noise_train = pd.concat(
            [train_df[hypergraph_features].reset_index(drop=True),
             noise_train.reset_index(drop=True)], axis=1
        )
        hyper_noise_test = pd.concat(
            [test_df[hypergraph_features].reset_index(drop=True),
             noise_test.reset_index(drop=True)], axis=1
        )
        hyper_noise_train.index = train_df.index
        hyper_noise_test.index  = test_df.index

        hyper_model  = tune_and_train_model(hyper_noise_train, y_train)
        hyper_imp    = perm_importance(hyper_model, hyper_noise_test, y_test)
        hyper_scores = score_model(hyper_model, hyper_noise_test, y_test)

        # ── Collate record ─────────────────────────────────────────────────
        record = {
            'split_index': split_idx,
            'seed_index':  seed_idx,
            'noise_seed':  noise_seed,
            'n_train':     len(train_df),
            'n_test':      len(test_df),
            # PR-AUC
            'rand_pr_auc':  rand_scores['pr_auc'],
            'pair_pr_auc':  pair_scores['pr_auc'],
            'hyper_pr_auc': hyper_scores['pr_auc'],
            # F1 @ 0.5
            'rand_f1':      rand_scores['f1'],
            'pair_f1':      pair_scores['f1'],
            'hyper_f1':     hyper_scores['f1'],
        }

        for feat, imp in rand_imp.items():
            record[f'rand__{feat}'] = imp
        for feat, imp in pair_imp.items():
            record[f'pair__{feat}'] = imp
        for feat, imp in hyper_imp.items():
            record[f'hyper__{feat}'] = imp

        records.append(record)

    return records


# =======================================================
# AGGREGATION
# =======================================================

def aggregate_importances(
    all_records: List[Dict],
    model_prefix: str,
    real_features: List[str],
    n_noise: int,
) -> pd.DataFrame:
    """
    Aggregate mean/std importance over (splits × seeds) for:
      - each real feature
      - each noise feature (pooled + individual)
    """
    prefix = model_prefix + '__'
    rows = []

    for feat in real_features:
        key  = prefix + feat
        vals = [r[key] for r in all_records if key in r]
        if vals:
            rows.append({
                'feature':  feat,
                'mean':     np.mean(vals),
                'std':      np.std(vals),
                'median':   np.median(vals),
                'n':        len(vals),
                'is_noise': False,
            })

    # Pooled noise row
    noise_vals = []
    for j in range(n_noise):
        key = prefix + f'rand_{j}'
        noise_vals.extend([r[key] for r in all_records if key in r])
    if noise_vals:
        rows.append({
            'feature':  f'[noise × {n_noise}]',
            'mean':     np.mean(noise_vals),
            'std':      np.std(noise_vals),
            'median':   np.median(noise_vals),
            'n':        len(noise_vals),
            'is_noise': True,
        })
        # Individual noise columns (used by rank table and seed-variability plot)
        for j in range(n_noise):
            key  = prefix + f'rand_{j}'
            vals = [r[key] for r in all_records if key in r]
            if vals:
                rows.append({
                    'feature':  f'rand_{j}',
                    'mean':     np.mean(vals),
                    'std':      np.std(vals),
                    'median':   np.median(vals),
                    'n':        len(vals),
                    'is_noise': True,
                })

    df      = pd.DataFrame(rows)
    real_df = df[~df['is_noise']].sort_values('mean', ascending=False)
    noise_df = df[df['is_noise']]
    return pd.concat([real_df, noise_df], ignore_index=True)


def build_noise_floor(all_records: List[Dict], model_prefix: str, n_noise: int) -> Dict:
    prefix = model_prefix + '__'
    vals   = []
    for j in range(n_noise):
        key = prefix + f'rand_{j}'
        vals.extend([r[key] for r in all_records if key in r])
    if not vals:
        return {'mean': 0.0, 'std': 0.0}
    return {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}


def build_prauc_f1_summary(all_records: List[Dict]) -> pd.DataFrame:
    """
    Per-split summary of PR-AUC and F1@0.5 for all three models.
    Seeds are collapsed to mean within each split.

    Columns: split_index, {rand,pair,hyper}_{pr_auc,f1}_{mean,std}
    Plus a final summary row with the cross-split mean ± std.
    """
    records_df = pd.DataFrame(all_records)
    metrics    = ['rand_pr_auc', 'pair_pr_auc', 'hyper_pr_auc',
                  'rand_f1',     'pair_f1',     'hyper_f1']

    # Mean over seeds within each split
    per_split = (
        records_df.groupby('split_index')[metrics]
        .agg(['mean', 'std'])
        .reset_index()
    )
    # Flatten multi-level columns
    per_split.columns = [
        '_'.join(c).strip('_') if c[1] else c[0]
        for c in per_split.columns
    ]

    # Summary row across splits
    summary_cols = [c for c in per_split.columns if c != 'split_index']
    summary_vals = per_split[summary_cols].mean().to_dict()
    summary_vals['split_index'] = 'MEAN_OVER_SPLITS'
    summary_row  = pd.DataFrame([summary_vals])

    return pd.concat([per_split, summary_row], ignore_index=True)


# =======================================================
# PLOTTING
# =======================================================

def plot_importance_vs_noise(
    pair_agg: pd.DataFrame,
    hyper_agg: pd.DataFrame,
    pair_noise_floor: Dict,
    hyper_noise_floor: Dict,
    output_dir: Path,
    n_noise: int,
):
    meta = get_task_meta()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    configs = [
        ('Pairwise + noise', pair_agg,   pair_noise_floor,  '#4393c3', axes[0]),
        ('Hypergraph + noise', hyper_agg, hyper_noise_floor, '#d6604d', axes[1]),
    ]

    for title, agg_df, noise_floor, colour, ax in configs:
        real_df = agg_df[~agg_df['is_noise']].copy().reset_index(drop=True)
        y_pos   = np.arange(len(real_df))

        nm, ns = noise_floor['mean'], noise_floor['std']
        ax.axvspan(nm - ns, nm + ns, alpha=0.2, color='grey', label=f'Noise ±1 SD')
        ax.axvline(nm, color='grey', linestyle='--', linewidth=1.5,
                   label=f'Noise mean ({nm:.4f})')

        ax.errorbar(
            real_df['mean'], y_pos,
            xerr=real_df['std'],
            fmt='o', color=colour, markersize=7, linewidth=1.5,
            capsize=4, label='Real features',
        )

        above_noise = real_df['mean'] > nm + ns
        for i, (_, row) in enumerate(real_df.iterrows()):
            lbl_colour = 'black' if above_noise.iloc[i] else '#888888'
            ax.text(
                real_df['mean'].max() * 1.02 + real_df['std'].max() * 0.5,
                i, row['feature'],
                va='center', ha='left', fontsize=11, color=lbl_colour,
            )

        ax.set_yticks([])
        ax.set_xlabel('Permutation Importance (PR-AUC drop)')
        ax.set_title(title)
        ax.legend(fontsize=11, loc='lower right')

        xmax = (real_df['mean'] + real_df['std']).max()
        ax.set_xlim(
            left=min(nm - ns * 2, real_df['mean'].min() - real_df['std'].max()),
            right=xmax * 1.5,
        )

    plt.suptitle(
        f'Feature Importance vs Noise Floor — {meta["DISPLAY_NAME"]}\n'
        f'({n_noise} noise features, {CONFIG["N_SEEDS"]} seeds, {CONFIG["MODEL_TYPE"]})',
        y=1.01,
    )
    plt.tight_layout()
    path = output_dir / 'noise_injection_importance.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {path.name}")


def plot_seed_variability(
    all_records: List[Dict],
    hypergraph_features: List[str],
    pairwise_features: List[str],
    n_noise: int,
    output_dir: Path,
):
    meta = get_task_meta()
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))

    for model_prefix, real_features, colour, ax, title in [
        ('pair',  pairwise_features,   '#4393c3', axes[0], 'Pairwise + noise'),
        ('hyper', hypergraph_features,  '#d6604d', axes[1], 'Hypergraph + noise'),
    ]:
        prefix    = model_prefix + '__'
        all_cols  = real_features + [f'rand_{j}' for j in range(n_noise)]

        # Collect data per feature, then sort by mean descending (real and
        # noise mixed together so rank position is honest)
        entries = []
        for feat in all_cols:
            key  = prefix + feat
            vals = [r[key] for r in all_records if key in r]
            entries.append({
                'feature':  feat,
                'vals':     vals,
                'mean':     np.mean(vals) if vals else 0.0,
                'is_noise': feat.startswith('rand_'),
            })
        entries.sort(key=lambda e: e['mean'], reverse=True)

        data    = [e['vals'] for e in entries]
        labels  = [e['feature'] for e in entries]
        colours = ['#cccccc' if e['is_noise'] else colour for e in entries]

        bp = ax.boxplot(data, vert=True, patch_artist=True,
                        medianprops=dict(color='black', linewidth=2))
        for patch, c in zip(bp['boxes'], colours):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=11)
        # Colour each tick label to match its box (real = model colour, noise = grey)
        for tick_label, c in zip(ax.get_xticklabels(), colours):
            tick_label.set_color(c)
        ax.set_ylabel('Permutation Importance (PR-AUC drop)')
        ax.set_title(title)
        ax.axhline(0, color='black', linewidth=0.8, linestyle=':')

        real_patch  = mpatches.Patch(color=colour,    alpha=0.7, label='Real features')
        noise_patch = mpatches.Patch(color='#cccccc', alpha=0.7, label='Noise features')
        ax.legend(handles=[real_patch, noise_patch], fontsize=11)

    plt.suptitle(
        f'Seed-to-Seed Variability — {meta["DISPLAY_NAME"]}\n'
        f'({n_noise} noise features × {CONFIG["N_SEEDS"]} seeds)',
        y=1.01,
    )
    plt.tight_layout()
    path = output_dir / 'noise_seed_variability.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {path.name}")


def plot_noise_floor_comparison(
    rand_floor: Dict,
    pair_floor: Dict,
    hyper_floor: Dict,
    output_dir: Path,
):
    meta   = get_task_meta()
    models = ['Pure random', 'Pairwise + noise', 'Hypergraph + noise']
    means  = [rand_floor['mean'], pair_floor['mean'], hyper_floor['mean']]
    stds   = [rand_floor['std'],  pair_floor['std'],  hyper_floor['std']]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(models))
    ax.bar(x, means, yerr=stds, capsize=6,
           color=['#aaaaaa', '#4393c3', '#d6604d'],
           alpha=0.8, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel('Mean noise feature importance\n(Permutation, PR-AUC drop)')
    ax.set_title(
        f'Noise Floor Comparison — {meta["DISPLAY_NAME"]}\n'
        '(Should be similar — confirms noise is genuinely uninformative)'
    )
    ax.axhline(0, color='black', linewidth=0.8, linestyle=':')
    plt.tight_layout()
    path = output_dir / 'noise_floor_comparison.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {path.name}")


# =======================================================
# RANKED TABLE BUILDER  (rank-based noise criterion)
# =======================================================

def compute_hit_counts(
    all_records: List[Dict],
    model_prefix: str,
    real_features: List[str],
    n_noise: int,
) -> pd.DataFrame:
    """
    Boruta-style hit counts (no auto-decision — we report the numbers).

    For each (split × seed) realisation, the "noise ceiling" is the MAXIMUM
    importance among the n_noise noise features in that realisation. A real
    feature scores a 'hit' whenever its importance exceeds that ceiling.

    Using the per-realisation max (rather than a pooled mean) is the Boruta
    convention: it folds the spread of the noise distribution into the
    threshold and gives automatic multiple-testing protection.

    Returns a DataFrame: feature, hits, n_realisations, hit_rate, binom_p
    where binom_p is a two-sided binomial test of hits vs the null hit
    rate of 0.5 (a non-informative feature beats the noise max ~half the
    time by chance under Boruta's framing).
    """
    from scipy.stats import binomtest

    prefix     = model_prefix + '__'
    noise_keys = [prefix + f'rand_{j}' for j in range(n_noise)]

    # Per-realisation noise ceiling
    rows = []
    for feat in real_features:
        fkey = prefix + feat
        hits = 0
        n    = 0
        for r in all_records:
            if fkey not in r:
                continue
            noise_vals = [r[k] for k in noise_keys if k in r]
            if not noise_vals:
                continue
            ceiling = max(noise_vals)
            n += 1
            if r[fkey] > ceiling:
                hits += 1
        if n == 0:
            continue
        p = binomtest(hits, n, p=0.5, alternative='two-sided').pvalue
        rows.append({
            'feature':         feat,
            'hits':            hits,
            'n_realisations':  n,
            'hit_rate':        hits / n,
            'binom_p':         p,
        })

    return pd.DataFrame(rows)


def build_ranked_table(
    imp_df: pd.DataFrame,
    hit_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Merge real features and individual noise features into one ranked table.
    The pooled [noise × N] row is excluded; individual rand_* rows are used
    so noise features appear at their actual rank positions.

    A feature is considered uninformative if it ranks at or below the
    highest-ranked noise feature — no numeric threshold needed.

    If hit_df (from compute_hit_counts) is supplied, Boruta-style columns
    (hit_rate, binom_p) are merged in for the real features. These are
    reported alongside the rank, not used to override it.
    """
    real  = imp_df[~imp_df['is_noise']].copy()
    noise = imp_df[imp_df['feature'].str.startswith('rand_')].copy()

    combined = pd.concat([real, noise], ignore_index=True)
    combined = combined.sort_values('mean', ascending=False).reset_index(drop=True)
    combined['rank'] = range(1, len(combined) + 1)
    combined['type'] = combined['feature'].apply(
        lambda f: 'NOISE' if f.startswith('rand_') else 'real'
    )

    cols = ['rank', 'type', 'feature', 'mean', 'std']
    if hit_df is not None and not hit_df.empty:
        combined = combined.merge(
            hit_df[['feature', 'hit_rate', 'binom_p']],
            on='feature', how='left',
        )
        cols += ['hit_rate', 'binom_p']

    return combined[cols]


# =======================================================
# SUMMARY WRITING
# =======================================================

def _format_importance_table(df: pd.DataFrame) -> str:
    has_hits = 'hit_rate' in df.columns
    lines = []
    if has_hits:
        header = (f"  {'Rank':<5} {'Type':<7} {'Feature':<38} "
                  f"{'Mean':>9}  {'Std':>9}  {'Hit rate':>9}  {'Binom p':>9}")
    else:
        header = f"  {'Rank':<5} {'Type':<7} {'Feature':<38} {'Mean':>9}  {'Std':>9}"
    lines.append(header)
    lines.append("  " + "-" * (94 if has_hits else 72))
    for _, row in df.iterrows():
        base = (f"  {int(row['rank']):<5} {row['type']:<7} {row['feature']:<38} "
                f"{row['mean']:>9.5f}  {row['std']:>9.5f}")
        if has_hits:
            hr = row.get('hit_rate')
            bp = row.get('binom_p')
            hr_s = f"{hr:>9.2f}" if pd.notna(hr) else f"{'—':>9}"
            bp_s = f"{bp:>9.4f}" if pd.notna(bp) else f"{'—':>9}"
            base += f"  {hr_s}  {bp_s}"
        lines.append(base)
    return "\n".join(lines)


def write_summary(
    rand_df: pd.DataFrame,
    pair_ranked: pd.DataFrame,
    hyper_ranked: pd.DataFrame,
    prauc_f1_df: pd.DataFrame,
    output_path: Path,
):
    meta = get_task_meta()
    sep  = "=" * 85

    pair_top_noise  = pair_ranked[pair_ranked['type'] == 'NOISE']['rank'].min()
    hyper_top_noise = hyper_ranked[hyper_ranked['type'] == 'NOISE']['rank'].min()

    pair_below  = pair_ranked[
        (pair_ranked['type'] == 'real') & (pair_ranked['rank'] >= pair_top_noise)
    ]
    hyper_below = hyper_ranked[
        (hyper_ranked['type'] == 'real') & (hyper_ranked['rank'] >= hyper_top_noise)
    ]

    lines = [
        sep,
        "  NOISE INJECTION — FEATURE IMPORTANCE RANKING SUMMARY",
        sep,
        f"  Task          : {meta['DISPLAY_NAME']}",
        f"  Model         : {CONFIG['MODEL_TYPE']}",
        f"  Splits × Seeds: {CONFIG.get('N_SPLITS', 15)} × {CONFIG['N_SEEDS']} "
        f"= {CONFIG.get('N_SPLITS', 15) * CONFIG['N_SEEDS']} realisations per feature",
        f"  Noise features: {CONFIG['N_NOISE']} (standard normal, i.i.d.)",
        "",
        "  Criterion: a real feature is considered uninformative if it ranks at",
        "  or below the highest-ranked noise feature (rank-based, no numeric",
        "  threshold required).",
        "",
        "  Hit rate / Binom p (Boruta-style, reported not decisive): per",
        "  realisation, hit rate is the fraction of (split × seed) runs in which",
        "  the feature's importance exceeds the MAXIMUM noise importance in that",
        "  run. Binom p tests hits against a 0.5 null. A high mean rank with a low",
        "  hit rate flags a feature whose ranking is fragile across runs.",
        "",
    ]

    # ── Section 0: PR-AUC / F1 summary ────────────────────────────────────
    lines += [
        sep,
        "  0. MODEL PERFORMANCE (PR-AUC and F1 @ 0.5 threshold)",
        sep,
        "  Mean ± std across seeds, then mean over splits in final row.",
        "  Note: F1@0.5 is a convenience metric — PR-AUC is the primary measure",
        "  for imbalanced tasks.",
        "",
    ]

    # Pull out the MEAN_OVER_SPLITS row and per-split rows separately
    summary_row = prauc_f1_df[prauc_f1_df['split_index'] == 'MEAN_OVER_SPLITS']
    per_split   = prauc_f1_df[prauc_f1_df['split_index'] != 'MEAN_OVER_SPLITS'].copy()

    # Header
    perf_header = (
        f"  {'Split':<8}  "
        f"{'Rand PR-AUC':>12}  {'Pair PR-AUC':>12}  {'Hyper PR-AUC':>13}  "
        f"{'Rand F1':>9}  {'Pair F1':>9}  {'Hyper F1':>9}"
    )
    lines.append(perf_header)
    lines.append("  " + "-" * 85)

    def fmt_metric(row, col_mean, col_std):
        m = row.get(col_mean, float('nan'))
        s = row.get(col_std,  float('nan'))
        if pd.isna(m):
            return f"{'—':>12}"
        return f"{m:>6.4f}±{s:>6.4f}"

    for _, row in per_split.iterrows():
        lines.append(
            f"  {str(row['split_index']):<8}  "
            f"{fmt_metric(row, 'rand_pr_auc_mean',  'rand_pr_auc_std'):>14}  "
            f"{fmt_metric(row, 'pair_pr_auc_mean',  'pair_pr_auc_std'):>14}  "
            f"{fmt_metric(row, 'hyper_pr_auc_mean', 'hyper_pr_auc_std'):>15}  "
            f"{fmt_metric(row, 'rand_f1_mean',  'rand_f1_std'):>11}  "
            f"{fmt_metric(row, 'pair_f1_mean',  'pair_f1_std'):>11}  "
            f"{fmt_metric(row, 'hyper_f1_mean', 'hyper_f1_std'):>11}"
        )

    if not summary_row.empty:
        row = summary_row.iloc[0]
        lines.append("  " + "-" * 85)
        lines.append(
            f"  {'MEAN':<8}  "
            f"{fmt_metric(row, 'rand_pr_auc_mean',  'rand_pr_auc_std'):>14}  "
            f"{fmt_metric(row, 'pair_pr_auc_mean',  'pair_pr_auc_std'):>14}  "
            f"{fmt_metric(row, 'hyper_pr_auc_mean', 'hyper_pr_auc_std'):>15}  "
            f"{fmt_metric(row, 'rand_f1_mean',  'rand_f1_std'):>11}  "
            f"{fmt_metric(row, 'pair_f1_mean',  'pair_f1_std'):>11}  "
            f"{fmt_metric(row, 'hyper_f1_mean', 'hyper_f1_std'):>11}"
        )
    lines.append("")

    # ── Section 1: Pure random sanity check ───────────────────────────────
    lines += [
        sep,
        "  1. PURE RANDOM MODEL (sanity check)",
        sep,
        "  All importances should be near zero — confirms noise is uninformative.",
        "",
    ]
    noise_only = rand_df[rand_df['feature'].str.startswith('rand_')][
        ['feature', 'mean', 'std']
    ].copy().sort_values('mean', ascending=False).reset_index(drop=True)
    noise_only['rank'] = range(1, len(noise_only) + 1)
    lines.append(f"  {'Rank':<5} {'Feature':<20} {'Mean':>9}  {'Std':>9}")
    lines.append("  " + "-" * 48)
    for _, row in noise_only.iterrows():
        lines.append(f"  {int(row['rank']):<5} {row['feature']:<20} "
                     f"{row['mean']:>9.5f}  {row['std']:>9.5f}")
    lines.append("")

    # ── Section 2: Pairwise ────────────────────────────────────────────────
    lines += [
        sep,
        "  2. PAIRWISE + NOISE — Full ranked list",
        sep,
        "  Noise features (rand_*) are interleaved at their actual rank position.",
        f"  Highest-ranked noise feature: rank {pair_top_noise}",
    ]
    if len(pair_below) == 0:
        lines.append("  All real features rank above noise.")
    else:
        lines.append(f"  Real features at or below noise: "
                     f"{', '.join(pair_below['feature'].tolist())}")
    lines.append("")
    lines.append(_format_importance_table(pair_ranked))
    lines.append("")

    # ── Section 3: Hypergraph ──────────────────────────────────────────────
    lines += [
        sep,
        "  3. HYPERGRAPH + NOISE — Full ranked list",
        sep,
        "  Noise features (rand_*) are interleaved at their actual rank position.",
        f"  Highest-ranked noise feature: rank {hyper_top_noise}",
    ]
    if len(hyper_below) == 0:
        lines.append("  All real features rank above noise.")
    else:
        lines.append(f"  Real features at or below noise: "
                     f"{', '.join(hyper_below['feature'].tolist())}")
    lines.append("")
    lines.append(_format_importance_table(hyper_ranked))
    lines.append("")

    text = "\n".join(lines)
    output_path.write_text(text)
    print(text)
    return text


# =======================================================
# CONSOLE SUMMARY (quick print during / after run)
# =======================================================

def print_noise_summary(
    pair_ranked:  pd.DataFrame,
    hyper_ranked: pd.DataFrame,
    pair_floor:   Dict,
    hyper_floor:  Dict,
    rand_floor:   Dict,
):
    """
    Console summary using the SAME rank-based criterion as the text file.
    Reports the noise floor for context, then the ranked tables. No
    mean+SD threshold or 'Signal?' verdict — the rank relative to the top
    noise feature is the criterion.
    """
    print(f"\n{'='*70}")
    print("  NOISE INJECTION — FEATURE IMPORTANCE SUMMARY (rank-based)")
    print(f"{'='*70}")
    print(f"\n  Noise floor (mean ± std across splits × seeds, context only):")
    print(f"    Pure random       : {rand_floor['mean']:+.5f} ± {rand_floor['std']:.5f}")
    print(f"    Pairwise + noise  : {pair_floor['mean']:+.5f} ± {pair_floor['std']:.5f}")
    print(f"    Hypergraph + noise: {hyper_floor['mean']:+.5f} ± {hyper_floor['std']:.5f}")

    for label, ranked in [
        ('PAIRWISE + noise',   pair_ranked),
        ('HYPERGRAPH + noise', hyper_ranked),
    ]:
        top_noise = ranked[ranked['type'] == 'NOISE']['rank'].min()
        below     = ranked[(ranked['type'] == 'real') & (ranked['rank'] >= top_noise)]
        print(f"\n  {label}  (top noise feature at rank {top_noise}):")
        if len(below) == 0:
            print("    All real features rank above noise.")
        else:
            print(f"    Real features at or below noise: "
                  f"{', '.join(below['feature'].tolist())}")
        print(_format_importance_table(ranked))
    print(f"\n{'='*70}")


# =======================================================
# SUMMARY-ONLY PATH  (reads existing CSVs, skips model fitting)
# =======================================================

def run_summary_only(output_dir: Path):
    print(f"\nLoading existing CSVs from: {output_dir}")
    hyper_df = pd.read_csv(output_dir / 'hypergraph_noise_importance.csv')
    pair_df  = pd.read_csv(output_dir / 'pairwise_noise_importance.csv')
    rand_df  = pd.read_csv(output_dir / 'random_noise_importance.csv')

    # Real feature lists are the non-noise rows of each importance table
    pair_feats  = pair_df[~pair_df['is_noise']]['feature'].tolist()
    hyper_feats = hyper_df[~hyper_df['is_noise']]['feature'].tolist()

    # Load records for PR-AUC / F1 table and Boruta-style hit counts
    records_path = output_dir / 'noise_injection_records.csv'
    pair_hits = hyper_hits = None
    if records_path.exists():
        all_records  = pd.read_csv(records_path).to_dict('records')
        prauc_f1_df  = build_prauc_f1_summary(all_records)
        prauc_f1_df.to_csv(output_dir / 'prauc_f1_summary.csv', index=False)
        print("   Saved: prauc_f1_summary.csv")
        pair_hits  = compute_hit_counts(all_records, 'pair',  pair_feats,  CONFIG["N_NOISE"])
        hyper_hits = compute_hit_counts(all_records, 'hyper', hyper_feats, CONFIG["N_NOISE"])
    else:
        print(f"   WARNING: {records_path.name} not found — "
              f"skipping performance table and hit counts.")
        prauc_f1_df = pd.DataFrame()

    pair_ranked  = build_ranked_table(pair_df,  pair_hits)
    hyper_ranked = build_ranked_table(hyper_df, hyper_hits)

    write_summary(
        rand_df, pair_ranked, hyper_ranked, prauc_f1_df,
        output_dir / 'noise_ranked_tables.txt',
    )

    pair_ranked.to_csv(output_dir  / 'noise_ranked_pairwise.csv',   index=False)
    hyper_ranked.to_csv(output_dir / 'noise_ranked_hypergraph.csv',  index=False)
    rand_df[rand_df['feature'].str.startswith('rand_')].to_csv(
        output_dir / 'noise_ranked_random.csv', index=False
    )

    # Regenerate seed-to-seed variability plot from existing records
    if records_path.exists():
        plot_seed_variability(
            all_records, hyper_feats, pair_feats,
            CONFIG["N_NOISE"], output_dir,
        )

    print(f"\nSaved to: {output_dir}")
    for f in ['noise_ranked_tables.txt', 'noise_ranked_pairwise.csv',
              'noise_ranked_hypergraph.csv', 'noise_ranked_random.csv']:
        print(f"  {f}")


# =======================================================
# MAIN
# =======================================================

if __name__ == '__main__':

    # Set to True to regenerate summary/plots from existing CSVs without refitting
    SUMMARY_ONLY = False

    output_dir = get_output_dir()
    meta       = get_task_meta()

    print(f"\n{'='*70}")
    print("  NOISE INJECTION — DIMENSIONALITY CONTROL")
    print(f"  Task   : {meta['DISPLAY_NAME']}")
    print(f"  Model  : {CONFIG['MODEL_TYPE']}")
    print(f"  N_NOISE: {CONFIG['N_NOISE']}   N_SEEDS: {CONFIG['N_SEEDS']}")
    print(f"  Output : {output_dir}")
    print(f"{'='*70}\n")

    if SUMMARY_ONLY:
        run_summary_only(output_dir)
    else:
        # ── Full run ───────────────────────────────────────────────────────
        start = time.time()
        print(f"Started at {time.strftime('%H:%M:%S')}")

        features_df   = load_all_features()
        splits_df     = load_splits()
        split_indices = sorted(splits_df['split_index'].unique())

        hypergraph_features = [f for f in CONFIG["FEATURES"]["HYPERGRAPH"]
                               if f in features_df.columns]
        pairwise_features   = [f for f in CONFIG["FEATURES"]["PAIRWISE"]
                               if f in features_df.columns]

        print(f"\n   Hypergraph features ({len(hypergraph_features)}): {hypergraph_features}")
        print(f"   Pairwise features   ({len(pairwise_features)}): {pairwise_features}")
        print(f"   Noise features      ({CONFIG['N_NOISE']}): "
              f"[standard normal, {CONFIG['N_SEEDS']} independent seeds per split]\n")

        all_feats = hypergraph_features + pairwise_features
        n_nans    = features_df[all_feats].isna().sum().sum()
        if n_nans:
            print(f"   Filling {n_nans} NaN values with 0.")
            features_df[all_feats] = features_df[all_feats].fillna(0)

        print(f"3. Running {len(split_indices)} splits × {CONFIG['N_SEEDS']} seeds "
              f"× 3 models = {len(split_indices) * CONFIG['N_SEEDS'] * 3} model fits...\n")

        all_records: List[Dict] = []

        for split_idx in split_indices:
            print(f"   Split {split_idx:>2}/{len(split_indices)}...", end=" ", flush=True)
            try:
                records = run_split_noise(
                    split_idx, features_df, splits_df,
                    hypergraph_features, pairwise_features,
                    CONFIG["N_NOISE"], CONFIG["N_SEEDS"],
                )
                all_records.extend(records)
                rand_mean  = np.mean([r['rand_pr_auc']  for r in records])
                pair_mean  = np.mean([r['pair_pr_auc']  for r in records])
                hyper_mean = np.mean([r['hyper_pr_auc'] for r in records])
                rand_f1    = np.mean([r['rand_f1']      for r in records])
                pair_f1    = np.mean([r['pair_f1']      for r in records])
                hyper_f1   = np.mean([r['hyper_f1']     for r in records])
                print(f"PR-AUC  rand={rand_mean:.3f}  pair={pair_mean:.3f}  hyper={hyper_mean:.3f}"
                      f"   F1@0.5  rand={rand_f1:.3f}  pair={pair_f1:.3f}  hyper={hyper_f1:.3f}")
            except Exception as e:
                print(f"ERROR: {e}")
                raise

        # Aggregate
        print("\n4. Aggregating importances...")
        pair_agg  = aggregate_importances(all_records, 'pair',  pairwise_features,   CONFIG["N_NOISE"])
        hyper_agg = aggregate_importances(all_records, 'hyper', hypergraph_features, CONFIG["N_NOISE"])
        rand_agg  = aggregate_importances(all_records, 'rand',  [],                  CONFIG["N_NOISE"])

        pair_floor  = build_noise_floor(all_records, 'pair',  CONFIG["N_NOISE"])
        hyper_floor = build_noise_floor(all_records, 'hyper', CONFIG["N_NOISE"])
        rand_floor  = build_noise_floor(all_records, 'rand',  CONFIG["N_NOISE"])

        # Boruta-style hit counts (reported alongside rank, not decisive)
        pair_hits  = compute_hit_counts(all_records, 'pair',  pairwise_features,   CONFIG["N_NOISE"])
        hyper_hits = compute_hit_counts(all_records, 'hyper', hypergraph_features, CONFIG["N_NOISE"])

        # Ranked tables (built once, used by both the console summary and text file)
        pair_ranked  = build_ranked_table(pair_agg,  pair_hits)
        hyper_ranked = build_ranked_table(hyper_agg, hyper_hits)

        print_noise_summary(pair_ranked, hyper_ranked, pair_floor, hyper_floor, rand_floor)

        # Plots
        print("\n5. Generating plots...")
        plot_importance_vs_noise(pair_agg, hyper_agg, pair_floor, hyper_floor,
                                 output_dir, CONFIG["N_NOISE"])
        plot_seed_variability(all_records, hypergraph_features, pairwise_features,
                              CONFIG["N_NOISE"], output_dir)
        plot_noise_floor_comparison(rand_floor, pair_floor, hyper_floor, output_dir)

        # Save CSVs
        print("\n6. Saving CSVs...")
        records_df = pd.DataFrame(all_records)
        records_df.to_csv(output_dir / 'noise_injection_records.csv', index=False)
        print("   Saved: noise_injection_records.csv")

        pair_agg.to_csv(output_dir  / 'pairwise_noise_importance.csv',  index=False)
        hyper_agg.to_csv(output_dir / 'hypergraph_noise_importance.csv', index=False)
        rand_agg.to_csv(output_dir  / 'random_noise_importance.csv',     index=False)
        print("   Saved: pairwise/hypergraph/random_noise_importance.csv")

        noise_floor_df = pd.DataFrame([
            {'model': 'pure_random',      **rand_floor},
            {'model': 'pairwise_noise',   **pair_floor},
            {'model': 'hypergraph_noise', **hyper_floor},
        ])
        noise_floor_df.to_csv(output_dir / 'noise_floor_summary.csv', index=False)
        print("   Saved: noise_floor_summary.csv")

        # PR-AUC / F1 summary
        prauc_f1_df = build_prauc_f1_summary(all_records)
        prauc_f1_df.to_csv(output_dir / 'prauc_f1_summary.csv', index=False)
        print("   Saved: prauc_f1_summary.csv")

        # Ranked summary (tables built earlier; written here once all CSVs exist)
        print("\n7. Writing ranked summary...")
        write_summary(
            rand_agg, pair_ranked, hyper_ranked, prauc_f1_df,
            output_dir / 'noise_ranked_tables.txt',
        )
        pair_ranked.to_csv(output_dir  / 'noise_ranked_pairwise.csv',   index=False)
        hyper_ranked.to_csv(output_dir / 'noise_ranked_hypergraph.csv',  index=False)
        rand_agg[rand_agg['feature'].str.startswith('rand_')].to_csv(
            output_dir / 'noise_ranked_random.csv', index=False
        )
        print("   Saved: noise_ranked_tables.txt / noise_ranked_*.csv")

        elapsed = time.time() - start
        print(f"\n{'='*70}")
        print(f"  COMPLETE — {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"{'='*70}")