"""
cp_noise_injection.py
=====================
Noise-injection dimensionality control experiment.

Supervisor's design:
  Three models are trained with the same random noise features injected alongside
  real features (or alone). Feature importance (permutation, PR-AUC drop) is
  compared across real features vs the noise-feature distribution.

  A real feature that consistently sits above the noise band is demonstrably
  carrying signal — independent of how many features the competing model has.

Models
------
  1. Pure random      — N_NOISE noise columns only.  Establishes the noise floor.
  2. Pairwise + noise — 4 pairwise features + N_NOISE noise columns.
  3. Hypergraph + noise — full hypergraph feature set + N_NOISE noise columns.

Key outputs
-----------
  - noise_injection_importance.csv      — per-split importance for all features
  - noise_injection_summary.csv         — mean/std importance aggregated over splits
  - noise_floor_summary.csv             — mean/std of noise feature importances per model
  - noise_injection_plot.png            — dot-plot with noise band per model
  - noise_seed_variability_plot.png     — box-plots showing seed-to-seed spread

Design notes
------------
  * N_NOISE = 4 (3–5 recommended by supervisor)
  * N_SEEDS = 3  — independent RNG seeds, each producing a different noise matrix.
                   Each seed gives one set of noise importances; this lets us
                   check that the noise floor estimate is stable.
  * Permutation importance is used (same as base pipeline) — scored by PR-AUC
    drop, so importances are on the same scale across all three models.
  * Random features are standard-normal i.i.d. — no relationship to labels.
  * Each (split × seed) combination is an independent model fit, so the noise
    importance estimates are not inflated by re-use.
"""

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
from sklearn.metrics import average_precision_score
from sklearn.inspection import permutation_importance
from scipy.stats import binomtest

# =======================================================
# Plotting style
# =======================================================
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 20,
})

# =======================================================
# CONFIGURATION
# =======================================================
CONFIG = {
    # --- Paths ---
    "DATA_DIR":        Path("/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/corum/"),
    "BASE_OUTPUT_DIR": Path("./randomforest/ess_noise_injection"),

    # --- Files ---
    "SPLITS_FILE":            "ess_protein_merged_splits.csv",
    "PROTEIN_FEATURES_FILE":  "hypergraph_features.csv",
    "PAIRWISE_FEATURES_FILE": "pairwise_features.csv",

    # --- Model ---
    "MODEL_TYPE":   "RandomForest",   # "RandomForest" | "LightGBM" | "XGBoost"
    "RANDOM_STATE": 42,
    "N_SPLITS_CV":  5,

    # --- Noise injection ---
    # N_NOISE: how many noise columns to inject (3–5 recommended).
    # N_SEEDS: how many independent RNG seeds to use.
    #   Each seed produces a different noise matrix → a distribution of
    #   noise importances, not a single point estimate.
    "N_NOISE":  4,
    "N_SEEDS":  3,

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
    print("2. Loading splits...")
    splits_df = pd.read_csv(CONFIG["DATA_DIR"] / CONFIG["SPLITS_FILE"])
    splits_df = splits_df.rename(columns={'UniProt_AC': 'ProteinId'})

    label_map = label_map = {'Essential': 1, 'Non-essential': 0, 'Drug_target': 1, 'Non_target': 0, 'Unknown': 0}
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
    """
    Generate n_noise standard-normal columns with a given RNG seed.
    prefix disambiguates noise columns across models in the same record.
    """
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_rows, n_noise))
    cols = [f'{prefix}_{j}' for j in range(n_noise)]
    return pd.DataFrame(data, columns=cols)


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
    feature importances for every real and noise feature.
    """
    # --- Prepare split data ---
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
        # Use a deterministic but varied seed per (split × seed)
        noise_seed = CONFIG["RANDOM_STATE"] + split_idx * 1000 + seed_idx

        # ── 1. PURE RANDOM model ───────────────────────────────────────────
        # Only noise features; all importances should cluster near zero.
        noise_train = make_noise_cols(len(train_df), n_noise, noise_seed, 'rand')
        noise_test  = make_noise_cols(len(test_df),  n_noise, noise_seed + 500, 'rand')
        noise_train.index = train_df.index
        noise_test.index  = test_df.index

        rand_model = tune_and_train_model(noise_train, y_train)
        rand_imp   = perm_importance(rand_model, noise_test, y_test)
        rand_prauc = average_precision_score(
            y_test, rand_model.predict_proba(noise_test)[:, 1]
        )

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

        pair_model = tune_and_train_model(pair_noise_train, y_train)
        pair_imp   = perm_importance(pair_model, pair_noise_test, y_test)
        pair_prauc = average_precision_score(
            y_test, pair_model.predict_proba(pair_noise_test)[:, 1]
        )

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

        hyper_model = tune_and_train_model(hyper_noise_train, y_train)
        hyper_imp   = perm_importance(hyper_model, hyper_noise_test, y_test)
        hyper_prauc = average_precision_score(
            y_test, hyper_model.predict_proba(hyper_noise_test)[:, 1]
        )

        # ── Collate record ─────────────────────────────────────────────────
        record = {
            'split_index': split_idx,
            'seed_index':  seed_idx,
            'noise_seed':  noise_seed,
            'n_train':     len(train_df),
            'n_test':      len(test_df),
            'rand_pr_auc':  rand_prauc,
            'pair_pr_auc':  pair_prauc,
            'hyper_pr_auc': hyper_prauc,
        }

        # Store importances with model prefix so they stay unambiguous when
        # the same noise column name appears in multiple models
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
      - each noise feature (pooled across all noise indices)

    Returns a DataFrame sorted by mean importance (descending),
    with a 'is_noise' column to distinguish real vs noise features.
    """
    noise_cols   = [f'rand_{j}' if model_prefix == 'rand' else f'rand_{j}'
                    for j in range(n_noise)]
    # Column names in records are prefixed: e.g. 'hyper__base_Degree'
    prefix = model_prefix + '__'

    rows = []
    # Real features
    for feat in real_features:
        key = prefix + feat
        vals = [r[key] for r in all_records if key in r]
        if vals:
            rows.append({
                'feature': feat,
                'mean': np.mean(vals),
                'std':  np.std(vals),
                'median': np.median(vals),
                'n':    len(vals),
                'is_noise': False,
            })

    # Noise features — pool all n_noise columns together for a richer sample
    noise_vals = []
    for j in range(n_noise):
        noise_key = prefix + f'rand_{j}'
        noise_vals.extend([r[noise_key] for r in all_records if noise_key in r])

    if noise_vals:
        rows.append({
            'feature': f'[noise × {n_noise}]',
            'mean':    np.mean(noise_vals),
            'std':     np.std(noise_vals),
            'median':  np.median(noise_vals),
            'n':       len(noise_vals),
            'is_noise': True,
        })
        # Also add individual noise columns for variability plot
        for j in range(n_noise):
            noise_key = prefix + f'rand_{j}'
            vals = [r[noise_key] for r in all_records if noise_key in r]
            if vals:
                rows.append({
                    'feature': f'rand_{j}',
                    'mean':    np.mean(vals),
                    'std':     np.std(vals),
                    'median':  np.median(vals),
                    'n':       len(vals),
                    'is_noise': True,
                })

    df = pd.DataFrame(rows)
    # Sort: real features by importance descending, noise at bottom
    real_df  = df[~df['is_noise']].sort_values('mean', ascending=False)
    noise_df = df[df['is_noise']]
    return pd.concat([real_df, noise_df], ignore_index=True)


def build_noise_floor(all_records: List[Dict], model_prefix: str, n_noise: int) -> Dict:
    """Returns mean and std of the pooled noise importance for a given model."""
    prefix = model_prefix + '__'
    vals = []
    for j in range(n_noise):
        key = prefix + f'rand_{j}'
        vals.extend([r[key] for r in all_records if key in r])
    if not vals:
        return {'mean': 0.0, 'std': 0.0}
    return {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}


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
    """
    Two-panel dot-plot: real feature importances with a noise reference band.

    The grey band shows mean ± 1 std of the pooled noise importance.
    Features above the band are carrying genuine signal.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    configs = [
        ('Pairwise + noise', pair_agg,   pair_noise_floor,  '#4393c3', axes[0]),
        ('Hypergraph + noise', hyper_agg, hyper_noise_floor, '#d6604d', axes[1]),
    ]

    for title, agg_df, noise_floor, colour, ax in configs:
        real_df = agg_df[~agg_df['is_noise']].copy().reset_index(drop=True)
        y_pos   = np.arange(len(real_df))

        # Noise reference band: mean ± 1 std
        nm = noise_floor['mean']
        ns = noise_floor['std']
        ax.axvspan(nm - ns, nm + ns, alpha=0.2, color='grey', label=f'Noise ±1 SD')
        ax.axvline(nm, color='grey', linestyle='--', linewidth=1.5,
                   label=f'Noise mean ({nm:.4f})')

        # Feature dots + error bars (std across splits × seeds)
        ax.errorbar(
            real_df['mean'], y_pos,
            xerr=real_df['std'],
            fmt='o', color=colour, markersize=7, linewidth=1.5,
            capsize=4, label='Real features',
        )

        # Colour labels above noise band differently
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

        # Extend x-axis so labels fit
        xmax = (real_df['mean'] + real_df['std']).max()
        ax.set_xlim(left=min(nm - ns * 2, real_df['mean'].min() - real_df['std'].max()),
                    right=xmax * 1.5)

    plt.suptitle(
        f'Feature Importance vs Noise Floor\n'
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
    """
    Box-plots showing seed-to-seed variability of feature importances.

    One figure with two panels (pairwise model / hypergraph model).
    Noise features shown in grey; real features in colour.
    Helps confirm that the noise floor estimate is stable across seeds.
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))

    for model_prefix, real_features, colour, ax, title in [
        ('pair',  pairwise_features,   '#4393c3', axes[0], 'Pairwise + noise'),
        ('hyper', hypergraph_features,  '#d6604d', axes[1], 'Hypergraph + noise'),
    ]:
        prefix = model_prefix + '__'
        all_cols  = real_features + [f'rand_{j}' for j in range(n_noise)]
        data      = []
        labels    = []
        colours   = []

        for feat in all_cols:
            key  = prefix + feat
            vals = [r[key] for r in all_records if key in r]
            data.append(vals)
            is_noise = feat.startswith('rand_')
            labels.append(feat)
            colours.append('#cccccc' if is_noise else colour)

        bp = ax.boxplot(data, vert=True, patch_artist=True,
                        medianprops=dict(color='black', linewidth=2))
        for patch, c in zip(bp['boxes'], colours):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=11)
        ax.set_ylabel('Permutation Importance (PR-AUC drop)')
        ax.set_title(title)
        ax.axhline(0, color='black', linewidth=0.8, linestyle=':')

        # Legend
        real_patch  = mpatches.Patch(color=colour,   alpha=0.7, label='Real features')
        noise_patch = mpatches.Patch(color='#cccccc', alpha=0.7, label='Noise features')
        ax.legend(handles=[real_patch, noise_patch], fontsize=11)

    plt.suptitle(
        f'Seed-to-Seed Variability ({n_noise} noise features × {CONFIG["N_SEEDS"]} seeds)',
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
    """
    Bar chart comparing the noise floor mean across all three models.
    All three should be similar — if the pure-random model's noise
    is very different from the others, something is wrong.
    """
    models = ['Pure random', 'Pairwise + noise', 'Hypergraph + noise']
    means  = [rand_floor['mean'], pair_floor['mean'], hyper_floor['mean']]
    stds   = [rand_floor['std'],  pair_floor['std'],  hyper_floor['std']]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(models))
    bars = ax.bar(x, means, yerr=stds, capsize=6,
                  color=['#aaaaaa', '#4393c3', '#d6604d'],
                  alpha=0.8, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel('Mean noise feature importance\n(Permutation, PR-AUC drop)')
    ax.set_title('Noise Floor Comparison Across Models\n'
                 '(Should be similar — confirms noise is genuinely uninformative)')
    ax.axhline(0, color='black', linewidth=0.8, linestyle=':')
    plt.tight_layout()
    path = output_dir / 'noise_floor_comparison.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {path.name}")


# =======================================================
# SUMMARY PRINTING
# =======================================================

def print_noise_summary(
    pair_agg:   pd.DataFrame,
    hyper_agg:  pd.DataFrame,
    pair_floor: Dict,
    hyper_floor: Dict,
    rand_floor:  Dict,
):
    print(f"\n{'='*70}")
    print("  NOISE INJECTION — FEATURE IMPORTANCE SUMMARY")
    print(f"{'='*70}")
    print(f"\n  Noise floor (mean ± std across splits × seeds):")
    print(f"    Pure random      : {rand_floor['mean']:+.5f} ± {rand_floor['std']:.5f}")
    print(f"    Pairwise + noise : {pair_floor['mean']:+.5f} ± {pair_floor['std']:.5f}")
    print(f"    Hypergraph + noise: {hyper_floor['mean']:+.5f} ± {hyper_floor['std']:.5f}")

    for label, agg_df, floor in [
        ('PAIRWISE + noise', pair_agg,  pair_floor),
        ('HYPERGRAPH + noise', hyper_agg, hyper_floor),
    ]:
        real_df = agg_df[~agg_df['is_noise']].copy()
        nm, ns  = floor['mean'], floor['std']
        print(f"\n  {label}:")
        print(f"  {'Feature':<35} {'Mean':>8}  {'Std':>8}  {'vs noise'}  {'Signal?'}")
        print(f"  {'-'*75}")
        for _, row in real_df.iterrows():
            delta  = row['mean'] - nm
            signal = '✓  ABOVE NOISE' if row['mean'] > nm + ns else '–  within noise'
            print(f"  {row['feature']:<35} {row['mean']:>8.5f}  {row['std']:>8.5f}"
                  f"  {delta:>+8.5f}  {signal}")

    print(f"\n{'='*70}")


# =======================================================
# MAIN
# =======================================================

if __name__ == '__main__':

    start = time.time()
    print(f"Started at {time.strftime('%H:%M:%S')}")

    # --- Output ---
    output_dir = CONFIG["BASE_OUTPUT_DIR"] / "drug_target_family_splits"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("  NOISE INJECTION — DIMENSIONALITY CONTROL")
    print(f"  Task   : Drug Target Prediction")
    print(f"  Model  : {CONFIG['MODEL_TYPE']}")
    print(f"  N_NOISE: {CONFIG['N_NOISE']}   N_SEEDS: {CONFIG['N_SEEDS']}")
    print(f"  Output : {output_dir}")
    print(f"{'='*70}\n")

    # --- Load data ---
    features_df = load_all_features()
    splits_df   = load_splits()
    split_indices = sorted(splits_df['split_index'].unique())

    # --- Resolve feature lists ---
    hypergraph_features = [f for f in CONFIG["FEATURES"]["HYPERGRAPH"]
                           if f in features_df.columns]
    pairwise_features   = [f for f in CONFIG["FEATURES"]["PAIRWISE"]
                           if f in features_df.columns]

    print(f"\n   Hypergraph features ({len(hypergraph_features)}): {hypergraph_features}")
    print(f"   Pairwise features   ({len(pairwise_features)}):   {pairwise_features}")
    print(f"   Noise features      ({CONFIG['N_NOISE']}):   "
          f"[standard normal, {CONFIG['N_SEEDS']} independent seeds per split]\n")

    # Fill NaNs
    all_feats = hypergraph_features + pairwise_features
    n_nans = features_df[all_feats].isna().sum().sum()
    if n_nans:
        print(f"   Filling {n_nans} NaN values with 0.")
        features_df[all_feats] = features_df[all_feats].fillna(0)

    # --- Main loop ---
    print(f"3. Running {len(split_indices)} splits × {CONFIG['N_SEEDS']} seeds "
          f"× 3 models = "
          f"{len(split_indices) * CONFIG['N_SEEDS'] * 3} model fits...\n")

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
            # Quick progress line: mean PR-AUC over seeds for this split
            rand_mean  = np.mean([r['rand_pr_auc']  for r in records])
            pair_mean  = np.mean([r['pair_pr_auc']  for r in records])
            hyper_mean = np.mean([r['hyper_pr_auc'] for r in records])
            print(f"rand={rand_mean:.4f}  pair={pair_mean:.4f}  hyper={hyper_mean:.4f}")
        except Exception as e:
            print(f"ERROR: {e}")
            raise

    # --- Aggregate importances ---
    print("\n4. Aggregating importances...")

    pair_agg  = aggregate_importances(all_records, 'pair',  pairwise_features,   CONFIG["N_NOISE"])
    hyper_agg = aggregate_importances(all_records, 'hyper', hypergraph_features, CONFIG["N_NOISE"])
    rand_agg  = aggregate_importances(all_records, 'rand',  [],                  CONFIG["N_NOISE"])

    pair_floor  = build_noise_floor(all_records, 'pair',  CONFIG["N_NOISE"])
    hyper_floor = build_noise_floor(all_records, 'hyper', CONFIG["N_NOISE"])
    rand_floor  = build_noise_floor(all_records, 'rand',  CONFIG["N_NOISE"])

    print_noise_summary(pair_agg, hyper_agg, pair_floor, hyper_floor, rand_floor)

    # --- Plots ---
    print("\n5. Generating plots...")
    plot_importance_vs_noise(pair_agg, hyper_agg, pair_floor, hyper_floor,
                             output_dir, CONFIG["N_NOISE"])
    plot_seed_variability(all_records, hypergraph_features, pairwise_features,
                          CONFIG["N_NOISE"], output_dir)
    plot_noise_floor_comparison(rand_floor, pair_floor, hyper_floor, output_dir)

    # --- Save CSVs ---
    print("\n6. Saving CSVs...")

    # Full per-(split × seed) records — wide format, one row per seed
    records_df = pd.DataFrame(all_records)
    records_df.to_csv(output_dir / 'noise_injection_records.csv', index=False)
    print("   Saved: noise_injection_records.csv")

    # Aggregated importance tables
    pair_agg.to_csv(output_dir  / 'pairwise_noise_importance.csv',  index=False)
    hyper_agg.to_csv(output_dir / 'hypergraph_noise_importance.csv', index=False)
    rand_agg.to_csv(output_dir  / 'random_noise_importance.csv',     index=False)
    print("   Saved: pairwise_noise_importance.csv")
    print("   Saved: hypergraph_noise_importance.csv")
    print("   Saved: random_noise_importance.csv")

    # Noise floor summary
    noise_floor_df = pd.DataFrame([
        {'model': 'pure_random',       **rand_floor},
        {'model': 'pairwise_noise',    **pair_floor},
        {'model': 'hypergraph_noise',  **hyper_floor},
    ])
    noise_floor_df.to_csv(output_dir / 'noise_floor_summary.csv', index=False)
    print("   Saved: noise_floor_summary.csv")

    elapsed = time.time() - start
    print(f"\n{'='*70}")
    print(f"  COMPLETE — {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'='*70}")