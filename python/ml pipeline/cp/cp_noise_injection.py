"""
Noise-injection dimensionality control — seed-to-seed variability figure.

Produces a single deliverable: noise_seed_variability.png, a two-panel
boxplot (pairwise+noise | hypergraph+noise) showing the distribution of
permutation importance across (splits x seeds) for every real feature and
every injected noise feature, with real and noise features interleaved by
rank so the separation is honest.

This is the trimmed version: it fits only the two models the figure needs
(pairwise+noise and hypergraph+noise) and drops the pure-random model, the
noise-floor / importance-vs-noise plots, and all ranked-table machinery.
"""

import time
import numpy as np
import pandas as pd
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

# =======================================================
# Plotting style  (all font sizes >= 12)
# =======================================================
plt.rcParams.update({
    'font.size':        16,
    'axes.titlesize':   18,
    'axes.labelsize':   16,
    'xtick.labelsize':  12,
    'ytick.labelsize':  14,
    'legend.fontsize':  12,
    'figure.titlesize': 20,
})

# =======================================================
# TASK METADATA
# =======================================================
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
    # --- Task: "ess" | "chembl" | "hpa" ---
    "TASK": "hpa",

    # --- Paths ---
    "DATA_DIR":        Path("/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/cp/"),
    "BASE_OUTPUT_DIR": Path("./randomforest"),

    # --- Feature files ---
    "PROTEIN_FEATURES_FILE":  "hypergraph_features.csv",
    "PAIRWISE_FEATURES_FILE": "pairwise_features.csv",

    # --- Model ---
    "MODEL_TYPE":   "RandomForest",   # "RandomForest" | "LightGBM" | "XGBoost"
    "RANDOM_STATE": 42,
    "N_SPLITS_CV":  5,

    # --- Noise injection ---
    "N_NOISE":  5,
    "N_SEEDS":  10,

    # --- Cap on number of splits used (None = all splits in the file, i.e. 50).
    #     15 is statistically ample for this diagnostic and much faster. ---
    "MAX_SPLITS": None,

    # --- Regenerate the figure from a saved records CSV without refitting ---
    "REGENERATE_PLOT_ONLY": False,

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
    # "HYPERGRAPH" here is the full 15-feature hb-graph set (structural +
    # stoichiometry + participation) that the "Hypergraph + noise" panel plots.
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
            'protein_RangeUniqueRatio',
            'protein_MedComplexNodes',
            'protein_RangeComplexNodes',
            'protein_NormUniqueSum',      # replaces protein_MedianUniqueRatio
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
# HELPERS
# =======================================================

def get_task_meta() -> Dict:
    task = CONFIG["TASK"]
    if task not in TASK_META:
        raise ValueError(f"Unknown TASK {task!r}. Choose from: {list(TASK_META)}")
    return TASK_META[task]


def get_output_dir() -> Path:
    task  = CONFIG["TASK"]
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
    print(f"   Combined shape: {combined.shape}")
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
    labelled = splits_df[splits_df['label_mask']].drop_duplicates('ProteinId')
    n_pos    = (labelled['target'] == 1).sum()
    print(f"   Splits in file: {n_splits}  |  "
          f"Labelled proteins: {len(labelled)} ({100*n_pos/len(labelled):.1f}% positive)")
    return splits_df


# =======================================================
# MODEL HELPERS
# =======================================================

def make_estimator(y_train: pd.Series):
    """Untuned estimator with fixed, data-dependent constructor args."""
    model_type = CONFIG["MODEL_TYPE"]
    if model_type == "RandomForest":
        return RandomForestClassifier(random_state=CONFIG["RANDOM_STATE"])
    elif model_type == "LightGBM":
        return LGBMClassifier(random_state=CONFIG["RANDOM_STATE"], n_jobs=1, verbose=-1)
    elif model_type == "XGBoost":
        pos = int((y_train == 1).sum())
        neg = int((y_train == 0).sum())
        spw = float(neg) / float(pos) if pos > 0 else 1.0
        return XGBClassifier(
            random_state=CONFIG["RANDOM_STATE"], n_jobs=-1,
            verbosity=0, eval_metric='logloss', scale_pos_weight=spw,
        )
    raise ValueError(f"Unknown MODEL_TYPE: {CONFIG['MODEL_TYPE']!r}")


def tune_hyperparams(X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
    """Grid-search once and return the best hyperparameters (not the model)."""
    gs = GridSearchCV(
        make_estimator(y_train),
        CONFIG["PARAM_GRIDS"][CONFIG["MODEL_TYPE"]],
        scoring='average_precision',
        cv=CONFIG["N_SPLITS_CV"], n_jobs=-1, verbose=0,
    )
    gs.fit(X_train, y_train)
    return gs.best_params_


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


def make_noise_cols(n_rows: int, n_noise: int, seed: int, prefix: str = 'rand') -> pd.DataFrame:
    rng  = np.random.default_rng(seed)
    data = rng.standard_normal((n_rows, n_noise))
    cols = [f'{prefix}_{j}' for j in range(n_noise)]
    return pd.DataFrame(data, columns=cols)


# =======================================================
# PER-SPLIT RUNNER  (pairwise+noise and hypergraph+noise only)
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
    For a single pre-assigned split, fit the pairwise+noise and
    hb-graph+noise models across n_seeds independent noise seeds, and record
    the permutation importance of every real and noise feature.

    Hyperparameters are tuned ONCE per (split, model) on the real feature set
    and held fixed across all seeds, so seed-to-seed variability reflects the
    noise draws rather than hyperparameter re-selection.
    """
    split_info = splits_df[splits_df['split_index'] == split_idx][
        ['ProteinId', 'split', 'target', 'label_mask']
    ].copy()

    df       = pd.merge(merged_df, split_info, on='ProteinId', how='inner')
    labelled = df[df['label_mask']].copy()
    train_df = labelled[labelled['split'] == 'train']
    test_df  = labelled[labelled['split'] == 'test']

    y_train = train_df['target'].astype(int)
    y_test  = test_df['target'].astype(int)

    # Tune once per (split, model) on the real feature set (no noise).
    model_specs = [('pair', pairwise_features), ('hyper', hypergraph_features)]
    best_params = {
        prefix: tune_hyperparams(train_df[feats], y_train)
        for prefix, feats in model_specs
    }

    records = []

    for seed_idx in range(n_seeds):
        noise_seed = CONFIG["RANDOM_STATE"] + split_idx * 1000 + seed_idx

        # Same noise columns injected into BOTH models within a seed (fair).
        noise_train = make_noise_cols(len(train_df), n_noise, noise_seed)
        noise_test  = make_noise_cols(len(test_df),  n_noise, noise_seed + 500)
        noise_train.index = train_df.index
        noise_test.index  = test_df.index

        record = {
            'split_index': split_idx,
            'seed_index':  seed_idx,
            'noise_seed':  noise_seed,
            'n_train':     len(train_df),
            'n_test':      len(test_df),
        }

        for prefix, feats in model_specs:
            X_train = pd.concat(
                [train_df[feats].reset_index(drop=True),
                 noise_train.reset_index(drop=True)], axis=1
            )
            X_test = pd.concat(
                [test_df[feats].reset_index(drop=True),
                 noise_test.reset_index(drop=True)], axis=1
            )
            X_train.index = train_df.index
            X_test.index  = test_df.index

            # Fit with the pre-tuned params for this split — no per-seed tuning.
            model = make_estimator(y_train).set_params(**best_params[prefix])
            model.fit(X_train, y_train)
            imp = perm_importance(model, X_test, y_test)

            proba = model.predict_proba(X_test)[:, 1]
            record[f'{prefix}_pr_auc'] = float(average_precision_score(y_test, proba))
            for feat, val in imp.items():
                record[f'{prefix}__{feat}'] = val

        records.append(record)

    return records


# =======================================================
# PLOT — seed-to-seed variability (the only deliverable)
# =======================================================

def plot_seed_variability(
    all_records: List[Dict],
    hypergraph_features: List[str],
    pairwise_features: List[str],
    n_noise: int,
    output_dir: Path,
):
    meta = get_task_meta()
    # Vertical stack: each panel spans the full figure width, giving the
    # 20-box hb-graph panel room to breathe. Tune figsize to taste — taller
    # reads larger in a two-column layout.
    fig, axes = plt.subplots(2, 1, figsize=(14, 13))

    for model_prefix, real_features, colour, ax, title in [
        ('pair',  pairwise_features,   '#4393c3', axes[0], 'Pairwise + noise'),
        ('hyper', hypergraph_features, '#d6604d', axes[1], 'Hb-graph + noise'),
    ]:
        prefix   = model_prefix + '__'
        all_cols = real_features + [f'rand_{j}' for j in range(n_noise)]

        # Collect importances per feature, then sort by mean descending so
        # real and noise features are interleaved at their honest rank.
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

        bp = ax.boxplot(data, patch_artist=True,
                        medianprops=dict(color='black', linewidth=2))
        for patch, c in zip(bp['boxes'], colours):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=12)
        for tick_label, c in zip(ax.get_xticklabels(), colours):
            tick_label.set_color(c)
        ax.set_ylabel('Permutation Importance (PR-AUC drop)')
        ax.set_title(title)
        ax.axhline(0, color='black', linewidth=0.8, linestyle=':')

        real_patch  = mpatches.Patch(color=colour,    alpha=0.7, label='Real features')
        noise_patch = mpatches.Patch(color='#cccccc', alpha=0.7, label='Noise features')
        ax.legend(handles=[real_patch, noise_patch], fontsize=12)

    plt.suptitle(
        f'Seed-to-Seed Variability — {meta["DISPLAY_NAME"]}\n'
        f'({n_noise} noise features × {CONFIG["N_SEEDS"]} seeds)',
        y=1.01,
    )
    plt.tight_layout()
    path = output_dir / 'noise_seed_variability.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {path}")


# =======================================================
# MAIN
# =======================================================

def main():
    output_dir = get_output_dir()
    meta       = get_task_meta()

    print(f"\n{'='*70}")
    print("  NOISE INJECTION — SEED-TO-SEED VARIABILITY FIGURE")
    print(f"  Task   : {meta['DISPLAY_NAME']}")
    print(f"  Model  : {CONFIG['MODEL_TYPE']}")
    print(f"  N_NOISE: {CONFIG['N_NOISE']}   N_SEEDS: {CONFIG['N_SEEDS']}")
    print(f"  Output : {output_dir}")
    print(f"{'='*70}\n")

    hypergraph_features = CONFIG["FEATURES"]["HYPERGRAPH"]
    pairwise_features   = CONFIG["FEATURES"]["PAIRWISE"]
    records_path        = output_dir / 'noise_injection_records.csv'

    # ── Fast path: rebuild the figure from a saved records CSV ─────────────
    if CONFIG["REGENERATE_PLOT_ONLY"]:
        if not records_path.exists():
            raise FileNotFoundError(f"No records CSV to plot: {records_path}")
        print(f"Regenerating figure from {records_path.name} (no refitting)...")
        all_records = pd.read_csv(records_path).to_dict('records')
        plot_seed_variability(all_records, hypergraph_features, pairwise_features,
                              CONFIG["N_NOISE"], output_dir)
        return

    # ── Full run ───────────────────────────────────────────────────────────
    start = time.time()
    print(f"Started at {time.strftime('%H:%M:%S')}")

    features_df   = load_all_features()
    splits_df     = load_splits()
    split_indices = sorted(splits_df['split_index'].unique())
    if CONFIG["MAX_SPLITS"] is not None:
        split_indices = split_indices[:CONFIG["MAX_SPLITS"]]

    # Keep only features present in the CSVs (guards against renames).
    hypergraph_features = [f for f in hypergraph_features if f in features_df.columns]
    pairwise_features   = [f for f in pairwise_features   if f in features_df.columns]
    print(f"\n   Hypergraph features ({len(hypergraph_features)}): {hypergraph_features}")
    print(f"   Pairwise features   ({len(pairwise_features)}): {pairwise_features}")

    all_feats = hypergraph_features + pairwise_features
    n_nans    = features_df[all_feats].isna().sum().sum()
    if n_nans:
        print(f"   Filling {n_nans} NaN values with 0.")
        features_df[all_feats] = features_df[all_feats].fillna(0)

    n_splits = len(split_indices)
    print(f"\n3. Tuning 2 models once per split ({2 * n_splits} grid searches), "
          f"then {n_splits} splits × {CONFIG['N_SEEDS']} seeds × 2 models = "
          f"{n_splits * CONFIG['N_SEEDS'] * 2} fits (no per-seed re-tuning)...\n")

    all_records: List[Dict] = []
    for split_idx in split_indices:
        print(f"   Split {split_idx:>2}/{len(split_indices)}...", end=" ", flush=True)
        records = run_split_noise(
            split_idx, features_df, splits_df,
            hypergraph_features, pairwise_features,
            CONFIG["N_NOISE"], CONFIG["N_SEEDS"],
        )
        all_records.extend(records)
        pair_mean  = np.mean([r['pair_pr_auc']  for r in records])
        hyper_mean = np.mean([r['hyper_pr_auc'] for r in records])
        print(f"PR-AUC  pair={pair_mean:.3f}  hyper={hyper_mean:.3f}")

    # Save raw records (lets you re-plot later via REGENERATE_PLOT_ONLY).
    print("\n4. Saving records + figure...")
    pd.DataFrame(all_records).to_csv(records_path, index=False)
    print(f"   Saved: {records_path.name}")

    plot_seed_variability(all_records, hypergraph_features, pairwise_features,
                          CONFIG["N_NOISE"], output_dir)

    elapsed = time.time() - start
    print(f"\n{'='*70}")
    print(f"  COMPLETE — {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()