import os
import pandas as pd
import numpy as np
import matplotlib .pyplot as plt
from pathlib import Path
from typing import List, Dict
import time

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sklearn.metrics import classification_report, average_precision_score
from sklearn.inspection import permutation_importance
from scipy.stats import binomtest

# =======================================================
# Plotting Style Configuration
# =======================================================
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 20
})

# =======================================================
# CONFIGURATION
# =======================================================
CONFIG = {
    # --- Paths ---
    "DATA_DIR": Path("/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/cp/"),
    "BASE_OUTPUT_DIR": Path("./randomforest/hpa_two_excl_stoich_features"),

    # --- File Names ---
    "SPLITS_FILE":           "hpa_protein_merged_splits.csv",
    "PROTEIN_FEATURES_FILE": "hypergraph_features.csv",
    "PAIRWISE_FEATURES_FILE":"pairwise_features.csv",

    # --- Model ---
    # Options: "RandomForest" | "LightGBM" | "XGBoost"
    "MODEL_TYPE": "RandomForest",

    # --- Fixed settings ---
    "RANDOM_STATE": 42,
    "N_SPLITS_CV":  5,
    "N_RANDOM_SEEDS": 5,   # Independent random-padding runs to average per split
    "N_RANDOM_PAD": 2,     # Number of noise columns to add (keep low to avoid drowning signal)

    # --- Model-Specific Hyperparameter Grids for GridSearchCV ---
    "PARAM_GRIDS": {
        "RandomForest": {
            'n_estimators':      [80, 100, 200],
            'max_depth':         [None, 5, 10],
            'min_samples_split': [2, 5, 10],
            'class_weight':      ['balanced']
        },
        "LightGBM": {
            'n_estimators':  [80, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth':     [None, 5, 10],
            'num_leaves':    [30, 50, 100],
            'class_weight':  ['balanced']
        },
        "XGBoost": {
            'n_estimators':  [80, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth':     [None, 5, 10],
            'subsample':     [0.75, 0.8, 1.0],
            # scale_pos_weight is set automatically from training data (see tune_and_train_model)
        }
    },

    # --- Feature Selection ---
    # Comment/uncomment individual features to include or exclude them.
    "FEATURES": {
        "HYPERGRAPH": [
            # --- Base / native hypergraph metrics ---
            'base_Degree',
            'base_LocalClustCoeff',
            # 'base_ComponentSize',
            # 'base_ComponentEdgeNodeRatio',
            'base_TriangleCount',
            'base_UniquePartners',
            'base_AvgNeighbourDegree',
            # 'base_BetweennessCentrality',
            # 'base_EigenvectorCentrality',
            # 'base_KatzCentrality',

            # --- Stoichiometry-based metrics ---
            'stoich_WeightedTriangles',
            'stoich_AvgNeighbourDegreeStoich',
            'stoich_RangeComplexSize',
            'stoich_MedComplexSize',
            'stoich_MedianRatio',
            'stoich_RangeRatio',

            # --- Protein-participation metrics ---
            'protein_MedianUniqueRatio',
            'protein_RangeUniqueRatio',
            'protein_MedComplexNodes',
            'protein_RangeComplexNodes',
        ],

        # Stoichiometry features to ablate — must be a subset of HYPERGRAPH above.
        # The no-stoich hypergraph feature set is derived automatically as
        # HYPERGRAPH minus STOICHIOMETRY_FEATURES.
        "STOICHIOMETRY_FEATURES": [
            'stoich_WeightedTriangles',
            'stoich_AvgNeighbourDegreeStoich',
            'stoich_RangeComplexSize',
            'stoich_MedComplexSize',
            'stoich_MedianRatio',
            'stoich_RangeRatio',
        ],
        "PAIRWISE": [
            'pair_Degree',
            'pair_LocalClustCoeff',
            'pair_TriangleCount',
            # 'pair_ComponentSize',
            # 'pair_EigenvectorCentrality',
            # 'pair_BetweennessCentrality',
            # 'pair_KatzCentrality',
            'pair_AvgNeighborDegree',
        ],

        # Hypergraph-exclusive features — those with NO pairwise analogue.
        # Used for the dimensionality control experiment: if these 5 features
        # (which pairwise cannot express) outperform the 4 pairwise features,
        # the advantage is genuine higher-order signal, not a feature-count artefact.
        #
        # Excluded from this list (have pairwise analogues):
        #   base_UniquePartners  ≈ pair_Degree (both count distinct co-members)
        #   base_AvgNeighbourDegree ≈ pair_AvgNeighborDegree (conceptually similar)
        #   base_TriangleCount   ≈ pair_TriangleCount (related structural property)
        #   base_LocalClustCoeff ≈ pair_LocalClustCoeff
        #
        # Included (no pairwise equivalent):
        #   base_Degree = number of complexes (hyperedges) — pairwise has no notion of this
        #   protein_* features = complex-level participation statistics
        "HYPER_EXCLUSIVE": [
            'base_Degree',                  # hyperedge membership count
            'protein_MedianUniqueRatio',    # median unique-to-total member ratio
            'protein_RangeUniqueRatio',     # spread of that ratio
            'protein_MedComplexNodes',      # median complex size (hyperedge cardinality)
            'protein_RangeComplexNodes',    # spread of complex sizes
        ],
    }
}

splits_path = CONFIG["DATA_DIR"] / CONFIG["SPLITS_FILE"]
print(f"   Splits file last modified: {pd.Timestamp(os.path.getmtime(splits_path), unit='s')}")
print(f"   Splits file rows: {pd.read_csv(splits_path).shape}")


# =======================================================
# DATA LOADING
# =======================================================

def load_all_features() -> pd.DataFrame:
    """Loads hypergraph and pairwise feature CSVs and merges them on ProteinId."""
    print("1. Loading feature data...")

    hg_df   = pd.read_csv(CONFIG["DATA_DIR"] / CONFIG["PROTEIN_FEATURES_FILE"])
    pair_df = pd.read_csv(CONFIG["DATA_DIR"] / CONFIG["PAIRWISE_FEATURES_FILE"])

    combined = pd.merge(hg_df, pair_df, on='ProteinId', how='inner')

    print(f"   Hypergraph features shape : {hg_df.shape}")
    print(f"   Pairwise features shape   : {pair_df.shape}")
    print(f"   Combined shape            : {combined.shape}")
    return combined


def load_splits() -> pd.DataFrame:
    """
    Loads the pre-assigned family-level splits file.

    Expected columns:
        split_index   — integer 1..N identifying which split
        UniProt_AC    — protein identifier (matches ProteinId in feature files)
        split         — 'train' or 'test'
        protein_label — 'Essential' | 'Non-essential' | 'Unknown'
        label_mask    — bool; False for Unknown proteins (excluded from metrics)
    """
    print("2. Loading pre-assigned splits...")
    splits_df = pd.read_csv(CONFIG["DATA_DIR"] / CONFIG["SPLITS_FILE"])

    # Rename to match feature file key
    splits_df = splits_df.rename(columns={'UniProt_AC': 'ProteinId'})

    # Encode binary target: Drug_target=1, Non_target=0; Unknown kept as NaN
    label_map = {'Drug_target': 1, 'Non_target': 0}
    splits_df['target'] = splits_df['protein_label'].map(label_map)

    n_splits = splits_df['split_index'].nunique()
    print(f"   Splits file rows  : {len(splits_df)}")
    print(f"   Unique proteins   : {splits_df['ProteinId'].nunique()}")
    print(f"   Number of splits  : {n_splits}")

    labelled = splits_df[splits_df['label_mask']].drop_duplicates('ProteinId')
    n_ess = (labelled['target'] == 1).sum()
    n_tot = len(labelled)
    print(f"   Labelled proteins : {n_tot}  ({100*n_ess/n_tot:.1f}% essential)")

    return splits_df


# =======================================================
# MODEL TRAINING & EVALUATION
# =======================================================

def tune_and_train_model(X_train: pd.DataFrame, y_train: pd.Series):
    """Hyperparameter search + fit.  Returns (best_estimator, best_params)."""
    model_type = CONFIG["MODEL_TYPE"]

    if model_type == "RandomForest":
        base_model = RandomForestClassifier(random_state=CONFIG["RANDOM_STATE"])
        param_grid = CONFIG["PARAM_GRIDS"]["RandomForest"]

    elif model_type == "LightGBM":
        base_model = LGBMClassifier(
            random_state=CONFIG["RANDOM_STATE"], n_jobs=1, verbose=-1
        )
        param_grid = CONFIG["PARAM_GRIDS"]["LightGBM"]

    elif model_type == "XGBoost":
        pos = int((y_train == 1).sum())
        neg = int((y_train == 0).sum())
        spw = float(neg) / float(pos) if pos > 0 else 1.0
        base_model = XGBClassifier(
            random_state=CONFIG["RANDOM_STATE"],
            n_jobs=-1,
            verbosity=0,
            eval_metric='logloss',
            scale_pos_weight=spw
        )
        param_grid = CONFIG["PARAM_GRIDS"]["XGBoost"]

    else:
        raise ValueError(f"Unknown MODEL_TYPE: '{model_type}'")

    gs = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='average_precision',
        cv=CONFIG["N_SPLITS_CV"],
        n_jobs=-1,
        verbose=0
    )
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
    """Returns PR-AUC, F1 for the positive class, and predicted probabilities."""
    y_pred       = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(
        y_test, y_pred,
        target_names=['Non-Essential', 'Essential'],
        output_dict=True
    )

    return {
        'pr_auc':       average_precision_score(y_test, y_pred_proba),
        'f1':           report['Essential']['f1-score'],
        'y_pred_proba': y_pred_proba
    }


def compute_permutation_importance(
    model, X_test: pd.DataFrame, y_test: pd.Series, n_repeats: int = 10
) -> Dict[str, float]:
    """Permutation importance scored by average_precision (PR-AUC drop)."""
    result = permutation_importance(
        model, X_test, y_test,
        scoring='average_precision',
        n_repeats=n_repeats,
        random_state=CONFIG["RANDOM_STATE"],
        n_jobs=-1
    )
    return dict(zip(X_test.columns, result.importances_mean))


# =======================================================
# PER-SPLIT RUNNER
# =======================================================

def run_split(
    split_idx: int,
    merged_df: pd.DataFrame,
    splits_df: pd.DataFrame,
    hypergraph_features: List[str],
    hypergraph_no_stoich_features: List[str],
    pairwise_features: List[str],
    hyper_exclusive_features: List[str],
) -> Dict:
    """
    Runs both hypergraph and pairwise models for a single pre-assigned split.

    merged_df   — feature matrix (ProteinId + all feature columns)
    splits_df   — full splits table (all split indices)
    Returns a results dict with PR-AUC, F1, importances, and per-protein predictions.
    """
    # --- Extract this split's assignments ---
    split_mask = splits_df['split_index'] == split_idx
    split_info = splits_df[split_mask][['ProteinId', 'split', 'target', 'label_mask']].copy()

    # Merge features with split assignments
    df = pd.merge(merged_df, split_info, on='ProteinId', how='inner')

    # Only use labelled proteins for training/evaluation
    labelled_df = df[df['label_mask']].copy()

    train_df = labelled_df[labelled_df['split'] == 'train']
    test_df  = labelled_df[labelled_df['split'] == 'test']

    y_train = train_df['target'].astype(int)
    y_test  = test_df['target'].astype(int)

    results = {
        'split_index':  split_idx,
        'n_train':      len(train_df),
        'n_test':       len(test_df),
        'train_ess_pct': 100 * y_train.mean(),
        'test_ess_pct':  100 * y_test.mean(),
    }

    # --- Hypergraph model ---
    X_hyper_train = train_df[hypergraph_features]
    X_hyper_test  = test_df[hypergraph_features]

    hyper_model, hyper_params = tune_and_train_model(X_hyper_train, y_train)
    hyper_eval = evaluate_model(hyper_model, X_hyper_test, y_test)

    results['hypergraph_pr_auc']    = hyper_eval['pr_auc']
    results['hypergraph_f1']        = hyper_eval['f1']
    results['hypergraph_best_params'] = hyper_params
    results['hypergraph_importance'] = compute_permutation_importance(
        hyper_model, X_hyper_test, y_test
    )

    # Store per-protein predictions (hypergraph)
    hyper_preds = test_df[['ProteinId']].copy()
    hyper_preds['split_index']       = split_idx
    hyper_preds['true_label']        = y_test.values
    hyper_preds['hyper_pred_proba']  = hyper_eval['y_pred_proba']
    results['hypergraph_predictions'] = hyper_preds

    # --- Hypergraph WITHOUT stoichiometry model ---
    X_nostoich_train = train_df[hypergraph_no_stoich_features]
    X_nostoich_test  = test_df[hypergraph_no_stoich_features]

    nostoich_model, nostoich_params = tune_and_train_model(X_nostoich_train, y_train)
    nostoich_eval = evaluate_model(nostoich_model, X_nostoich_test, y_test)

    results['hyper_nostoich_pr_auc']      = nostoich_eval['pr_auc']
    results['hyper_nostoich_f1']          = nostoich_eval['f1']
    results['hyper_nostoich_best_params'] = nostoich_params
    results['hyper_nostoich_importance']  = compute_permutation_importance(
        nostoich_model, X_nostoich_test, y_test
    )

    nostoich_preds = test_df[['ProteinId']].copy()
    nostoich_preds['split_index']            = split_idx
    nostoich_preds['true_label']             = y_test.values
    nostoich_preds['nostoich_pred_proba']    = nostoich_eval['y_pred_proba']
    results['hyper_nostoich_predictions']    = nostoich_preds

    # --- Pairwise model ---
    X_pair_train = train_df[pairwise_features]
    X_pair_test  = test_df[pairwise_features]

    pair_model, pair_params = tune_and_train_model(X_pair_train, y_train)
    pair_eval = evaluate_model(pair_model, X_pair_test, y_test)

    results['pairwise_pr_auc']    = pair_eval['pr_auc']
    results['pairwise_f1']        = pair_eval['f1']
    results['pairwise_best_params'] = pair_params
    results['pairwise_importance'] = compute_permutation_importance(
        pair_model, X_pair_test, y_test
    )

    # Store per-protein predictions (pairwise)
    pair_preds = test_df[['ProteinId']].copy()
    pair_preds['split_index']      = split_idx
    pair_preds['true_label']       = y_test.values
    pair_preds['pair_pred_proba']  = pair_eval['y_pred_proba']
    results['pairwise_predictions'] = pair_preds

    # --- Hypergraph-exclusive model (dimensionality control #1) ---
    # Features that have NO pairwise analogue (5 features vs 4 pairwise).
    X_excl_train = train_df[hyper_exclusive_features]
    X_excl_test  = test_df[hyper_exclusive_features]

    excl_model, excl_params = tune_and_train_model(X_excl_train, y_train)
    excl_eval = evaluate_model(excl_model, X_excl_test, y_test)

    results['hyper_exclusive_pr_auc']       = excl_eval['pr_auc']
    results['hyper_exclusive_f1']           = excl_eval['f1']
    results['hyper_exclusive_best_params']  = excl_params
    results['hyper_exclusive_importance']   = compute_permutation_importance(
        excl_model, X_excl_test, y_test
    )

    # --- Pairwise + random padding (dimensionality control #2) ---
    # Pad pairwise features with a small number of random noise columns.
    # If noise columns don't help, the hypergraph advantage isn't a
    # dimensionality artefact.  N_RANDOM_PAD is kept low (default 2) to
    # avoid drowning the 4 real pairwise features in noise.
    n_pad = CONFIG.get("N_RANDOM_PAD", 2)
    n_seeds = CONFIG.get("N_RANDOM_SEEDS", 5)
    padded_pr_aucs = []
    padded_f1s     = []

    for seed in range(n_seeds):
        rng = np.random.default_rng(CONFIG["RANDOM_STATE"] + seed + split_idx * 1000)

        train_pad = pd.DataFrame(
            rng.standard_normal((len(train_df), n_pad)),
            columns=[f'random_{j}' for j in range(n_pad)],
            index=train_df.index,
        )
        test_pad = pd.DataFrame(
            rng.standard_normal((len(test_df), n_pad)),
            columns=[f'random_{j}' for j in range(n_pad)],
            index=test_df.index,
        )

        X_padded_train = pd.concat([train_df[pairwise_features], train_pad], axis=1)
        X_padded_test  = pd.concat([test_df[pairwise_features],  test_pad],  axis=1)

        padded_model, _ = tune_and_train_model(X_padded_train, y_train)
        padded_eval     = evaluate_model(padded_model, X_padded_test, y_test)
        padded_pr_aucs.append(padded_eval['pr_auc'])
        padded_f1s.append(padded_eval['f1'])

    results['pairwise_padded_pr_auc']      = float(np.mean(padded_pr_aucs))
    results['pairwise_padded_pr_auc_std']  = float(np.std(padded_pr_aucs))
    results['pairwise_padded_f1']          = float(np.mean(padded_f1s))
    results['pairwise_padded_n_pad']       = n_pad
    results['pairwise_padded_n_seeds']     = n_seeds

    # Differences
    results['pr_auc_diff']          = results['hypergraph_pr_auc'] - results['pairwise_pr_auc']
    results['f1_diff']              = results['hypergraph_f1']     - results['pairwise_f1']
    results['stoich_pr_auc_diff']   = results['hypergraph_pr_auc'] - results['hyper_nostoich_pr_auc']
    results['stoich_f1_diff']       = results['hypergraph_f1']     - results['hyper_nostoich_f1']
    results['excl_vs_pair_pr_auc_diff']  = results['hyper_exclusive_pr_auc'] - results['pairwise_pr_auc']
    results['padded_vs_hyper_pr_auc_diff'] = results['hypergraph_pr_auc'] - results['pairwise_padded_pr_auc']

    return results


# =======================================================
# STATISTICAL COMPARISON
# =======================================================

def run_sign_test_comparison(all_results: List[Dict]) -> Dict:
    """Sign test (binomial) on paired PR-AUC wins/losses across splits.
    Covers three paired comparisons:
      1. Hypergraph (full) vs Pairwise
      2. Hypergraph (full) vs Hypergraph no-stoich  — stoichiometry ablation
      3. Hypergraph no-stoich vs Pairwise
    """
    hyper_vals    = np.array([r['hypergraph_pr_auc']      for r in all_results])
    nostoich_vals = np.array([r['hyper_nostoich_pr_auc']  for r in all_results])
    pair_vals     = np.array([r['pairwise_pr_auc']        for r in all_results])

    def _sign_test(a, b):
        diffs   = a - b
        n_wins  = int(np.sum(diffs > 0))
        n_loss  = int(np.sum(diffs < 0))
        n_ties  = int(np.sum(diffs == 0))
        n_valid = n_wins + n_loss
        if n_valid > 0:
            p_greater   = binomtest(n_wins, n_valid, 0.5, alternative='greater').pvalue
            p_two_sided = binomtest(n_wins, n_valid, 0.5, alternative='two-sided').pvalue
        else:
            p_greater = p_two_sided = 1.0
        return dict(wins=n_wins, losses=n_loss, ties=n_ties,
                    mean_diff=float(np.mean(diffs)), std_diff=float(np.std(diffs)),
                    p_greater=p_greater, p_two_sided=p_two_sided)

    hyper_vs_pair     = _sign_test(hyper_vals, pair_vals)
    stoich_vs_nostoich = _sign_test(hyper_vals, nostoich_vals)
    nostoich_vs_pair  = _sign_test(nostoich_vals, pair_vals)

    # --- Dimensionality control arms ---
    excl_vals   = np.array([r['hyper_exclusive_pr_auc']   for r in all_results])
    padded_vals = np.array([r['pairwise_padded_pr_auc']   for r in all_results])

    excl_vs_pair     = _sign_test(excl_vals, pair_vals)
    hyper_vs_padded  = _sign_test(hyper_vals, padded_vals)
    padded_vs_pair   = _sign_test(padded_vals, pair_vals)

    return {
        'n_runs': len(all_results),
        'hypergraph_pr_auc_mean':   float(np.mean(hyper_vals)),
        'hypergraph_pr_auc_std':    float(np.std(hyper_vals)),
        'hyper_nostoich_pr_auc_mean': float(np.mean(nostoich_vals)),
        'hyper_nostoich_pr_auc_std':  float(np.std(nostoich_vals)),
        'pairwise_pr_auc_mean':     float(np.mean(pair_vals)),
        'pairwise_pr_auc_std':      float(np.std(pair_vals)),
        'hyper_exclusive_pr_auc_mean': float(np.mean(excl_vals)),
        'hyper_exclusive_pr_auc_std':  float(np.std(excl_vals)),
        'pairwise_padded_pr_auc_mean': float(np.mean(padded_vals)),
        'pairwise_padded_pr_auc_std':  float(np.std(padded_vals)),
        # Legacy keys kept for downstream plot compatibility
        'mean_difference':          hyper_vs_pair['mean_diff'],
        'std_difference':           hyper_vs_pair['std_diff'],
        'hypergraph_wins':          hyper_vs_pair['wins'],
        'pairwise_wins':            hyper_vs_pair['losses'],
        'ties':                     hyper_vs_pair['ties'],
        'sign_test_p_greater':      hyper_vs_pair['p_greater'],
        'sign_test_p_two_sided':    hyper_vs_pair['p_two_sided'],
        # Stoichiometry ablation
        'stoich_ablation':          stoich_vs_nostoich,
        'nostoich_vs_pair':         nostoich_vs_pair,
        # Dimensionality controls
        'excl_vs_pair':             excl_vs_pair,
        'hyper_vs_padded':          hyper_vs_padded,
        'padded_vs_pair':           padded_vs_pair,
    }


# =======================================================
# FEATURE IMPORTANCE AGGREGATION
# =======================================================

def aggregate_feature_importance(
    all_results: List[Dict], representation: str
) -> pd.DataFrame:
    """
    Aggregates permutation importance across all splits.
    representation: 'hypergraph' or 'pairwise'
    """
    key = f'{representation}_importance'
    records = []
    for r in all_results:
        if key in r:
            for feat, imp in r[key].items():
                records.append({'split_index': r['split_index'],
                                'feature': feat, 'importance': imp})

    if not records:
        return pd.DataFrame()

    imp_df = pd.DataFrame(records)
    agg_df = (
        imp_df.groupby('feature')['importance']
        .agg(mean='mean', std='std', median='median',
             min='min', max='max', n_splits='count')
        .reset_index()
        .sort_values('mean', ascending=False)
        .reset_index(drop=True)
    )
    agg_df['rank'] = range(1, len(agg_df) + 1)
    return agg_df


# =======================================================
# PRINTING
# =======================================================

def print_statistical_summary(stats: Dict):
    print(f"\n{'='*70}")
    print("  STATISTICAL COMPARISON")
    print(f"{'='*70}")
    print(f"\n  Number of splits: {stats['n_runs']}")
    print(f"\n  {'Metric':<30} {'Mean ± Std'}")
    print(f"  {'-'*60}")
    print(f"  {'Hypergraph (full)':<30} "
          f"{stats['hypergraph_pr_auc_mean']:.4f} ± {stats['hypergraph_pr_auc_std']:.4f}")
    print(f"  {'Hypergraph (no stoich)':<30} "
          f"{stats['hyper_nostoich_pr_auc_mean']:.4f} ± {stats['hyper_nostoich_pr_auc_std']:.4f}")
    print(f"  {'Pairwise':<30} "
          f"{stats['pairwise_pr_auc_mean']:.4f} ± {stats['pairwise_pr_auc_std']:.4f}")

    def _print_comparison(label, d):
        print(f"\n  --- {label} ---")
        print(f"  Mean diff : {d['mean_diff']:+.4f} ± {d['std_diff']:.4f}")
        print(f"  Wins/Losses/Ties : {d['wins']}/{d['losses']}/{d['ties']}")
        print(f"  Sign test p (one-sided) : {d['p_greater']:.6f}")
        print(f"  Sign test p (two-sided) : {d['p_two_sided']:.6f}")

    _print_comparison("Hypergraph (full) vs Pairwise",
                      {'mean_diff': stats['mean_difference'],
                       'std_diff':  stats['std_difference'],
                       'wins':      stats['hypergraph_wins'],
                       'losses':    stats['pairwise_wins'],
                       'ties':      stats['ties'],
                       'p_greater': stats['sign_test_p_greater'],
                       'p_two_sided': stats['sign_test_p_two_sided']})
    _print_comparison("Hypergraph (full) vs Hypergraph (no stoich) — stoichiometry effect",
                      stats['stoich_ablation'])
    _print_comparison("Hypergraph (no stoich) vs Pairwise — representation effect alone",
                      stats['nostoich_vs_pair'])

    # --- Dimensionality controls ---
    if 'hyper_exclusive_pr_auc_mean' in stats:
        print(f"\n  {'Hyper-exclusive only':<30} "
              f"{stats['hyper_exclusive_pr_auc_mean']:.4f} ± {stats['hyper_exclusive_pr_auc_std']:.4f}")
        print(f"  {'Pairwise + random padding':<30} "
              f"{stats['pairwise_padded_pr_auc_mean']:.4f} ± {stats['pairwise_padded_pr_auc_std']:.4f}")
        _print_comparison("Hyper-exclusive (5) vs Pairwise (4) — higher-order signal only",
                          stats['excl_vs_pair'])
        _print_comparison("Hypergraph (full) vs Pairwise (padded) — dimensionality control",
                          stats['hyper_vs_padded'])
        _print_comparison("Pairwise (padded) vs Pairwise (unpadded) — noise sanity check",
                          stats['padded_vs_pair'])
    print(f"{'='*70}")


def print_feature_importance_summary(
    hyper_imp_df: pd.DataFrame, pair_imp_df: pd.DataFrame, top_n: int = 10
):
    print(f"\n{'='*70}")
    print("  FEATURE IMPORTANCE (Permutation — PR-AUC drop)")
    print(f"{'='*70}")
    for label, df in [("Hypergraph", hyper_imp_df), ("Pairwise", pair_imp_df)]:
        print(f"\n  Top {top_n} {label} Features:")
        print(f"  {'Rank':<6} {'Feature':<35} {'Mean':<12} {'Std':<10}")
        print(f"  {'-'*65}")
        for _, row in df.head(top_n).iterrows():
            print(f"  {int(row['rank']):<6} {row['feature']:<35} "
                  f"{row['mean']:.4f}       {row['std']:.4f}")
    print(f"\n  Note: Higher = more important; negative = possible noise.")
    print(f"{'='*70}")


# =======================================================
# PLOTTING
# =======================================================

def plot_paired_comparison(all_results: List[Dict], stats: Dict, output_dir: Path):
    """Three-panel comparison plot: histogram of diffs, scatter, and boxplot."""
    hyper_vals = np.array([r['hypergraph_pr_auc'] for r in all_results])
    pair_vals  = np.array([r['pairwise_pr_auc']   for r in all_results])
    diffs      = hyper_vals - pair_vals

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: histogram of differences
    ax1 = axes[0]
    ax1.hist(diffs, bins=10, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(0, color='red',   linestyle='--', linewidth=2, label='No difference')
    ax1.axvline(diffs.mean(), color='green', linestyle='-', linewidth=2,
                label=f'Mean diff: {diffs.mean():.4f}')
    ax1.set_xlabel('PR-AUC Difference (Hypergraph − Pairwise)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Paired Differences')
    ax1.legend()

    # Panel 2: scatter (one point per split)
    ax2 = axes[1]
    ax2.scatter(pair_vals, hyper_vals, alpha=0.7, s=60, zorder=3)
    lo = min(pair_vals.min(), hyper_vals.min()) - 0.02
    hi = max(pair_vals.max(), hyper_vals.max()) + 0.02
    ax2.plot([lo, hi], [lo, hi], 'r--', linewidth=2, label='y = x')
    ax2.set_xlabel('Pairwise PR-AUC')
    ax2.set_ylabel('Hypergraph PR-AUC')
    ax2.set_title('Paired Comparison — One Point per Split')
    ax2.set_xlim(lo, hi)
    ax2.set_ylim(lo, hi)
    ax2.set_aspect('equal')
    above = int(np.sum(hyper_vals > pair_vals))
    below = int(np.sum(hyper_vals < pair_vals))
    ax2.text(0.95, 0.05,
             f'Hypergraph wins: {above}\nPairwise wins: {below}',
             transform=ax2.transAxes, ha='right', va='bottom',
             bbox=dict(facecolor='lightgreen', alpha=0.5))

    # Panel 3: boxplot
    ax3 = axes[2]
    bp = ax3.boxplot([pair_vals, hyper_vals],
                     labels=['Pairwise', 'Hypergraph'],
                     patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgray')
    bp['boxes'][1].set_facecolor('steelblue')
    ax3.set_ylabel('PR-AUC')
    ax3.set_title('Distribution Comparison')
    rng = np.random.default_rng(0)
    for i, data in enumerate([pair_vals, hyper_vals]):
        x = rng.normal(i + 1, 0.04, size=len(data))
        ax3.scatter(x, data, alpha=0.4, s=20, color='black')

    plt.tight_layout()
    plt.savefig(output_dir / 'paired_comparison.png', dpi=300)
    plt.close()
    print("   Saved: paired_comparison.png")


def plot_stoich_ablation(all_results: List[Dict], stats: Dict, output_dir: Path):
    """
    Two-panel stoichiometry ablation figure matching the poster's Fig 5 style.

    Panel 1 — Scatter: full hypergraph vs no-stoich PR-AUC, one point per split.
               Points above the diagonal = stoichiometry helps.
    Panel 2 — Boxplot: pairwise / no-stoich / full hypergraph distributions,
               showing the stepwise improvement from adding stoichiometry.
    """
    hyper_vals    = np.array([r['hypergraph_pr_auc']     for r in all_results])
    nostoich_vals = np.array([r['hyper_nostoich_pr_auc'] for r in all_results])
    pair_vals     = np.array([r['pairwise_pr_auc']       for r in all_results])

    stoich_wins   = int(np.sum(hyper_vals > nostoich_vals))
    stoich_losses = int(np.sum(hyper_vals < nostoich_vals))

    ab  = stats['stoich_ablation']
    p_one = ab['p_greater']
    p_two = ab['p_two_sided']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Panel 1: scatter — full hypergraph vs no-stoich ──────────────────────
    ax1 = axes[0]
    lo = min(nostoich_vals.min(), hyper_vals.min()) - 0.02
    hi = max(nostoich_vals.max(), hyper_vals.max()) + 0.02
    ax1.scatter(nostoich_vals, hyper_vals, alpha=0.7, s=60, zorder=3,
                color='steelblue')
    ax1.plot([lo, hi], [lo, hi], 'r--', linewidth=2, label='y = x (no difference)')
    ax1.set_xlabel('Hypergraph (no stoich) PR-AUC')
    ax1.set_ylabel('Hypergraph (full) PR-AUC')
    ax1.set_title('Stoichiometry Ablation — One Point per Split')
    ax1.set_xlim(lo, hi)
    ax1.set_ylim(lo, hi)
    ax1.set_aspect('equal')
    ax1.legend(fontsize=12)
    ax1.text(0.97, 0.03,
             f'Stoich wins: {stoich_wins}\nNo-stoich wins: {stoich_losses}',
             transform=ax1.transAxes, ha='right', va='bottom', fontsize=12,
             bbox=dict(facecolor='lightgreen', alpha=0.5))

    # ── Panel 2: boxplot — pairwise / no-stoich / full hypergraph ─────────────
    ax2 = axes[1]
    colours = ['lightgray', 'lightsteelblue', 'steelblue']
    labels  = ['Pairwise', 'Hypergraph\n(no stoich)', 'Hypergraph\n(full)']
    bp = ax2.boxplot(
        [pair_vals, nostoich_vals, hyper_vals],
        labels=labels,
        patch_artist=True,
        medianprops=dict(color='black', linewidth=2),
    )
    for patch, colour in zip(bp['boxes'], colours):
        patch.set_facecolor(colour)
    ax2.set_ylabel('PR-AUC')
    ax2.set_title('Distribution Comparison')
    rng = np.random.default_rng(0)
    for i, data in enumerate([pair_vals, nostoich_vals, hyper_vals]):
        x = rng.normal(i + 1, 0.04, size=len(data))
        ax2.scatter(x, data, alpha=0.4, s=20, color='black', zorder=3)

    # Annotate with mean ± std for each box
    for i, (vals, lbl) in enumerate(zip(
            [pair_vals, nostoich_vals, hyper_vals],
            ['Pairwise', 'No-stoich', 'Full'])):
        ax2.text(i + 1, vals.max() + 0.005,
                 f'{vals.mean():.3f}±{vals.std():.3f}',
                 ha='center', va='bottom', fontsize=10)

    # Sign test annotation
    sig_label = (f'Stoich effect\np (one-sided) = {p_one:.4f}\n'
                 f'p (two-sided) = {p_two:.4f}')
    ax2.text(0.97, 0.03, sig_label,
             transform=ax2.transAxes, ha='right', va='bottom', fontsize=10,
             bbox=dict(facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_dir / 'stoich_ablation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   Saved: stoich_ablation.png")


def plot_feature_importance(
    hyper_imp_df: pd.DataFrame,
    pair_imp_df: pd.DataFrame,
    output_dir: Path,
    top_n: int = 15
):
    """Side-by-side horizontal bar charts of permutation importance."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, df, title, colour in zip(
        axes,
        [hyper_imp_df, pair_imp_df],
        [f'Top {top_n} Hypergraph Features', f'Top {top_n} Pairwise Features'],
        ['steelblue', 'gray']
    ):
        top = df.head(top_n)
        colors = [colour if v > 0 else 'lightcoral' for v in top['mean']]
        ax.barh(range(len(top)), top['mean'], xerr=top['std'],
                color=colors, edgecolor='black', capsize=3)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top['feature'])
        ax.invert_yaxis()
        ax.set_xlabel('Mean Permutation Importance (PR-AUC drop)')
        ax.set_title(title)
        ax.axvline(0, color='gray', linestyle='--', linewidth=1)

    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance_comparison.png', dpi=300)
    plt.close()
    print("   Saved: feature_importance_comparison.png")


def plot_dimensionality_control(all_results: List[Dict], stats: Dict, output_dir: Path):
    """
    Two-figure dimensionality control output.

    Figure 1 — Grouped bar chart: PR-AUC per split for all arms side by side.
               Shows complementary signal from pairwise and hyper-exclusive features,
               and that the full hypergraph combines both.
    Figure 2 — Two-panel scatter plots for the key paired comparisons.
    """
    hyper_vals    = np.array([r['hypergraph_pr_auc']        for r in all_results])
    pair_vals     = np.array([r['pairwise_pr_auc']          for r in all_results])
    excl_vals     = np.array([r['hyper_exclusive_pr_auc']   for r in all_results])
    padded_vals   = np.array([r['pairwise_padded_pr_auc']   for r in all_results])
    nostoich_vals = np.array([r['hyper_nostoich_pr_auc']    for r in all_results])

    n_splits = len(all_results)
    n_pad    = all_results[0].get('pairwise_padded_n_pad', '?')
    n_seeds  = all_results[0].get('pairwise_padded_n_seeds', '?')
    split_labels = [str(r['split_index']) for r in all_results]

    # ================================================================
    # Figure 1: Grouped bar chart — PR-AUC per split, all arms
    # ================================================================
    arms = {
        'Pairwise (4)':            {'vals': pair_vals,     'colour': '#b0b0b0'},
        f'Pair + noise (4+{n_pad})': {'vals': padded_vals,   'colour': '#f4a582'},
        'Hyper-exclusive (5)':     {'vals': excl_vals,     'colour': '#9e7bb5'},
        'Hyper no-stoich (7)':     {'vals': nostoich_vals, 'colour': '#92c5de'},
        'Hypergraph full (13)':    {'vals': hyper_vals,    'colour': '#4393c3'},
    }
    n_arms = len(arms)
    bar_width = 0.15
    x = np.arange(n_splits)

    fig, ax = plt.subplots(figsize=(max(14, n_splits * 1.2), 6))
    for i, (label, arm) in enumerate(arms.items()):
        offset = (i - n_arms / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, arm['vals'], bar_width,
                       label=label, color=arm['colour'], edgecolor='white', linewidth=0.5)

    ax.set_xlabel('Split index')
    ax.set_ylabel('PR-AUC')
    ax.set_title('Complementary Signal: Pairwise vs Hyper-exclusive vs Full Hypergraph')
    ax.set_xticks(x)
    ax.set_xticklabels(split_labels)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.set_ylim(0, None)

    # Add mean ± std summary text
    summary_lines = []
    for label, arm in arms.items():
        v = arm['vals']
        summary_lines.append(f"{label}: {v.mean():.3f}\u00b1{v.std():.3f}")
    ax.text(0.02, 0.98, '\n'.join(summary_lines),
            transform=ax.transAxes, va='top', ha='left', fontsize=10,
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='lightgray'),
            family='monospace')

    plt.tight_layout()
    plt.savefig(output_dir / 'dimensionality_grouped_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   Saved: dimensionality_grouped_bar.png")

    # ================================================================
    # Figure 2: Two-panel scatter — key paired comparisons
    # ================================================================
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # ── Panel 1: hyper-exclusive vs pairwise ─────────────────────────
    ax1 = axes[0]
    lo = min(pair_vals.min(), excl_vals.min()) - 0.02
    hi = max(pair_vals.max(), excl_vals.max()) + 0.02
    ax1.scatter(pair_vals, excl_vals, alpha=0.7, s=60, zorder=3, color='#9e7bb5')
    ax1.plot([lo, hi], [lo, hi], 'r--', linewidth=2, label='y = x')
    ax1.set_xlabel('Pairwise PR-AUC (4 features)')
    ax1.set_ylabel('Hyper-exclusive PR-AUC (5 features)')
    ax1.set_title('Higher-order Signal\n(information pairwise cannot express)')
    ax1.set_xlim(lo, hi)
    ax1.set_ylim(lo, hi)
    ax1.set_aspect('equal')
    ax1.legend(fontsize=11)
    ep = stats.get('excl_vs_pair', {})
    ax1.text(0.97, 0.03,
             f'Excl wins: {ep.get("wins", 0)}\n'
             f'Pair wins: {ep.get("losses", 0)}\n'
             f'p (one-sided) = {ep.get("p_greater", 1.0):.4f}',
             transform=ax1.transAxes, ha='right', va='bottom', fontsize=11,
             bbox=dict(facecolor='thistle', alpha=0.5))

    # ── Panel 2: pairwise vs pairwise-padded (noise sanity check) ────
    ax2 = axes[1]
    lo = min(pair_vals.min(), padded_vals.min()) - 0.02
    hi = max(pair_vals.max(), padded_vals.max()) + 0.02
    ax2.scatter(pair_vals, padded_vals, alpha=0.7, s=60, zorder=3, color='#f4a582')
    ax2.plot([lo, hi], [lo, hi], 'r--', linewidth=2, label='y = x')
    ax2.set_xlabel('Pairwise PR-AUC (4 features)')
    ax2.set_ylabel(f'Pairwise + {n_pad} noise PR-AUC\n(avg over {n_seeds} seeds)')
    ax2.set_title('Noise Padding Sanity Check\n(extra dimensions without signal)')
    ax2.set_xlim(lo, hi)
    ax2.set_ylim(lo, hi)
    ax2.set_aspect('equal')
    ax2.legend(fontsize=11)
    pp = stats.get('padded_vs_pair', {})
    ax2.text(0.97, 0.03,
             f'Padded wins: {pp.get("wins", 0)}\n'
             f'Pair wins: {pp.get("losses", 0)}\n'
             f'p (one-sided) = {pp.get("p_greater", 1.0):.4f}',
             transform=ax2.transAxes, ha='right', va='bottom', fontsize=11,
             bbox=dict(facecolor='peachpuff', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'dimensionality_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   Saved: dimensionality_scatter.png")


# =======================================================
# MAIN
# =======================================================

if __name__ == "__main__":

    start_time = time.time()
    print(f"Process started at {time.strftime('%H:%M:%S', time.localtime(start_time))}")

    # --- Output directory ---
    output_dir = CONFIG["BASE_OUTPUT_DIR"] / "drug_target_family_splits"
    output_dir.mkdir(parents=True, exist_ok=True)
    CONFIG["OUTPUT_DIR"] = output_dir

    print(f"\n{'='*70}")
    print(f"  PAIRED COMPARISON: HYPERGRAPH vs PAIRWISE")
    print(f"  Task   : Drug Target Prediction")
    print(f"  Model  : {CONFIG['MODEL_TYPE']}")
    print(f"  Splits : pre-assigned family-level (protein_splits_drug_target_strat.csv)")
    print(f"  Output : {output_dir}")
    print(f"{'='*70}\n")

    # --- Load data ---
    features_df = load_all_features()
    splits_df   = load_splits()

    split_indices = sorted(splits_df['split_index'].unique())
    print(f"\n   Running {len(split_indices)} splits: {split_indices}\n")

    # --- Resolve active features (only keep those actually present in features_df) ---
    hypergraph_features = [f for f in CONFIG["FEATURES"]["HYPERGRAPH"]
                           if f in features_df.columns]
    pairwise_features   = [f for f in CONFIG["FEATURES"]["PAIRWISE"]
                           if f in features_df.columns]

    stoich_features = CONFIG["FEATURES"].get("STOICHIOMETRY_FEATURES", [])
    hypergraph_no_stoich_features = [f for f in hypergraph_features
                                     if f not in stoich_features]

    hyper_exclusive_features = [f for f in CONFIG["FEATURES"].get("HYPER_EXCLUSIVE", [])
                                if f in features_df.columns]

    missing_hg   = [f for f in CONFIG["FEATURES"]["HYPERGRAPH"] if f not in features_df.columns]
    missing_pair = [f for f in CONFIG["FEATURES"]["PAIRWISE"]   if f not in features_df.columns]
    if missing_hg:
        print(f"   WARNING: {len(missing_hg)} hypergraph features not found in data: {missing_hg}")
    if missing_pair:
        print(f"   WARNING: {len(missing_pair)} pairwise features not found in data: {missing_pair}")

    print(f"   Active hypergraph features ({len(hypergraph_features)}):")
    for f in hypergraph_features:
        tag = " [stoich]" if f in stoich_features else ""
        print(f"     - {f}{tag}")
    print(f"   Active hypergraph NO-STOICH features ({len(hypergraph_no_stoich_features)}):")
    for f in hypergraph_no_stoich_features:
        print(f"     - {f}")
    print(f"   Active pairwise features ({len(pairwise_features)}):")
    for f in pairwise_features:
        print(f"     - {f}")
    print(f"   Active hyper-exclusive features ({len(hyper_exclusive_features)}):")
    for f in hyper_exclusive_features:
        print(f"     - {f}")
    print(f"   Random padding: {CONFIG['N_RANDOM_PAD']} "
          f"noise columns × {CONFIG['N_RANDOM_SEEDS']} seeds")

    # --- Fill any NaNs in feature columns ---
    all_feature_cols = hypergraph_features + pairwise_features
    n_nans = features_df[all_feature_cols].isna().sum().sum()
    if n_nans > 0:
        print(f"   Filling {n_nans} missing feature values with 0.")
        features_df[all_feature_cols] = features_df[all_feature_cols].fillna(0)

    # --- Main loop over splits ---
    print(f"\n3. Running paired comparisons across {len(split_indices)} splits...\n")
    all_results = []

    for split_idx in split_indices:
        print(f"   Split {split_idx:>2}/{len(split_indices)}...", end=" ", flush=True)
        try:
            result = run_split(
                split_idx, features_df, splits_df,
                hypergraph_features, hypergraph_no_stoich_features, pairwise_features,
                hyper_exclusive_features
            )
            all_results.append(result)
            winner = ("Hyper" if result['pr_auc_diff'] > 0
                      else "Pair" if result['pr_auc_diff'] < 0 else "Tie")
            print(f"train={result['n_train']} ({result['train_ess_pct']:.1f}% ess)  "
                  f"test={result['n_test']} ({result['test_ess_pct']:.1f}% ess)  |  "
                  f"Hyper: {result['hypergraph_pr_auc']:.4f}, "
                  f"NoStoich: {result['hyper_nostoich_pr_auc']:.4f}, "
                  f"Pair: {result['pairwise_pr_auc']:.4f}, "
                  f"Excl: {result['hyper_exclusive_pr_auc']:.4f}, "
                  f"Padded: {result['pairwise_padded_pr_auc']:.4f} "
                  f"[{winner}]")
        except Exception as e:
            print(f"ERROR: {e}")

    # --- Statistical comparison ---
    print("\n4. Statistical analysis...")
    stats = run_sign_test_comparison(all_results)
    print_statistical_summary(stats)

    # --- Plots ---
    print("\n5. Generating plots...")
    plot_paired_comparison(all_results, stats, output_dir)
    plot_stoich_ablation(all_results, stats, output_dir)
    plot_dimensionality_control(all_results, stats, output_dir)

    # --- Feature importance ---
    print("\n6. Aggregating feature importance...")
    hyper_imp_df = aggregate_feature_importance(all_results, 'hypergraph')
    pair_imp_df  = aggregate_feature_importance(all_results, 'pairwise')
    print_feature_importance_summary(hyper_imp_df, pair_imp_df, top_n=10)
    plot_feature_importance(hyper_imp_df, pair_imp_df, output_dir, top_n=15)

    # --- Save CSVs ---
    print("\n7. Saving outputs...")

    # Per-split summary (no nested dicts)
    summary_cols = ['split_index', 'n_train', 'n_test', 'train_ess_pct', 'test_ess_pct',
                    'hypergraph_pr_auc', 'hypergraph_f1',
                    'hyper_nostoich_pr_auc', 'hyper_nostoich_f1',
                    'pairwise_pr_auc',   'pairwise_f1',
                    'hyper_exclusive_pr_auc', 'hyper_exclusive_f1',
                    'pairwise_padded_pr_auc', 'pairwise_padded_f1',
                    'pairwise_padded_pr_auc_std',
                    'pr_auc_diff', 'f1_diff',
                    'stoich_pr_auc_diff', 'stoich_f1_diff',
                    'excl_vs_pair_pr_auc_diff', 'padded_vs_hyper_pr_auc_diff']
    summary_df = pd.DataFrame([{k: r[k] for k in summary_cols} for r in all_results])
    summary_df.to_csv(output_dir / 'split_results.csv', index=False)
    print("   Saved: split_results.csv")

    # Per-protein predictions — hypergraph
    hyper_preds_all = pd.concat(
        [r['hypergraph_predictions'] for r in all_results], ignore_index=True
    )
    hyper_preds_all.to_csv(output_dir / 'hypergraph_predictions.csv', index=False)
    print("   Saved: hypergraph_predictions.csv")

    # Per-protein predictions — pairwise
    pair_preds_all = pd.concat(
        [r['pairwise_predictions'] for r in all_results], ignore_index=True
    )
    pair_preds_all.to_csv(output_dir / 'pairwise_predictions.csv', index=False)
    print("   Saved: pairwise_predictions.csv")

    # Per-protein predictions — hypergraph no-stoich
    nostoich_preds_all = pd.concat(
        [r['hyper_nostoich_predictions'] for r in all_results], ignore_index=True
    )
    nostoich_preds_all.to_csv(output_dir / 'hyper_nostoich_predictions.csv', index=False)
    print("   Saved: hyper_nostoich_predictions.csv")

    # Feature importance
    hyper_imp_df.to_csv(output_dir / 'hypergraph_feature_importance.csv', index=False)
    pair_imp_df.to_csv(output_dir / 'pairwise_feature_importance.csv', index=False)
    nostoich_imp_df = aggregate_feature_importance(all_results, 'hyper_nostoich')
    nostoich_imp_df.to_csv(output_dir / 'hyper_nostoich_feature_importance.csv', index=False)
    excl_imp_df = aggregate_feature_importance(all_results, 'hyper_exclusive')
    excl_imp_df.to_csv(output_dir / 'hyper_exclusive_feature_importance.csv', index=False)
    print("   Saved: hypergraph_feature_importance.csv")
    print("   Saved: pairwise_feature_importance.csv")
    print("   Saved: hyper_nostoich_feature_importance.csv")
    print("   Saved: hyper_exclusive_feature_importance.csv")

    with open(output_dir / 'statistical_summary.txt', 'w') as f:
            f.write("PAIRED COMPARISON: HYPERGRAPH vs PAIRWISE (with stoichiometry ablation)\n")
            f.write("Task: Gene Essentiality\n")
            f.write(f"Model: {CONFIG['MODEL_TYPE']}\n")
            f.write(f"Number of splits: {stats['n_runs']}\n\n")
            f.write(f"Hypergraph features ({len(hypergraph_features)}):\n")
            for feat in hypergraph_features:
                tag = " [stoich]" if feat in stoich_features else ""
                f.write(f"  - {feat}{tag}\n")
            f.write(f"\nHypergraph no-stoich features ({len(hypergraph_no_stoich_features)}):\n")
            for feat in hypergraph_no_stoich_features:
                f.write(f"  - {feat}\n")
            f.write(f"\nPairwise features ({len(pairwise_features)}):\n")
            for feat in pairwise_features:
                f.write(f"  - {feat}\n")
            f.write(f"\nPR-AUC Mean ± Std:\n")
            f.write(f"  Hypergraph (full)    : {stats['hypergraph_pr_auc_mean']:.4f} ± {stats['hypergraph_pr_auc_std']:.4f}\n")
            f.write(f"  Hypergraph (no stoich): {stats['hyper_nostoich_pr_auc_mean']:.4f} ± {stats['hyper_nostoich_pr_auc_std']:.4f}\n")
            f.write(f"  Pairwise             : {stats['pairwise_pr_auc_mean']:.4f} ± {stats['pairwise_pr_auc_std']:.4f}\n")

            def _write_comparison(label, d):
                f.write(f"\n{label}:\n")
                f.write(f"  Mean diff : {d['mean_diff']:+.4f} ± {d['std_diff']:.4f}\n")
                f.write(f"  Wins/Losses/Ties : {d['wins']}/{d['losses']}/{d['ties']}\n")
                f.write(f"  Sign test p (one-sided) : {d['p_greater']:.6f}\n")
                f.write(f"  Sign test p (two-sided) : {d['p_two_sided']:.6f}\n")

            _write_comparison("Hypergraph (full) vs Pairwise",
                              {'mean_diff': stats['mean_difference'],
                               'std_diff':  stats['std_difference'],
                               'wins':      stats['hypergraph_wins'],
                               'losses':    stats['pairwise_wins'],
                               'ties':      stats['ties'],
                               'p_greater': stats['sign_test_p_greater'],
                               'p_two_sided': stats['sign_test_p_two_sided']})
            _write_comparison("Hypergraph (full) vs Hypergraph (no stoich) — stoichiometry effect",
                              stats['stoich_ablation'])
            _write_comparison("Hypergraph (no stoich) vs Pairwise — representation effect alone",
                              stats['nostoich_vs_pair'])

            # Dimensionality controls
            f.write(f"\nHyper-exclusive features ({len(hyper_exclusive_features)}):\n")
            for feat in hyper_exclusive_features:
                f.write(f"  - {feat}\n")
            f.write(f"\nHyper-exclusive only     : {stats['hyper_exclusive_pr_auc_mean']:.4f} ± {stats['hyper_exclusive_pr_auc_std']:.4f}\n")
            f.write(f"Pairwise + random padding: {stats['pairwise_padded_pr_auc_mean']:.4f} ± {stats['pairwise_padded_pr_auc_std']:.4f}\n")
            _write_comparison("Hyper-exclusive (5) vs Pairwise (4) — higher-order signal only",
                              stats['excl_vs_pair'])
            _write_comparison("Hypergraph (full) vs Pairwise (padded) — dimensionality control",
                              stats['hyper_vs_padded'])
            _write_comparison("Pairwise (padded) vs Pairwise (unpadded) — noise sanity check",
                              stats['padded_vs_pair'])

    print(f"\n{'='*70}")
    print("  COMPLETE")
    print(f"{'='*70}")

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")