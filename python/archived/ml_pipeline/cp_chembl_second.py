import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    "BASE_OUTPUT_DIR": Path("./randomforest/cp_chembl_second"),

    # --- File Names ---
    "SPLITS_FILE":           "chembl_protein_merged_splits.csv",
    "PROTEIN_FEATURES_FILE": "hypergraph_features.csv",
    "PAIRWISE_FEATURES_FILE":"pairwise_features.csv",

    # --- Model ---
    # Options: "RandomForest" | "LightGBM" | "XGBoost"
    "MODEL_TYPE": "RandomForest",

    # --- Fixed settings ---
    "RANDOM_STATE": 42,
    "N_SPLITS_CV":  5,

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
    #
    # Three nested representations are compared (pairwise ⊂ hypergraph ⊂ hb_graph):
    #   PAIRWISE   — dyadic PPI features only.
    #   HB_GRAPH   — full higher-order feature set INCLUDING stoichiometry
    #                (multiset hyperedges; the hb-graph representation).
    #   HYPERGRAPH — the hb-graph feature set MINUS stoichiometry, i.e. the
    #                set-based hypergraph. Derived automatically as
    #                HB_GRAPH minus STOICHIOMETRY_FEATURES (do not list separately).
    "FEATURES": {
        "HB_GRAPH": [
            # --- Base / native higher-order metrics ---
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

            # --- Stoichiometry-based metrics (hb-graph only) ---
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

        # Stoichiometry features to ablate — must be a subset of HB_GRAPH above.
        # The set-based hypergraph feature set is derived automatically as
        # HB_GRAPH minus STOICHIOMETRY_FEATURES.
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
        ]
    }
}

splits_path = CONFIG["DATA_DIR"] / CONFIG["SPLITS_FILE"]
print(f"   Splits file last modified: {pd.Timestamp(os.path.getmtime(splits_path), unit='s')}")
print(f"   Splits file rows: {pd.read_csv(splits_path).shape}")


# =======================================================
# DATA LOADING
# =======================================================

def load_all_features() -> pd.DataFrame:
    """Loads higher-order (hb-graph) and pairwise feature CSVs and merges on ProteinId."""
    print("1. Loading feature data...")

    hg_df   = pd.read_csv(CONFIG["DATA_DIR"] / CONFIG["PROTEIN_FEATURES_FILE"])
    pair_df = pd.read_csv(CONFIG["DATA_DIR"] / CONFIG["PAIRWISE_FEATURES_FILE"])

    combined = pd.merge(hg_df, pair_df, on='ProteinId', how='inner')

    print(f"   Higher-order (hb-graph) features shape : {hg_df.shape}")
    print(f"   Pairwise features shape               : {pair_df.shape}")
    print(f"   Combined shape                        : {combined.shape}")
    return combined


def load_splits() -> pd.DataFrame:
    """
    Loads the pre-assigned family-level splits file.

    Expected columns:
        split_index   — integer 1..N identifying which split
        UniProt_AC    — protein identifier (matches ProteinId in feature files)
        split         — 'train' or 'test'
        protein_label — 'Drug_target' | 'Non_target' | 'Unknown'
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
    n_dt = (labelled['target'] == 1).sum()
    n_tot = len(labelled)
    print(f"   Labelled proteins : {n_tot}  ({100*n_dt/n_tot:.1f}% drug targets)")

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
        target_names=['Non_target', 'Drug_target'],
        output_dict=True
    )

    return {
        'pr_auc':       average_precision_score(y_test, y_pred_proba),
        'f1':           report['Drug_target']['f1-score'],
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
    hb_graph_features: List[str],
    hypergraph_features: List[str],
    pairwise_features: List[str]
) -> Dict:
    """
    Runs the three nested representations for a single pre-assigned split:
      pairwise   — dyadic PPI features
      hypergraph — set-based higher-order features (no stoichiometry)
      hb_graph   — hypergraph features PLUS stoichiometry (multiset hyperedges)

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
        'train_dt_pct': 100 * y_train.mean(),
        'test_dt_pct':  100 * y_test.mean(),
    }

    # --- Pairwise model ---
    X_pair_train = train_df[pairwise_features]
    X_pair_test  = test_df[pairwise_features]

    pair_model, pair_params = tune_and_train_model(X_pair_train, y_train)
    pair_eval = evaluate_model(pair_model, X_pair_test, y_test)

    results['pairwise_pr_auc']      = pair_eval['pr_auc']
    results['pairwise_f1']          = pair_eval['f1']
    results['pairwise_best_params'] = pair_params
    results['pairwise_importance']  = compute_permutation_importance(
        pair_model, X_pair_test, y_test
    )

    # Store per-protein predictions (pairwise)
    pair_preds = test_df[['ProteinId']].copy()
    pair_preds['split_index']      = split_idx
    pair_preds['true_label']       = y_test.values
    pair_preds['pair_pred_proba']  = pair_eval['y_pred_proba']
    results['pairwise_predictions'] = pair_preds

    # --- Hypergraph model (set-based, no stoichiometry) ---
    X_hyper_train = train_df[hypergraph_features]
    X_hyper_test  = test_df[hypergraph_features]

    hyper_model, hyper_params = tune_and_train_model(X_hyper_train, y_train)
    hyper_eval = evaluate_model(hyper_model, X_hyper_test, y_test)

    results['hypergraph_pr_auc']      = hyper_eval['pr_auc']
    results['hypergraph_f1']          = hyper_eval['f1']
    results['hypergraph_best_params'] = hyper_params
    results['hypergraph_importance']  = compute_permutation_importance(
        hyper_model, X_hyper_test, y_test
    )

    hyper_preds = test_df[['ProteinId']].copy()
    hyper_preds['split_index']       = split_idx
    hyper_preds['true_label']        = y_test.values
    hyper_preds['hyper_pred_proba']  = hyper_eval['y_pred_proba']
    results['hypergraph_predictions'] = hyper_preds

    # --- HB-graph model (hypergraph + stoichiometry) ---
    X_hbg_train = train_df[hb_graph_features]
    X_hbg_test  = test_df[hb_graph_features]

    hbg_model, hbg_params = tune_and_train_model(X_hbg_train, y_train)
    hbg_eval = evaluate_model(hbg_model, X_hbg_test, y_test)

    results['hb_graph_pr_auc']      = hbg_eval['pr_auc']
    results['hb_graph_f1']          = hbg_eval['f1']
    results['hb_graph_best_params'] = hbg_params
    results['hb_graph_importance']  = compute_permutation_importance(
        hbg_model, X_hbg_test, y_test
    )

    # Store per-protein predictions (hb-graph)
    hbg_preds = test_df[['ProteinId']].copy()
    hbg_preds['split_index']      = split_idx
    hbg_preds['true_label']       = y_test.values
    hbg_preds['hbg_pred_proba']   = hbg_eval['y_pred_proba']
    results['hb_graph_predictions'] = hbg_preds

    # Differences
    # Headline representation contrast: hb_graph vs pairwise
    results['pr_auc_diff']        = results['hb_graph_pr_auc'] - results['pairwise_pr_auc']
    results['f1_diff']            = results['hb_graph_f1']     - results['pairwise_f1']
    # Stoichiometry effect: hb_graph vs hypergraph (adding multiset stoichiometry)
    results['stoich_pr_auc_diff'] = results['hb_graph_pr_auc'] - results['hypergraph_pr_auc']
    results['stoich_f1_diff']     = results['hb_graph_f1']     - results['hypergraph_f1']

    return results


# =======================================================
# STATISTICAL COMPARISON
# =======================================================

def run_sign_test_comparison(all_results: List[Dict]) -> Dict:
    """Sign test (binomial) on paired PR-AUC wins/losses across splits.
    Covers three paired comparisons:
      1. HB-graph vs Pairwise                        — headline representation effect
      2. HB-graph vs Hypergraph  — stoichiometry effect (adding multiset stoichiometry)
      3. Hypergraph vs Pairwise  — set-based representation effect alone
    """
    pair_vals  = np.array([r['pairwise_pr_auc']   for r in all_results])
    hyper_vals = np.array([r['hypergraph_pr_auc'] for r in all_results])
    hbg_vals   = np.array([r['hb_graph_pr_auc']   for r in all_results])

    # F1 values (positive class) per representation
    pair_f1  = np.array([r['pairwise_f1']   for r in all_results])
    hyper_f1 = np.array([r['hypergraph_f1'] for r in all_results])
    hbg_f1   = np.array([r['hb_graph_f1']   for r in all_results])

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

    hbg_vs_pair   = _sign_test(hbg_vals,   pair_vals)   # headline
    stoich_effect = _sign_test(hbg_vals,   hyper_vals)  # hb_graph vs hypergraph
    hyper_vs_pair = _sign_test(hyper_vals, pair_vals)   # representation effect alone

    # Random-classifier PR-AUC baseline = positive-class prevalence in test set
    base_key = 'test_ess_pct' if 'test_ess_pct' in all_results[0] else 'test_dt_pct'
    random_baseline = float(np.mean([r[base_key] for r in all_results])) / 100.0

    return {
        'n_runs': len(all_results),
        'random_baseline': random_baseline,
        # --- PR-AUC mean ± std per representation ---
        'pairwise_pr_auc_mean':   float(np.mean(pair_vals)),
        'pairwise_pr_auc_std':    float(np.std(pair_vals)),
        'hypergraph_pr_auc_mean': float(np.mean(hyper_vals)),
        'hypergraph_pr_auc_std':  float(np.std(hyper_vals)),
        'hb_graph_pr_auc_mean':   float(np.mean(hbg_vals)),
        'hb_graph_pr_auc_std':    float(np.std(hbg_vals)),
        # --- F1 mean ± std per representation (reported in main-paper table) ---
        'pairwise_f1_mean':   float(np.mean(pair_f1)),
        'pairwise_f1_std':    float(np.std(pair_f1)),
        'hypergraph_f1_mean': float(np.mean(hyper_f1)),
        'hypergraph_f1_std':  float(np.std(hyper_f1)),
        'hb_graph_f1_mean':   float(np.mean(hbg_f1)),
        'hb_graph_f1_std':    float(np.std(hbg_f1)),
        # --- Headline comparison: HB-graph vs Pairwise ---
        'mean_difference':       hbg_vs_pair['mean_diff'],
        'std_difference':        hbg_vs_pair['std_diff'],
        'hb_graph_wins':         hbg_vs_pair['wins'],
        'pairwise_wins':         hbg_vs_pair['losses'],
        'ties':                  hbg_vs_pair['ties'],
        'sign_test_p_greater':   hbg_vs_pair['p_greater'],
        'sign_test_p_two_sided': hbg_vs_pair['p_two_sided'],
        # --- Stoichiometry effect: HB-graph vs Hypergraph ---
        'stoich_effect':         stoich_effect,
        # --- Representation effect alone: Hypergraph vs Pairwise ---
        'hyper_vs_pair':         hyper_vs_pair,
    }


# =======================================================
# FEATURE IMPORTANCE AGGREGATION
# =======================================================

def aggregate_feature_importance(
    all_results: List[Dict], representation: str
) -> pd.DataFrame:
    """
    Aggregates permutation importance across all splits.
    representation: 'pairwise', 'hypergraph', or 'hb_graph'
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
    print(f"  Random baseline (PR-AUC = positive-class prevalence): "
          f"{stats['random_baseline']:.4f}")

    # --- PR-AUC (ordered pairwise -> hypergraph -> hb_graph) ---
    print(f"\n  PR-AUC")
    print(f"  {'Representation':<20} {'Mean ± Std'}")
    print(f"  {'-'*45}")
    print(f"  {'Pairwise':<20} "
          f"{stats['pairwise_pr_auc_mean']:.4f} ± {stats['pairwise_pr_auc_std']:.4f}")
    print(f"  {'Hypergraph':<20} "
          f"{stats['hypergraph_pr_auc_mean']:.4f} ± {stats['hypergraph_pr_auc_std']:.4f}")
    print(f"  {'HB-graph':<20} "
          f"{stats['hb_graph_pr_auc_mean']:.4f} ± {stats['hb_graph_pr_auc_std']:.4f}")

    # --- F1 (positive class; reported in main-paper table) ---
    print(f"\n  F1 (positive class)")
    print(f"  {'Representation':<20} {'Mean ± Std'}")
    print(f"  {'-'*45}")
    print(f"  {'Pairwise':<20} "
          f"{stats['pairwise_f1_mean']:.4f} ± {stats['pairwise_f1_std']:.4f}")
    print(f"  {'Hypergraph':<20} "
          f"{stats['hypergraph_f1_mean']:.4f} ± {stats['hypergraph_f1_std']:.4f}")
    print(f"  {'HB-graph':<20} "
          f"{stats['hb_graph_f1_mean']:.4f} ± {stats['hb_graph_f1_std']:.4f}")

    def _print_comparison(label, d):
        print(f"\n  --- {label} ---")
        print(f"  Mean diff : {d['mean_diff']:+.4f} ± {d['std_diff']:.4f}")
        print(f"  Wins/Losses/Ties : {d['wins']}/{d['losses']}/{d['ties']}")
        print(f"  Sign test p (one-sided) : {d['p_greater']:.6f}")
        print(f"  Sign test p (two-sided) : {d['p_two_sided']:.6f}")

    _print_comparison("HB-graph vs Pairwise — headline representation effect",
                      {'mean_diff': stats['mean_difference'],
                       'std_diff':  stats['std_difference'],
                       'wins':      stats['hb_graph_wins'],
                       'losses':    stats['pairwise_wins'],
                       'ties':      stats['ties'],
                       'p_greater': stats['sign_test_p_greater'],
                       'p_two_sided': stats['sign_test_p_two_sided']})
    _print_comparison("HB-graph vs Hypergraph — stoichiometry effect",
                      stats['stoich_effect'])
    _print_comparison("Hypergraph vs Pairwise — representation effect alone",
                      stats['hyper_vs_pair'])
    print(f"{'='*70}")


def print_feature_importance_summary(
    imp_dfs: List[tuple], top_n: int = 10
):
    """imp_dfs: list of (label, importance_df) tuples, in display order."""
    print(f"\n{'='*70}")
    print("  FEATURE IMPORTANCE (Permutation — PR-AUC drop)")
    print(f"{'='*70}")
    for label, df in imp_dfs:
        if df.empty:
            continue
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

def get_random_baseline(all_results: List[Dict]) -> float:
    """
    PR-AUC of a random classifier = positive-class prevalence in the test set.

    (Note: 0.5 is the *ROC-AUC* baseline, not the PR-AUC baseline. Because
    PR-AUC depends on class balance, this baseline is task-specific and must be
    quoted alongside PR-AUC values.)

    Averaged across splits; the per-split test prevalence is already stored as
    'test_ess_pct' (essentiality) or 'test_dt_pct' (drug target).
    """
    key = 'test_ess_pct' if 'test_ess_pct' in all_results[0] else 'test_dt_pct'
    return float(np.mean([r[key] for r in all_results])) / 100.0


def plot_paired_comparison(all_results: List[Dict], stats: Dict, output_dir: Path):
    """Two-panel comparison plot: paired scatter (headline contrast) and 3-way boxplot.

    Axes are fixed to [0, 1] so panels are directly comparable across tasks/files.
    """
    pair_vals  = np.array([r['pairwise_pr_auc']   for r in all_results])
    hyper_vals = np.array([r['hypergraph_pr_auc'] for r in all_results])
    hbg_vals   = np.array([r['hb_graph_pr_auc']   for r in all_results])

    baseline = get_random_baseline(all_results)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # Panel 1: paired scatter — headline contrast (HB-graph vs Pairwise), one point per split
    ax1 = axes[0]
    ax1.scatter(pair_vals, hbg_vals, alpha=0.7, s=60, zorder=3)
    ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, label='y = x')
    # Random-classifier baseline (= positive-class prevalence)
    ax1.axhline(baseline, color='dimgray', linestyle=':', linewidth=1.8, zorder=1,
                label=f'Random baseline ({baseline:.3f})')
    ax1.axvline(baseline, color='dimgray', linestyle=':', linewidth=1.8, zorder=1)
    ax1.set_xlabel('Pairwise PR-AUC')
    ax1.set_ylabel('HB-graph PR-AUC')
    ax1.set_title('Paired Comparison — One Point per Split')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect('equal')
    ax1.legend(loc='upper left')
    above = int(np.sum(hbg_vals > pair_vals))
    below = int(np.sum(hbg_vals < pair_vals))
    ax1.text(0.95, 0.05,
             f'HB-graph wins: {above}\nPairwise wins: {below}',
             transform=ax1.transAxes, ha='right', va='bottom',
             bbox=dict(facecolor='lightgreen', alpha=0.5))

    # Panel 2: boxplot across all three representations
    ax2 = axes[1]
    box_data   = [pair_vals, hyper_vals, hbg_vals]
    box_labels = ['Pairwise', 'Hypergraph', 'HB-graph']
    box_colors = ['lightgray', 'skyblue', 'steelblue']
    bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
    for patch, colour in zip(bp['boxes'], box_colors):
        patch.set_facecolor(colour)
    ax2.set_ylabel('PR-AUC')
    ax2.set_title('Distribution Comparison')
    ax2.set_ylim(0, 1)
    rng = np.random.default_rng(0)
    for i, data in enumerate(box_data):
        x = rng.normal(i + 1, 0.04, size=len(data))
        ax2.scatter(x, data, alpha=0.4, s=20, color='black')
    # Random-classifier baseline (= positive-class prevalence)
    ax2.axhline(baseline, color='dimgray', linestyle=':', linewidth=1.8, zorder=1,
                label=f'Random baseline ({baseline:.3f})')
    ax2.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'paired_comparison.png', dpi=300)
    plt.close()
    print("   Saved: paired_comparison.png")


def plot_stoich_ablation(all_results: List[Dict], stats: Dict, output_dir: Path):
    """
    Two-panel stoichiometry ablation figure matching the poster's Fig 5 style.

    Panel 1 — Scatter: hb-graph vs hypergraph PR-AUC, one point per split.
               Points above the diagonal = stoichiometry helps.
    Panel 2 — Boxplot: pairwise / hypergraph / hb-graph distributions,
               showing the stepwise improvement from adding stoichiometry.
    """
    pair_vals  = np.array([r['pairwise_pr_auc']   for r in all_results])
    hyper_vals = np.array([r['hypergraph_pr_auc'] for r in all_results])
    hbg_vals   = np.array([r['hb_graph_pr_auc']   for r in all_results])

    stoich_wins   = int(np.sum(hbg_vals > hyper_vals))
    stoich_losses = int(np.sum(hbg_vals < hyper_vals))

    baseline = get_random_baseline(all_results)

    ab  = stats['stoich_effect']
    p_one = ab['p_greater']
    p_two = ab['p_two_sided']

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # ── Panel 1: scatter — hb-graph vs hypergraph ────────────────────────────
    ax1 = axes[0]
    ax1.scatter(hyper_vals, hbg_vals, alpha=0.7, s=60, zorder=3,
                color='steelblue')
    ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, label='y = x (no difference)')
    # Random-classifier baseline (= positive-class prevalence)
    ax1.axhline(baseline, color='dimgray', linestyle=':', linewidth=1.8, zorder=1,
                label=f'Random baseline ({baseline:.3f})')
    ax1.axvline(baseline, color='dimgray', linestyle=':', linewidth=1.8, zorder=1)
    ax1.set_xlabel('Hypergraph PR-AUC')
    ax1.set_ylabel('HB-graph PR-AUC')
    ax1.set_title('Stoichiometry Ablation — One Point per Split')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect('equal')
    ax1.legend(fontsize=12, loc='upper left')
    ax1.text(0.97, 0.03,
             f'HB-graph wins: {stoich_wins}\nHypergraph wins: {stoich_losses}',
             transform=ax1.transAxes, ha='right', va='bottom', fontsize=12,
             bbox=dict(facecolor='lightgreen', alpha=0.5))

    # ── Panel 2: boxplot — pairwise / hypergraph / hb-graph ──────────────────
    ax2 = axes[1]
    colours = ['lightgray', 'skyblue', 'steelblue']
    labels  = ['Pairwise', 'Hypergraph', 'HB-graph']
    box_data = [pair_vals, hyper_vals, hbg_vals]
    bp = ax2.boxplot(
        box_data,
        labels=labels,
        patch_artist=True,
        medianprops=dict(color='black', linewidth=2),
    )
    for patch, colour in zip(bp['boxes'], colours):
        patch.set_facecolor(colour)
    ax2.set_ylabel('PR-AUC')
    ax2.set_title('Distribution Comparison')
    ax2.set_ylim(0, 1)
    rng = np.random.default_rng(0)
    for i, data in enumerate(box_data):
        x = rng.normal(i + 1, 0.04, size=len(data))
        ax2.scatter(x, data, alpha=0.4, s=20, color='black', zorder=3)
    # Random-classifier baseline (= positive-class prevalence)
    ax2.axhline(baseline, color='dimgray', linestyle=':', linewidth=1.8, zorder=1,
                label=f'Random baseline ({baseline:.3f})')
    ax2.legend(loc='upper left', fontsize=10)

    # Annotate with mean ± std for each box
    for i, vals in enumerate(box_data):
        ax2.text(i + 1, 0.02,
                 f'{vals.mean():.3f}±{vals.std():.3f}',
                 ha='center', va='bottom', fontsize=10)

    # Sign test annotation (stoichiometry effect: hb-graph vs hypergraph)
    sig_label = (f'Stoich effect\np (one-sided) = {p_one:.4f}\n'
                 f'p (two-sided) = {p_two:.4f}')
    ax2.text(0.97, 0.97, sig_label,
             transform=ax2.transAxes, ha='right', va='top', fontsize=10,
             bbox=dict(facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_dir / 'stoich_ablation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   Saved: stoich_ablation.png")


def plot_feature_importance(
    imp_dfs: List[tuple],
    output_dir: Path,
    top_n: int = 15
):
    """Side-by-side horizontal bar charts of permutation importance.

    imp_dfs: list of (label, importance_df, colour) tuples, in display order.
    """
    n = len(imp_dfs)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 8))
    if n == 1:
        axes = [axes]

    for ax, (label, df, colour) in zip(axes, imp_dfs):
        top = df.head(top_n)
        colors = [colour if v > 0 else 'lightcoral' for v in top['mean']]
        ax.barh(range(len(top)), top['mean'], xerr=top['std'],
                color=colors, edgecolor='black', capsize=3)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top['feature'])
        ax.invert_yaxis()
        ax.set_xlabel('Mean Permutation Importance (PR-AUC drop)')
        ax.set_title(f'Top {top_n} {label} Features')
        ax.axvline(0, color='gray', linestyle='--', linewidth=1)

    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance_comparison.png', dpi=300)
    plt.close()
    print("   Saved: feature_importance_comparison.png")


# =======================================================
# MAIN
# =======================================================

if __name__ == "__main__":

    start_time = time.time()
    print(f"Process started at {time.strftime('%H:%M:%S', time.localtime(start_time))}")

    # --- Output directory ---
    output_dir = CONFIG["BASE_OUTPUT_DIR"]
    output_dir.mkdir(parents=True, exist_ok=True)
    CONFIG["OUTPUT_DIR"] = output_dir

    print(f"\n{'='*70}")
    print(f"  REPRESENTATION COMPARISON: PAIRWISE vs HYPERGRAPH vs HB-GRAPH")
    print(f"  Task   : Drug Target Prediction (ChEMBL)")
    print(f"  Model  : {CONFIG['MODEL_TYPE']}")
    print(f"  Splits : pre-assigned family-level")
    print(f"  Output : {output_dir}")
    print(f"{'='*70}\n")

    # --- Load data ---
    features_df = load_all_features()
    splits_df   = load_splits()

    split_indices = sorted(splits_df['split_index'].unique())
    print(f"\n   Running {len(split_indices)} splits: {split_indices}\n")

    # --- Resolve active features (only keep those actually present in features_df) ---
    # hb_graph   = full higher-order feature set (includes stoichiometry)
    # hypergraph = hb_graph minus stoichiometry (set-based representation)
    hb_graph_features = [f for f in CONFIG["FEATURES"]["HB_GRAPH"]
                         if f in features_df.columns]
    pairwise_features = [f for f in CONFIG["FEATURES"]["PAIRWISE"]
                         if f in features_df.columns]

    stoich_features = CONFIG["FEATURES"].get("STOICHIOMETRY_FEATURES", [])
    hypergraph_features = [f for f in hb_graph_features
                           if f not in stoich_features]

    missing_hbg  = [f for f in CONFIG["FEATURES"]["HB_GRAPH"] if f not in features_df.columns]
    missing_pair = [f for f in CONFIG["FEATURES"]["PAIRWISE"] if f not in features_df.columns]
    if missing_hbg:
        print(f"   WARNING: {len(missing_hbg)} hb-graph features not found in data: {missing_hbg}")
    if missing_pair:
        print(f"   WARNING: {len(missing_pair)} pairwise features not found in data: {missing_pair}")

    print(f"   Active pairwise features ({len(pairwise_features)}):")
    for f in pairwise_features:
        print(f"     - {f}")
    print(f"   Active hypergraph features ({len(hypergraph_features)}):")
    for f in hypergraph_features:
        print(f"     - {f}")
    print(f"   Active hb-graph features ({len(hb_graph_features)}):")
    for f in hb_graph_features:
        tag = " [stoich]" if f in stoich_features else ""
        print(f"     - {f}{tag}")

    # --- Fill any NaNs in feature columns ---
    all_feature_cols = hb_graph_features + pairwise_features
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
                hb_graph_features, hypergraph_features, pairwise_features
            )
            all_results.append(result)
            winner = ("HB-graph" if result['pr_auc_diff'] > 0
                      else "Pair" if result['pr_auc_diff'] < 0 else "Tie")
            print(f"train={result['n_train']} ({result['train_dt_pct']:.1f}% dt)  "
                  f"test={result['n_test']} ({result['test_dt_pct']:.1f}% dt)  |  "
                  f"Pair: {result['pairwise_pr_auc']:.4f}, "
                  f"Hyper: {result['hypergraph_pr_auc']:.4f}, "
                  f"HB-graph: {result['hb_graph_pr_auc']:.4f}, "
                  f"Diff(stoich): {result['stoich_pr_auc_diff']:+.4f} [{winner}]")
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

    # --- Feature importance ---
    print("\n6. Aggregating feature importance...")
    pair_imp_df  = aggregate_feature_importance(all_results, 'pairwise')
    hyper_imp_df = aggregate_feature_importance(all_results, 'hypergraph')
    hbg_imp_df   = aggregate_feature_importance(all_results, 'hb_graph')
    print_feature_importance_summary(
        [("Pairwise", pair_imp_df),
         ("Hypergraph", hyper_imp_df),
         ("HB-graph", hbg_imp_df)],
        top_n=10
    )
    plot_feature_importance(
        [("Pairwise", pair_imp_df, 'gray'),
         ("Hypergraph", hyper_imp_df, 'skyblue'),
         ("HB-graph", hbg_imp_df, 'steelblue')],
        output_dir, top_n=15
    )

    # --- Save CSVs ---
    print("\n7. Saving outputs...")

    # Per-split summary (no nested dicts), ordered pairwise -> hypergraph -> hb_graph
    summary_cols = ['split_index', 'n_train', 'n_test', 'train_dt_pct', 'test_dt_pct',
                    'pairwise_pr_auc',   'pairwise_f1',
                    'hypergraph_pr_auc', 'hypergraph_f1',
                    'hb_graph_pr_auc',   'hb_graph_f1',
                    'pr_auc_diff', 'f1_diff',
                    'stoich_pr_auc_diff', 'stoich_f1_diff']
    summary_df = pd.DataFrame([{k: r[k] for k in summary_cols} for r in all_results])
    summary_df.to_csv(output_dir / 'split_results.csv', index=False)
    print("   Saved: split_results.csv")

    # Per-protein predictions — pairwise
    pair_preds_all = pd.concat(
        [r['pairwise_predictions'] for r in all_results], ignore_index=True
    )
    pair_preds_all.to_csv(output_dir / 'pairwise_predictions.csv', index=False)
    print("   Saved: pairwise_predictions.csv")

    # Per-protein predictions — hypergraph
    hyper_preds_all = pd.concat(
        [r['hypergraph_predictions'] for r in all_results], ignore_index=True
    )
    hyper_preds_all.to_csv(output_dir / 'hypergraph_predictions.csv', index=False)
    print("   Saved: hypergraph_predictions.csv")

    # Per-protein predictions — hb-graph
    hbg_preds_all = pd.concat(
        [r['hb_graph_predictions'] for r in all_results], ignore_index=True
    )
    hbg_preds_all.to_csv(output_dir / 'hb_graph_predictions.csv', index=False)
    print("   Saved: hb_graph_predictions.csv")

    # Feature importance
    pair_imp_df.to_csv(output_dir / 'pairwise_feature_importance.csv', index=False)
    hyper_imp_df.to_csv(output_dir / 'hypergraph_feature_importance.csv', index=False)
    hbg_imp_df.to_csv(output_dir / 'hb_graph_feature_importance.csv', index=False)
    print("   Saved: pairwise_feature_importance.csv")
    print("   Saved: hypergraph_feature_importance.csv")
    print("   Saved: hb_graph_feature_importance.csv")

    with open(output_dir / 'statistical_summary.txt', 'w') as f:
            f.write("REPRESENTATION COMPARISON: PAIRWISE vs HYPERGRAPH vs HB-GRAPH\n")
            f.write("Task: Drug Target Prediction (ChEMBL)\n")
            f.write(f"Model: {CONFIG['MODEL_TYPE']}\n")
            f.write(f"Number of splits: {stats['n_runs']}\n\n")
            f.write(f"Pairwise features ({len(pairwise_features)}):\n")
            for feat in pairwise_features:
                f.write(f"  - {feat}\n")
            f.write(f"\nHypergraph features ({len(hypergraph_features)}):\n")
            for feat in hypergraph_features:
                f.write(f"  - {feat}\n")
            f.write(f"\nHB-graph features ({len(hb_graph_features)}):\n")
            for feat in hb_graph_features:
                tag = " [stoich]" if feat in stoich_features else ""
                f.write(f"  - {feat}{tag}\n")

            f.write(f"\nRandom baseline (PR-AUC of a random classifier\n")
            f.write(f"  = positive-class prevalence in test set): {stats['random_baseline']:.4f}\n")
            f.write(f"\nPR-AUC Mean ± Std:\n")
            f.write(f"  Pairwise   : {stats['pairwise_pr_auc_mean']:.4f} ± {stats['pairwise_pr_auc_std']:.4f}\n")
            f.write(f"  Hypergraph : {stats['hypergraph_pr_auc_mean']:.4f} ± {stats['hypergraph_pr_auc_std']:.4f}\n")
            f.write(f"  HB-graph   : {stats['hb_graph_pr_auc_mean']:.4f} ± {stats['hb_graph_pr_auc_std']:.4f}\n")

            f.write(f"\nF1 (positive class) Mean ± Std:\n")
            f.write(f"  Pairwise   : {stats['pairwise_f1_mean']:.4f} ± {stats['pairwise_f1_std']:.4f}\n")
            f.write(f"  Hypergraph : {stats['hypergraph_f1_mean']:.4f} ± {stats['hypergraph_f1_std']:.4f}\n")
            f.write(f"  HB-graph   : {stats['hb_graph_f1_mean']:.4f} ± {stats['hb_graph_f1_std']:.4f}\n")

            def _write_comparison(label, d):
                f.write(f"\n{label}:\n")
                f.write(f"  Mean diff : {d['mean_diff']:+.4f} ± {d['std_diff']:.4f}\n")
                f.write(f"  Wins/Losses/Ties : {d['wins']}/{d['losses']}/{d['ties']}\n")
                f.write(f"  Sign test p (one-sided) : {d['p_greater']:.6f}\n")
                f.write(f"  Sign test p (two-sided) : {d['p_two_sided']:.6f}\n")

            _write_comparison("HB-graph vs Pairwise — headline representation effect",
                              {'mean_diff': stats['mean_difference'],
                               'std_diff':  stats['std_difference'],
                               'wins':      stats['hb_graph_wins'],
                               'losses':    stats['pairwise_wins'],
                               'ties':      stats['ties'],
                               'p_greater': stats['sign_test_p_greater'],
                               'p_two_sided': stats['sign_test_p_two_sided']})
            _write_comparison("HB-graph vs Hypergraph — stoichiometry effect",
                              stats['stoich_effect'])
            _write_comparison("Hypergraph vs Pairwise — representation effect alone",
                              stats['hyper_vs_pair'])

    print(f"\n{'='*70}")
    print("  COMPLETE")
    print(f"{'='*70}")

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path
# from typing import List, Dict
# import time

# from sklearn.model_selection import GridSearchCV
# from sklearn.ensemble import RandomForestClassifier
# from lightgbm import LGBMClassifier
# from xgboost import XGBClassifier

# from sklearn.metrics import classification_report, average_precision_score
# from sklearn.inspection import permutation_importance
# from scipy.stats import binomtest

# # =======================================================
# # Plotting Style Configuration
# # =======================================================
# plt.rcParams.update({
#     'font.size': 16,
#     'axes.titlesize': 18,
#     'axes.labelsize': 16,
#     'xtick.labelsize': 14,
#     'ytick.labelsize': 14,
#     'legend.fontsize': 14,
#     'figure.titlesize': 20
# })

# # =======================================================
# # CONFIGURATION
# # =======================================================
# CONFIG = {
#     # --- Paths ---
#     "DATA_DIR": Path("/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/cp/"),
#     "BASE_OUTPUT_DIR": Path("./randomforest/cp_chembl_second"),

#     # --- File Names ---
#     "SPLITS_FILE":           "chembl_protein_merged_splits.csv",
#     "PROTEIN_FEATURES_FILE": "hypergraph_features.csv",
#     "PAIRWISE_FEATURES_FILE":"pairwise_features.csv",

#     # --- Model ---
#     # Options: "RandomForest" | "LightGBM" | "XGBoost"
#     "MODEL_TYPE": "RandomForest",

#     # --- Fixed settings ---
#     "RANDOM_STATE": 42,
#     "N_SPLITS_CV":  5,

#     # --- Model-Specific Hyperparameter Grids for GridSearchCV ---
#     "PARAM_GRIDS": {
#         "RandomForest": {
#             'n_estimators':      [80, 100, 200],
#             'max_depth':         [None, 5, 10],
#             'min_samples_split': [2, 5, 10],
#             'class_weight':      ['balanced']
#         },
#         "LightGBM": {
#             'n_estimators':  [80, 100, 200],
#             'learning_rate': [0.01, 0.05, 0.1],
#             'max_depth':     [None, 5, 10],
#             'num_leaves':    [30, 50, 100],
#             'class_weight':  ['balanced']
#         },
#         "XGBoost": {
#             'n_estimators':  [80, 100, 200],
#             'learning_rate': [0.01, 0.05, 0.1],
#             'max_depth':     [None, 5, 10],
#             'subsample':     [0.75, 0.8, 1.0],
#             # scale_pos_weight is set automatically from training data (see tune_and_train_model)
#         }
#     },

#     # --- Feature Selection ---
#     # Comment/uncomment individual features to include or exclude them.
#     #
#     # Three nested representations are compared (pairwise ⊂ hypergraph ⊂ hb_graph):
#     #   PAIRWISE   — dyadic PPI features only.
#     #   HB_GRAPH   — full higher-order feature set INCLUDING stoichiometry
#     #                (multiset hyperedges; the hb-graph representation).
#     #   HYPERGRAPH — the hb-graph feature set MINUS stoichiometry, i.e. the
#     #                set-based hypergraph. Derived automatically as
#     #                HB_GRAPH minus STOICHIOMETRY_FEATURES (do not list separately).
#     "FEATURES": {
#         "HB_GRAPH": [
#             # --- Base / native higher-order metrics ---
#             'base_Degree',
#             'base_LocalClustCoeff',
#             # 'base_ComponentSize',
#             # 'base_ComponentEdgeNodeRatio',
#             'base_TriangleCount',
#             'base_UniquePartners',
#             'base_AvgNeighbourDegree',
#             # 'base_BetweennessCentrality',
#             # 'base_EigenvectorCentrality',
#             # 'base_KatzCentrality',

#             # --- Stoichiometry-based metrics (hb-graph only) ---
#             'stoich_WeightedTriangles',
#             'stoich_AvgNeighbourDegreeStoich',
#             'stoich_RangeComplexSize',
#             'stoich_MedComplexSize',
#             'stoich_MedianRatio',
#             'stoich_RangeRatio',

#             # --- Protein-participation metrics ---
#             'protein_MedianUniqueRatio',
#             'protein_RangeUniqueRatio',
#             'protein_MedComplexNodes',
#             'protein_RangeComplexNodes',
#         ],

#         # Stoichiometry features to ablate — must be a subset of HB_GRAPH above.
#         # The set-based hypergraph feature set is derived automatically as
#         # HB_GRAPH minus STOICHIOMETRY_FEATURES.
#         "STOICHIOMETRY_FEATURES": [
#             'stoich_WeightedTriangles',
#             'stoich_AvgNeighbourDegreeStoich',
#             'stoich_RangeComplexSize',
#             'stoich_MedComplexSize',
#             'stoich_MedianRatio',
#             'stoich_RangeRatio',
#         ],
#         "PAIRWISE": [
#             'pair_Degree',
#             'pair_LocalClustCoeff',
#             'pair_TriangleCount',
#             # 'pair_ComponentSize',
#             # 'pair_EigenvectorCentrality',
#             # 'pair_BetweennessCentrality',
#             # 'pair_KatzCentrality',
#             'pair_AvgNeighborDegree',
#         ]
#     }
# }

# splits_path = CONFIG["DATA_DIR"] / CONFIG["SPLITS_FILE"]
# print(f"   Splits file last modified: {pd.Timestamp(os.path.getmtime(splits_path), unit='s')}")
# print(f"   Splits file rows: {pd.read_csv(splits_path).shape}")


# # =======================================================
# # DATA LOADING
# # =======================================================

# def load_all_features() -> pd.DataFrame:
#     """Loads higher-order (hb-graph) and pairwise feature CSVs and merges on ProteinId."""
#     print("1. Loading feature data...")

#     hg_df   = pd.read_csv(CONFIG["DATA_DIR"] / CONFIG["PROTEIN_FEATURES_FILE"])
#     pair_df = pd.read_csv(CONFIG["DATA_DIR"] / CONFIG["PAIRWISE_FEATURES_FILE"])

#     combined = pd.merge(hg_df, pair_df, on='ProteinId', how='inner')

#     print(f"   Higher-order (hb-graph) features shape : {hg_df.shape}")
#     print(f"   Pairwise features shape               : {pair_df.shape}")
#     print(f"   Combined shape                        : {combined.shape}")
#     return combined


# def load_splits() -> pd.DataFrame:
#     """
#     Loads the pre-assigned family-level splits file.

#     Expected columns:
#         split_index   — integer 1..N identifying which split
#         UniProt_AC    — protein identifier (matches ProteinId in feature files)
#         split         — 'train' or 'test'
#         protein_label — 'Drug_target' | 'Non_target' | 'Unknown'
#         label_mask    — bool; False for Unknown proteins (excluded from metrics)
#     """
#     print("2. Loading pre-assigned splits...")
#     splits_df = pd.read_csv(CONFIG["DATA_DIR"] / CONFIG["SPLITS_FILE"])

#     # Rename to match feature file key
#     splits_df = splits_df.rename(columns={'UniProt_AC': 'ProteinId'})

#     # Encode binary target: Drug_target=1, Non_target=0; Unknown kept as NaN
#     label_map = {'Drug_target': 1, 'Non_target': 0}
#     splits_df['target'] = splits_df['protein_label'].map(label_map)

#     n_splits = splits_df['split_index'].nunique()
#     print(f"   Splits file rows  : {len(splits_df)}")
#     print(f"   Unique proteins   : {splits_df['ProteinId'].nunique()}")
#     print(f"   Number of splits  : {n_splits}")

#     labelled = splits_df[splits_df['label_mask']].drop_duplicates('ProteinId')
#     n_dt = (labelled['target'] == 1).sum()
#     n_tot = len(labelled)
#     print(f"   Labelled proteins : {n_tot}  ({100*n_dt/n_tot:.1f}% drug targets)")

#     return splits_df


# # =======================================================
# # MODEL TRAINING & EVALUATION
# # =======================================================

# def tune_and_train_model(X_train: pd.DataFrame, y_train: pd.Series):
#     """Hyperparameter search + fit.  Returns (best_estimator, best_params)."""
#     model_type = CONFIG["MODEL_TYPE"]

#     if model_type == "RandomForest":
#         base_model = RandomForestClassifier(random_state=CONFIG["RANDOM_STATE"])
#         param_grid = CONFIG["PARAM_GRIDS"]["RandomForest"]

#     elif model_type == "LightGBM":
#         base_model = LGBMClassifier(
#             random_state=CONFIG["RANDOM_STATE"], n_jobs=1, verbose=-1
#         )
#         param_grid = CONFIG["PARAM_GRIDS"]["LightGBM"]

#     elif model_type == "XGBoost":
#         pos = int((y_train == 1).sum())
#         neg = int((y_train == 0).sum())
#         spw = float(neg) / float(pos) if pos > 0 else 1.0
#         base_model = XGBClassifier(
#             random_state=CONFIG["RANDOM_STATE"],
#             n_jobs=-1,
#             verbosity=0,
#             eval_metric='logloss',
#             scale_pos_weight=spw
#         )
#         param_grid = CONFIG["PARAM_GRIDS"]["XGBoost"]

#     else:
#         raise ValueError(f"Unknown MODEL_TYPE: '{model_type}'")

#     gs = GridSearchCV(
#         estimator=base_model,
#         param_grid=param_grid,
#         scoring='average_precision',
#         cv=CONFIG["N_SPLITS_CV"],
#         n_jobs=-1,
#         verbose=0
#     )
#     gs.fit(X_train, y_train)
#     return gs.best_estimator_, gs.best_params_


# def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
#     """Returns PR-AUC, F1 for the positive class, and predicted probabilities."""
#     y_pred       = model.predict(X_test)
#     y_pred_proba = model.predict_proba(X_test)[:, 1]

#     report = classification_report(
#         y_test, y_pred,
#         target_names=['Non_target', 'Drug_target'],
#         output_dict=True
#     )

#     return {
#         'pr_auc':       average_precision_score(y_test, y_pred_proba),
#         'f1':           report['Drug_target']['f1-score'],
#         'y_pred_proba': y_pred_proba
#     }


# def compute_permutation_importance(
#     model, X_test: pd.DataFrame, y_test: pd.Series, n_repeats: int = 10
# ) -> Dict[str, float]:
#     """Permutation importance scored by average_precision (PR-AUC drop)."""
#     result = permutation_importance(
#         model, X_test, y_test,
#         scoring='average_precision',
#         n_repeats=n_repeats,
#         random_state=CONFIG["RANDOM_STATE"],
#         n_jobs=-1
#     )
#     return dict(zip(X_test.columns, result.importances_mean))


# # =======================================================
# # PER-SPLIT RUNNER
# # =======================================================

# def run_split(
#     split_idx: int,
#     merged_df: pd.DataFrame,
#     splits_df: pd.DataFrame,
#     hb_graph_features: List[str],
#     hypergraph_features: List[str],
#     pairwise_features: List[str]
# ) -> Dict:
#     """
#     Runs the three nested representations for a single pre-assigned split:
#       pairwise   — dyadic PPI features
#       hypergraph — set-based higher-order features (no stoichiometry)
#       hb_graph   — hypergraph features PLUS stoichiometry (multiset hyperedges)

#     merged_df   — feature matrix (ProteinId + all feature columns)
#     splits_df   — full splits table (all split indices)
#     Returns a results dict with PR-AUC, F1, importances, and per-protein predictions.
#     """
#     # --- Extract this split's assignments ---
#     split_mask = splits_df['split_index'] == split_idx
#     split_info = splits_df[split_mask][['ProteinId', 'split', 'target', 'label_mask']].copy()

#     # Merge features with split assignments
#     df = pd.merge(merged_df, split_info, on='ProteinId', how='inner')

#     # Only use labelled proteins for training/evaluation
#     labelled_df = df[df['label_mask']].copy()

#     train_df = labelled_df[labelled_df['split'] == 'train']
#     test_df  = labelled_df[labelled_df['split'] == 'test']

#     y_train = train_df['target'].astype(int)
#     y_test  = test_df['target'].astype(int)

#     results = {
#         'split_index':  split_idx,
#         'n_train':      len(train_df),
#         'n_test':       len(test_df),
#         'train_dt_pct': 100 * y_train.mean(),
#         'test_dt_pct':  100 * y_test.mean(),
#     }

#     # --- Pairwise model ---
#     X_pair_train = train_df[pairwise_features]
#     X_pair_test  = test_df[pairwise_features]

#     pair_model, pair_params = tune_and_train_model(X_pair_train, y_train)
#     pair_eval = evaluate_model(pair_model, X_pair_test, y_test)

#     results['pairwise_pr_auc']      = pair_eval['pr_auc']
#     results['pairwise_f1']          = pair_eval['f1']
#     results['pairwise_best_params'] = pair_params
#     results['pairwise_importance']  = compute_permutation_importance(
#         pair_model, X_pair_test, y_test
#     )

#     # Store per-protein predictions (pairwise)
#     pair_preds = test_df[['ProteinId']].copy()
#     pair_preds['split_index']      = split_idx
#     pair_preds['true_label']       = y_test.values
#     pair_preds['pair_pred_proba']  = pair_eval['y_pred_proba']
#     results['pairwise_predictions'] = pair_preds

#     # --- Hypergraph model (set-based, no stoichiometry) ---
#     X_hyper_train = train_df[hypergraph_features]
#     X_hyper_test  = test_df[hypergraph_features]

#     hyper_model, hyper_params = tune_and_train_model(X_hyper_train, y_train)
#     hyper_eval = evaluate_model(hyper_model, X_hyper_test, y_test)

#     results['hypergraph_pr_auc']      = hyper_eval['pr_auc']
#     results['hypergraph_f1']          = hyper_eval['f1']
#     results['hypergraph_best_params'] = hyper_params
#     results['hypergraph_importance']  = compute_permutation_importance(
#         hyper_model, X_hyper_test, y_test
#     )

#     hyper_preds = test_df[['ProteinId']].copy()
#     hyper_preds['split_index']       = split_idx
#     hyper_preds['true_label']        = y_test.values
#     hyper_preds['hyper_pred_proba']  = hyper_eval['y_pred_proba']
#     results['hypergraph_predictions'] = hyper_preds

#     # --- HB-graph model (hypergraph + stoichiometry) ---
#     X_hbg_train = train_df[hb_graph_features]
#     X_hbg_test  = test_df[hb_graph_features]

#     hbg_model, hbg_params = tune_and_train_model(X_hbg_train, y_train)
#     hbg_eval = evaluate_model(hbg_model, X_hbg_test, y_test)

#     results['hb_graph_pr_auc']      = hbg_eval['pr_auc']
#     results['hb_graph_f1']          = hbg_eval['f1']
#     results['hb_graph_best_params'] = hbg_params
#     results['hb_graph_importance']  = compute_permutation_importance(
#         hbg_model, X_hbg_test, y_test
#     )

#     # Store per-protein predictions (hb-graph)
#     hbg_preds = test_df[['ProteinId']].copy()
#     hbg_preds['split_index']      = split_idx
#     hbg_preds['true_label']       = y_test.values
#     hbg_preds['hbg_pred_proba']   = hbg_eval['y_pred_proba']
#     results['hb_graph_predictions'] = hbg_preds

#     # Differences
#     # Headline representation contrast: hb_graph vs pairwise
#     results['pr_auc_diff']        = results['hb_graph_pr_auc'] - results['pairwise_pr_auc']
#     results['f1_diff']            = results['hb_graph_f1']     - results['pairwise_f1']
#     # Stoichiometry effect: hb_graph vs hypergraph (adding multiset stoichiometry)
#     results['stoich_pr_auc_diff'] = results['hb_graph_pr_auc'] - results['hypergraph_pr_auc']
#     results['stoich_f1_diff']     = results['hb_graph_f1']     - results['hypergraph_f1']

#     return results


# # =======================================================
# # STATISTICAL COMPARISON
# # =======================================================

# def run_sign_test_comparison(all_results: List[Dict]) -> Dict:
#     """Sign test (binomial) on paired PR-AUC wins/losses across splits.
#     Covers three paired comparisons:
#       1. HB-graph vs Pairwise                        — headline representation effect
#       2. HB-graph vs Hypergraph  — stoichiometry effect (adding multiset stoichiometry)
#       3. Hypergraph vs Pairwise  — set-based representation effect alone
#     """
#     pair_vals  = np.array([r['pairwise_pr_auc']   for r in all_results])
#     hyper_vals = np.array([r['hypergraph_pr_auc'] for r in all_results])
#     hbg_vals   = np.array([r['hb_graph_pr_auc']   for r in all_results])

#     # F1 values (positive class) per representation
#     pair_f1  = np.array([r['pairwise_f1']   for r in all_results])
#     hyper_f1 = np.array([r['hypergraph_f1'] for r in all_results])
#     hbg_f1   = np.array([r['hb_graph_f1']   for r in all_results])

#     def _sign_test(a, b):
#         diffs   = a - b
#         n_wins  = int(np.sum(diffs > 0))
#         n_loss  = int(np.sum(diffs < 0))
#         n_ties  = int(np.sum(diffs == 0))
#         n_valid = n_wins + n_loss
#         if n_valid > 0:
#             p_greater   = binomtest(n_wins, n_valid, 0.5, alternative='greater').pvalue
#             p_two_sided = binomtest(n_wins, n_valid, 0.5, alternative='two-sided').pvalue
#         else:
#             p_greater = p_two_sided = 1.0
#         return dict(wins=n_wins, losses=n_loss, ties=n_ties,
#                     mean_diff=float(np.mean(diffs)), std_diff=float(np.std(diffs)),
#                     p_greater=p_greater, p_two_sided=p_two_sided)

#     hbg_vs_pair   = _sign_test(hbg_vals,   pair_vals)   # headline
#     stoich_effect = _sign_test(hbg_vals,   hyper_vals)  # hb_graph vs hypergraph
#     hyper_vs_pair = _sign_test(hyper_vals, pair_vals)   # representation effect alone

#     return {
#         'n_runs': len(all_results),
#         # --- PR-AUC mean ± std per representation ---
#         'pairwise_pr_auc_mean':   float(np.mean(pair_vals)),
#         'pairwise_pr_auc_std':    float(np.std(pair_vals)),
#         'hypergraph_pr_auc_mean': float(np.mean(hyper_vals)),
#         'hypergraph_pr_auc_std':  float(np.std(hyper_vals)),
#         'hb_graph_pr_auc_mean':   float(np.mean(hbg_vals)),
#         'hb_graph_pr_auc_std':    float(np.std(hbg_vals)),
#         # --- F1 mean ± std per representation (reported in main-paper table) ---
#         'pairwise_f1_mean':   float(np.mean(pair_f1)),
#         'pairwise_f1_std':    float(np.std(pair_f1)),
#         'hypergraph_f1_mean': float(np.mean(hyper_f1)),
#         'hypergraph_f1_std':  float(np.std(hyper_f1)),
#         'hb_graph_f1_mean':   float(np.mean(hbg_f1)),
#         'hb_graph_f1_std':    float(np.std(hbg_f1)),
#         # --- Headline comparison: HB-graph vs Pairwise ---
#         'mean_difference':       hbg_vs_pair['mean_diff'],
#         'std_difference':        hbg_vs_pair['std_diff'],
#         'hb_graph_wins':         hbg_vs_pair['wins'],
#         'pairwise_wins':         hbg_vs_pair['losses'],
#         'ties':                  hbg_vs_pair['ties'],
#         'sign_test_p_greater':   hbg_vs_pair['p_greater'],
#         'sign_test_p_two_sided': hbg_vs_pair['p_two_sided'],
#         # --- Stoichiometry effect: HB-graph vs Hypergraph ---
#         'stoich_effect':         stoich_effect,
#         # --- Representation effect alone: Hypergraph vs Pairwise ---
#         'hyper_vs_pair':         hyper_vs_pair,
#     }


# # =======================================================
# # FEATURE IMPORTANCE AGGREGATION
# # =======================================================

# def aggregate_feature_importance(
#     all_results: List[Dict], representation: str
# ) -> pd.DataFrame:
#     """
#     Aggregates permutation importance across all splits.
#     representation: 'pairwise', 'hypergraph', or 'hb_graph'
#     """
#     key = f'{representation}_importance'
#     records = []
#     for r in all_results:
#         if key in r:
#             for feat, imp in r[key].items():
#                 records.append({'split_index': r['split_index'],
#                                 'feature': feat, 'importance': imp})

#     if not records:
#         return pd.DataFrame()

#     imp_df = pd.DataFrame(records)
#     agg_df = (
#         imp_df.groupby('feature')['importance']
#         .agg(mean='mean', std='std', median='median',
#              min='min', max='max', n_splits='count')
#         .reset_index()
#         .sort_values('mean', ascending=False)
#         .reset_index(drop=True)
#     )
#     agg_df['rank'] = range(1, len(agg_df) + 1)
#     return agg_df


# # =======================================================
# # PRINTING
# # =======================================================

# def print_statistical_summary(stats: Dict):
#     print(f"\n{'='*70}")
#     print("  STATISTICAL COMPARISON")
#     print(f"{'='*70}")
#     print(f"\n  Number of splits: {stats['n_runs']}")

#     # --- PR-AUC (ordered pairwise -> hypergraph -> hb_graph) ---
#     print(f"\n  PR-AUC")
#     print(f"  {'Representation':<20} {'Mean ± Std'}")
#     print(f"  {'-'*45}")
#     print(f"  {'Pairwise':<20} "
#           f"{stats['pairwise_pr_auc_mean']:.4f} ± {stats['pairwise_pr_auc_std']:.4f}")
#     print(f"  {'Hypergraph':<20} "
#           f"{stats['hypergraph_pr_auc_mean']:.4f} ± {stats['hypergraph_pr_auc_std']:.4f}")
#     print(f"  {'HB-graph':<20} "
#           f"{stats['hb_graph_pr_auc_mean']:.4f} ± {stats['hb_graph_pr_auc_std']:.4f}")

#     # --- F1 (positive class; reported in main-paper table) ---
#     print(f"\n  F1 (positive class)")
#     print(f"  {'Representation':<20} {'Mean ± Std'}")
#     print(f"  {'-'*45}")
#     print(f"  {'Pairwise':<20} "
#           f"{stats['pairwise_f1_mean']:.4f} ± {stats['pairwise_f1_std']:.4f}")
#     print(f"  {'Hypergraph':<20} "
#           f"{stats['hypergraph_f1_mean']:.4f} ± {stats['hypergraph_f1_std']:.4f}")
#     print(f"  {'HB-graph':<20} "
#           f"{stats['hb_graph_f1_mean']:.4f} ± {stats['hb_graph_f1_std']:.4f}")

#     def _print_comparison(label, d):
#         print(f"\n  --- {label} ---")
#         print(f"  Mean diff : {d['mean_diff']:+.4f} ± {d['std_diff']:.4f}")
#         print(f"  Wins/Losses/Ties : {d['wins']}/{d['losses']}/{d['ties']}")
#         print(f"  Sign test p (one-sided) : {d['p_greater']:.6f}")
#         print(f"  Sign test p (two-sided) : {d['p_two_sided']:.6f}")

#     _print_comparison("HB-graph vs Pairwise — headline representation effect",
#                       {'mean_diff': stats['mean_difference'],
#                        'std_diff':  stats['std_difference'],
#                        'wins':      stats['hb_graph_wins'],
#                        'losses':    stats['pairwise_wins'],
#                        'ties':      stats['ties'],
#                        'p_greater': stats['sign_test_p_greater'],
#                        'p_two_sided': stats['sign_test_p_two_sided']})
#     _print_comparison("HB-graph vs Hypergraph — stoichiometry effect",
#                       stats['stoich_effect'])
#     _print_comparison("Hypergraph vs Pairwise — representation effect alone",
#                       stats['hyper_vs_pair'])
#     print(f"{'='*70}")


# def print_feature_importance_summary(
#     imp_dfs: List[tuple], top_n: int = 10
# ):
#     """imp_dfs: list of (label, importance_df) tuples, in display order."""
#     print(f"\n{'='*70}")
#     print("  FEATURE IMPORTANCE (Permutation — PR-AUC drop)")
#     print(f"{'='*70}")
#     for label, df in imp_dfs:
#         if df.empty:
#             continue
#         print(f"\n  Top {top_n} {label} Features:")
#         print(f"  {'Rank':<6} {'Feature':<35} {'Mean':<12} {'Std':<10}")
#         print(f"  {'-'*65}")
#         for _, row in df.head(top_n).iterrows():
#             print(f"  {int(row['rank']):<6} {row['feature']:<35} "
#                   f"{row['mean']:.4f}       {row['std']:.4f}")
#     print(f"\n  Note: Higher = more important; negative = possible noise.")
#     print(f"{'='*70}")


# # =======================================================
# # PLOTTING
# # =======================================================

# def plot_paired_comparison(all_results: List[Dict], stats: Dict, output_dir: Path):
#     """Two-panel comparison plot: paired scatter (headline contrast) and 3-way boxplot.

#     Axes are fixed to [0, 1] so panels are directly comparable across tasks/files.
#     """
#     pair_vals  = np.array([r['pairwise_pr_auc']   for r in all_results])
#     hyper_vals = np.array([r['hypergraph_pr_auc'] for r in all_results])
#     hbg_vals   = np.array([r['hb_graph_pr_auc']   for r in all_results])

#     fig, axes = plt.subplots(1, 2, figsize=(13, 6))

#     # Panel 1: paired scatter — headline contrast (HB-graph vs Pairwise), one point per split
#     ax1 = axes[0]
#     ax1.scatter(pair_vals, hbg_vals, alpha=0.7, s=60, zorder=3)
#     ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, label='y = x')
#     ax1.set_xlabel('Pairwise PR-AUC')
#     ax1.set_ylabel('HB-graph PR-AUC')
#     ax1.set_title('Paired Comparison — One Point per Split')
#     ax1.set_xlim(0, 1)
#     ax1.set_ylim(0, 1)
#     ax1.set_aspect('equal')
#     ax1.legend(loc='upper left')
#     above = int(np.sum(hbg_vals > pair_vals))
#     below = int(np.sum(hbg_vals < pair_vals))
#     ax1.text(0.95, 0.05,
#              f'HB-graph wins: {above}\nPairwise wins: {below}',
#              transform=ax1.transAxes, ha='right', va='bottom',
#              bbox=dict(facecolor='lightgreen', alpha=0.5))

#     # Panel 2: boxplot across all three representations
#     ax2 = axes[1]
#     box_data   = [pair_vals, hyper_vals, hbg_vals]
#     box_labels = ['Pairwise', 'Hypergraph', 'HB-graph']
#     box_colors = ['lightgray', 'skyblue', 'steelblue']
#     bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
#     for patch, colour in zip(bp['boxes'], box_colors):
#         patch.set_facecolor(colour)
#     ax2.set_ylabel('PR-AUC')
#     ax2.set_title('Distribution Comparison')
#     ax2.set_ylim(0, 1)
#     rng = np.random.default_rng(0)
#     for i, data in enumerate(box_data):
#         x = rng.normal(i + 1, 0.04, size=len(data))
#         ax2.scatter(x, data, alpha=0.4, s=20, color='black')

#     plt.tight_layout()
#     plt.savefig(output_dir / 'paired_comparison.png', dpi=300)
#     plt.close()
#     print("   Saved: paired_comparison.png")


# def plot_stoich_ablation(all_results: List[Dict], stats: Dict, output_dir: Path):
#     """
#     Two-panel stoichiometry ablation figure matching the poster's Fig 5 style.

#     Panel 1 — Scatter: hb-graph vs hypergraph PR-AUC, one point per split.
#                Points above the diagonal = stoichiometry helps.
#     Panel 2 — Boxplot: pairwise / hypergraph / hb-graph distributions,
#                showing the stepwise improvement from adding stoichiometry.
#     """
#     pair_vals  = np.array([r['pairwise_pr_auc']   for r in all_results])
#     hyper_vals = np.array([r['hypergraph_pr_auc'] for r in all_results])
#     hbg_vals   = np.array([r['hb_graph_pr_auc']   for r in all_results])

#     stoich_wins   = int(np.sum(hbg_vals > hyper_vals))
#     stoich_losses = int(np.sum(hbg_vals < hyper_vals))

#     ab  = stats['stoich_effect']
#     p_one = ab['p_greater']
#     p_two = ab['p_two_sided']

#     fig, axes = plt.subplots(1, 2, figsize=(12, 6))

#     # ── Panel 1: scatter — hb-graph vs hypergraph ────────────────────────────
#     ax1 = axes[0]
#     ax1.scatter(hyper_vals, hbg_vals, alpha=0.7, s=60, zorder=3,
#                 color='steelblue')
#     ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, label='y = x (no difference)')
#     ax1.set_xlabel('Hypergraph PR-AUC')
#     ax1.set_ylabel('HB-graph PR-AUC')
#     ax1.set_title('Stoichiometry Ablation — One Point per Split')
#     ax1.set_xlim(0, 1)
#     ax1.set_ylim(0, 1)
#     ax1.set_aspect('equal')
#     ax1.legend(fontsize=12, loc='upper left')
#     ax1.text(0.97, 0.03,
#              f'HB-graph wins: {stoich_wins}\nHypergraph wins: {stoich_losses}',
#              transform=ax1.transAxes, ha='right', va='bottom', fontsize=12,
#              bbox=dict(facecolor='lightgreen', alpha=0.5))

#     # ── Panel 2: boxplot — pairwise / hypergraph / hb-graph ──────────────────
#     ax2 = axes[1]
#     colours = ['lightgray', 'skyblue', 'steelblue']
#     labels  = ['Pairwise', 'Hypergraph', 'HB-graph']
#     box_data = [pair_vals, hyper_vals, hbg_vals]
#     bp = ax2.boxplot(
#         box_data,
#         labels=labels,
#         patch_artist=True,
#         medianprops=dict(color='black', linewidth=2),
#     )
#     for patch, colour in zip(bp['boxes'], colours):
#         patch.set_facecolor(colour)
#     ax2.set_ylabel('PR-AUC')
#     ax2.set_title('Distribution Comparison')
#     ax2.set_ylim(0, 1)
#     rng = np.random.default_rng(0)
#     for i, data in enumerate(box_data):
#         x = rng.normal(i + 1, 0.04, size=len(data))
#         ax2.scatter(x, data, alpha=0.4, s=20, color='black', zorder=3)

#     # Annotate with mean ± std for each box
#     for i, vals in enumerate(box_data):
#         ax2.text(i + 1, 0.02,
#                  f'{vals.mean():.3f}±{vals.std():.3f}',
#                  ha='center', va='bottom', fontsize=10)

#     # Sign test annotation (stoichiometry effect: hb-graph vs hypergraph)
#     sig_label = (f'Stoich effect\np (one-sided) = {p_one:.4f}\n'
#                  f'p (two-sided) = {p_two:.4f}')
#     ax2.text(0.97, 0.97, sig_label,
#              transform=ax2.transAxes, ha='right', va='top', fontsize=10,
#              bbox=dict(facecolor='lightyellow', alpha=0.8))

#     plt.tight_layout()
#     plt.savefig(output_dir / 'stoich_ablation.png', dpi=300, bbox_inches='tight')
#     plt.close()
#     print("   Saved: stoich_ablation.png")


# def plot_feature_importance(
#     imp_dfs: List[tuple],
#     output_dir: Path,
#     top_n: int = 15
# ):
#     """Side-by-side horizontal bar charts of permutation importance.

#     imp_dfs: list of (label, importance_df, colour) tuples, in display order.
#     """
#     n = len(imp_dfs)
#     fig, axes = plt.subplots(1, n, figsize=(8 * n, 8))
#     if n == 1:
#         axes = [axes]

#     for ax, (label, df, colour) in zip(axes, imp_dfs):
#         top = df.head(top_n)
#         colors = [colour if v > 0 else 'lightcoral' for v in top['mean']]
#         ax.barh(range(len(top)), top['mean'], xerr=top['std'],
#                 color=colors, edgecolor='black', capsize=3)
#         ax.set_yticks(range(len(top)))
#         ax.set_yticklabels(top['feature'])
#         ax.invert_yaxis()
#         ax.set_xlabel('Mean Permutation Importance (PR-AUC drop)')
#         ax.set_title(f'Top {top_n} {label} Features')
#         ax.axvline(0, color='gray', linestyle='--', linewidth=1)

#     plt.tight_layout()
#     plt.savefig(output_dir / 'feature_importance_comparison.png', dpi=300)
#     plt.close()
#     print("   Saved: feature_importance_comparison.png")


# # =======================================================
# # MAIN
# # =======================================================

# if __name__ == "__main__":

#     start_time = time.time()
#     print(f"Process started at {time.strftime('%H:%M:%S', time.localtime(start_time))}")

#     # --- Output directory ---
#     output_dir = CONFIG["BASE_OUTPUT_DIR"]
#     output_dir.mkdir(parents=True, exist_ok=True)
#     CONFIG["OUTPUT_DIR"] = output_dir

#     print(f"\n{'='*70}")
#     print(f"  REPRESENTATION COMPARISON: PAIRWISE vs HYPERGRAPH vs HB-GRAPH")
#     print(f"  Task   : Drug Target Prediction (ChEMBL)")
#     print(f"  Model  : {CONFIG['MODEL_TYPE']}")
#     print(f"  Splits : pre-assigned family-level")
#     print(f"  Output : {output_dir}")
#     print(f"{'='*70}\n")

#     # --- Load data ---
#     features_df = load_all_features()
#     splits_df   = load_splits()

#     split_indices = sorted(splits_df['split_index'].unique())
#     print(f"\n   Running {len(split_indices)} splits: {split_indices}\n")

#     # --- Resolve active features (only keep those actually present in features_df) ---
#     # hb_graph   = full higher-order feature set (includes stoichiometry)
#     # hypergraph = hb_graph minus stoichiometry (set-based representation)
#     hb_graph_features = [f for f in CONFIG["FEATURES"]["HB_GRAPH"]
#                          if f in features_df.columns]
#     pairwise_features = [f for f in CONFIG["FEATURES"]["PAIRWISE"]
#                          if f in features_df.columns]

#     stoich_features = CONFIG["FEATURES"].get("STOICHIOMETRY_FEATURES", [])
#     hypergraph_features = [f for f in hb_graph_features
#                            if f not in stoich_features]

#     missing_hbg  = [f for f in CONFIG["FEATURES"]["HB_GRAPH"] if f not in features_df.columns]
#     missing_pair = [f for f in CONFIG["FEATURES"]["PAIRWISE"] if f not in features_df.columns]
#     if missing_hbg:
#         print(f"   WARNING: {len(missing_hbg)} hb-graph features not found in data: {missing_hbg}")
#     if missing_pair:
#         print(f"   WARNING: {len(missing_pair)} pairwise features not found in data: {missing_pair}")

#     print(f"   Active pairwise features ({len(pairwise_features)}):")
#     for f in pairwise_features:
#         print(f"     - {f}")
#     print(f"   Active hypergraph features ({len(hypergraph_features)}):")
#     for f in hypergraph_features:
#         print(f"     - {f}")
#     print(f"   Active hb-graph features ({len(hb_graph_features)}):")
#     for f in hb_graph_features:
#         tag = " [stoich]" if f in stoich_features else ""
#         print(f"     - {f}{tag}")

#     # --- Fill any NaNs in feature columns ---
#     all_feature_cols = hb_graph_features + pairwise_features
#     n_nans = features_df[all_feature_cols].isna().sum().sum()
#     if n_nans > 0:
#         print(f"   Filling {n_nans} missing feature values with 0.")
#         features_df[all_feature_cols] = features_df[all_feature_cols].fillna(0)

#     # --- Main loop over splits ---
#     print(f"\n3. Running paired comparisons across {len(split_indices)} splits...\n")
#     all_results = []

#     for split_idx in split_indices:
#         print(f"   Split {split_idx:>2}/{len(split_indices)}...", end=" ", flush=True)
#         try:
#             result = run_split(
#                 split_idx, features_df, splits_df,
#                 hb_graph_features, hypergraph_features, pairwise_features
#             )
#             all_results.append(result)
#             winner = ("HB-graph" if result['pr_auc_diff'] > 0
#                       else "Pair" if result['pr_auc_diff'] < 0 else "Tie")
#             print(f"train={result['n_train']} ({result['train_dt_pct']:.1f}% dt)  "
#                   f"test={result['n_test']} ({result['test_dt_pct']:.1f}% dt)  |  "
#                   f"Pair: {result['pairwise_pr_auc']:.4f}, "
#                   f"Hyper: {result['hypergraph_pr_auc']:.4f}, "
#                   f"HB-graph: {result['hb_graph_pr_auc']:.4f}, "
#                   f"Diff(stoich): {result['stoich_pr_auc_diff']:+.4f} [{winner}]")
#         except Exception as e:
#             print(f"ERROR: {e}")

#     # --- Statistical comparison ---
#     print("\n4. Statistical analysis...")
#     stats = run_sign_test_comparison(all_results)
#     print_statistical_summary(stats)

#     # --- Plots ---
#     print("\n5. Generating plots...")
#     plot_paired_comparison(all_results, stats, output_dir)
#     plot_stoich_ablation(all_results, stats, output_dir)

#     # --- Feature importance ---
#     print("\n6. Aggregating feature importance...")
#     pair_imp_df  = aggregate_feature_importance(all_results, 'pairwise')
#     hyper_imp_df = aggregate_feature_importance(all_results, 'hypergraph')
#     hbg_imp_df   = aggregate_feature_importance(all_results, 'hb_graph')
#     print_feature_importance_summary(
#         [("Pairwise", pair_imp_df),
#          ("Hypergraph", hyper_imp_df),
#          ("HB-graph", hbg_imp_df)],
#         top_n=10
#     )
#     plot_feature_importance(
#         [("Pairwise", pair_imp_df, 'gray'),
#          ("Hypergraph", hyper_imp_df, 'skyblue'),
#          ("HB-graph", hbg_imp_df, 'steelblue')],
#         output_dir, top_n=15
#     )

#     # --- Save CSVs ---
#     print("\n7. Saving outputs...")

#     # Per-split summary (no nested dicts), ordered pairwise -> hypergraph -> hb_graph
#     summary_cols = ['split_index', 'n_train', 'n_test', 'train_dt_pct', 'test_dt_pct',
#                     'pairwise_pr_auc',   'pairwise_f1',
#                     'hypergraph_pr_auc', 'hypergraph_f1',
#                     'hb_graph_pr_auc',   'hb_graph_f1',
#                     'pr_auc_diff', 'f1_diff',
#                     'stoich_pr_auc_diff', 'stoich_f1_diff']
#     summary_df = pd.DataFrame([{k: r[k] for k in summary_cols} for r in all_results])
#     summary_df.to_csv(output_dir / 'split_results.csv', index=False)
#     print("   Saved: split_results.csv")

#     # Per-protein predictions — pairwise
#     pair_preds_all = pd.concat(
#         [r['pairwise_predictions'] for r in all_results], ignore_index=True
#     )
#     pair_preds_all.to_csv(output_dir / 'pairwise_predictions.csv', index=False)
#     print("   Saved: pairwise_predictions.csv")

#     # Per-protein predictions — hypergraph
#     hyper_preds_all = pd.concat(
#         [r['hypergraph_predictions'] for r in all_results], ignore_index=True
#     )
#     hyper_preds_all.to_csv(output_dir / 'hypergraph_predictions.csv', index=False)
#     print("   Saved: hypergraph_predictions.csv")

#     # Per-protein predictions — hb-graph
#     hbg_preds_all = pd.concat(
#         [r['hb_graph_predictions'] for r in all_results], ignore_index=True
#     )
#     hbg_preds_all.to_csv(output_dir / 'hb_graph_predictions.csv', index=False)
#     print("   Saved: hb_graph_predictions.csv")

#     # Feature importance
#     pair_imp_df.to_csv(output_dir / 'pairwise_feature_importance.csv', index=False)
#     hyper_imp_df.to_csv(output_dir / 'hypergraph_feature_importance.csv', index=False)
#     hbg_imp_df.to_csv(output_dir / 'hb_graph_feature_importance.csv', index=False)
#     print("   Saved: pairwise_feature_importance.csv")
#     print("   Saved: hypergraph_feature_importance.csv")
#     print("   Saved: hb_graph_feature_importance.csv")

#     with open(output_dir / 'statistical_summary.txt', 'w') as f:
#             f.write("REPRESENTATION COMPARISON: PAIRWISE vs HYPERGRAPH vs HB-GRAPH\n")
#             f.write("Task: Drug Target Prediction (ChEMBL)\n")
#             f.write(f"Model: {CONFIG['MODEL_TYPE']}\n")
#             f.write(f"Number of splits: {stats['n_runs']}\n\n")
#             f.write(f"Pairwise features ({len(pairwise_features)}):\n")
#             for feat in pairwise_features:
#                 f.write(f"  - {feat}\n")
#             f.write(f"\nHypergraph features ({len(hypergraph_features)}):\n")
#             for feat in hypergraph_features:
#                 f.write(f"  - {feat}\n")
#             f.write(f"\nHB-graph features ({len(hb_graph_features)}):\n")
#             for feat in hb_graph_features:
#                 tag = " [stoich]" if feat in stoich_features else ""
#                 f.write(f"  - {feat}{tag}\n")

#             f.write(f"\nPR-AUC Mean ± Std:\n")
#             f.write(f"  Pairwise   : {stats['pairwise_pr_auc_mean']:.4f} ± {stats['pairwise_pr_auc_std']:.4f}\n")
#             f.write(f"  Hypergraph : {stats['hypergraph_pr_auc_mean']:.4f} ± {stats['hypergraph_pr_auc_std']:.4f}\n")
#             f.write(f"  HB-graph   : {stats['hb_graph_pr_auc_mean']:.4f} ± {stats['hb_graph_pr_auc_std']:.4f}\n")

#             f.write(f"\nF1 (positive class) Mean ± Std:\n")
#             f.write(f"  Pairwise   : {stats['pairwise_f1_mean']:.4f} ± {stats['pairwise_f1_std']:.4f}\n")
#             f.write(f"  Hypergraph : {stats['hypergraph_f1_mean']:.4f} ± {stats['hypergraph_f1_std']:.4f}\n")
#             f.write(f"  HB-graph   : {stats['hb_graph_f1_mean']:.4f} ± {stats['hb_graph_f1_std']:.4f}\n")

#             def _write_comparison(label, d):
#                 f.write(f"\n{label}:\n")
#                 f.write(f"  Mean diff : {d['mean_diff']:+.4f} ± {d['std_diff']:.4f}\n")
#                 f.write(f"  Wins/Losses/Ties : {d['wins']}/{d['losses']}/{d['ties']}\n")
#                 f.write(f"  Sign test p (one-sided) : {d['p_greater']:.6f}\n")
#                 f.write(f"  Sign test p (two-sided) : {d['p_two_sided']:.6f}\n")

#             _write_comparison("HB-graph vs Pairwise — headline representation effect",
#                               {'mean_diff': stats['mean_difference'],
#                                'std_diff':  stats['std_difference'],
#                                'wins':      stats['hb_graph_wins'],
#                                'losses':    stats['pairwise_wins'],
#                                'ties':      stats['ties'],
#                                'p_greater': stats['sign_test_p_greater'],
#                                'p_two_sided': stats['sign_test_p_two_sided']})
#             _write_comparison("HB-graph vs Hypergraph — stoichiometry effect",
#                               stats['stoich_effect'])
#             _write_comparison("Hypergraph vs Pairwise — representation effect alone",
#                               stats['hyper_vs_pair'])

#     print(f"\n{'='*70}")
#     print("  COMPLETE")
#     print(f"{'='*70}")

#     elapsed = time.time() - start_time
#     print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")