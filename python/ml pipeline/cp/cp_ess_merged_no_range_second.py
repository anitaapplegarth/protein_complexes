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
    "BASE_OUTPUT_DIR": Path("./randomforest/ess_two_hop_no_range_stoich_features"),

    # --- File Names ---
    "SPLITS_FILE":           "ess_protein_merged_splits.csv",
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
            # 'stoich_RangeComplexSize',
            'stoich_MedComplexSize',
            'stoich_MedianRatio',
            # 'stoich_RangeRatio',

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
            # 'stoich_RangeComplexSize',
            'stoich_MedComplexSize',
            'stoich_MedianRatio',
            # 'stoich_RangeRatio',
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

    # Encode binary target: Essential=1, Non-essential=0; Unknown kept as NaN
    label_map = {'Essential': 1, 'Non-essential': 0}
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
    pairwise_features: List[str]
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

    # Differences
    results['pr_auc_diff']          = results['hypergraph_pr_auc'] - results['pairwise_pr_auc']
    results['f1_diff']              = results['hypergraph_f1']     - results['pairwise_f1']
    results['stoich_pr_auc_diff']   = results['hypergraph_pr_auc'] - results['hyper_nostoich_pr_auc']
    results['stoich_f1_diff']       = results['hypergraph_f1']     - results['hyper_nostoich_f1']

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

    return {
        'n_runs': len(all_results),
        'hypergraph_pr_auc_mean':   float(np.mean(hyper_vals)),
        'hypergraph_pr_auc_std':    float(np.std(hyper_vals)),
        'hyper_nostoich_pr_auc_mean': float(np.mean(nostoich_vals)),
        'hyper_nostoich_pr_auc_std':  float(np.std(nostoich_vals)),
        'pairwise_pr_auc_mean':     float(np.mean(pair_vals)),
        'pairwise_pr_auc_std':      float(np.std(pair_vals)),
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


# =======================================================
# MAIN
# =======================================================

if __name__ == "__main__":

    start_time = time.time()
    print(f"Process started at {time.strftime('%H:%M:%S', time.localtime(start_time))}")

    # --- Output directory ---
    output_dir = CONFIG["BASE_OUTPUT_DIR"] / "essentiality_family_splits"
    output_dir.mkdir(parents=True, exist_ok=True)
    CONFIG["OUTPUT_DIR"] = output_dir

    print(f"\n{'='*70}")
    print(f"  PAIRED COMPARISON: HYPERGRAPH vs PAIRWISE")
    print(f"  Task   : Gene Essentiality")
    print(f"  Model  : {CONFIG['MODEL_TYPE']}")
    print(f"  Splits : pre-assigned family-level (protein_splits_all_strat.csv)")
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
                hypergraph_features, hypergraph_no_stoich_features, pairwise_features
            )
            all_results.append(result)
            winner = ("Hyper" if result['pr_auc_diff'] > 0
                      else "Pair" if result['pr_auc_diff'] < 0 else "Tie")
            print(f"train={result['n_train']} ({result['train_ess_pct']:.1f}% ess)  "
                  f"test={result['n_test']} ({result['test_ess_pct']:.1f}% ess)  |  "
                  f"Hyper: {result['hypergraph_pr_auc']:.4f}, "
                  f"Hyper(no-stoich): {result['hyper_nostoich_pr_auc']:.4f}, "
                  f"Pair: {result['pairwise_pr_auc']:.4f}, "
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
                    'pr_auc_diff', 'f1_diff',
                    'stoich_pr_auc_diff', 'stoich_f1_diff']
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
    print("   Saved: hypergraph_feature_importance.csv")
    print("   Saved: pairwise_feature_importance.csv")

    with open(output_dir / 'statistical_summary.txt', 'w') as f:
            f.write("PAIRED COMPARISON: HYPERGRAPH vs PAIRWISE (with stoichiometry ablation)\n")
            f.write("Task: Gene Essentiality\n")
            f.write(f"Model: {CONFIG['MODEL_TYPE']}\n")
            f.write(f"Number of splits: {stats['n_runs']}\n\n")
            f.write(f"Hypergraph features ({len(hypergraph_features)}):\n")
            for feat in hypergraph_features:
                tag = ' [stoich]' if feat in stoich_features else ''
                f.write(f"  - {feat}{tag}\n")
            f.write(f"\nHypergraph no-stoich features ({len(hypergraph_no_stoich_features)}):\n")
            for feat in hypergraph_no_stoich_features:
                f.write(f"  - {feat}\n")
            f.write(f"\nPairwise features ({len(pairwise_features)}):\n")
            for feat in pairwise_features:
                f.write(f"  - {feat}\n")
            f.write(f"\nPR-AUC Mean \u00b1 Std:\n")
            f.write(f"  Hypergraph (full)     : {stats['hypergraph_pr_auc_mean']:.4f} \u00b1 {stats['hypergraph_pr_auc_std']:.4f}\n")
            f.write(f"  Hypergraph (no stoich): {stats['hyper_nostoich_pr_auc_mean']:.4f} \u00b1 {stats['hyper_nostoich_pr_auc_std']:.4f}\n")
            f.write(f"  Pairwise             : {stats['pairwise_pr_auc_mean']:.4f} \u00b1 {stats['pairwise_pr_auc_std']:.4f}\n")

            def _write_comparison(label, d):
                f.write(f"\n{label}:\n")
                f.write(f"  Mean diff : {d['mean_diff']:+.4f} \u00b1 {d['std_diff']:.4f}\n")
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
            _write_comparison("Hypergraph (full) vs Hypergraph (no stoich) \u2014 stoichiometry effect",
                              stats['stoich_ablation'])
            _write_comparison("Hypergraph (no stoich) vs Pairwise \u2014 representation effect alone",
                              stats['nostoich_vs_pair'])

    print(f"\n{'='*70}")
    print("  COMPLETE")
    print(f"{'='*70}")

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")