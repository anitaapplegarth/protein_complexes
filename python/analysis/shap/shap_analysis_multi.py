"""
SHAP Analysis for Hypergraph vs Pairwise Protein Complex Models
================================================================
Standalone script that retrains on the best-performing split and generates:
  1. SHAP beeswarm (summary) plots — direction + magnitude for all features
  2. SHAP dependence plots — for key features of interest
  3. Feature distribution comparison (essential vs non-essential, target vs non-target)

Usage:
  Configure the ANALYSIS block below, then run:
    python shap_analysis.py

Requirements:
  pip install shap matplotlib pandas numpy scikit-learn lightgbm xgboost
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap as shap_lib  # renamed to avoid notebook shadowing
import warnings
from pathlib import Path
from typing import List, Tuple

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score

warnings.filterwarnings('ignore')

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
# ANALYSIS CONFIGURATION
# =======================================================
# Change these settings for each analysis you want to run.
# The script will:
#   1. Load splits + features
#   2. Find the best-performing split (highest hypergraph PR-AUC)
#   3. Retrain the model on that split
#   4. Compute SHAP values on the test set
#   5. Generate plots

ANALYSIS = {
    # --- Which analysis to run ---
    # Options: "cp_essentiality", "cp_drug_hpa", "cp_drug_chembl",
    #          "corum_essentiality", "corum_drug_hpa", "corum_drug_chembl"
    "TASK": "corum_essentiality",

    # --- Model ---
    # Options: "RandomForest", "LightGBM", "XGBoost"
    "MODEL_TYPE": "RandomForest",

    # --- Paths (adjust to your local setup) ---
    "DATA_DIR": Path("../../../data/lookup_tables"),

    # --- Where to find the split_results.csv from your pipeline ---
    # This is used to identify the best-performing split.
    # Set to None to use split_index=1 as default.
    "RESULTS_CSV": Path("../randomforest/corum_two_hop_features/essentiality_family_splits/split_results.csv"),

    # --- Output ---
    "OUTPUT_DIR": Path("./shap_analysis"),

    # --- Which split to use (None = auto-select best from RESULTS_CSV) ---
    "SPLIT_INDEX": None,

    # --- Number of top splits to analyse (by hypergraph PR-AUC) ---
    "N_TOP_SPLITS": 15,

    # --- Features to generate dependence plots for ---
    # Set to None to auto-select top 4 from SHAP importance
    "DEPENDENCE_FEATURES": None,

    # --- Random state ---
    "RANDOM_STATE": 42,
    "N_SPLITS_CV": 5,
}

# =======================================================
# TASK-SPECIFIC CONFIGURATION
# =======================================================
# Maps task name -> (splits file, label column logic, feature sets, has_stoich)

TASK_CONFIG = {
    "cp_essentiality": {
        "splits_file": "cp_ess_protein_splits.csv",
        "hg_features_file": "cp_hypergraph_features.csv",
        "pw_features_file": "cp_pairwise_features.csv",
        "label_col": "protein_label",
        "label_map": {"Essential": 1, "Non-essential": 0},
        "positive_label": "Essential",
        "has_stoich": True,
        "hypergraph_features": [
            'base_Degree', 'base_LocalClustCoeff',
            'base_TriangleCount', 'base_UniquePartners',
            'base_AvgNeighbourDegree',
            'stoich_WeightedTriangles', 'stoich_AvgNeighbourDegreeStoich',
            'stoich_RangeComplexSize', 'stoich_MedComplexSize',
            'stoich_MedianRatio', 'stoich_RangeRatio',
            'protein_MedianUniqueRatio', 'protein_RangeUniqueRatio',
            'protein_MedComplexNodes', 'protein_RangeComplexNodes',
        ],
        "pairwise_features": [
            'pair_Degree',
            'pair_LocalClustCoeff',
            'pair_TriangleCount',
            'pair_AvgNeighborDegree',
        ],
    },
    "cp_drug_hpa": {
        "splits_file": "cp_drug_hpa_protein_splits.csv",
        "hg_features_file": "cp_hypergraph_features.csv",
        "pw_features_file": "cp_pairwise_features.csv",
        "label_col": "protein_label",
        "label_map": {"Drug_target": 1, "Non_target": 0},
        "positive_label": "Drug_target",
        "has_stoich": True,
        "hypergraph_features": [
            'base_Degree', 'base_LocalClustCoeff',
            'base_TriangleCount', 'base_UniquePartners',
            'base_AvgNeighbourDegree',
            'stoich_WeightedTriangles', 'stoich_AvgNeighbourDegreeStoich',
            'stoich_RangeComplexSize', 'stoich_MedComplexSize',
            'stoich_MedianRatio', 'stoich_RangeRatio',
            'protein_MedianUniqueRatio', 'protein_RangeUniqueRatio',
            'protein_MedComplexNodes', 'protein_RangeComplexNodes',
        ],
        "pairwise_features": [
            'pair_Degree',
            'pair_LocalClustCoeff',
            'pair_TriangleCount',
            'pair_AvgNeighborDegree',
        ],
    },
    "cp_drug_chembl": {
        "splits_file": "cp_drug_chembl_protein_splits.csv",
        "hg_features_file": "cp_hypergraph_features.csv",
        "pw_features_file": "cp_pairwise_features.csv",
        "label_col": "protein_label",
        "label_map": {"Drug_target": 1, "Non_target": 0},
        "positive_label": "Drug_target",
        "has_stoich": True,
        "hypergraph_features": [
            'base_Degree', 'base_LocalClustCoeff',
            'base_TriangleCount', 'base_UniquePartners',
            'base_AvgNeighbourDegree',
            'stoich_WeightedTriangles', 'stoich_AvgNeighbourDegreeStoich',
            'stoich_RangeComplexSize', 'stoich_MedComplexSize',
            'stoich_MedianRatio', 'stoich_RangeRatio',
            'protein_MedianUniqueRatio', 'protein_RangeUniqueRatio',
            'protein_MedComplexNodes', 'protein_RangeComplexNodes',
        ],
        "pairwise_features": [
            'pair_Degree',
            'pair_LocalClustCoeff',
            'pair_TriangleCount',
            'pair_AvgNeighborDegree',
        ],
    },
    "corum_essentiality": {
        "splits_file": "corum_ess_protein_splits.csv",
        "hg_features_file": "corum_hypergraph_features.csv",
        "pw_features_file": "corum_pairwise_features.csv",
        "label_col": "protein_label",
        "label_map": {"Essential": 1, "Non-essential": 0},
        "positive_label": "Essential",
        "has_stoich": False,
        "hypergraph_features": [
            'base_Degree', 'base_LocalClustCoeff',
            'base_TriangleCount', 'base_UniquePartners',
            'base_AvgNeighbourDegree',
            'protein_MedianUniqueRatio', 'protein_RangeUniqueRatio',
            'protein_MedComplexNodes', 'protein_RangeComplexNodes',
        ],
        "pairwise_features": [
            'pair_Degree',
            'pair_LocalClustCoeff',
            'pair_TriangleCount',
            'pair_AvgNeighborDegree',
        ],
    },
    "corum_drug_hpa": {
        "splits_file": "corum_drug_hpa_protein_splits.csv",
        "hg_features_file": "corum_hypergraph_features.csv",
        "pw_features_file": "corum_pairwise_features.csv",
        "label_col": "protein_label",
        "label_map": {"Drug_target": 1, "Non_target": 0},
        "positive_label": "Drug_target",
        "has_stoich": False,
        "hypergraph_features": [
            'base_Degree', 'base_LocalClustCoeff',
            'base_TriangleCount', 'base_UniquePartners',
            'base_AvgNeighbourDegree',
            'protein_MedianUniqueRatio', 'protein_RangeUniqueRatio',
            'protein_MedComplexNodes', 'protein_RangeComplexNodes',
        ],
        "pairwise_features": [
            'pair_Degree',
            'pair_LocalClustCoeff',
            'pair_TriangleCount',
            'pair_AvgNeighborDegree',
        ],
    },
    "corum_drug_chembl": {
        "splits_file": "corum_drug_chembl_protein_splits.csv",
        "hg_features_file": "corum_hypergraph_features.csv",
        "pw_features_file": "corum_pairwise_features.csv",
        "label_col": "protein_label",
        "label_map": {"Drug_target": 1, "Non_target": 0},
        "positive_label": "Drug_target",
        "has_stoich": False,
        "hypergraph_features": [
            'base_Degree', 'base_LocalClustCoeff',
            'base_TriangleCount', 'base_UniquePartners',
            'base_AvgNeighbourDegree',
            'protein_MedianUniqueRatio', 'protein_RangeUniqueRatio',
            'protein_MedComplexNodes', 'protein_RangeComplexNodes',
        ],
        "pairwise_features": [
            'pair_Degree',
            'pair_LocalClustCoeff',
            'pair_TriangleCount',
            'pair_AvgNeighborDegree',
        ],
    },
}

# =======================================================
# PARAM GRIDS (matching your pipeline)
# =======================================================
PARAM_GRIDS = {
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
    }
}


# =======================================================
# HELPER FUNCTIONS
# =======================================================

def load_data(task_cfg: dict, data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and merge feature files and splits."""
    hg_df = pd.read_csv(data_dir / task_cfg["hg_features_file"])
    pw_df = pd.read_csv(data_dir / task_cfg["pw_features_file"])
    features_df = pd.merge(hg_df, pw_df, on='ProteinId', how='inner')

    splits_df = pd.read_csv(data_dir / task_cfg["splits_file"])
    splits_df = splits_df.rename(columns={'UniProt_AC': 'ProteinId'})
    splits_df['target'] = splits_df[task_cfg["label_col"]].map(task_cfg["label_map"])

    return features_df, splits_df


def find_best_split(results_csv: Path) -> int:
    """Find the split with highest hypergraph PR-AUC."""
    df = pd.read_csv(results_csv)
    best_idx = df.loc[df['hypergraph_pr_auc'].idxmax(), 'split_index']
    best_prauc = df.loc[df['hypergraph_pr_auc'].idxmax(), 'hypergraph_pr_auc']
    print(f"   Best split: {int(best_idx)} (PR-AUC = {best_prauc:.4f})")
    return int(best_idx)


def find_top_n_splits(results_csv: Path, n: int = 3) -> List[int]:
    """Find the top N splits by hypergraph PR-AUC."""
    df = pd.read_csv(results_csv)
    top = df.nlargest(n, 'hypergraph_pr_auc')
    split_indices = top['split_index'].astype(int).tolist()
    praucs = top['hypergraph_pr_auc'].tolist()
    for idx, prauc in zip(split_indices, praucs):
        print(f"   Split {idx}: PR-AUC = {prauc:.4f}")
    return split_indices


def get_train_test(
    features_df: pd.DataFrame,
    splits_df: pd.DataFrame,
    split_idx: int,
    feature_cols: List[str]
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Extract train/test X, y for a given split and feature set."""
    split_info = splits_df[splits_df['split_index'] == split_idx][
        ['ProteinId', 'split', 'target', 'label_mask']
    ].copy()

    df = pd.merge(features_df, split_info, on='ProteinId', how='inner')
    labelled = df[df['label_mask']].copy()

    train = labelled[labelled['split'] == 'train']
    test = labelled[labelled['split'] == 'test']

    # Only keep features that exist in the data
    valid_features = [f for f in feature_cols if f in features_df.columns]
    if len(valid_features) < len(feature_cols):
        missing = set(feature_cols) - set(valid_features)
        print(f"   WARNING: Features not found in data (skipped): {missing}")

    X_train = train[valid_features].fillna(0)
    X_test = test[valid_features].fillna(0)
    y_train = train['target'].astype(int)
    y_test = test['target'].astype(int)

    return X_train, y_train, X_test, y_test


def train_model(X_train, y_train, model_type: str, random_state: int = 42, n_cv: int = 5):
    """Train with GridSearchCV (matching your pipeline exactly)."""
    if model_type == "RandomForest":
        base = RandomForestClassifier(random_state=random_state)
    elif model_type == "LightGBM":
        base = LGBMClassifier(random_state=random_state, n_jobs=1, verbose=-1)
    elif model_type == "XGBoost":
        pos = int((y_train == 1).sum())
        neg = int((y_train == 0).sum())
        spw = float(neg) / float(pos) if pos > 0 else 1.0
        base = XGBClassifier(
            random_state=random_state, n_jobs=-1, verbosity=0,
            eval_metric='logloss', scale_pos_weight=spw
        )
    else:
        raise ValueError(f"Unknown model: {model_type}")

    gs = GridSearchCV(
        estimator=base,
        param_grid=PARAM_GRIDS[model_type],
        scoring='average_precision',
        cv=n_cv, n_jobs=-1, verbose=0
    )
    gs.fit(X_train, y_train)
    print(f"   Best params: {gs.best_params_}")
    print(f"   CV PR-AUC: {gs.best_score_:.4f}")
    return gs.best_estimator_


def compute_shap_values(model, X_test: pd.DataFrame, model_type: str):
    """Compute SHAP values, compatible with both old and new SHAP APIs."""
    try:
        explainer = shap_lib.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        # Handle different return formats:
        # - list [class_0_array, class_1_array] (older shap + RF)
        # - 3D array (n_samples, n_features, n_classes) (newer shap + RF)
        # - 2D array (n_samples, n_features) (LightGBM/XGBoost)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
    except AttributeError:
        explainer = shap_lib.Explainer(model, X_test)
        explanation = explainer(X_test)
        shap_values = explanation.values
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

    return shap_values, explainer


# =======================================================
# PLOTTING FUNCTIONS
# =======================================================

def plot_beeswarm(shap_values, X_test, title: str, output_path: Path):
    """SHAP beeswarm (summary) plot — shows direction + magnitude."""
    plt.figure(figsize=(10, 8))
    try:
        # New API (shap >= 0.43): create Explanation object
        explanation = shap_lib.Explanation(
            values=shap_values,
            data=X_test.values,
            feature_names=list(X_test.columns)
        )
        shap_lib.plots.beeswarm(explanation, show=False, max_display=20)
    except (AttributeError, TypeError):
        # Legacy API
        shap_lib.summary_plot(
            shap_values, X_test,
            plot_type="dot", show=False, max_display=20
        )
    plt.title(title, fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {output_path.name}")


def plot_bar_importance(shap_values, X_test, title: str, output_path: Path):
    """SHAP bar plot — mean absolute SHAP value per feature."""
    plt.figure(figsize=(10, 8))
    try:
        explanation = shap_lib.Explanation(
            values=shap_values,
            data=X_test.values,
            feature_names=list(X_test.columns)
        )
        shap_lib.plots.bar(explanation, show=False, max_display=20)
    except (AttributeError, TypeError):
        shap_lib.summary_plot(
            shap_values, X_test,
            plot_type="bar", show=False, max_display=20
        )
    plt.title(title, fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {output_path.name}")


def plot_dependence(
    shap_values, X_test, feature: str,
    title: str, output_path: Path,
    interaction_feature: str = "auto"
):
    """SHAP dependence plot for a single feature."""
    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        # Try legacy API first — it's more reliable for dependence plots
        shap_lib.dependence_plot(
            feature, shap_values, X_test,
            interaction_index=interaction_feature,
            show=False, ax=ax
        )
    except (AttributeError, TypeError):
        # Manual fallback: scatter feature value vs SHAP value
        feat_idx = list(X_test.columns).index(feature)
        ax.scatter(
            X_test[feature].values, shap_values[:, feat_idx],
            alpha=0.3, s=10, c='steelblue'
        )
        ax.set_xlabel(feature)
        ax.set_ylabel(f'SHAP value for {feature}')
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {output_path.name}")


def plot_feature_distributions(
    X_test: pd.DataFrame, y_test: pd.Series,
    features: List[str], positive_label: str,
    output_path: Path
):
    """Violin plots comparing feature distributions for positive vs negative class."""
    n_features = len(features)
    n_cols = 2
    n_rows = (n_features + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = axes.flatten() if n_features > 2 else [axes] if n_features == 1 else axes.flatten()

    for i, feat in enumerate(features):
        ax = axes[i]
        pos_vals = X_test.loc[y_test == 1, feat]
        neg_vals = X_test.loc[y_test == 0, feat]

        parts = ax.violinplot(
            [neg_vals.values, pos_vals.values],
            positions=[0, 1], showmeans=True, showmedians=True
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f'Not {positive_label}', positive_label])
        ax.set_ylabel(feat)
        ax.set_title(feat, fontsize=12)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f'Feature Distributions: {positive_label} vs Not', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {output_path.name}")


# =======================================================
# MAIN
# =======================================================

def run_single_split(
    split_idx: int,
    features_df: pd.DataFrame,
    splits_df: pd.DataFrame,
    task_cfg: dict,
    model_type: str,
    output_dir: Path,
    task_label: str,
    random_state: int = 42,
    n_cv: int = 5,
    generate_beeswarm: bool = False,
) -> dict:
    """Run SHAP analysis for a single split. Returns dict with SHAP values and metadata."""

    hg_features = task_cfg["hypergraph_features"]
    pw_features = task_cfg["pairwise_features"]

    split_dir = output_dir / f"split_{split_idx}"
    split_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n   --- Split {split_idx} ---")

    X_train_hg, y_train, X_test_hg, y_test = get_train_test(
        features_df, splits_df, split_idx, hg_features
    )
    X_train_pw, _, X_test_pw, _ = get_train_test(
        features_df, splits_df, split_idx, pw_features
    )

    print(f"   Train: {len(X_train_hg)} samples ({100*y_train.mean():.1f}% positive)")
    print(f"   Test:  {len(X_test_hg)} samples ({100*y_test.mean():.1f}% positive)")

    # Train hypergraph model
    print(f"   Training hypergraph {model_type}...")
    hg_model = train_model(X_train_hg, y_train, model_type, random_state, n_cv)
    hg_prauc = average_precision_score(y_test, hg_model.predict_proba(X_test_hg)[:, 1])
    print(f"   Hypergraph test PR-AUC: {hg_prauc:.4f}")

    # Train pairwise model
    print(f"   Training pairwise {model_type}...")
    pw_model = train_model(X_train_pw, y_train, model_type, random_state, n_cv)
    pw_prauc = average_precision_score(y_test, pw_model.predict_proba(X_test_pw)[:, 1])
    print(f"   Pairwise test PR-AUC: {pw_prauc:.4f}")

    # Compute SHAP values
    print(f"   Computing SHAP values...")
    hg_shap, _ = compute_shap_values(hg_model, X_test_hg, model_type)
    pw_shap, _ = compute_shap_values(pw_model, X_test_pw, model_type)

    # Beeswarm plots — only for selected splits (best/worst)
    if generate_beeswarm:
        plot_beeswarm(
            hg_shap, X_test_hg,
            f"Hypergraph SHAP — {task_label} ({model_type}) — Split {split_idx}",
            split_dir / "shap_beeswarm_hypergraph.png"
        )
        plot_beeswarm(
            pw_shap, X_test_pw,
            f"Pairwise SHAP — {task_label} ({model_type}) — Split {split_idx}",
            split_dir / "shap_beeswarm_pairwise.png"
        )

    # Save per-split SHAP values
    hg_shap_df = pd.DataFrame(hg_shap, columns=X_test_hg.columns)
    hg_shap_df['true_label'] = y_test.values
    hg_shap_df.to_csv(split_dir / "shap_values_hypergraph.csv", index=False)

    pw_shap_df = pd.DataFrame(pw_shap, columns=X_test_pw.columns)
    pw_shap_df['true_label'] = y_test.values
    pw_shap_df.to_csv(split_dir / "shap_values_pairwise.csv", index=False)

    return {
        "split_index": split_idx,
        "hg_prauc": hg_prauc,
        "pw_prauc": pw_prauc,
        "hg_shap": hg_shap,
        "pw_shap": pw_shap,
        "hg_features": list(X_test_hg.columns),
        "pw_features": list(X_test_pw.columns),
        "hg_mean_abs_shap": np.abs(hg_shap).mean(axis=0),
        "pw_mean_abs_shap": np.abs(pw_shap).mean(axis=0),
        "X_test_hg": X_test_hg,
        "X_test_pw": X_test_pw,
    }


def main():
    task_name = ANALYSIS["TASK"]
    model_type = ANALYSIS["MODEL_TYPE"]
    data_dir = ANALYSIS["DATA_DIR"]
    task_cfg = TASK_CONFIG[task_name]
    n_top_splits = ANALYSIS.get("N_TOP_SPLITS", 3)

    # Output directory
    output_dir = ANALYSIS["OUTPUT_DIR"] / f"{task_name}_{model_type.lower()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  SHAP ANALYSIS (Top {n_top_splits} splits)")
    print(f"  Task  : {task_name}")
    print(f"  Model : {model_type}")
    print(f"  Output: {output_dir}")
    print(f"{'='*70}\n")

    # --- Load data ---
    print("1. Loading data...")
    features_df, splits_df = load_data(task_cfg, data_dir)
    print(f"   Features shape: {features_df.shape}")
    print(f"   Splits entries: {len(splits_df)}")

    # --- Select splits ---
    print(f"\n2. Selecting top {n_top_splits} splits...")
    if ANALYSIS["SPLIT_INDEX"] is not None:
        # Single split override — run just that one
        split_indices = [ANALYSIS["SPLIT_INDEX"]]
        print(f"   Using specified split: {split_indices[0]}")
    elif ANALYSIS["RESULTS_CSV"] is not None and ANALYSIS["RESULTS_CSV"].exists():
        split_indices = find_top_n_splits(ANALYSIS["RESULTS_CSV"], n=n_top_splits)
    else:
        split_indices = sorted(splits_df['split_index'].unique())[:n_top_splits]
        print(f"   No results CSV found — using first {n_top_splits} splits: {split_indices}")

    task_label = task_name.replace("_", " ").title()

    # --- Run each split ---
    print(f"\n3. Running SHAP analysis across {len(split_indices)} splits...")
    all_results = []
    for split_idx in split_indices:
        result = run_single_split(
            split_idx, features_df, splits_df, task_cfg,
            model_type, output_dir, task_label,
            ANALYSIS["RANDOM_STATE"], ANALYSIS["N_SPLITS_CV"],
            generate_beeswarm=False,  # We'll generate for best/worst after
        )
        all_results.append(result)

    # --- Generate beeswarm plots for best and worst performing splits ---
    best_result = max(all_results, key=lambda r: r["hg_prauc"])
    worst_result = min(all_results, key=lambda r: r["hg_prauc"])

    for label, r in [("best", best_result), ("worst", worst_result)]:
        split_dir = output_dir / f"split_{r['split_index']}"
        split_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n   Generating beeswarm for {label} split {r['split_index']} "
              f"(PR-AUC = {r['hg_prauc']:.4f})...")
        plot_beeswarm(
            r["hg_shap"], r["X_test_hg"],
            f"Hypergraph SHAP — {task_label} ({model_type}) — "
            f"Split {r['split_index']} ({label}, PR-AUC={r['hg_prauc']:.3f})",
            split_dir / "shap_beeswarm_hypergraph.png"
        )
        plot_beeswarm(
            r["pw_shap"], r["X_test_pw"],
            f"Pairwise SHAP — {task_label} ({model_type}) — "
            f"Split {r['split_index']} ({label}, PR-AUC={r['pw_prauc']:.3f})",
            split_dir / "shap_beeswarm_pairwise.png"
        )

    # --- Aggregate SHAP importance across splits ---
    print(f"\n4. Aggregating SHAP importance across {len(all_results)} splits...")

    # Hypergraph aggregation
    hg_feature_names = all_results[0]["hg_features"]
    hg_importance_per_split = np.array([r["hg_mean_abs_shap"] for r in all_results])
    hg_mean = hg_importance_per_split.mean(axis=0)
    hg_std = hg_importance_per_split.std(axis=0)

    hg_agg_df = pd.DataFrame({
        "feature": hg_feature_names,
        "mean_abs_shap": hg_mean,
        "std_abs_shap": hg_std,
    })
    for i, r in enumerate(all_results):
        hg_agg_df[f"split_{r['split_index']}"] = r["hg_mean_abs_shap"]
    hg_agg_df = hg_agg_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    hg_agg_df["rank"] = range(1, len(hg_agg_df) + 1)
    hg_agg_df.to_csv(output_dir / "aggregated_shap_hypergraph.csv", index=False)

    # Pairwise aggregation
    pw_feature_names = all_results[0]["pw_features"]
    pw_importance_per_split = np.array([r["pw_mean_abs_shap"] for r in all_results])
    pw_mean = pw_importance_per_split.mean(axis=0)
    pw_std = pw_importance_per_split.std(axis=0)

    pw_agg_df = pd.DataFrame({
        "feature": pw_feature_names,
        "mean_abs_shap": pw_mean,
        "std_abs_shap": pw_std,
    })
    for i, r in enumerate(all_results):
        pw_agg_df[f"split_{r['split_index']}"] = r["pw_mean_abs_shap"]
    pw_agg_df = pw_agg_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    pw_agg_df["rank"] = range(1, len(pw_agg_df) + 1)
    pw_agg_df.to_csv(output_dir / "aggregated_shap_pairwise.csv", index=False)

    # --- Print aggregated summary ---
    print(f"\n   Aggregated Hypergraph SHAP Importance (mean ± std across {len(all_results)} splits):")
    print(f"   {'Rank':<6} {'Feature':<35} {'Mean |SHAP|':<14} {'Std':>8}")
    print(f"   {'-'*65}")
    for _, row in hg_agg_df.iterrows():
        print(f"   {int(row['rank']):<6} {row['feature']:<35} {row['mean_abs_shap']:<14.4f} {row['std_abs_shap']:>8.4f}")

    print(f"\n   Aggregated Pairwise SHAP Importance (mean ± std across {len(all_results)} splits):")
    print(f"   {'Rank':<6} {'Feature':<35} {'Mean |SHAP|':<14} {'Std':>8}")
    print(f"   {'-'*65}")
    for _, row in pw_agg_df.iterrows():
        print(f"   {int(row['rank']):<6} {row['feature']:<35} {row['mean_abs_shap']:<14.4f} {row['std_abs_shap']:>8.4f}")

    # --- Check ranking stability ---
    print(f"\n   Ranking stability check (top 5 hypergraph features per split):")
    for r in all_results:
        top5_idx = np.argsort(r["hg_mean_abs_shap"])[::-1][:5]
        top5_names = [r["hg_features"][i] for i in top5_idx]
        print(f"   Split {r['split_index']}: {top5_names}")

    print(f"\n   Ranking stability check (top 3 pairwise features per split):")
    for r in all_results:
        top3_idx = np.argsort(r["pw_mean_abs_shap"])[::-1][:3]
        top3_names = [r["pw_features"][i] for i in top3_idx]
        print(f"   Split {r['split_index']}: {top3_names}")

    # --- Aggregated bar plot with error bars ---
    print(f"\n5. Generating aggregated plots...")

    # SHAP-style red colour
    SHAP_RED = '#FF0051'

    # Hypergraph aggregated bar plot
    fig, ax = plt.subplots(figsize=(10, 8))
    top_n = min(15, len(hg_agg_df))
    top = hg_agg_df.head(top_n)
    ax.barh(range(len(top)), top['mean_abs_shap'], xerr=top['std_abs_shap'],
            color=SHAP_RED, edgecolor='none', capsize=3)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top['feature'])
    ax.invert_yaxis()
    ax.set_xlabel(f'Mean |SHAP value| (± std across {len(all_results)} splits)')
    ax.set_title(f'Hypergraph SHAP Importance — {task_label} ({model_type})\n'
                 f'Aggregated across {len(all_results)} splits')
    ax.axvline(0, color='gray', linestyle='--', linewidth=1)
    # Annotate bars with mean values
    for i, (_, row) in enumerate(top.iterrows()):
        ax.text(row['mean_abs_shap'] + row['std_abs_shap'] + 0.001, i,
                f"+{row['mean_abs_shap']:.2f}", va='center', fontsize=11, color=SHAP_RED)
    plt.tight_layout()
    plt.savefig(output_dir / "aggregated_shap_bar_hypergraph.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: aggregated_shap_bar_hypergraph.png")

    # Pairwise aggregated bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    top_pw = pw_agg_df.head(len(pw_agg_df))
    ax.barh(range(len(top_pw)), top_pw['mean_abs_shap'], xerr=top_pw['std_abs_shap'],
            color=SHAP_RED, edgecolor='none', capsize=3)
    ax.set_yticks(range(len(top_pw)))
    ax.set_yticklabels(top_pw['feature'])
    ax.invert_yaxis()
    ax.set_xlabel(f'Mean |SHAP value| (± std across {len(all_results)} splits)')
    ax.set_title(f'Pairwise SHAP Importance — {task_label} ({model_type})\n'
                 f'Aggregated across {len(all_results)} splits')
    ax.axvline(0, color='gray', linestyle='--', linewidth=1)
    for i, (_, row) in enumerate(top_pw.iterrows()):
        ax.text(row['mean_abs_shap'] + row['std_abs_shap'] + 0.001, i,
                f"+{row['mean_abs_shap']:.2f}", va='center', fontsize=11, color=SHAP_RED)
    plt.tight_layout()
    plt.savefig(output_dir / "aggregated_shap_bar_pairwise.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: aggregated_shap_bar_pairwise.png")

    # --- Summary CSV ---
    summary_rows = []
    for r in all_results:
        summary_rows.append({
            "split_index": r["split_index"],
            "hypergraph_pr_auc": r["hg_prauc"],
            "pairwise_pr_auc": r["pw_prauc"],
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "multi_split_summary.csv", index=False)

    print(f"\n{'='*70}")
    print(f"  COMPLETE — outputs in {output_dir}")
    print(f"  Analysed {len(all_results)} splits: {[r['split_index'] for r in all_results]}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()