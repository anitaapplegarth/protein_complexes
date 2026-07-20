import os
import argparse
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
    "BASE_OUTPUT_DIR": Path("./randomforest/cp_ess_second_testA"),

    # --- TEST A: annotation-presence control -------------------------------
    # A fourth tier, HYPER_FLAG = HYPERGRAPH + a single binary feature marking
    # whether the protein has ANY curated stoichiometry. This decomposes the
    # apparent stoichiometry effect into two parts:
    #
    #     hb_graph - hypergraph   =  [hyper_flag - hypergraph]   (annotation presence)
    #                             +  [hb_graph  - hyper_flag ]   (stoichiometry VALUES)
    #
    # The flag is recovered exactly from the existing features: a true
    # stoich_MedianRatio lies in (0, 1], so a value of 0 is out of range and
    # uniquely marks "this protein has no curated stoichiometry anywhere".
    # NOTE: the fillna(0) encoding is deliberately left UNCHANGED — this test
    # measures what the current encoding is buying, so it must not be fixed here.
    "ANNOTATION_FLAG":     "flag_HasStoich",
    "FLAG_SOURCE_FEATURE": "stoich_MedianRatio",
    # Optional sanity check against the raw incidence file. Set to None to skip.
    "RAW_STOICH_FILE": Path("/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/stoich_protein.csv"),

    # --- Split sets (n = 50 splits each) ------------------------------------
    # Two arms share this pipeline and differ only in the splits file:
    #   'strat'   — splits stratified by complex-size bucket (main analysis)
    #   'unstrat' — stratify_by_bucket = False (bucket ablation)
    # Unstratified runs are written to BASE_OUTPUT_DIR + '_unstrat' so the two
    # arms never overwrite one another. Override either name with --splits-file.
    "SPLIT_SETS": {
        "strat":   "ess_protein_merged_splits.csv",
        "unstrat": "ess_protein_merged_splits_unstrat.csv",
    },
    # --- Run controls (used when no command-line arguments are given, i.e.
    #     when you just hit Run in VS Code) --------------------------------
    # These two are the ONLY knobs you need to touch for a normal run.
    #
    # SPLIT_SET     : 'strat'   -> main analysis    -> ./<output>
    #                 'unstrat' -> bucket ablation  -> ./<output>_unstrat
    # LIMIT_SPLITS  : None      -> full 50-split run
    #                 3         -> smoke test on the first 3 splits, written to
    #                              ./<output>_smoke3 so it cannot overwrite
    #                              real results.
    "SPLIT_SET":    "strat",
    "LIMIT_SPLITS": None,
    # Sanity check only — the pipeline reads however many splits the file holds.
    "EXPECTED_N_SPLITS": 50,

    # --- File Names ---
    # SPLITS_FILE is set at runtime from SPLIT_SETS / --splits-file.
    "SPLITS_FILE":           None,
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
            # 'protein_MedianUniqueRatio',   # DROPPED: near-exact reciprocal of
            #                                # protein_MedComplexNodes (Spearman -0.9965).
            #                                # MedComplexNodes retained.
            'protein_RangeUniqueRatio',
            'protein_MedComplexNodes',
            'protein_RangeComplexNodes',
            # 'protein_NormUniqueSum'
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

# NOTE: the splits file is no longer probed at import time — it is resolved in
# main() once the split set (strat / unstrat) is known. See resolve_split_set().


def resolve_split_set(split_set: str, splits_file_override: str | None = None) -> Path:
    """Sets CONFIG['SPLITS_FILE'] and returns the full path to it."""
    if splits_file_override:
        CONFIG["SPLITS_FILE"] = splits_file_override
    else:
        if split_set not in CONFIG["SPLIT_SETS"]:
            raise ValueError(
                f"Unknown split set '{split_set}'. "
                f"Options: {list(CONFIG['SPLIT_SETS'])}"
            )
        CONFIG["SPLITS_FILE"] = CONFIG["SPLIT_SETS"][split_set]

    path = CONFIG["DATA_DIR"] / CONFIG["SPLITS_FILE"]
    if not path.exists():
        raise FileNotFoundError(
            f"Splits file not found: {path}\n"
            f"   Check CONFIG['SPLIT_SETS'] or pass --splits-file <name.csv>."
        )
    return path

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

    combined = derive_annotation_flag(combined)
    return combined


def derive_annotation_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds CONFIG['ANNOTATION_FLAG'] (binary): 1 if the protein has at least one
    complex with curated stoichiometry, 0 otherwise.

    Derived from stoich_MedianRatio. In cp_hypergraph_features.ipynb the ratio is
    computed only over complexes where the protein's own Stoichiometry != 0, and
    falls back to 0 when that list is empty. Since a genuine ratio is in (0, 1],
    a value of exactly 0 is unambiguous: no curated stoichiometry anywhere.
    """
    flag = CONFIG["ANNOTATION_FLAG"]
    src  = CONFIG["FLAG_SOURCE_FEATURE"]

    if src not in df.columns:
        raise KeyError(f"Cannot derive {flag}: '{src}' not in feature file.")

    df[flag] = (df[src] > 0).astype(int)

    n_annot = int(df[flag].sum())
    print(f"\n   Derived '{flag}' from '{src}':")
    print(f"     curated   : {n_annot} / {len(df)}  ({100*n_annot/len(df):.1f}%)")
    print(f"     uncurated : {len(df)-n_annot} / {len(df)}  ({100*(1-n_annot/len(df)):.1f}%)")

    # --- Optional cross-check against the raw incidence file ---
    raw_path = CONFIG.get("RAW_STOICH_FILE")
    if raw_path and Path(raw_path).exists():
        raw = pd.read_csv(raw_path)
        truth = (raw.assign(a=(raw['Stoichiometry'] > 0).astype(int))
                    .groupby('ProteinId')['a'].max())
        chk = df[['ProteinId', flag]].merge(
            truth.rename('truth'), left_on='ProteinId', right_index=True, how='inner')
        n_mismatch = int((chk[flag] != chk['truth']).sum())
        if n_mismatch == 0:
            print(f"     cross-check vs {Path(raw_path).name}: exact match "
                  f"on all {len(chk)} proteins ✓")
        else:
            print(f"     WARNING: cross-check disagrees on {n_mismatch}/{len(chk)} proteins.")
    else:
        print(f"     (raw-file cross-check skipped — RAW_STOICH_FILE not found)")

    return df


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
    splits_path = CONFIG["DATA_DIR"] / CONFIG["SPLITS_FILE"]
    print(f"   File              : {splits_path.name}")
    print(f"   Last modified     : "
          f"{pd.Timestamp(os.path.getmtime(splits_path), unit='s')}")
    splits_df = pd.read_csv(splits_path)

    # Rename to match feature file key
    splits_df = splits_df.rename(columns={'UniProt_AC': 'ProteinId'})

    # Encode binary target: Essential=1, Non-essential=0; Unknown kept as NaN
    label_map = {'Essential': 1, 'Non-essential': 0}
    splits_df['target'] = splits_df['protein_label'].map(label_map)

    n_splits = splits_df['split_index'].nunique()
    expected = CONFIG.get("EXPECTED_N_SPLITS")
    if expected is not None and n_splits != expected:
        print(f"   WARNING: found {n_splits} splits, expected {expected}. "
              f"Check that the splits file is the regenerated one.")
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
    hb_graph_features: List[str],
    hypergraph_features: List[str],
    pairwise_features: List[str],
    hyper_flag_features: List[str]
) -> Dict:
    """
    Runs FOUR tiers for a single pre-assigned split:
      pairwise   — dyadic PPI features
      hypergraph — set-based higher-order features (no stoichiometry)
      hyper_flag — hypergraph + the binary annotation-presence flag  [TEST A]
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
        'train_ess_pct': 100 * y_train.mean(),
        'test_ess_pct':  100 * y_test.mean(),
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

    # --- Hyper+flag model (hypergraph + annotation-presence flag) [TEST A] ---
    X_hf_train = train_df[hyper_flag_features]
    X_hf_test  = test_df[hyper_flag_features]

    hf_model, hf_params = tune_and_train_model(X_hf_train, y_train)
    hf_eval = evaluate_model(hf_model, X_hf_test, y_test)

    results['hyper_flag_pr_auc']      = hf_eval['pr_auc']
    results['hyper_flag_f1']          = hf_eval['f1']
    results['hyper_flag_best_params'] = hf_params
    results['hyper_flag_importance']  = compute_permutation_importance(
        hf_model, X_hf_test, y_test
    )

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

    # --- TEST A decomposition of the stoichiometry effect ---
    # annotation presence alone:
    results['annot_pr_auc_diff'] = results['hyper_flag_pr_auc'] - results['hypergraph_pr_auc']
    # stoichiometry VALUES, net of annotation presence:
    results['value_pr_auc_diff'] = results['hb_graph_pr_auc']   - results['hyper_flag_pr_auc']

    return results

# =======================================================
# STATISTICAL COMPARISON
# =======================================================

def run_sign_test_comparison(all_results: List[Dict]) -> Dict:
    """One-sided sign test (binomial on wins/losses) on the paired PR-AUC
    differences across splits: does the better representation win on
    significantly more than half of the splits? Cohen's dz is reported
    alongside as a descriptive effect size (it is not a test).
    Covers three paired comparisons:
      1. HB-graph vs Pairwise                        — headline representation effect
      2. HB-graph vs Hypergraph  — stoichiometry effect (adding multiset stoichiometry)
      3. Hypergraph vs Pairwise  — set-based representation effect alone
    """
    pair_vals  = np.array([r['pairwise_pr_auc']   for r in all_results])
    hyper_vals = np.array([r['hypergraph_pr_auc'] for r in all_results])
    hf_vals    = np.array([r['hyper_flag_pr_auc'] for r in all_results])
    hbg_vals   = np.array([r['hb_graph_pr_auc']   for r in all_results])

    # F1 values (positive class) per representation
    pair_f1  = np.array([r['pairwise_f1']   for r in all_results])
    hyper_f1 = np.array([r['hypergraph_f1'] for r in all_results])
    hf_f1    = np.array([r['hyper_flag_f1'] for r in all_results])
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

        # --- Paired standardised effect size (Cohen's dz) ---
        sd = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
        dz = float(np.mean(diffs) / sd) if sd > 0 else 0.0

        return dict(wins=n_wins, losses=n_loss, ties=n_ties,
                    mean_diff=float(np.mean(diffs)),
                    std_diff=sd,   # sample SD (ddof=1), consistent with Cohen's dz
                    p_greater=p_greater, p_two_sided=p_two_sided,
                    cohens_dz=dz)

    hbg_vs_pair   = _sign_test(hbg_vals,   pair_vals)   # headline
    stoich_effect = _sign_test(hbg_vals,   hyper_vals)  # hb_graph vs hypergraph
    hyper_vs_pair = _sign_test(hyper_vals, pair_vals)   # representation effect alone

    # --- TEST A: split the stoichiometry effect into its two parts ---
    annot_effect = _sign_test(hf_vals,  hyper_vals)  # what annotation PRESENCE buys
    value_effect = _sign_test(hbg_vals, hf_vals)     # what stoichiometry VALUES buy

    return {
        'n_runs': len(all_results),
        'hyper_flag_pr_auc_mean': float(np.mean(hf_vals)),
        'hyper_flag_pr_auc_std':  float(np.std(hf_vals)),
        'hyper_flag_f1_mean':     float(np.mean(hf_f1)),
        'hyper_flag_f1_std':      float(np.std(hf_f1)),
        'annot_effect': annot_effect,
        'value_effect': value_effect,
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
        'cohens_dz':             hbg_vs_pair['cohens_dz'],
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

    # --- PR-AUC (ordered pairwise -> hypergraph -> hb_graph) ---
    print(f"\n  PR-AUC")
    print(f"  {'Representation':<20} {'Mean ± Std'}")
    print(f"  {'-'*45}")
    print(f"  {'Pairwise':<20} "
          f"{stats['pairwise_pr_auc_mean']:.4f} ± {stats['pairwise_pr_auc_std']:.4f}")
    print(f"  {'Hypergraph':<20} "
          f"{stats['hypergraph_pr_auc_mean']:.4f} ± {stats['hypergraph_pr_auc_std']:.4f}")
    print(f"  {'Hypergraph + flag':<20} "
          f"{stats['hyper_flag_pr_auc_mean']:.4f} ± {stats['hyper_flag_pr_auc_std']:.4f}")
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
    print(f"  {'Hypergraph + flag':<20} "
          f"{stats['hyper_flag_f1_mean']:.4f} ± {stats['hyper_flag_f1_std']:.4f}")
    print(f"  {'HB-graph':<20} "
          f"{stats['hb_graph_f1_mean']:.4f} ± {stats['hb_graph_f1_std']:.4f}")

    def _print_comparison(label, d):
        print(f"\n  --- {label} ---")
        print(f"  Mean diff : {d['mean_diff']:+.4f} ± {d['std_diff']:.4f}"
              f"   (Cohen's dz = {d.get('cohens_dz', float('nan')):+.3f})")
        print(f"  Wins/Losses/Ties : {d['wins']}/{d['losses']}/{d['ties']}")
        print(f"  Sign test p (one-sided) : {d['p_greater']:.6f}   "
              f"(two-sided: {d['p_two_sided']:.6f})")

    _print_comparison("HB-graph vs Pairwise — headline representation effect",
                      {'mean_diff': stats['mean_difference'],
                       'std_diff':  stats['std_difference'],
                       'wins':      stats['hb_graph_wins'],
                       'losses':    stats['pairwise_wins'],
                       'ties':      stats['ties'],
                       'p_greater': stats['sign_test_p_greater'],
                       'p_two_sided': stats['sign_test_p_two_sided'],
                       'cohens_dz':   stats['cohens_dz']})
    _print_comparison("HB-graph vs Hypergraph — stoichiometry effect",
                      stats['stoich_effect'])
    _print_comparison("Hypergraph vs Pairwise — representation effect alone",
                      stats['hyper_vs_pair'])

    # ---------------- TEST A: decomposition ----------------
    ann = stats['annot_effect']
    val = stats['value_effect']
    tot = stats['stoich_effect']

    _print_comparison("[TEST A] Hyper+flag vs Hypergraph — ANNOTATION PRESENCE alone",
                      ann)
    _print_comparison("[TEST A] HB-graph vs Hyper+flag — STOICHIOMETRY VALUES, "
                      "net of annotation", val)

    print(f"\n{'='*70}")
    print("  TEST A — DECOMPOSITION OF THE STOICHIOMETRY EFFECT")
    print(f"{'='*70}")
    print(f"\n  {'Component':<44} {'ΔPR-AUC':>9} {'W/L':>7} {'p (1-sided)':>12}")
    print(f"  {'-'*78}")
    print(f"  {'Annotation presence   (flag − hypergraph)':<44} "
          f"{ann['mean_diff']:>+9.4f} {ann['wins']:>3}/{ann['losses']:<3} "
          f"{ann['p_greater']:>12.4f}")
    print(f"  {'Stoichiometry values  (hb-graph − flag)':<44} "
          f"{val['mean_diff']:>+9.4f} {val['wins']:>3}/{val['losses']:<3} "
          f"{val['p_greater']:>12.4f}")
    print(f"  {'-'*78}")
    print(f"  {'TOTAL  (hb-graph − hypergraph)':<44} "
          f"{tot['mean_diff']:>+9.4f} {tot['wins']:>3}/{tot['losses']:<3} "
          f"{tot['p_greater']:>12.4f}")

    total = tot['mean_diff']
    if abs(total) > 1e-9:
        share = 100 * ann['mean_diff'] / total
        print(f"\n  Annotation presence accounts for {share:.0f}% of the total "
              f"stoichiometry effect.")

    print("\n  VERDICT:")
    if val['p_greater'] < 0.05:
        print("    Stoichiometry VALUES add signal beyond annotation presence.")
        print("    Finding 2 survives — but report it net of the flag, and confirm with Test B")
        print("    (re-run restricted to curated proteins only).")
    else:
        print("    Stoichiometry VALUES add NO significant signal beyond annotation presence.")
        print("    The apparent stoichiometry effect is substantially an ascertainment artefact.")
        print("    Finding 2 cannot be claimed as stated.")
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

def plot_paired_comparison(all_results: List[Dict], stats: Dict, output_dir: Path):
    """Two-panel comparison plot: paired scatter (headline contrast) and 3-way boxplot.

    Axes are fixed to [0, 1] so panels are directly comparable across tasks/files.
    """
    pair_vals  = np.array([r['pairwise_pr_auc']   for r in all_results])
    hyper_vals = np.array([r['hypergraph_pr_auc'] for r in all_results])
    hbg_vals   = np.array([r['hb_graph_pr_auc']   for r in all_results])

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # Panel 1: paired scatter — headline contrast (HB-graph vs Pairwise), one point per split
    ax1 = axes[0]
    ax1.scatter(pair_vals, hbg_vals, alpha=0.7, s=60, zorder=3)
    ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, label='y = x')
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

    ab  = stats['stoich_effect']
    p_one = ab['p_greater']
    p_two = ab['p_two_sided']

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # ── Panel 1: scatter — hb-graph vs hypergraph ────────────────────────────
    ax1 = axes[0]
    ax1.scatter(hyper_vals, hbg_vals, alpha=0.7, s=60, zorder=3,
                color='steelblue')
    ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, label='y = x (no difference)')
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


def plot_testA_decomposition(all_results: List[Dict], stats: Dict, output_dir: Path):
    """
    TEST A figure.

    Panel 1 — Boxplot of all four tiers. The gap between 'Hypergraph + flag' and
              'HB-graph' is the only part of the stoichiometry effect that the
              stoichiometry VALUES can claim.
    Panel 2 — Paired scatter, hb-graph vs hypergraph+flag, one point per split.
              Points on the diagonal = the values add nothing beyond the flag.
    """
    pair_vals  = np.array([r['pairwise_pr_auc']   for r in all_results])
    hyper_vals = np.array([r['hypergraph_pr_auc'] for r in all_results])
    hf_vals    = np.array([r['hyper_flag_pr_auc'] for r in all_results])
    hbg_vals   = np.array([r['hb_graph_pr_auc']   for r in all_results])

    ann, val = stats['annot_effect'], stats['value_effect']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    box_data = [pair_vals, hyper_vals, hf_vals, hbg_vals]
    labels   = ['Pairwise', 'Hypergraph', 'Hypergraph\n+ flag', 'HB-graph']
    colours  = ['lightgray', 'skyblue', 'gold', 'steelblue']
    bp = ax.boxplot(box_data, labels=labels, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2))
    for patch, c in zip(bp['boxes'], colours):
        patch.set_facecolor(c)
    rng = np.random.default_rng(0)
    for i, d in enumerate(box_data):
        ax.scatter(rng.normal(i + 1, 0.04, size=len(d)), d,
                   alpha=0.4, s=20, color='black', zorder=3)
    for i, v in enumerate(box_data):
        ax.text(i + 1, 0.02, f'{v.mean():.3f}', ha='center', va='bottom', fontsize=12)
    ax.set_ylabel('PR-AUC')
    ax.set_ylim(0, 1)
    ax.set_title('Test A — four nested tiers')

    ax = axes[1]
    ax.scatter(hf_vals, hbg_vals, alpha=0.75, s=60, color='steelblue', zorder=3)
    lo = min(hf_vals.min(), hbg_vals.min()) - 0.05
    hi = max(hf_vals.max(), hbg_vals.max()) + 0.05
    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=2, label='y = x (values add nothing)')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal')
    ax.set_xlabel('Hypergraph + flag PR-AUC')
    ax.set_ylabel('HB-graph PR-AUC')
    ax.set_title('Stoichiometry values, net of annotation')
    ax.legend(fontsize=12, loc='upper left')
    ax.text(0.97, 0.03,
            f"Annotation:  {ann['mean_diff']:+.4f}  ({ann['wins']}/{ann['losses']}, "
            f"p={ann['p_greater']:.4f})\n"
            f"Values:      {val['mean_diff']:+.4f}  ({val['wins']}/{val['losses']}, "
            f"p={val['p_greater']:.4f})",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=11,
            family='monospace',
            bbox=dict(facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()
    plt.savefig(output_dir / 'testA_decomposition.png', dpi=300)
    plt.close()
    print("   Saved: testA_decomposition.png")


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

    # --- Command-line arguments -------------------------------------------
    parser = argparse.ArgumentParser(
        description="Test A pipeline (pairwise / hypergraph / hyper+flag / hb-graph)."
    )
    parser.add_argument(
        "--split-set", choices=list(CONFIG["SPLIT_SETS"]),
        default=CONFIG["SPLIT_SET"],
        help="Which n=50 split set to run: 'strat' (main) or 'unstrat' "
             "(bucket ablation). Unstratified output goes to <output_dir>_unstrat."
    )
    parser.add_argument(
        "--splits-file", default=None,
        help="Override the splits file name (relative to DATA_DIR)."
    )
    parser.add_argument(
        "--model", choices=["RandomForest", "LightGBM", "XGBoost"],
        default=CONFIG["MODEL_TYPE"], help="Model type."
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Override the output directory entirely."
    )
    parser.add_argument(
        "--limit", type=int, default=CONFIG["LIMIT_SPLITS"],
        help="Run only the first N splits (smoke test). Defaults to "
             "CONFIG['LIMIT_SPLITS'], so a smoke test can be set in the file "
             "and run straight from VS Code with no arguments."
    )
    # parse_known_args (not parse_args) so that stray arguments injected by the
    # VS Code interactive window / debugger do not kill the run.
    args, _unknown = parser.parse_known_args()

    CONFIG["MODEL_TYPE"] = args.model
    splits_path = resolve_split_set(args.split_set, args.splits_file)

    # --- Output directory ---
    # The bucket-ablation arm writes to a parallel '_unstrat' directory so the
    # two arms can never overwrite each other.
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        suffix = "" if args.split_set == "strat" else f"_{args.split_set}"
        if args.limit:
            suffix += f"_smoke{args.limit}"
        output_dir = Path(str(CONFIG["BASE_OUTPUT_DIR"]) + suffix)
    output_dir.mkdir(parents=True, exist_ok=True)
    CONFIG["OUTPUT_DIR"] = output_dir

    print(f"\n{'='*70}")
    print(f"  REPRESENTATION COMPARISON: PAIRWISE vs HYPERGRAPH vs HB-GRAPH")
    print(f"  Task   : Gene Essentiality")
    print(f"  Model  : {CONFIG['MODEL_TYPE']}")
    print(f"  Splits : pre-assigned family-level  [{args.split_set}]")
    print(f"           {splits_path}")
    print(f"  Output : {output_dir}")
    print(f"{'='*70}\n")

    # --- Load data ---
    features_df = load_all_features()
    splits_df   = load_splits()

    split_indices = sorted(splits_df['split_index'].unique())
    if args.limit:
        split_indices = split_indices[:args.limit]
        print(f"\n   *** SMOKE TEST: {args.limit} of "
              f"{splits_df['split_index'].nunique()} splits only. ***")
        print(f"   *** Results are NOT publishable and are written to "
              f"{output_dir}. ***")
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

    # TEST A tier: hypergraph + the single annotation-presence flag
    flag = CONFIG["ANNOTATION_FLAG"]
    hyper_flag_features = hypergraph_features + [flag]

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
    print(f"   Active hypergraph + flag features ({len(hyper_flag_features)}):  [TEST A]")
    for f in hyper_flag_features:
        tag = " [annotation flag]" if f == flag else ""
        print(f"     - {f}{tag}")
    print(f"   Active hb-graph features ({len(hb_graph_features)}):")
    for f in hb_graph_features:
        tag = " [stoich]" if f in stoich_features else ""
        print(f"     - {f}{tag}")

    # --- Fill any NaNs in feature columns ---
    # NB: encoding deliberately UNCHANGED from the original script. Test A measures
    # what the current fillna(0) encoding is buying; fixing it here would defeat that.
    all_feature_cols = hb_graph_features + pairwise_features
    n_nans = features_df[all_feature_cols].isna().sum().sum()
    if n_nans > 0:
        print(f"   Filling {n_nans} missing feature values with 0.")
        features_df[all_feature_cols] = features_df[all_feature_cols].fillna(0)

    # --- Main loop over splits ---
    print(f"\n3. Running paired comparisons across {len(split_indices)} splits...\n")
    all_results = []
    failed_splits = []

    # Per-split checkpoint: a 50-split run is long, so results are appended to
    # disk as they complete rather than only at the very end.
    checkpoint_path = output_dir / 'split_results_checkpoint.csv'
    checkpoint_cols = ['split_index', 'n_train', 'n_test',
                       'train_ess_pct', 'test_ess_pct',
                       'pairwise_pr_auc', 'hypergraph_pr_auc',
                       'hyper_flag_pr_auc', 'hb_graph_pr_auc']
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    for i, split_idx in enumerate(split_indices, start=1):
        print(f"   Split {split_idx:>3} ({i:>2}/{len(split_indices)})...",
              end=" ", flush=True)
        try:
            result = run_split(
                split_idx, features_df, splits_df,
                hb_graph_features, hypergraph_features, pairwise_features,
                hyper_flag_features
            )
            all_results.append(result)
            pd.DataFrame([{k: result[k] for k in checkpoint_cols}]).to_csv(
                checkpoint_path, mode='a', index=False,
                header=not checkpoint_path.exists()
            )
            winner = ("HB-graph" if result['pr_auc_diff'] > 0
                      else "Pair" if result['pr_auc_diff'] < 0 else "Tie")
            print(f"train={result['n_train']} ({result['train_ess_pct']:.1f}% ess)  "
                  f"test={result['n_test']} ({result['test_ess_pct']:.1f}% ess)  |  "
                  f"Pair: {result['pairwise_pr_auc']:.4f}, "
                  f"Hyper: {result['hypergraph_pr_auc']:.4f}, "
                  f"Hyper+flag: {result['hyper_flag_pr_auc']:.4f}, "
                  f"HB-graph: {result['hb_graph_pr_auc']:.4f}  |  "
                  f"annot: {result['annot_pr_auc_diff']:+.4f}, "
                  f"values: {result['value_pr_auc_diff']:+.4f} [{winner}]")
        except Exception as e:
            failed_splits.append(split_idx)
            print(f"ERROR: {e}")

    # --- Fail loudly rather than silently analysing a partial run ---
    if failed_splits:
        print(f"\n   WARNING: {len(failed_splits)} split(s) FAILED and are excluded "
              f"from all statistics: {failed_splits}")
    if not all_results:
        raise RuntimeError("No splits completed successfully — nothing to analyse.")
    print(f"\n   Completed {len(all_results)}/{len(split_indices)} splits.")

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
    hf_imp_df    = aggregate_feature_importance(all_results, 'hyper_flag')
    hbg_imp_df   = aggregate_feature_importance(all_results, 'hb_graph')
    print_feature_importance_summary(
        [("Pairwise", pair_imp_df),
         ("Hypergraph", hyper_imp_df),
         ("Hypergraph + flag", hf_imp_df),
         ("HB-graph", hbg_imp_df)],
        top_n=10
    )

    # Where does the annotation flag rank on its own?
    if not hf_imp_df.empty and flag in set(hf_imp_df['feature']):
        row = hf_imp_df[hf_imp_df['feature'] == flag].iloc[0]
        print(f"\n   >>> '{flag}' ranks {int(row['rank'])} of {len(hf_imp_df)} "
              f"in the hypergraph+flag model (mean importance {row['mean']:.5f})")
    plot_feature_importance(
        [("Pairwise", pair_imp_df, 'gray'),
         ("Hypergraph", hyper_imp_df, 'skyblue'),
         ("HB-graph", hbg_imp_df, 'steelblue')],
        output_dir, top_n=15
    )

    # --- Save CSVs ---
    print("\n7. Saving outputs...")

    # Per-split summary (no nested dicts), ordered pairwise -> hypergraph -> hb_graph
    plot_testA_decomposition(all_results, stats, output_dir)

    summary_cols = ['split_index', 'n_train', 'n_test', 'train_ess_pct', 'test_ess_pct',
                    'pairwise_pr_auc',   'pairwise_f1',
                    'hypergraph_pr_auc', 'hypergraph_f1',
                    'hyper_flag_pr_auc', 'hyper_flag_f1',
                    'hb_graph_pr_auc',   'hb_graph_f1',
                    'pr_auc_diff', 'f1_diff',
                    'stoich_pr_auc_diff', 'stoich_f1_diff',
                    'annot_pr_auc_diff', 'value_pr_auc_diff']
    summary_df = pd.DataFrame([{k: r[k] for k in summary_cols} for r in all_results])
    summary_df.to_csv(output_dir / 'split_results.csv', index=False)
    print("   Saved: split_results.csv")

    # Tidy one-row-per-comparison stats table (feeds the LaTeX results tables)
    comparison_map = {
        'hb_graph_vs_pairwise': {
            'mean_diff':  stats['mean_difference'],
            'std_diff':   stats['std_difference'],
            'wins':       stats['hb_graph_wins'],
            'losses':     stats['pairwise_wins'],
            'ties':       stats['ties'],
            'p_greater':  stats['sign_test_p_greater'],
            'p_two_sided': stats['sign_test_p_two_sided'],
            'cohens_dz':  stats['cohens_dz'],
        },
        'hb_graph_vs_hypergraph':  stats['stoich_effect'],
        'hypergraph_vs_pairwise':  stats['hyper_vs_pair'],
        'hyper_flag_vs_hypergraph': stats['annot_effect'],
        'hb_graph_vs_hyper_flag':   stats['value_effect'],
    }
    stats_rows = []
    for name, d in comparison_map.items():
        row = {'comparison': name, 'split_set': args.split_set,
               'n_splits': stats['n_runs'], 'model': CONFIG['MODEL_TYPE']}
        row.update({k: d[k] for k in
                    ['mean_diff', 'std_diff', 'cohens_dz', 'wins', 'losses', 'ties',
                     'p_greater', 'p_two_sided']})
        stats_rows.append(row)
    pd.DataFrame(stats_rows).to_csv(output_dir / 'comparison_stats.csv', index=False)
    print("   Saved: comparison_stats.csv")

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

    hf_imp_df.to_csv(output_dir / 'hyper_flag_feature_importance.csv', index=False)
    print("   Saved: hyper_flag_feature_importance.csv")
    print("   Saved: hb_graph_feature_importance.csv")

    with open(output_dir / 'statistical_summary.txt', 'w') as f:
            f.write("REPRESENTATION COMPARISON: PAIRWISE vs HYPERGRAPH vs HYPER+FLAG vs HB-GRAPH\n")
            f.write("Task: Gene Essentiality\n")
            if args.limit:
                f.write("\n*** SMOKE TEST — PARTIAL RUN, NOT PUBLISHABLE ***\n")
                f.write(f"*** Only the first {args.limit} splits were run. ***\n\n")

            f.write("\nRUN PROVENANCE\n")
            f.write(f"{'-'*70}\n")
            f.write(f"Run at              : "
                    f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")
            f.write(f"Model               : {CONFIG['MODEL_TYPE']}\n")
            f.write(f"Random state        : {CONFIG['RANDOM_STATE']}\n")
            f.write(f"Inner CV folds      : {CONFIG['N_SPLITS_CV']}\n")
            f.write(f"Split set           : {args.split_set}\n")
            f.write(f"Splits file         : {splits_path}\n")
            f.write(f"Splits file mtime   : "
                    f"{pd.Timestamp(os.path.getmtime(splits_path), unit='s')}\n")
            f.write(f"Data directory      : {CONFIG['DATA_DIR']}\n")
            f.write(f"Feature files       : {CONFIG['PROTEIN_FEATURES_FILE']}, "
                    f"{CONFIG['PAIRWISE_FEATURES_FILE']}\n")
            f.write(f"Splits in file      : "
                    f"{splits_df['split_index'].nunique()}\n")
            f.write(f"Splits attempted    : {len(split_indices)}\n")
            f.write(f"Splits completed    : {stats['n_runs']}\n")
            if failed_splits:
                f.write(f"FAILED splits (excluded from all statistics): "
                        f"{failed_splits}\n")
            f.write(f"Annotation flag     : {CONFIG['ANNOTATION_FLAG']} "
                    f"(derived from {CONFIG['FLAG_SOURCE_FEATURE']})\n")
            f.write(f"Hyperparameter grid : "
                    f"{CONFIG['PARAM_GRIDS'][CONFIG['MODEL_TYPE']]}\n")
            f.write("\n")

            f.write("CLASS BALANCE (mean over splits)\n")
            f.write(f"{'-'*70}\n")
            f.write(f"Train positives     : {summary_df['train_ess_pct'].mean():.2f}% "
                    f"(range {summary_df['train_ess_pct'].min():.2f}"
                    f"-{summary_df['train_ess_pct'].max():.2f}%)\n")
            f.write(f"Test  positives     : {summary_df['test_ess_pct'].mean():.2f}% "
                    f"(range {summary_df['test_ess_pct'].min():.2f}"
                    f"-{summary_df['test_ess_pct'].max():.2f}%)\n")
            f.write(f"Mean train / test n : {summary_df['n_train'].mean():.0f} / "
                    f"{summary_df['n_test'].mean():.0f}\n\n")
            f.write(f"Pairwise features ({len(pairwise_features)}):\n")
            for feat in pairwise_features:
                f.write(f"  - {feat}\n")
            f.write(f"\nHypergraph features ({len(hypergraph_features)}):\n")
            for feat in hypergraph_features:
                f.write(f"  - {feat}\n")
            f.write(f"\nHB-graph features ({len(hb_graph_features)}):\n")
            for feat in hb_graph_features:
                tag = ' [stoich]' if feat in stoich_features else ''
                f.write(f"  - {feat}{tag}\n")

            f.write(f"\nPR-AUC Mean \u00b1 Std:\n")
            f.write(f"  Pairwise   : {stats['pairwise_pr_auc_mean']:.4f} \u00b1 {stats['pairwise_pr_auc_std']:.4f}\n")
            f.write(f"  Hypergraph : {stats['hypergraph_pr_auc_mean']:.4f} \u00b1 {stats['hypergraph_pr_auc_std']:.4f}\n")
            f.write(f"  Hyper+flag : {stats['hyper_flag_pr_auc_mean']:.4f} \u00b1 {stats['hyper_flag_pr_auc_std']:.4f}\n")
            f.write(f"  HB-graph   : {stats['hb_graph_pr_auc_mean']:.4f} \u00b1 {stats['hb_graph_pr_auc_std']:.4f}\n")

            f.write(f"\nF1 (positive class) Mean \u00b1 Std:\n")
            f.write(f"  Pairwise   : {stats['pairwise_f1_mean']:.4f} \u00b1 {stats['pairwise_f1_std']:.4f}\n")
            f.write(f"  Hypergraph : {stats['hypergraph_f1_mean']:.4f} \u00b1 {stats['hypergraph_f1_std']:.4f}\n")
            f.write(f"  Hyper+flag : {stats['hyper_flag_f1_mean']:.4f} \u00b1 {stats['hyper_flag_f1_std']:.4f}\n")
            f.write(f"  HB-graph   : {stats['hb_graph_f1_mean']:.4f} \u00b1 {stats['hb_graph_f1_std']:.4f}\n")

            def _write_comparison(label, d):
                f.write(f"\n{label}:\n")
                f.write(f"  Mean diff : {d['mean_diff']:+.4f} \u00b1 {d['std_diff']:.4f}\n")
                f.write(f"  Cohen's dz : {d.get('cohens_dz', float('nan')):+.3f}\n")
                f.write(f"  Wins/Losses/Ties : {d['wins']}/{d['losses']}/{d['ties']}\n")
                f.write(f"  Sign test p (one-sided) : {d['p_greater']:.6f}\n")
                f.write(f"  Sign test p (two-sided) : {d['p_two_sided']:.6f}\n")

            _write_comparison("HB-graph vs Pairwise \u2014 headline representation effect",
                              {'mean_diff': stats['mean_difference'],
                               'std_diff':  stats['std_difference'],
                               'wins':      stats['hb_graph_wins'],
                               'losses':    stats['pairwise_wins'],
                               'ties':      stats['ties'],
                               'p_greater': stats['sign_test_p_greater'],
                               'p_two_sided': stats['sign_test_p_two_sided'],
                               'cohens_dz':   stats['cohens_dz']})
            _write_comparison("HB-graph vs Hypergraph \u2014 stoichiometry effect",
                              stats['stoich_effect'])
            _write_comparison("Hypergraph vs Pairwise \u2014 representation effect alone",
                              stats['hyper_vs_pair'])

            # ---------------- TEST A: decomposition ----------------
            ann, val, tot = stats['annot_effect'], stats['value_effect'], stats['stoich_effect']
            _write_comparison("[TEST A] Hyper+flag vs Hypergraph \u2014 ANNOTATION PRESENCE alone",
                              ann)
            _write_comparison("[TEST A] HB-graph vs Hyper+flag \u2014 STOICHIOMETRY VALUES, "
                              "net of annotation", val)

            f.write(f"\n{'='*70}\n")
            f.write("TEST A \u2014 DECOMPOSITION OF THE STOICHIOMETRY EFFECT\n")
            f.write(f"{'='*70}\n\n")
            f.write(f"{'Component':<44} {'dPR-AUC':>9} {'W/L':>7} {'p (1-sided)':>12}\n")
            f.write(f"{'-'*78}\n")
            f.write(f"{'Annotation presence   (flag - hypergraph)':<44} "
                    f"{ann['mean_diff']:>+9.4f} {ann['wins']:>3}/{ann['losses']:<3} "
                    f"{ann['p_greater']:>12.4f}\n")
            f.write(f"{'Stoichiometry values  (hb-graph - flag)':<44} "
                    f"{val['mean_diff']:>+9.4f} {val['wins']:>3}/{val['losses']:<3} "
                    f"{val['p_greater']:>12.4f}\n")
            f.write(f"{'-'*78}\n")
            f.write(f"{'TOTAL  (hb-graph - hypergraph)':<44} "
                    f"{tot['mean_diff']:>+9.4f} {tot['wins']:>3}/{tot['losses']:<3} "
                    f"{tot['p_greater']:>12.4f}\n")

            if abs(tot['mean_diff']) > 1e-9:
                share = 100 * ann['mean_diff'] / tot['mean_diff']
                f.write(f"\nAnnotation presence accounts for {share:.0f}% of the total "
                        f"stoichiometry effect.\n")

            f.write("\nVERDICT:\n")
            if val['p_greater'] < 0.05:
                f.write("  Stoichiometry VALUES add signal beyond annotation presence.\n")
                f.write("  Finding 2 survives \u2014 report net of the flag; confirm with Test B.\n")
            else:
                f.write("  Stoichiometry VALUES add NO significant signal beyond annotation "
                        "presence.\n")
                f.write("  The apparent stoichiometry effect is substantially an ascertainment "
                        "artefact.\n")
            f.write(f"{'='*70}\n")

            # ---------------- FEATURE IMPORTANCE (all four tiers) -------------
            f.write("\n\n")
            f.write(f"{'='*70}\n")
            f.write("FEATURE IMPORTANCE (permutation \u2014 mean PR-AUC drop)\n")
            f.write(f"{'='*70}\n")
            for label, imp_df in [("Pairwise", pair_imp_df),
                                  ("Hypergraph", hyper_imp_df),
                                  ("Hypergraph + flag [TEST A]", hf_imp_df),
                                  ("HB-graph", hbg_imp_df)]:
                if imp_df.empty:
                    continue
                f.write(f"\n{label}\n")
                f.write(f"{'Rank':<6} {'Feature':<36} {'Mean':>10} {'Std':>10} "
                        f"{'Median':>10}\n")
                f.write(f"{'-'*74}\n")
                for _, row in imp_df.iterrows():
                    f.write(f"{int(row['rank']):<6} {row['feature']:<36} "
                            f"{row['mean']:>10.5f} {row['std']:>10.5f} "
                            f"{row['median']:>10.5f}\n")
            if not hf_imp_df.empty and flag in set(hf_imp_df['feature']):
                _row = hf_imp_df[hf_imp_df['feature'] == flag].iloc[0]
                f.write(f"\n'{flag}' ranks {int(_row['rank'])} of {len(hf_imp_df)} "
                        f"in the hypergraph+flag model "
                        f"(mean importance {_row['mean']:.5f}).\n")
            f.write("\nNote: higher = more important; negative = likely noise.\n")

            # ---------------- PER-SPLIT RESULTS -------------------------------
            f.write("\n\n")
            f.write(f"{'='*70}\n")
            f.write("PER-SPLIT PR-AUC (also in split_results.csv)\n")
            f.write(f"{'='*70}\n\n")
            f.write(f"{'Split':>6} {'Pairwise':>10} {'Hyper':>10} {'Hyper+flag':>11} "
                    f"{'HB-graph':>10} {'HBG-Pair':>10} {'HBG-Hyper':>10}\n")
            f.write(f"{'-'*72}\n")
            for _, row in summary_df.sort_values('split_index').iterrows():
                f.write(f"{int(row['split_index']):>6} "
                        f"{row['pairwise_pr_auc']:>10.4f} "
                        f"{row['hypergraph_pr_auc']:>10.4f} "
                        f"{row['hyper_flag_pr_auc']:>11.4f} "
                        f"{row['hb_graph_pr_auc']:>10.4f} "
                        f"{row['pr_auc_diff']:>+10.4f} "
                        f"{row['stoich_pr_auc_diff']:>+10.4f}\n")

            f.write(f"\n\nRuntime to this point: "
                    f"{(time.time() - start_time)/60:.1f} min\n")

    print(f"\n{'='*70}")
    print("  COMPLETE")
    print(f"{'='*70}")

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")