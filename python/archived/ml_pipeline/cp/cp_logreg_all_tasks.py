"""
=======================================================================
LOGISTIC REGRESSION BASELINE — REPRESENTATION COMPARISON (Complex Portal)
=======================================================================

Runs the same four-tier representation comparison as the tree-ensemble
scripts (cp_ess_second.py / cp_hpa_second.py / cp_chembl_second.py), but
with an L2-penalised logistic regression, across ALL THREE tasks in one
run:

    gene essentiality  |  HPA drug targets  |  ChEMBL drug targets

Tiers (identical to the tree scripts, nested pairwise ⊂ hypergraph ⊂ hb_graph):
    pairwise   — dyadic PPI features
    hypergraph — set-based higher-order features (no stoichiometry)
    hyper_flag — hypergraph + binary annotation-presence flag   [TEST A]
    hyper_recip— hypergraph + 1/(1+complex size)                [TEST C]
    hb_graph   — hypergraph + stoichiometry (multiset hyperedges)

TEST C — the reciprocal-of-size control
    Among curated CP proteins stoich_MedianRatio correlates with
    protein_MedComplexNodes at Spearman -0.918 (CORUM -0.659), and
    stoich_WeightedTriangles at +0.942. The mechanism is arithmetic: for a
    complex of n roughly equimolar subunits, one protein's share of total
    stoichiometry is about 1/n, so stoich_MedianRatio ~= 1/MedComplexNodes.

    A reciprocal is monotone but strongly NONLINEAR. L2 logistic regression
    on x cannot represent 1/x, so supplying stoich_MedianRatio hands it a
    basis function it structurally lacked; a random forest already
    approximates 1/x by splitting repeatedly on x. That is exactly the
    observed pattern (large LR stoichiometry gains, RF null, one feature
    carrying the whole effect), so the two must be told apart.

    The hyper_recip tier adds recip_MedComplexNodes = 1/(1 + MedComplexNodes),
    a deterministic transform of a feature the hypergraph tier ALREADY has
    and which contains NO stoichiometry information. The contrast that
    matters is then hb_graph - hyper_recip: if it collapses to zero, the
    apparent stoichiometric gain was a size recoding.

WHAT DIFFERS FROM THE TREE SCRIPTS
    1. The estimator is a Pipeline: StandardScaler -> LogisticRegression.
       Scaling is MANDATORY here (unlike for trees) because the features span
       orders of magnitude — base_Degree is an unbounded count, stoich_MedianRatio
       is bounded in (0, 1] — and an L2 penalty applied to unscaled features
       penalises them by an arbitrary amount that depends only on their units.
       The scaler sits INSIDE the pipeline, so it is refitted on each inner CV
       training fold and never sees the test split.
    2. The hyperparameter grid is over C (inverse regularisation strength) only.
    3. Feature importance is permutation importance (PR-AUC drop), exactly as
       in the tree scripts, so the numbers are directly comparable. Coefficients
       are deliberately NOT reported: with correlated features they are not a
       reliable importance ranking.

EVERYTHING ELSE IS HELD FIXED so the comparison is like-for-like:
    same splits files, same frozen feature sets, same fillna(0) encoding,
    same average_precision scorer, same 5-fold inner CV, same PR-AUC / F1
    metrics, same one-sided sign tests, same output file names.

USAGE
    Set DATA_DIR and BASE_OUTPUT_DIR below, then:  python lr_cp_all_tasks.py
    Outputs land in BASE_OUTPUT_DIR/<task_key>/ , one directory per task,
    plus a cross-task summary at BASE_OUTPUT_DIR/all_tasks_summary.csv.
"""

import os
import time
import warnings
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, average_precision_score
from sklearn.inspection import permutation_importance
from sklearn.exceptions import ConvergenceWarning
from scipy.stats import binomtest, spearmanr

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
    # --- Database identity (used in printouts and output filenames) ---
    "DATABASE": "CP",

    # --- Paths ---
    "DATA_DIR": Path("/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/cp/"),
    "BASE_OUTPUT_DIR": Path("./logistic/cp_second_testA"),

    "PROTEIN_FEATURES_FILE":  "hypergraph_features.csv",
    "PAIRWISE_FEATURES_FILE": "pairwise_features.csv",

    # Optional sanity check of the annotation flag against the raw incidence file.
    # Must sit in the SAME subdirectory as DATA_DIR. Set to None to skip.
    "RAW_STOICH_FILE": Path("/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/cp/stoich_protein.csv"),

    # --- TASKS ---------------------------------------------------------------
    # All three tasks run in a single invocation. Each writes to its own
    # subdirectory of BASE_OUTPUT_DIR. To run only a subset, comment one out.
    #
    # 'label_map' must match the protein_label strings in that task's splits
    # file; anything not in the map is treated as Unknown (NaN) and excluded
    # by label_mask, exactly as in the tree scripts.
    "TASKS": {
        "ess": {
            "name":        "Gene Essentiality",
            "splits_file": "ess_protein_splits.csv",
            "label_map":   {'Essential': 1, 'Non-essential': 0},
            "pos_name":    "Essential",
            "neg_name":    "Non-Essential",
        },
        "hpa": {
            "name":        "Drug Target Prediction (HPA)",
            "splits_file": "hpa_protein_splits.csv",
            "label_map":   {'Drug_target': 1, 'Non_target': 0},
            "pos_name":    "Drug_target",
            "neg_name":    "Non_target",
        },
        "chembl": {
            "name":        "Drug Target Prediction (ChEMBL)",
            "splits_file": "chembl_protein_splits.csv",
            "label_map":   {'Drug_target': 1, 'Non_target': 0},
            "pos_name":    "Drug_target",
            "neg_name":    "Non_target",
        },
    },

    # --- TEST A: annotation-presence control ---------------------------------
    # A fourth tier, HYPER_FLAG = HYPERGRAPH + a single binary feature marking
    # whether the protein has ANY curated stoichiometry. This decomposes the
    # apparent stoichiometry effect into two parts:
    #
    #     hb_graph - hypergraph  =  [hyper_flag - hypergraph]  (annotation presence)
    #                            +  [hb_graph   - hyper_flag]  (stoichiometry VALUES)
    #
    # The flag is recovered exactly from the existing features: a true
    # stoich_MedianRatio lies in (0, 1], so a value of 0 uniquely marks
    # "this protein has no curated stoichiometry anywhere".
    #
    # SET INCLUDE_HYPER_FLAG = False FOR CORUM if the flag is near-constant
    # there (CORUM stoichiometry coverage is ~4% vs CP's ~60%). A near-degenerate
    # binary predictor is harmless to a tree but can produce a large, unstable
    # coefficient under regularised logistic regression. The script also
    # auto-skips the tier per split if the flag has <2 levels in the training
    # fold, or if either level has fewer than MIN_FLAG_COUNT proteins.
    "INCLUDE_HYPER_FLAG":  True,
    "MIN_FLAG_COUNT":      10,
    "ANNOTATION_FLAG":     "flag_HasStoich",
    "FLAG_SOURCE_FEATURE": "stoich_MedianRatio",

    # --- TEST C: reciprocal-of-size control (see header) ---------------------
    # recip = 1/(1 + protein_MedComplexNodes). 1/(1+n) rather than 1/n because
    # MedComplexNodes can be 0 for proteins whose only complexes are
    # homo-oligomers; the +1 keeps the transform finite and monotone throughout.
    "INCLUDE_HYPER_RECIP": True,
    "RECIP_FEATURE":       "recip_MedComplexNodes",
    "RECIP_SOURCE_FEATURE": "protein_MedComplexNodes",

    # --- Fixed settings (matched to the tree scripts) ---
    "RANDOM_STATE": 42,
    "N_SPLITS_CV":  5,
    "PERM_REPEATS": 10,

    # --- Logistic regression ---
    # lbfgs with the default L2 penalty is the standard, fast choice and
    # converges reliably on standardised features. class_weight='balanced'
    # matches the tree scripts. max_iter is set high because the smallest C
    # values converge slowly.
    # NB: 'penalty' is deliberately NOT passed. L2 is the default in every
    # sklearn version, and passing penalty='l2' explicitly raises a
    # FutureWarning from sklearn 1.8 onwards.
    "LOGREG_KWARGS": {
        "solver":       "lbfgs",
        "class_weight": "balanced",
        "max_iter":     5000,
    },
    "PARAM_GRID": {
        "clf__C": list(np.logspace(-6, 4, 11))
    },

    # --- Feature Selection (FROZEN — must match the tree scripts exactly) ---
    "FEATURES": {
        "HB_GRAPH": [
            # --- Base / native higher-order metrics ---
            'base_Degree',
            'base_LocalClustCoeff',
            'base_TriangleCount',
            'base_UniquePartners',
            'base_AvgNeighbourDegree',

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
            'protein_RangeUniqueRatio',
            'protein_MedComplexNodes',
            'protein_RangeComplexNodes',
        ],
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
            'pair_AvgNeighborDegree',
        ]
    }
}

# Tier order used everywhere (printouts, CSV columns, plots).
TIERS = ['pairwise', 'hypergraph', 'hyper_flag', 'hyper_recip', 'hb_graph']
TIER_LABELS = {
    'pairwise':   'Pairwise',
    'hypergraph': 'Hypergraph',
    'hyper_flag': 'Hypergraph + flag',
    'hyper_recip': 'Hypergraph + 1/size',
    'hb_graph':   'HB-graph',
}


# =======================================================
# DATA LOADING
# =======================================================

def load_all_features() -> pd.DataFrame:
    """Loads higher-order (hb-graph) and pairwise feature CSVs, merges on ProteinId."""
    print("1. Loading feature data...")

    hg_df   = pd.read_csv(CONFIG["DATA_DIR"] / CONFIG["PROTEIN_FEATURES_FILE"])
    pair_df = pd.read_csv(CONFIG["DATA_DIR"] / CONFIG["PAIRWISE_FEATURES_FILE"])

    combined = pd.merge(hg_df, pair_df, on='ProteinId', how='inner')

    print(f"   Higher-order (hb-graph) features shape : {hg_df.shape}")
    print(f"   Pairwise features shape               : {pair_df.shape}")
    print(f"   Combined shape                        : {combined.shape}")

    combined = derive_annotation_flag(combined)
    combined = derive_reciprocal(combined)
    return combined


def derive_reciprocal(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the TEST C control feature: 1 / (1 + complex size).

    Contains no stoichiometry information — it is a deterministic transform
    of a feature the hypergraph tier already holds. Its purpose is to give a
    linear model the 1/n basis function that stoich_MedianRatio would
    otherwise smuggle in.
    """
    if not CONFIG["INCLUDE_HYPER_RECIP"]:
        return df

    recip = CONFIG["RECIP_FEATURE"]
    src   = CONFIG["RECIP_SOURCE_FEATURE"]
    if src not in df.columns:
        raise KeyError(f"Cannot derive {recip}: '{src}' not in feature file.")

    df[recip] = 1.0 / (1.0 + df[src])

    print(f"\n   Derived '{recip}' = 1 / (1 + {src})  [TEST C control]")

    # Report how closely it tracks the stoichiometry features. The comparison
    # that matters is among CURATED proteins only: across all proteins the
    # correlation is dominated by the shared zero/non-zero structure.
    flag = CONFIG["ANNOTATION_FLAG"]
    cur = df[df[flag] == 1] if flag in df.columns else df
    print(f"     Spearman vs stoichiometry features "
          f"(curated proteins only, n = {len(cur):,}):")
    for f_ in CONFIG["FEATURES"]["STOICHIOMETRY_FEATURES"]:
        if f_ in cur.columns and len(cur) > 10 and cur[f_].nunique() > 1:
            rho, _ = spearmanr(cur[recip], cur[f_])
            mark = "   <-- near-deterministic" if abs(rho) > 0.85 else ""
            print(f"       {f_:<34} {rho:+.3f}{mark}")

    return df


def derive_annotation_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds CONFIG['ANNOTATION_FLAG'] (binary): 1 if the protein has at least one
    complex with curated stoichiometry, 0 otherwise. Derived from
    stoich_MedianRatio, where a genuine ratio lies in (0, 1] and a value of
    exactly 0 unambiguously means "no curated stoichiometry anywhere".
    """
    flag = CONFIG["ANNOTATION_FLAG"]
    src  = CONFIG["FLAG_SOURCE_FEATURE"]

    if src not in df.columns:
        raise KeyError(f"Cannot derive {flag}: '{src}' not in feature file.")

    df[flag] = (df[src] > 0).astype(int)

    n_annot = int(df[flag].sum())
    print(f"\n   Derived '{flag}' from '{src}':")
    print(f"     curated   : {n_annot} / {len(df)}  ({100*n_annot/len(df):.1f}%)")
    print(f"     uncurated : {len(df)-n_annot} / {len(df)}  "
          f"({100*(1-n_annot/len(df)):.1f}%)")

    # Loud warning if the flag is near-degenerate — relevant for CORUM.
    frac = n_annot / len(df) if len(df) else 0.0
    if CONFIG["INCLUDE_HYPER_FLAG"] and (frac < 0.05 or frac > 0.95):
        print(f"     WARNING: '{flag}' is near-constant ({100*frac:.1f}% curated).")
        print(f"              The hyper_flag tier will be unstable under a "
              f"regularised linear model.")
        print(f"              Consider setting INCLUDE_HYPER_FLAG = False.")

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
            print(f"     WARNING: cross-check disagrees on "
                  f"{n_mismatch}/{len(chk)} proteins.")
    else:
        print(f"     (raw-file cross-check skipped — RAW_STOICH_FILE not found)")

    return df


def load_splits(task_key: str) -> pd.DataFrame:
    """
    Loads the pre-assigned group-atomic splits file for one task.

    Expected columns:
        split_index   — integer identifying which split
        UniProt_AC    — protein identifier (matches ProteinId in feature files)
        split         — 'train' or 'test'
        protein_label — task-specific label strings (see CONFIG['TASKS'])
        label_mask    — bool; False for Unknown proteins (excluded from metrics)
    """
    task = CONFIG["TASKS"][task_key]
    splits_path = CONFIG["DATA_DIR"] / task["splits_file"]

    print("2. Loading pre-assigned splits...")
    print(f"   File              : {splits_path.name}")
    print(f"   Last modified     : "
          f"{pd.Timestamp(os.path.getmtime(splits_path), unit='s')}")

    splits_df = pd.read_csv(splits_path)
    splits_df = splits_df.rename(columns={'UniProt_AC': 'ProteinId'})

    unmapped = set(splits_df['protein_label'].unique()) - set(task["label_map"]) 
    splits_df['target'] = splits_df['protein_label'].map(task["label_map"])

    print(f"   Splits file rows  : {len(splits_df)}")
    print(f"   Unique proteins   : {splits_df['ProteinId'].nunique()}")
    print(f"   Number of splits  : {splits_df['split_index'].nunique()}")
    if unmapped:
        print(f"   Unmapped labels (treated as Unknown): {sorted(unmapped)}")

    labelled = splits_df[splits_df['label_mask']].drop_duplicates('ProteinId')
    n_pos, n_tot = int((labelled['target'] == 1).sum()), len(labelled)
    print(f"   Labelled proteins : {n_tot}  ({100*n_pos/n_tot:.1f}% positive)")

    # Guard: a label_mask=True row with a NaN target means the label_map is out
    # of date relative to the splits file. Fail loudly rather than silently
    # dropping proteins from the analysis.
    n_bad = int(splits_df.loc[splits_df['label_mask'], 'target'].isna().sum())
    if n_bad:
        raise ValueError(
            f"{n_bad} rows have label_mask=True but an unmapped protein_label. "
            f"Check CONFIG['TASKS']['{task_key}']['label_map'] against "
            f"{splits_path.name}."
        )

    return splits_df


# =======================================================
# MODEL TRAINING & EVALUATION
# =======================================================

def build_pipeline() -> Pipeline:
    """StandardScaler -> LogisticRegression, so scaling is refitted per CV fold."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            random_state=CONFIG["RANDOM_STATE"], **CONFIG["LOGREG_KWARGS"]
        )),
    ])


def tune_and_train_model(X_train: pd.DataFrame, y_train: pd.Series):
    """Grid search over C + fit. Returns (best_estimator, best_params)."""
    gs = GridSearchCV(
        estimator=build_pipeline(),
        param_grid=CONFIG["PARAM_GRID"],
        scoring='average_precision',
        cv=CONFIG["N_SPLITS_CV"],
        n_jobs=-1,
        verbose=0
    )
    # Small C values on near-separable folds can hit the iteration cap; that is
    # expected and those folds simply lose the grid search on PR-AUC.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series,
                   pos_name: str, neg_name: str) -> Dict:
    """Returns PR-AUC, F1 for the positive class, and predicted probabilities."""
    y_pred       = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(
        y_test, y_pred,
        labels=[0, 1],
        target_names=[neg_name, pos_name],
        output_dict=True,
        zero_division=0
    )

    return {
        'pr_auc':       average_precision_score(y_test, y_pred_proba),
        'f1':           report[pos_name]['f1-score'],
        'y_pred_proba': y_pred_proba
    }


def compute_permutation_importance(
    model, X_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, float]:
    """Permutation importance scored by average_precision (PR-AUC drop).

    Permuting the raw column and passing it through the fitted pipeline is the
    correct thing to do: the scaler is a fixed monotone transform at this point,
    so permuting before or after scaling gives the same importance.
    """
    result = permutation_importance(
        model, X_test, y_test,
        scoring='average_precision',
        n_repeats=CONFIG["PERM_REPEATS"],
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
    feature_sets: Dict[str, List[str]],
    task: Dict
) -> Dict:
    """Runs every configured tier for a single pre-assigned split."""
    split_mask = splits_df['split_index'] == split_idx
    split_info = splits_df[split_mask][
        ['ProteinId', 'split', 'target', 'label_mask']].copy()

    df = pd.merge(merged_df, split_info, on='ProteinId', how='inner')
    labelled_df = df[df['label_mask']].copy()

    train_df = labelled_df[labelled_df['split'] == 'train']
    test_df  = labelled_df[labelled_df['split'] == 'test']

    y_train = train_df['target'].astype(int)
    y_test  = test_df['target'].astype(int)

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError(
            f"split {split_idx}: train or test fold contains a single class "
            f"(train pos={int(y_train.sum())}/{len(y_train)}, "
            f"test pos={int(y_test.sum())}/{len(y_test)})."
        )

    results = {
        'split_index':   split_idx,
        'n_train':       len(train_df),
        'n_test':        len(test_df),
        'train_pos_pct': 100 * y_train.mean(),
        'test_pos_pct':  100 * y_test.mean(),
    }

    for tier, feats in feature_sets.items():
        if not feats:
            continue

        # Guard the TEST A tier against a degenerate flag in this training fold.
        flag = CONFIG["ANNOTATION_FLAG"]
        if tier == 'hyper_flag' and flag in feats:
            counts = train_df[flag].value_counts()
            if len(counts) < 2 or counts.min() < CONFIG["MIN_FLAG_COUNT"]:
                results['hyper_flag_pr_auc'] = np.nan
                results['hyper_flag_f1']     = np.nan
                results['hyper_flag_skipped'] = True
                continue

        X_train, X_test = train_df[feats], test_df[feats]

        model, params = tune_and_train_model(X_train, y_train)
        ev = evaluate_model(model, X_test, y_test,
                            task["pos_name"], task["neg_name"])

        results[f'{tier}_pr_auc']      = ev['pr_auc']
        results[f'{tier}_f1']          = ev['f1']
        results[f'{tier}_best_params'] = params
        results[f'{tier}_importance']  = compute_permutation_importance(
            model, X_test, y_test)

        preds = test_df[['ProteinId']].copy()
        preds['split_index'] = split_idx
        preds['true_label']  = y_test.values
        preds[f'{tier}_pred_proba'] = ev['y_pred_proba']
        results[f'{tier}_predictions'] = preds

    # --- Differences ---
    # Headline representation contrast: hb_graph vs pairwise
    results['pr_auc_diff'] = results['hb_graph_pr_auc'] - results['pairwise_pr_auc']
    results['f1_diff']     = results['hb_graph_f1']     - results['pairwise_f1']
    # Stoichiometry effect: hb_graph vs hypergraph
    results['stoich_pr_auc_diff'] = (results['hb_graph_pr_auc']
                                     - results['hypergraph_pr_auc'])
    results['stoich_f1_diff']     = (results['hb_graph_f1']
                                     - results['hypergraph_f1'])
    # TEST A decomposition
    hf = results.get('hyper_flag_pr_auc', np.nan)
    results['annot_pr_auc_diff'] = hf - results['hypergraph_pr_auc']
    results['value_pr_auc_diff'] = results['hb_graph_pr_auc'] - hf
    # TEST C: stoichiometry net of a pure reciprocal-of-size recoding
    hr = results.get('hyper_recip_pr_auc', np.nan)
    results['recip_pr_auc_diff']     = hr - results['hypergraph_pr_auc']
    results['net_recip_pr_auc_diff'] = results['hb_graph_pr_auc'] - hr

    return results


# =======================================================
# STATISTICAL COMPARISON
# =======================================================

def _sign_test(a: np.ndarray, b: np.ndarray) -> Dict:
    """One-sided sign test on paired differences, with Cohen's dz alongside
    as a descriptive effect size (it is not itself a test).

    Splits where either arm is NaN (e.g. a skipped hyper_flag tier) are dropped
    pairwise, so the comparison stays properly paired.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = ~(np.isnan(a) | np.isnan(b))
    diffs = a[ok] - b[ok]

    n_wins  = int(np.sum(diffs > 0))
    n_loss  = int(np.sum(diffs < 0))
    n_ties  = int(np.sum(diffs == 0))
    n_valid = n_wins + n_loss

    if n_valid > 0:
        p_greater   = binomtest(n_wins, n_valid, 0.5, alternative='greater').pvalue
        p_two_sided = binomtest(n_wins, n_valid, 0.5, alternative='two-sided').pvalue
    else:
        p_greater = p_two_sided = 1.0

    sd = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    dz = float(np.mean(diffs) / sd) if sd > 0 else 0.0

    return dict(
        n_pairs=int(ok.sum()), wins=n_wins, losses=n_loss, ties=n_ties,
        mean_diff=float(np.mean(diffs)) if len(diffs) else float('nan'),
        std_diff=sd, p_greater=p_greater, p_two_sided=p_two_sided, cohens_dz=dz
    )


def run_sign_test_comparison(all_results: List[Dict]) -> Dict:
    """Paired PR-AUC comparisons across splits, matching the tree scripts:
      1. HB-graph  vs Pairwise    — headline representation effect
      2. HB-graph  vs Hypergraph  — stoichiometry effect
      3. Hypergraph vs Pairwise   — set-based representation effect alone
      4. Hyper+flag vs Hypergraph — annotation presence alone       [TEST A]
      5. HB-graph  vs Hyper+flag  — stoichiometry values, net of presence [TEST A]
    """
    def col(tier, metric):
        return np.array([r.get(f'{tier}_{metric}', np.nan) for r in all_results],
                        dtype=float)

    stats = {'n_runs': len(all_results)}

    for tier in TIERS:
        pr, f1 = col(tier, 'pr_auc'), col(tier, 'f1')
        stats[f'{tier}_pr_auc_mean'] = float(np.nanmean(pr)) if not np.all(np.isnan(pr)) else float('nan')
        stats[f'{tier}_pr_auc_std']  = float(np.nanstd(pr))  if not np.all(np.isnan(pr)) else float('nan')
        stats[f'{tier}_f1_mean']     = float(np.nanmean(f1)) if not np.all(np.isnan(f1)) else float('nan')
        stats[f'{tier}_f1_std']      = float(np.nanstd(f1))  if not np.all(np.isnan(f1)) else float('nan')

    stats['comparisons'] = {
        'hb_graph_vs_pairwise':     _sign_test(col('hb_graph', 'pr_auc'),
                                               col('pairwise', 'pr_auc')),
        'hb_graph_vs_hypergraph':   _sign_test(col('hb_graph', 'pr_auc'),
                                               col('hypergraph', 'pr_auc')),
        'hypergraph_vs_pairwise':   _sign_test(col('hypergraph', 'pr_auc'),
                                               col('pairwise', 'pr_auc')),
        'hyper_flag_vs_hypergraph': _sign_test(col('hyper_flag', 'pr_auc'),
                                               col('hypergraph', 'pr_auc')),
        'hb_graph_vs_hyper_flag':   _sign_test(col('hb_graph', 'pr_auc'),
                                               col('hyper_flag', 'pr_auc')),
        'hyper_recip_vs_hypergraph': _sign_test(col('hyper_recip', 'pr_auc'),
                                                col('hypergraph', 'pr_auc')),
        'hb_graph_vs_hyper_recip':  _sign_test(col('hb_graph', 'pr_auc'),
                                               col('hyper_recip', 'pr_auc')),
    }
    return stats


# =======================================================
# FEATURE IMPORTANCE AGGREGATION
# =======================================================

def aggregate_feature_importance(all_results: List[Dict],
                                 tier: str) -> pd.DataFrame:
    """Aggregates permutation importance across all splits for one tier."""
    key = f'{tier}_importance'
    records = [
        {'split_index': r['split_index'], 'feature': feat, 'importance': imp}
        for r in all_results if key in r
        for feat, imp in r[key].items()
    ]
    if not records:
        return pd.DataFrame()

    agg_df = (
        pd.DataFrame(records)
        .groupby('feature')['importance']
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

COMPARISON_LABELS = {
    'hb_graph_vs_pairwise':     "HB-graph vs Pairwise — headline representation effect",
    'hb_graph_vs_hypergraph':   "HB-graph vs Hypergraph — stoichiometry effect",
    'hypergraph_vs_pairwise':   "Hypergraph vs Pairwise — representation effect alone",
    'hyper_flag_vs_hypergraph': "[TEST A] Hyper+flag vs Hypergraph — ANNOTATION PRESENCE alone",
    'hb_graph_vs_hyper_flag':   "[TEST A] HB-graph vs Hyper+flag — STOICHIOMETRY VALUES, net of annotation",
    'hyper_recip_vs_hypergraph': "[TEST C] Hyper+1/size vs Hypergraph — what a pure size recoding buys",
    'hb_graph_vs_hyper_recip':  "[TEST C] HB-graph vs Hyper+1/size — STOICHIOMETRY net of size recoding",
}


def _format_comparison(label: str, d: Dict) -> str:
    return (
        f"\n  --- {label} ---\n"
        f"  Mean diff : {d['mean_diff']:+.4f} ± {d['std_diff']:.4f}"
        f"   (Cohen's dz = {d['cohens_dz']:+.3f})\n"
        f"  Wins/Losses/Ties : {d['wins']}/{d['losses']}/{d['ties']}"
        f"   (n paired = {d['n_pairs']})\n"
        f"  Sign test p (one-sided) : {d['p_greater']:.6f}   "
        f"(two-sided: {d['p_two_sided']:.6f})\n"
    )


def format_statistical_summary(stats: Dict, task: Dict) -> str:
    out = [f"\n{'='*70}", "  STATISTICAL COMPARISON", f"{'='*70}",
           f"\n  Task            : {task['name']}",
           f"  Model           : Logistic regression (L2, standardised)",
           f"  Number of splits: {stats['n_runs']}"]

    for metric, title in [('pr_auc', 'PR-AUC'), ('f1', 'F1 (positive class)')]:
        out += [f"\n  {title}", f"  {'Representation':<20} {'Mean ± Std'}",
                f"  {'-'*45}"]
        for tier in TIERS:
            m, s = stats[f'{tier}_{metric}_mean'], stats[f'{tier}_{metric}_std']
            val = "not run" if np.isnan(m) else f"{m:.4f} ± {s:.4f}"
            out.append(f"  {TIER_LABELS[tier]:<20} {val}")

    for name, d in stats['comparisons'].items():
        if d['n_pairs'] == 0:
            continue
        out.append(_format_comparison(COMPARISON_LABELS[name], d))

    # --- TEST A decomposition ---
    ann = stats['comparisons']['hyper_flag_vs_hypergraph']
    val = stats['comparisons']['hb_graph_vs_hyper_flag']
    tot = stats['comparisons']['hb_graph_vs_hypergraph']

    if ann['n_pairs'] > 0 and val['n_pairs'] > 0:
        out += [f"\n{'='*70}",
                "  TEST A — DECOMPOSITION OF THE STOICHIOMETRY EFFECT",
                f"{'='*70}", "",
                f"  {'Component':<44} {'dPR-AUC':>9} {'W/L':>7} {'p (1-sided)':>12}",
                f"  {'-'*78}",
                f"  {'Annotation presence   (flag - hypergraph)':<44} "
                f"{ann['mean_diff']:>+9.4f} {ann['wins']:>3}/{ann['losses']:<3} "
                f"{ann['p_greater']:>12.4f}",
                f"  {'Stoichiometry values  (hb-graph - flag)':<44} "
                f"{val['mean_diff']:>+9.4f} {val['wins']:>3}/{val['losses']:<3} "
                f"{val['p_greater']:>12.4f}",
                f"  {'-'*78}",
                f"  {'TOTAL  (hb-graph - hypergraph)':<44} "
                f"{tot['mean_diff']:>+9.4f} {tot['wins']:>3}/{tot['losses']:<3} "
                f"{tot['p_greater']:>12.4f}"]
        if abs(tot['mean_diff']) > 1e-9:
            share = 100 * ann['mean_diff'] / tot['mean_diff']
            out.append(f"\n  Annotation presence accounts for {share:.0f}% of the "
                       f"total stoichiometry effect.")
        out.append(f"{'='*70}")
    else:
        out.append("\n  TEST A decomposition skipped "
                   "(hyper_flag tier not run or degenerate).")

    # --- TEST C: is the stoichiometry gain just a reciprocal of size? ---
    rec = stats['comparisons']['hyper_recip_vs_hypergraph']
    net = stats['comparisons']['hb_graph_vs_hyper_recip']

    if rec['n_pairs'] > 0 and net['n_pairs'] > 0:
        out += [f"\n{'='*70}",
                "  TEST C — RECIPROCAL-OF-SIZE CONTROL",
                f"{'='*70}", "",
                f"  {'Component':<44} {'dPR-AUC':>9} {'W/L':>7} {'p (1-sided)':>12}",
                f"  {'-'*78}",
                f"  {'1/size alone   (recip - hypergraph)':<44} "
                f"{rec['mean_diff']:>+9.4f} {rec['wins']:>3}/{rec['losses']:<3} "
                f"{rec['p_greater']:>12.4f}",
                f"  {'Stoichiometry  (hb-graph - recip)':<44} "
                f"{net['mean_diff']:>+9.4f} {net['wins']:>3}/{net['losses']:<3} "
                f"{net['p_greater']:>12.4f}",
                f"  {'-'*78}",
                f"  {'TOTAL  (hb-graph - hypergraph)':<44} "
                f"{tot['mean_diff']:>+9.4f} {tot['wins']:>3}/{tot['losses']:<3} "
                f"{tot['p_greater']:>12.4f}"]

        if tot['mean_diff'] > 0.001:
            absorbed = 100 * (1 - net['mean_diff'] / tot['mean_diff'])
            out.append(f"\n  The reciprocal absorbs {absorbed:.0f}% of the total "
                       f"apparent stoichiometry effect.")
        else:
            out.append("\n  (Absorbed share not shown: the total effect is not "
                       "positive, so the ratio\n   is uninterpretable.)")

        if net['p_greater'] < 0.05 and net['mean_diff'] > 0:
            out += ["\n  VERDICT: SURVIVES. The stoichiometry features add signal "
                    "beyond a pure\n  reciprocal-of-size recoding, so the gain is "
                    "not an artefact of the linear\n  model lacking a 1/n basis "
                    "function."]
        else:
            out += ["\n  VERDICT: DOES NOT SURVIVE. The apparent stoichiometric "
                    "gain is accounted\n  for by 1/(complex size), which carries no "
                    "copy-number information. H2\n  cannot be supported from these "
                    "features under this model."]
        out.append(f"{'='*70}")
    else:
        out.append("\n  TEST C control skipped (hyper_recip tier not run).")

    return "\n".join(out)


def format_feature_importance(imp_dfs: List[tuple], top_n: Optional[int] = None) -> str:
    out = [f"\n{'='*70}",
           "  FEATURE IMPORTANCE (Permutation — mean PR-AUC drop)",
           f"{'='*70}"]
    for label, df in imp_dfs:
        if df.empty:
            continue
        show = df if top_n is None else df.head(top_n)
        out += [f"\n  {label}",
                f"  {'Rank':<6} {'Feature':<36} {'Mean':>10} {'Std':>10} {'Median':>10}",
                f"  {'-'*74}"]
        for _, row in show.iterrows():
            out.append(f"  {int(row['rank']):<6} {row['feature']:<36} "
                       f"{row['mean']:>10.5f} {row['std']:>10.5f} "
                       f"{row['median']:>10.5f}")
    out.append("\n  Note: higher = more important; negative = likely noise.")
    return "\n".join(out)


# =======================================================
# PLOTTING
# =======================================================

def plot_tier_comparison(all_results: List[Dict], stats: Dict,
                         output_dir: Path, task: Dict):
    """Paired scatter (headline contrast) + boxplot across all tiers."""
    pair = np.array([r['pairwise_pr_auc']   for r in all_results], dtype=float)
    hbg  = np.array([r['hb_graph_pr_auc']   for r in all_results], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # Panel 1: paired scatter, one point per split
    axes[0].scatter(pair, hbg, s=60, alpha=0.7, edgecolor='k', linewidth=0.5,
                    color='steelblue')
    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1)
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1)
    axes[0].set_xlabel('Pairwise PR-AUC')
    axes[0].set_ylabel('HB-graph PR-AUC')
    axes[0].set_title('Paired splits')
    d = stats['comparisons']['hb_graph_vs_pairwise']
    axes[0].text(0.04, 0.94, f"HB-graph wins {d['wins']}/{d['n_pairs']}\n"
                             f"p = {d['p_greater']:.4f}",
                 transform=axes[0].transAxes, va='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    # Panel 2: boxplot across tiers
    data, labels = [], []
    for tier in TIERS:
        vals = np.array([r.get(f'{tier}_pr_auc', np.nan) for r in all_results],
                        dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals):
            data.append(vals)
            labels.append(TIER_LABELS[tier].replace(' + ', '\n+ '))
    # 'labels' was renamed 'tick_labels' in Matplotlib 3.9; try the new name
    # first so the script is quiet on both old and new versions.
    try:
        bp = axes[1].boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6)
    except TypeError:
        bp = axes[1].boxplot(data, labels=labels, patch_artist=True, widths=0.6)
    for patch, colour in zip(bp['boxes'],
                             ['gray', 'lightsteelblue', 'cornflowerblue',
                              'lightseagreen', 'steelblue']):
        patch.set_facecolor(colour)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel('PR-AUC')
    axes[1].set_title('Distribution across splits')
    axes[1].tick_params(axis='x', labelsize=12)

    fig.suptitle(f"{CONFIG['DATABASE']} — {task['name']} — logistic regression")
    fig.tight_layout()
    fig.savefig(output_dir / 'tier_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_feature_importance(imp_dfs: List[tuple], output_dir: Path,
                            task: Dict, top_n: int = 15):
    """Horizontal bar chart of mean permutation importance per tier."""
    panels = [(lab, df, c) for lab, df, c in imp_dfs if not df.empty]
    if not panels:
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 7))
    if len(panels) == 1:
        axes = [axes]

    for ax, (label, df, colour) in zip(axes, panels):
        top = df.head(top_n).iloc[::-1]
        ax.barh(top['feature'], top['mean'], xerr=top['std'],
                color=colour, edgecolor='k', linewidth=0.5, capsize=3)
        ax.axvline(0, color='k', linewidth=1)
        ax.set_xlabel('Mean PR-AUC drop')
        ax.set_title(label)
        ax.tick_params(axis='y', labelsize=12)

    fig.suptitle(f"{CONFIG['DATABASE']} — {task['name']} — permutation importance")
    fig.tight_layout()
    fig.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


# =======================================================
# PER-TASK RUNNER
# =======================================================

def run_task(task_key: str, features_df: pd.DataFrame,
             feature_sets: Dict[str, List[str]], base_output_dir: Path,
             start_time: float) -> Dict:
    """Runs one task end-to-end and writes its outputs. Returns summary rows."""
    task = CONFIG["TASKS"][task_key]
    output_dir = base_output_dir / task_key
    output_dir.mkdir(parents=True, exist_ok=True)

    splits_path = CONFIG["DATA_DIR"] / task["splits_file"]

    print(f"\n{'#'*70}")
    print(f"  TASK: {task['name']}   [{task_key}]")
    print(f"  Splits : {splits_path}")
    print(f"  Output : {output_dir}")
    print(f"{'#'*70}\n")

    splits_df = load_splits(task_key)
    split_indices = sorted(splits_df['split_index'].unique())
    print(f"\n   Running {len(split_indices)} splits.\n")

    # --- Main loop over splits ---
    all_results, failed_splits = [], []
    checkpoint_path = output_dir / 'split_results_checkpoint.csv'
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    for i, split_idx in enumerate(split_indices, start=1):
        print(f"   Split {split_idx:>3} ({i:>2}/{len(split_indices)})...",
              end=" ", flush=True)
        try:
            result = run_split(split_idx, features_df, splits_df,
                               feature_sets, task)
            all_results.append(result)

            ckpt = {k: result.get(k, np.nan) for k in
                    ['split_index', 'n_train', 'n_test',
                     'train_pos_pct', 'test_pos_pct'] +
                    [f'{t}_pr_auc' for t in TIERS]}
            pd.DataFrame([ckpt]).to_csv(
                checkpoint_path, mode='a', index=False,
                header=not checkpoint_path.exists())

            winner = ("HB-graph" if result['pr_auc_diff'] > 0
                      else "Pair" if result['pr_auc_diff'] < 0 else "Tie")
            hf = result.get('hyper_flag_pr_auc', np.nan)
            hf_str = "skip" if np.isnan(hf) else f"{hf:.4f}"
            hr = result.get('hyper_recip_pr_auc', np.nan)
            hr_str = "skip" if np.isnan(hr) else f"{hr:.4f}"
            print(f"train={result['n_train']} ({result['train_pos_pct']:.1f}% pos)  "
                  f"test={result['n_test']} ({result['test_pos_pct']:.1f}% pos)  |  "
                  f"Pair: {result['pairwise_pr_auc']:.4f}, "
                  f"Hyper: {result['hypergraph_pr_auc']:.4f}, "
                  f"Hyper+flag: {hf_str}, "
                  f"Hyper+1/sz: {hr_str}, "
                  f"HB-graph: {result['hb_graph_pr_auc']:.4f}  [{winner}]")
        except Exception as e:
            failed_splits.append(split_idx)
            print(f"ERROR: {e}")

    if failed_splits:
        print(f"\n   WARNING: {len(failed_splits)} split(s) FAILED and are "
              f"excluded from all statistics: {failed_splits}")
    if not all_results:
        raise RuntimeError(f"[{task_key}] No splits completed — nothing to analyse.")
    print(f"\n   Completed {len(all_results)}/{len(split_indices)} splits.")

    n_skipped = sum(1 for r in all_results if r.get('hyper_flag_skipped'))
    if n_skipped:
        print(f"   NOTE: hyper_flag tier skipped on {n_skipped} split(s) "
              f"(flag degenerate in the training fold).")

    # --- Statistics ---
    print("\n4. Statistical analysis...")
    stats = run_sign_test_comparison(all_results)
    summary_text = format_statistical_summary(stats, task)
    print(summary_text)

    # --- Plots ---
    print("\n5. Generating plots...")
    plot_tier_comparison(all_results, stats, output_dir, task)

    # --- Feature importance ---
    print("\n6. Aggregating feature importance...")
    imp = {tier: aggregate_feature_importance(all_results, tier) for tier in TIERS}
    imp_text = format_feature_importance(
        [(TIER_LABELS[t], imp[t]) for t in TIERS])
    print(imp_text)

    flag = CONFIG["ANNOTATION_FLAG"]
    hf_imp = imp['hyper_flag']
    if not hf_imp.empty and flag in set(hf_imp['feature']):
        row = hf_imp[hf_imp['feature'] == flag].iloc[0]
        print(f"\n   >>> '{flag}' ranks {int(row['rank'])} of {len(hf_imp)} in the "
              f"hypergraph+flag model (mean importance {row['mean']:.5f})")

    plot_feature_importance(
        [(TIER_LABELS['pairwise'],   imp['pairwise'],   'gray'),
         (TIER_LABELS['hypergraph'], imp['hypergraph'], 'skyblue'),
         (TIER_LABELS['hb_graph'],   imp['hb_graph'],   'steelblue')],
        output_dir, task)

    # --- Save CSVs ---
    print("\n7. Saving outputs...")

    summary_cols = (['split_index', 'n_train', 'n_test',
                     'train_pos_pct', 'test_pos_pct'] +
                    [f'{t}_{m}' for t in TIERS for m in ('pr_auc', 'f1')] +
                    ['pr_auc_diff', 'f1_diff', 'stoich_pr_auc_diff',
                     'stoich_f1_diff', 'annot_pr_auc_diff', 'value_pr_auc_diff',
                     'recip_pr_auc_diff', 'net_recip_pr_auc_diff'])
    summary_df = pd.DataFrame(
        [{k: r.get(k, np.nan) for k in summary_cols} for r in all_results])
    summary_df.to_csv(output_dir / 'split_results.csv', index=False)
    print("   Saved: split_results.csv")

    stats_rows = []
    for name, d in stats['comparisons'].items():
        stats_rows.append({
            'database': CONFIG['DATABASE'], 'task': task_key,
            'comparison': name, 'splits_file': task['splits_file'],
            'n_splits': stats['n_runs'], 'model': 'LogisticRegression',
            **{k: d[k] for k in ['mean_diff', 'std_diff', 'cohens_dz', 'wins',
                                 'losses', 'ties', 'n_pairs', 'p_greater',
                                 'p_two_sided']}
        })
    pd.DataFrame(stats_rows).to_csv(output_dir / 'comparison_stats.csv', index=False)
    print("   Saved: comparison_stats.csv")

    for tier in TIERS:
        key = f'{tier}_predictions'
        frames = [r[key] for r in all_results if key in r]
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(
                output_dir / f'{tier}_predictions.csv', index=False)
            print(f"   Saved: {tier}_predictions.csv")
        if not imp[tier].empty:
            imp[tier].to_csv(output_dir / f'{tier}_feature_importance.csv',
                             index=False)
            print(f"   Saved: {tier}_feature_importance.csv")

    # Best C per tier per split — worth eyeballing to check the grid isn't
    # railed at an endpoint (which would mean the C range needs widening).
    c_rows = []
    for r in all_results:
        row = {'split_index': r['split_index']}
        for tier in TIERS:
            bp = r.get(f'{tier}_best_params')
            row[f'{tier}_C'] = bp['clf__C'] if bp else np.nan
        c_rows.append(row)
    c_df = pd.DataFrame(c_rows)
    c_df.to_csv(output_dir / 'best_C_per_split.csv', index=False)
    print("   Saved: best_C_per_split.csv")

    grid_C = CONFIG['PARAM_GRID']['clf__C']
    for tier in TIERS:
        vals = c_df[f'{tier}_C'].dropna()
        if len(vals) and (vals == min(grid_C)).mean() > 0.5:
            print(f"   WARNING: {tier} selected the smallest C on "
                  f"{100*(vals == min(grid_C)).mean():.0f}% of splits — "
                  f"consider extending the grid downwards.")
        if len(vals) and (vals == max(grid_C)).mean() > 0.5:
            print(f"   WARNING: {tier} selected the largest C on "
                  f"{100*(vals == max(grid_C)).mean():.0f}% of splits — "
                  f"consider extending the grid upwards.")

    # --- Text report ---
    with open(output_dir / 'statistical_summary.txt', 'w') as f:
        f.write("REPRESENTATION COMPARISON — LOGISTIC REGRESSION BASELINE\n")
        f.write(f"Database: {CONFIG['DATABASE']}\n")
        f.write(f"Task: {task['name']}\n")
        f.write("\nRUN PROVENANCE\n")
        f.write(f"{'-'*70}\n")
        f.write(f"Run at              : "
                f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")
        f.write(f"Model               : LogisticRegression "
                f"(StandardScaler pipeline)\n")
        f.write(f"LogReg settings     : {CONFIG['LOGREG_KWARGS']}\n")
        f.write(f"Hyperparameter grid : {CONFIG['PARAM_GRID']}\n")
        f.write(f"Random state        : {CONFIG['RANDOM_STATE']}\n")
        f.write(f"Inner CV folds      : {CONFIG['N_SPLITS_CV']}\n")
        f.write(f"Permutation repeats : {CONFIG['PERM_REPEATS']}\n")
        f.write(f"Splits file         : {splits_path}\n")
        f.write(f"Splits file mtime   : "
                f"{pd.Timestamp(os.path.getmtime(splits_path), unit='s')}\n")
        f.write(f"Data directory      : {CONFIG['DATA_DIR']}\n")
        f.write(f"Feature files       : {CONFIG['PROTEIN_FEATURES_FILE']}, "
                f"{CONFIG['PAIRWISE_FEATURES_FILE']}\n")
        f.write(f"Splits in file      : {splits_df['split_index'].nunique()}\n")
        f.write(f"Splits attempted    : {len(split_indices)}\n")
        f.write(f"Splits completed    : {stats['n_runs']}\n")
        if failed_splits:
            f.write(f"FAILED splits (excluded from all statistics): "
                    f"{failed_splits}\n")
        f.write(f"Annotation flag     : {flag} "
                f"(derived from {CONFIG['FLAG_SOURCE_FEATURE']})\n")
        f.write(f"hyper_recip tier    : "
                f"{'included' if CONFIG['INCLUDE_HYPER_RECIP'] else 'DISABLED'}"
                f" ({CONFIG['RECIP_FEATURE']} = 1/(1+"
                f"{CONFIG['RECIP_SOURCE_FEATURE']}))\n")
        f.write(f"hyper_flag tier     : "
                f"{'included' if CONFIG['INCLUDE_HYPER_FLAG'] else 'DISABLED'}"
                f"{f'; skipped on {n_skipped} split(s)' if n_skipped else ''}\n\n")

        f.write("CLASS BALANCE (mean over splits)\n")
        f.write(f"{'-'*70}\n")
        f.write(f"Train positives     : {summary_df['train_pos_pct'].mean():.2f}% "
                f"(range {summary_df['train_pos_pct'].min():.2f}"
                f"-{summary_df['train_pos_pct'].max():.2f}%)\n")
        f.write(f"Test  positives     : {summary_df['test_pos_pct'].mean():.2f}% "
                f"(range {summary_df['test_pos_pct'].min():.2f}"
                f"-{summary_df['test_pos_pct'].max():.2f}%)\n")
        f.write(f"Mean train / test n : {summary_df['n_train'].mean():.0f} / "
                f"{summary_df['n_test'].mean():.0f}\n\n")

        for tier in TIERS:
            feats = feature_sets.get(tier, [])
            f.write(f"{TIER_LABELS[tier]} features ({len(feats)}):\n")
            for feat in feats:
                tag = ''
                if feat in CONFIG['FEATURES']['STOICHIOMETRY_FEATURES']:
                    tag = ' [stoich]'
                elif feat == flag:
                    tag = ' [annotation flag]'
                elif feat == CONFIG['RECIP_FEATURE']:
                    tag = ' [1/size control]'
                f.write(f"  - {feat}{tag}\n")
            f.write("\n")

        f.write(summary_text.replace('±', '+/-'))
        f.write("\n\n")
        f.write(imp_text)
        f.write("\n\n")
        f.write(f"{'='*70}\n")
        f.write("PER-SPLIT PR-AUC (also in split_results.csv)\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"{'Split':>6} {'Pairwise':>10} {'Hyper':>10} {'Hyper+flag':>11} "
                f"{'Hyper+1/sz':>11} {'HB-graph':>10} {'HBG-Pair':>10} "
                f"{'HBG-Hyper':>10} {'HBG-1/sz':>10}\n")
        f.write(f"{'-'*94}\n")
        for _, row in summary_df.sort_values('split_index').iterrows():
            def _fmt(v, w=11):
                return " " * (w - 1) + "-" if pd.isna(v) else f"{v:>{w}.4f}"
            f.write(f"{int(row['split_index']):>6} "
                    f"{row['pairwise_pr_auc']:>10.4f} "
                    f"{row['hypergraph_pr_auc']:>10.4f} "
                    f"{_fmt(row['hyper_flag_pr_auc'])} "
                    f"{_fmt(row['hyper_recip_pr_auc'])} "
                    f"{row['hb_graph_pr_auc']:>10.4f} "
                    f"{row['pr_auc_diff']:>+10.4f} "
                    f"{row['stoich_pr_auc_diff']:>+10.4f} "
                    f"{_fmt(row['net_recip_pr_auc_diff'], 10)}\n")
        f.write(f"\n\nRuntime to this point: "
                f"{(time.time() - start_time)/60:.1f} min\n")
    print("   Saved: statistical_summary.txt")

    # --- Rows for the cross-task summary table ---
    perf_rows = [{
        'database': CONFIG['DATABASE'], 'task': task_key,
        'task_name': task['name'], 'model': 'LogisticRegression',
        'n_splits': stats['n_runs'], 'representation': TIER_LABELS[tier],
        'pr_auc_mean': stats[f'{tier}_pr_auc_mean'],
        'pr_auc_std':  stats[f'{tier}_pr_auc_std'],
        'f1_mean':     stats[f'{tier}_f1_mean'],
        'f1_std':      stats[f'{tier}_f1_std'],
    } for tier in TIERS]

    return {'performance': perf_rows, 'comparisons': stats_rows}


# =======================================================
# MAIN
# =======================================================

if __name__ == "__main__":

    start_time = time.time()
    print(f"Process started at "
          f"{time.strftime('%H:%M:%S', time.localtime(start_time))}")

    base_output_dir = Path(CONFIG["BASE_OUTPUT_DIR"])
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # --- Pre-flight: every splits file must exist before any work begins ---
    for key, task in CONFIG["TASKS"].items():
        p = CONFIG["DATA_DIR"] / task["splits_file"]
        if not p.exists():
            raise FileNotFoundError(
                f"Splits file for task '{key}' not found: {p}\n"
                f"   Check CONFIG['TASKS']['{key}']['splits_file'].")

    print(f"\n{'='*70}")
    print(f"  LOGISTIC REGRESSION BASELINE")
    print(f"  Database : {CONFIG['DATABASE']}")
    print(f"  Tasks    : {', '.join(CONFIG['TASKS'])}")
    print(f"  Output   : {base_output_dir}")
    print(f"{'='*70}\n")

    # --- Load features once; they are shared across all three tasks ---
    features_df = load_all_features()

    hb_graph_features = [f for f in CONFIG["FEATURES"]["HB_GRAPH"]
                         if f in features_df.columns]
    pairwise_features = [f for f in CONFIG["FEATURES"]["PAIRWISE"]
                         if f in features_df.columns]
    stoich_features   = CONFIG["FEATURES"].get("STOICHIOMETRY_FEATURES", [])
    hypergraph_features = [f for f in hb_graph_features
                           if f not in stoich_features]

    flag  = CONFIG["ANNOTATION_FLAG"]
    recip = CONFIG["RECIP_FEATURE"]
    feature_sets = {
        'pairwise':   pairwise_features,
        'hypergraph': hypergraph_features,
        'hyper_flag': (hypergraph_features + [flag]
                       if CONFIG["INCLUDE_HYPER_FLAG"] else []),
        'hyper_recip': (hypergraph_features + [recip]
                        if CONFIG["INCLUDE_HYPER_RECIP"] else []),
        'hb_graph':   hb_graph_features,
    }

    missing_hbg  = [f for f in CONFIG["FEATURES"]["HB_GRAPH"]
                    if f not in features_df.columns]
    missing_pair = [f for f in CONFIG["FEATURES"]["PAIRWISE"]
                    if f not in features_df.columns]
    if missing_hbg:
        print(f"   WARNING: {len(missing_hbg)} hb-graph features not found "
              f"in data: {missing_hbg}")
    if missing_pair:
        print(f"   WARNING: {len(missing_pair)} pairwise features not found "
              f"in data: {missing_pair}")

    for tier in TIERS:
        feats = feature_sets[tier]
        if not feats:
            print(f"\n   {TIER_LABELS[tier]} tier: DISABLED")
            continue
        print(f"\n   Active {TIER_LABELS[tier]} features ({len(feats)}):")
        for f_ in feats:
            tag = (' [stoich]' if f_ in stoich_features
                   else ' [annotation flag]' if f_ == flag
                   else ' [1/size control]' if f_ == recip else '')
            print(f"     - {f_}{tag}")

    # --- Fill any NaNs in feature columns ---
    # NB: encoding deliberately UNCHANGED from the tree scripts. Test A measures
    # what the current fillna(0) encoding is buying; fixing it here would defeat that.
    all_feature_cols = sorted(set(hb_graph_features + pairwise_features))
    if CONFIG["INCLUDE_HYPER_RECIP"] and recip in features_df.columns:
        all_feature_cols = sorted(set(all_feature_cols + [recip]))
    n_nans = int(features_df[all_feature_cols].isna().sum().sum())
    if n_nans > 0:
        print(f"\n   Filling {n_nans} missing feature values with 0.")
        features_df[all_feature_cols] = features_df[all_feature_cols].fillna(0)

    # --- Run every task ---
    perf_rows, comp_rows, failed_tasks = [], [], []
    for task_key in CONFIG["TASKS"]:
        try:
            out = run_task(task_key, features_df, feature_sets,
                           base_output_dir, start_time)
            perf_rows += out['performance']
            comp_rows += out['comparisons']
        except Exception as e:
            failed_tasks.append(task_key)
            print(f"\n   ERROR: task '{task_key}' failed: {e}\n")

    # --- Cross-task summary ---
    if perf_rows:
        perf_df = pd.DataFrame(perf_rows)
        perf_df.to_csv(base_output_dir / 'all_tasks_summary.csv', index=False)
        pd.DataFrame(comp_rows).to_csv(
            base_output_dir / 'all_tasks_comparison_stats.csv', index=False)

        print(f"\n{'='*70}")
        print(f"  CROSS-TASK SUMMARY — {CONFIG['DATABASE']} — "
              f"logistic regression PR-AUC")
        print(f"{'='*70}\n")
        print(f"  {'Task':<12} {'Representation':<20} {'PR-AUC':>18} {'F1':>18}")
        print(f"  {'-'*70}")
        for _, r in perf_df.iterrows():
            if np.isnan(r['pr_auc_mean']):
                continue
            print(f"  {r['task']:<12} {r['representation']:<20} "
                  f"{r['pr_auc_mean']:>10.4f} ± {r['pr_auc_std']:.4f}  "
                  f"{r['f1_mean']:>8.4f} ± {r['f1_std']:.4f}")
        print(f"\n  Saved: all_tasks_summary.csv, all_tasks_comparison_stats.csv")

    if failed_tasks:
        print(f"\n  WARNING: {len(failed_tasks)} task(s) failed: {failed_tasks}")

    print(f"\n{'='*70}")
    print("  COMPLETE")
    print(f"{'='*70}")
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")