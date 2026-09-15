"""
=======================================================================
MODEL-VARIANCE SWEEP  —  supplementary analysis
=======================================================================
Runs the SAME three feature tiers across many learners, to show that the
ordering pairwise < hypergraph < hb-graph does not depend on the model.

This is deliberately NARROWER than cp_ess_second.py:
  - three tiers only (no hyper_flag / Test A decomposition)
  - PR-AUC only (no F1 — thresholded metrics are not comparable across
    learners with different class-imbalance handling)
  - no permutation importance, no plots (see analyse_model_sweep.py)

FAIRNESS PROTOCOL — the claim is invariance ACROSS TIERS WITHIN a model,
so everything that could differ between tiers is held fixed:
  - identical splits file, identical train/test protein sets
  - identical CV folds (StratifiedKFold, unshuffled, fixed n_splits) —
    matches cp_hpa_second.py's plain-integer cv=5, which resolves to the
    same unshuffled StratifiedKFold internally
  - a single fixed RANDOM_STATE for every (model, split, tier) cell —
    matches cp_hpa_second.py's single CONFIG["RANDOM_STATE"] used across
    all 50 splits. Earlier versions of this script varied the seed per
    (model, split); that confounded model-seed variance with the
    data-partition variance the 50-split design is meant to isolate, and
    is why RandomForest's hb-graph numbers didn't reproduce the paper
    until this was fixed — see git history / lab notes if the reasoning
    is needed again.
  - identical candidate hyperparameter configurations — the search
    sampler (grid or random) uses that same fixed seed, so all three
    tiers are offered the same configurations, and every split is offered
    the same configurations too
  - identical search budget n_iter, constant across tiers
  - identical preprocessing (train-only quantile transform of the
    continuous features, refitted inside each CV fold)

Note that the budget being constant across tiers of DIFFERENT
dimensionality (4 / 8 / 14) means the 14-dimensional hb-graph space is
searched least densely. Any ordering observed is therefore conservative.

Budgets differ BETWEEN models. That is not unfairness: n_estimators has
no Gaussian-process counterpart, so "the same grid" is not definable
across learners. What is defensible, and what is enforced here, is a
constant budget within each model across the three tiers.

Output: one long-format CSV, appended after every (model, split).
Re-running skips any (model, split) already present, so the sweep is
resumable after a crash or an interrupted GP.
=======================================================================
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold
from sklearn.metrics import average_precision_score
from scipy.stats import loguniform, uniform

# =======================================================
# TASK SWITCH  — set this, everything else follows
# =======================================================
TASK = "ess"          # "ess" | "hpa" | "chembl"

TASKS = {
    "ess": dict(
        SPLITS_FILE="ess_protein_splits.csv",
        LABEL_MAP={'Essential': 1, 'Non-essential': 0},
        OUTPUT_DIR=Path("./model_sweep/cp_ess"),
    ),
    "hpa": dict(
        SPLITS_FILE="hpa_protein_splits.csv",
        LABEL_MAP={'Drug_target': 1, 'Non_target': 0},
        OUTPUT_DIR=Path("./model_sweep/cp_hpa"),
    ),
    "chembl": dict(
        SPLITS_FILE="chembl_protein_splits.csv",
        LABEL_MAP={'Drug_target': 1, 'Non_target': 0},
        OUTPUT_DIR=Path("./model_sweep/cp_chembl"),
    ),
}

# =======================================================
# CONFIGURATION
# =======================================================
CONFIG = {
    # --- Paths (mirror cp_ess_second.py) ---
    "DATA_DIR":    Path("../../../data/lookup_tables/cp/"),
    "RESULTS_CSV":      "sweep_results_tabpfn_svm.csv",       # long format, appended
    "PREDICTIONS_CSV":  "predictions.csv",         # per-protein scores, appended

    "PROTEIN_FEATURES_FILE":  "hypergraph_features.csv",
    "PAIRWISE_FEATURES_FILE": "pairwise_features.csv",

    # --- Fixed settings ---
    "RANDOM_STATE": 42,
    "N_SPLITS_CV":  5,

    # Which models to run, in order. Comment out to skip.
    # Anything whose library is not installed is dropped with a warning.
    "MODELS_TO_RUN": [
        "LogisticRegression",
        # "EBM",  excluded: ~21 min/split (~17.5 h over 50 splits).
        "RBF-SVM",
        # "GaussianProcess",
        "MLP",
        # "MLP-lbfgs"
        "RandomForest",
        "XGBoost",
        "LightGBM",
        # "TabPFN",   # slowest (~7.5 min/split); last so the rest land first
    ],

    # Cap the number of splits while piloting; None = all splits in the file.
    "MAX_SPLITS": None,

    # --- Features -------------------------------------------------------
    # Matches the headline tiers in cp_ess_second.py. Three tiers:
    #   PAIRWISE   (4)  — dyadic PPI features
    #   HYPERGRAPH (8)  — HB_GRAPH minus stoichiometry values
    #   HB_GRAPH   (14) — everything (no annotation flags)
    "FEATURES": {
        "HB_GRAPH": [
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
        ],
    },
}

CONFIG.update(TASKS[TASK])
CONFIG["TASK"] = TASK

# =======================================================
# MODEL REGISTRY
# =======================================================
# Each entry: (family, factory, param_distributions, n_iter, needs_scaling)
#
#   factory(y_train, seed) -> unfitted estimator
#   n_iter = None, params={}        -> no search; fitted directly (e.g. GaussianProcess)
#   n_iter = None, search="grid"    -> exhaustive GridSearchCV over params
#                                       (e.g. RandomForest, XGBoost, LightGBM —
#                                       matches cp_hpa_second.py's GridSearchCV)
#   n_iter = <int>                  -> RandomizedSearchCV, that many candidates
#   needs_scaling                   -> wrap in a train-only quantile transform

def build_registry(random_state: int) -> Dict[str, dict]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

    reg: Dict[str, dict] = {}

    # ---------------- Linear / additive ----------------
    reg["LogisticRegression"] = dict(
        family="Linear",
        factory=lambda y, s: LogisticRegression(
            solver='lbfgs', max_iter=5000,
            class_weight='balanced', random_state=s),
        params={'C': loguniform(1e-3, 1e2)},
        n_iter=20,
        scale=True,
    )

    try:
        from interpret.glassbox import ExplainableBoostingClassifier
        reg["EBM"] = dict(
            family="Additive",
            factory=lambda y, s: ExplainableBoostingClassifier(
                random_state=s, n_jobs=1),
            params={
                'max_bins':      [128, 256],
                'interactions':  [0, 5, 10],
                'learning_rate': loguniform(5e-3, 2e-1),
            },
            n_iter=10,
            scale=False,
        )
    except ImportError:
        print("   [skip] EBM — `pip install interpret` to include it.")

    # ---------------- Kernel ----------------
    # probability=False: the average_precision scorer falls back to
    # decision_function, which avoids SVC's internal 5-fold Platt
    # calibration and is ~5x faster. Ranking is unaffected.
    # reg["RBF-SVM"] = dict(
    #     family="Kernel",
    #     factory=lambda y, s: SVC(
    #         kernel='rbf', probability=False,
    #         class_weight='balanced', random_state=s),
    #     params={'C': loguniform(1e-2, 1e3), 'gamma': loguniform(1e-4, 1e1)},
    #     n_iter=20,
    #     scale=True,
    # )
    reg["RBF-SVM"] = dict(
    family="Kernel",
    factory=lambda y, s: SVC(
        kernel='rbf', probability=False,
        class_weight='balanced', random_state=s),
    params={
        'C':     [0.1, 1, 10, 100, 1000],
        'gamma': ['scale', 0.001, 0.01, 0.1, 1],
    },
    n_iter=None,
    scale=True,
    search="grid",
    )

    # Fixed kernel, no hyperparameter optimisation, no search. Features are
    # quantile-transformed to a common scale, so a single shared length
    # scale is defensible. Without this the L-BFGS marginal-likelihood
    # optimisation runs dozens of O(n^3) Choleskys per fit.
    reg["GaussianProcess"] = dict(
        family="Kernel",
        factory=lambda y, s: GaussianProcessClassifier(
            kernel=ConstantKernel(1.0) * RBF(length_scale=1.0)
                   + WhiteKernel(noise_level=1e-3,
                                 noise_level_bounds='fixed'),
            optimizer=None, random_state=s),
        params={},
        n_iter=None,
        scale=True,
    )

    # ---------------- Neural (adam, existing) ----------------
    reg["MLP"] = dict(
        family="Neural",
        factory=lambda y, s: MLPClassifier(
            max_iter=1000, early_stopping=True, n_iter_no_change=20,
            random_state=s),
        params={
            'hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64)],
            'alpha':              loguniform(1e-5, 1e-1),
            'learning_rate_init': loguniform(1e-4, 1e-2),
        },
        n_iter=20,
        scale=True,
    )

    # ---------------- Neural (lbfgs, comparison run) ----------------
    # sklearn recommends lbfgs over adam/sgd for small datasets (~thousands
    # of rows, which this is) — it's a full-batch quasi-Newton optimizer,
    # generally converges more reliably at this scale, and removes
    # learning_rate_init as a tunable entirely (it only applies to sgd/adam).
    # early_stopping / n_iter_no_change are also sgd/adam-only, so they're
    # dropped here rather than silently ignored.
    reg["MLP-lbfgs"] = dict(
        family="Neural",
        factory=lambda y, s: MLPClassifier(
            solver='lbfgs', max_iter=5000, random_state=s),
        params={
            'hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64)],
            'alpha':              loguniform(1e-5, 1e-1),
        },
        n_iter=20,
        scale=True,
    )

    # ---------------- Bagged trees ----------------
    # Grid matches cp_hpa_second.py's PARAM_GRIDS["RandomForest"] exactly, so
    # RF sweep results are directly comparable to the main-paper numbers.
    # max_features is deliberately NOT a search dimension here — the original
    # never tuned it, so it stayed at sklearn's default ('sqrt') for every
    # fit. Adding it as a free dimension is what let RandomizedSearchCV drift
    # to overfit configs on the 14-dim hb_graph tier; leaving it fixed removes
    # that failure mode without needing a wider exhaustive search.
    reg["RandomForest"] = dict(
        family="Bagged trees",
        factory=lambda y, s: RandomForestClassifier(
            class_weight='balanced', n_jobs=1, random_state=s),
        params={
            'n_estimators':      [80, 100, 200],
            'max_depth':         [None, 5, 10],
            'min_samples_split': [2, 5, 10],
        },
        n_iter=None,     # NEW: exhaustive — matches GridSearchCV in the original
        scale=False,
        search="grid",
    )

    # ---------------- Boosted trees ----------------
    try:
        from xgboost import XGBClassifier

        def _xgb(y, s):
            pos, neg = int((y == 1).sum()), int((y == 0).sum())
            return XGBClassifier(
                n_jobs=1, verbosity=0, eval_metric='logloss',
                scale_pos_weight=(neg / pos if pos else 1.0), random_state=s)

        # Grid matches cp_hpa_second.py's PARAM_GRIDS["XGBoost"] exactly, so
        # results are directly comparable to the main-paper numbers.
        # colsample_bytree is deliberately NOT a search dimension — the
        # original never tuned it, so it stayed at XGBoost's default (1.0)
        # for every fit.
        #
        # NOTE: max_depth=None here does NOT mean "unlimited" the way it
        # does for sklearn's RandomForestClassifier. XGBoost's sklearn
        # wrapper drops None params before passing them to the booster, so
        # max_depth=None silently falls back to XGBoost's own default (6),
        # not to unlimited depth. That matches what actually happened in
        # cp_hpa_second.py's own GridSearchCV run (same None in the same
        # grid, same fallback) — so it's faithful to the original, not a new
        # discrepancy — but worth knowing before reading too much into which
        # max_depth value "wins" in best_params for this model.
        reg["XGBoost"] = dict(
            family="Boosted trees",
            factory=_xgb,
            params={
                'n_estimators':  [80, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth':     [None, 5, 10],
                'subsample':     [0.75, 0.8, 1.0],
            },
            n_iter=None,     # exhaustive — matches GridSearchCV in the original
            scale=False,
            search="grid",
        )
    except ImportError:
        print("   [skip] XGBoost — not installed.")

    try:
        from lightgbm import LGBMClassifier

        # Grid matches cp_hpa_second.py's PARAM_GRIDS["LightGBM"] exactly.
        # min_child_samples is deliberately NOT a search dimension — the
        # original never tuned it, so it stayed at LightGBM's default (20)
        # for every fit.
        reg["LightGBM"] = dict(
            family="Boosted trees",
            factory=lambda y, s: LGBMClassifier(
                class_weight='balanced', n_jobs=1, verbose=-1, random_state=s),
            params={
                'n_estimators':  [80, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth':     [None, 5, 10],
                'num_leaves':    [30, 50, 100],
            },
            n_iter=None,     # exhaustive — matches GridSearchCV in the original
            scale=False,
            search="grid",
        )
    except ImportError:
        print("   [skip] LightGBM — not installed.")

    # ---------------- Amortised / in-context ----------------
    # TabPFN v2 is a transformer pre-trained on synthetic tabular priors.
    # It has no hyperparameters to tune — that is the method, not a
    # concession, so n_iter is None. Also covers the "attention-based
    # architecture" the supervisor asked for.
    try:
        from tabpfn import TabPFNClassifier
        reg["TabPFN"] = dict(
            family="Amortised",
            factory=lambda y, s: TabPFNClassifier(device='cpu',
                                                  random_state=s),
            params={},
            n_iter=None,
            scale=False,      # TabPFN normalises internally
        )
    except ImportError:
        print("   [skip] TabPFN — `pip install tabpfn` to include it.")

    return reg


# =======================================================
# DATA
# =======================================================
def load_features() -> pd.DataFrame:
    hg   = pd.read_csv(CONFIG["DATA_DIR"] / CONFIG["PROTEIN_FEATURES_FILE"])
    pair = pd.read_csv(CONFIG["DATA_DIR"] / CONFIG["PAIRWISE_FEATURES_FILE"])
    df = pd.merge(hg, pair, on='ProteinId', how='inner')
    print(f"   Features merged: {df.shape}")
    return df


def load_splits() -> pd.DataFrame:
    path = CONFIG["DATA_DIR"] / CONFIG["SPLITS_FILE"]
    df = pd.read_csv(path).rename(columns={'UniProt_AC': 'ProteinId'})
    df['target'] = df['protein_label'].map(CONFIG["LABEL_MAP"])
    print(f"   Splits file    : {path.name}  "
          f"({df['split_index'].nunique()} splits)")
    return df


def resolve_tiers(features_df: pd.DataFrame) -> Dict[str, List[str]]:
    F = CONFIG["FEATURES"]
    hb   = [f for f in F["HB_GRAPH"] if f in features_df.columns]
    pair = [f for f in F["PAIRWISE"] if f in features_df.columns]

    missing = [f for f in F["HB_GRAPH"] + F["PAIRWISE"]
               if f not in features_df.columns]
    if missing:
        raise KeyError(f"Features absent from the feature files: {missing}")

    # Hypergraph = hb-graph minus the stoichiometry-value features (the
    # set-based representation). No annotation flags are used, so the three
    # tiers match the headline comparison in cp_ess_second.py exactly (4/8/14).
    hyper = [f for f in hb if f not in set(F["STOICHIOMETRY_FEATURES"])]

    assert set(hyper) < set(hb), "Tiers are not nested"
    tiers = {'pairwise': pair, 'hypergraph': hyper, 'hb_graph': hb}
    for name, cols in tiers.items():
        print(f"   {name:<11}: {len(cols)} features")
    return tiers


# =======================================================
# ESTIMATOR ASSEMBLY
# =======================================================
def make_estimator(spec: dict, y_train: pd.Series, seed: int,
                   columns: List[str]):
    """Returns (estimator, param_distributions_with_pipeline_prefix)."""
    clf = spec["factory"](y_train, seed)

    if not spec["scale"]:
        return clf, dict(spec["params"])

    # Quantile-transform the features (all continuous; no binary flags).
    # Wrapped in a Pipeline so the transform is
    # refitted on the training part of each CV fold — never on test data.
    # Sized from the CV training fold, not the full training set, otherwise
    # sklearn clips it and warns on every fit.
    k = CONFIG["N_SPLITS_CV"]
    n_q = max(10, min(500, int(len(y_train) * (k - 1) / k)))

    pre = ColumnTransformer(
        transformers=[
            ('qt', QuantileTransformer(output_distribution='normal',
                                       n_quantiles=n_q,
                                       random_state=seed), list(columns)),
        ],
        remainder='drop',
    )
    pipe = Pipeline([('pre', pre), ('clf', clf)])
    params = {f'clf__{k}': v for k, v in spec["params"].items()}
    return pipe, params

def _predict_score(model, X_test) -> np.ndarray:
    """Ranking score for PR-AUC: predict_proba where available, else decision_function."""
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X_test)[:, 1]
    return model.decision_function(X_test)

def fit_one(spec: dict, X_train, y_train, X_test, y_test, seed: int):
    """Fit one (model, tier, split) cell. Returns (pr_auc, best_params, y_score)."""
    est, params = make_estimator(spec, y_train, seed, list(X_train.columns))

    if not params:
        est.fit(X_train, y_train)
        y_score = _predict_score(est, X_test)
        return float(average_precision_score(y_test, y_score)), {}, y_score

    cv = StratifiedKFold(n_splits=CONFIG["N_SPLITS_CV"])

    if spec.get("search") == "grid":
        search = GridSearchCV(
            estimator=est, param_grid=params, scoring='average_precision',
            cv=cv, n_jobs=-1, error_score=np.nan)
    else:
        search = RandomizedSearchCV(
            estimator=est, param_distributions=params, n_iter=spec["n_iter"],
            scoring='average_precision', cv=cv, random_state=seed,
            n_jobs=-1, error_score=np.nan)
    search.fit(X_train, y_train)
    y_score = _predict_score(search.best_estimator_, X_test)
    return (float(average_precision_score(y_test, y_score)),
            search.best_params_, y_score)

# =======================================================
# MAIN
# =======================================================
def cell_seed(model_name: str, split_idx: int) -> int:
    """Fixed across every (model, split) cell — deliberately NOT a function of
    split_idx. cp_hpa_second.py uses a single RANDOM_STATE for every split
    (see tune_and_train_model); varying the model's random_state per split
    here confounds model-seed variance with data-partition variance, which is
    exactly the axis the 50-split design is meant to isolate. Signature kept
    unchanged so every call site (fit_one, make_estimator, factories) is
    unaffected — they all just receive the same seed now."""
    return CONFIG["RANDOM_STATE"]

def main():
    t0 = time.time()
    print("=" * 70)
    print(f"  MODEL-VARIANCE SWEEP  —  {CONFIG['TASK'].upper()}")
    print("=" * 70)

    out_dir = CONFIG["OUTPUT_DIR"]
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / CONFIG["RESULTS_CSV"]
    predictions_path = out_dir / CONFIG["PREDICTIONS_CSV"]

    print("\n1. Loading data")
    features_df = load_features()
    splits_df   = load_splits()
    tiers       = resolve_tiers(features_df)

    all_cols = sorted({c for cols in tiers.values() for c in cols})
    n_nan = int(features_df[all_cols].isna().sum().sum())
    if n_nan:
        print(f"   Filling {n_nan} missing feature values with 0.")
        features_df[all_cols] = features_df[all_cols].fillna(0)

    split_indices = sorted(splits_df['split_index'].unique())
    if CONFIG["MAX_SPLITS"]:
        split_indices = split_indices[:CONFIG["MAX_SPLITS"]]

    print("\n2. Building model registry")
    registry = build_registry(CONFIG["RANDOM_STATE"])
    models = [m for m in CONFIG["MODELS_TO_RUN"] if m in registry]
    print(f"   Running {len(models)} models x {len(split_indices)} splits "
          f"x {len(tiers)} tiers = "
          f"{len(models) * len(split_indices) * len(tiers)} cells")

    # --- Resume: skip (model, split) pairs already on disk ---
    done = set()
    if results_path.exists():
        prev = pd.read_csv(results_path)
        done = set(zip(prev['model'], prev['split_index']))
        print(f"   Resuming: {len(done)} (model, split) pairs already done")

    print("\n3. Running sweep\n")
    for model_name in models:
        spec = registry[model_name]
        budget = 'none' if spec['n_iter'] is None else spec['n_iter']
        print(f"   {model_name}  [{spec['family']}]  "
              f"budget n_iter={budget}, scaled={spec['scale']}")
        m_start = time.time()
        for split_idx in split_indices:
            if (model_name, split_idx) in done:
                continue

            info = splits_df[splits_df['split_index'] == split_idx][
                ['ProteinId', 'split', 'target', 'label_mask']]
            df = features_df.merge(info, on='ProteinId', how='inner')
            df = df[df['label_mask']]

            train = df[df['split'] == 'train']
            test  = df[df['split'] == 'test']
            y_train = train['target'].astype(int)
            y_test  = test['target'].astype(int)

            seed = cell_seed(model_name, split_idx)
            rows, pred_frames, c_start = [], [], time.time()

            for tier_name, cols in tiers.items():
                try:
                    pr_auc, best, y_score = fit_one(
                        spec, train[cols], y_train, test[cols], y_test, seed)
                except Exception as exc:                # noqa: BLE001
                    print(f"      split {split_idx} / {tier_name} FAILED: "
                          f"{type(exc).__name__}: {exc}")
                    pr_auc, best, y_score = np.nan, {}, np.full(len(test), np.nan)

                rows.append({
                    'model':       model_name,
                    'family':      spec['family'],
                    'tier':        tier_name,
                    'split_index': split_idx,
                    'pr_auc':      pr_auc,
                    'n_train':     len(train),
                    'n_test':      len(test),
                    'n_features':  len(cols),
                    'seed':        seed,
                    'best_params': str(best),
                })

                pred_frames.append(pd.DataFrame({
                    'model':       model_name,
                    'tier':        tier_name,
                    'split_index': split_idx,
                    'ProteinId':   test['ProteinId'].values,
                    'true_label':  y_test.values,
                    'pred_proba':  y_score,
                }))

            # Append immediately so an interrupted run loses at most one split.
            pd.DataFrame(rows).to_csv(
                results_path, mode='a', index=False,
                header=not results_path.exists())
            pd.concat(pred_frames, ignore_index=True).to_csv(
                predictions_path, mode='a', index=False,
                header=not predictions_path.exists())

            if split_idx == split_indices[0] or split_idx % 10 == 0:
                vals = {r['tier']: r['pr_auc'] for r in rows}
                print(f"      split {split_idx:>3}: "
                      f"pair {vals['pairwise']:.4f}  "
                      f"hyper {vals['hypergraph']:.4f}  "
                      f"hb {vals['hb_graph']:.4f}   "
                      f"({time.time() - c_start:.1f}s)")

        print(f"      -> {model_name} done in "
              f"{(time.time() - m_start) / 60:.1f} min\n")

    print("=" * 70)
    print(f"  COMPLETE — {(time.time() - t0) / 60:.1f} min")
    print(f"  Results:     {results_path}")
    print(f"  Predictions: {predictions_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()