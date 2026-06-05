"""
cp_boruta.py
============
Boruta robustness check, complementary to cp_noise_injection.py.

Runs Boruta (Kursa & Rudnicki 2010, J. Stat. Softw.) on the same splits as
the noise-injection pipeline, separately for the pairwise and hypergraph
feature sets. For each split, Boruta returns confirmed / tentative / rejected
verdicts per feature; results are aggregated across the 15 splits to give a
confirmation rate per feature, providing an independent robustness check
on the noise-injection rank criterion.

Key differences from cp_noise_injection.py:
  * Boruta uses Z-scored impurity importance, not permutation PR-AUC drop.
  * Shadows are permuted copies of real features, not Gaussian noise — so
    the marginal distribution of each feature is preserved.
  * Boruta iterates: features clearly above / below shadow get permanently
    confirmed / rejected each round, and the algorithm re-runs on the rest.
  * Output is a 3-way verdict (confirmed / tentative / rejected) per feature
    per split, aggregated to a confirmation rate (k of N splits confirmed).

Usage
-----
  python cp_boruta.py

  Set TASK in CONFIG below to one of: "ess", "chembl", "hpa".

Outputs (per task, in BASE_OUTPUT_DIR / {model}_{task}_boruta/)
  boruta_per_split.csv           — per-(split × feature) verdict and ranking
  boruta_confirmation_rates.csv  — per-feature confirmation rate across splits
  boruta_summary.txt             — plain-text report, with side-by-side comparison
                                    to noise-injection ranked tables (if present)

Install
-------
  pip install Boruta

  If you hit `AttributeError: module 'numpy' has no attribute 'int'`, that's
  a known compatibility issue between BorutaPy and recent numpy. The
  workaround (already applied below) is to alias np.int = int before import.
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict

# --- BorutaPy compatibility shim for newer numpy ----------------------------
# BorutaPy 0.3 still references np.int; numpy >= 1.20 removed it.
if not hasattr(np, 'int'):
    np.int = int  # noqa: NPY001
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'bool'):
    np.bool = bool
# ----------------------------------------------------------------------------

from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier


# =======================================================
# TASK METADATA — keep in sync with cp_noise_injection.py
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
    # --- Task ---
    "TASK": "chembl",

    # --- Paths ---
    "DATA_DIR":        Path("/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/cp/"),
    "BASE_OUTPUT_DIR": Path("./randomforest"),

    # --- Files ---
    "PROTEIN_FEATURES_FILE":  "hypergraph_features.csv",
    "PAIRWISE_FEATURES_FILE": "pairwise_features.csv",

    # --- Model (impurity importance — Boruta's default) ---
    "MODEL_TYPE":   "RandomForest",
    "RANDOM_STATE": 42,

    # --- Boruta settings ---
    # max_iter: more iterations resolve borderline features. 100 is standard.
    # perc:     percentile of shadow importance used as the threshold (100 = max,
    #           the original Boruta; lower values are more permissive).
    # alpha:    family-wise significance level (Bonferroni-corrected internally).
    "BORUTA_MAX_ITER":         100,
    "BORUTA_PERC":             100,
    "BORUTA_ALPHA":            0.05,
    "RF_N_ESTIMATORS":         200,
    "RF_MAX_DEPTH":            5,

    # --- Feature lists (keep in sync with cp_noise_injection.py) ---
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
    d = CONFIG["BASE_OUTPUT_DIR"] / f"{model}_{task}_boruta"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_noise_injection_dir() -> Path:
    """Where cp_noise_injection.py writes its outputs (for cross-comparison)."""
    task  = CONFIG["TASK"]
    model = CONFIG["MODEL_TYPE"].lower()
    return CONFIG["BASE_OUTPUT_DIR"] / f"{model}_{task}_noise_injection"


# =======================================================
# DATA LOADING (mirrors cp_noise_injection.py)
# =======================================================

def load_all_features() -> pd.DataFrame:
    print("1. Loading feature data...")
    hg_df   = pd.read_csv(CONFIG["DATA_DIR"] / CONFIG["PROTEIN_FEATURES_FILE"])
    pair_df = pd.read_csv(CONFIG["DATA_DIR"] / CONFIG["PAIRWISE_FEATURES_FILE"])
    combined = pd.merge(hg_df, pair_df, on='ProteinId', how='inner')
    print(f"   Combined feature shape: {combined.shape}")
    return combined


def load_splits() -> pd.DataFrame:
    meta = get_task_meta()
    print(f"2. Loading splits ({meta['SPLITS_FILE']})...")
    splits_df = pd.read_csv(CONFIG["DATA_DIR"] / meta["SPLITS_FILE"])
    splits_df = splits_df.rename(columns={'UniProt_AC': 'ProteinId'})

    label_map = {'Essential': 1, 'Non-essential': 0,
                 'Drug_target': 1, 'Non_target': 0, 'Unknown': 0}
    splits_df['target'] = splits_df['protein_label'].map(label_map)
    print(f"   Splits: {splits_df['split_index'].nunique()}")
    return splits_df


# =======================================================
# BORUTA RUNNER
# =======================================================

def run_boruta_split(
    split_idx: int,
    merged_df: pd.DataFrame,
    splits_df: pd.DataFrame,
    feature_list: List[str],
) -> pd.DataFrame:
    """
    Run Boruta on a single train/test split. Returns one row per feature
    with the Boruta verdict and ranking for this split.
    """
    split_info = splits_df[splits_df['split_index'] == split_idx][
        ['ProteinId', 'split', 'target', 'label_mask']
    ].copy()
    df = pd.merge(merged_df, split_info, on='ProteinId', how='inner')

    labelled = df[df['label_mask']].copy()
    train_df = labelled[labelled['split'] == 'train']

    X = train_df[feature_list].fillna(0).values
    y = train_df['target'].astype(int).values

    rf = RandomForestClassifier(
        n_estimators=CONFIG["RF_N_ESTIMATORS"],
        max_depth=CONFIG["RF_MAX_DEPTH"],
        class_weight='balanced',
        n_jobs=-1,
        random_state=CONFIG["RANDOM_STATE"] + split_idx,
    )

    boruta = BorutaPy(
        rf,
        n_estimators='auto',
        max_iter=CONFIG["BORUTA_MAX_ITER"],
        perc=CONFIG["BORUTA_PERC"],
        alpha=CONFIG["BORUTA_ALPHA"],
        verbose=0,
        random_state=CONFIG["RANDOM_STATE"] + split_idx,
    )
    boruta.fit(X, y)

    # Build verdict per feature
    rows = []
    for i, feat in enumerate(feature_list):
        if boruta.support_[i]:
            verdict = 'confirmed'
        elif boruta.support_weak_[i]:
            verdict = 'tentative'
        else:
            verdict = 'rejected'
        rows.append({
            'split_index':   split_idx,
            'feature':       feat,
            'verdict':       verdict,
            'boruta_rank':   int(boruta.ranking_[i]),  # 1 = best, ties allowed
        })
    return pd.DataFrame(rows)


# =======================================================
# AGGREGATION
# =======================================================

def aggregate_confirmations(per_split_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per feature: count of confirmed / tentative / rejected verdicts across
    splits, and the confirmation rate (confirmed / N splits).
    """
    n_splits = per_split_df['split_index'].nunique()
    rows = []
    for feat, sub in per_split_df.groupby('feature'):
        counts = sub['verdict'].value_counts()
        n_conf = int(counts.get('confirmed', 0))
        n_tent = int(counts.get('tentative', 0))
        n_rej  = int(counts.get('rejected',  0))
        rows.append({
            'feature':          feat,
            'n_splits':         n_splits,
            'n_confirmed':      n_conf,
            'n_tentative':      n_tent,
            'n_rejected':       n_rej,
            'confirmation_rate': n_conf / n_splits,
            'mean_rank':        sub['boruta_rank'].mean(),
        })
    df = pd.DataFrame(rows)
    return df.sort_values(['confirmation_rate', 'mean_rank'],
                          ascending=[False, True]).reset_index(drop=True)


# =======================================================
# SUMMARY WRITING
# =======================================================

def _format_boruta_table(df: pd.DataFrame) -> str:
    lines = []
    header = (f"  {'Feature':<38} {'Conf':>5} {'Tent':>5} {'Rej':>5}  "
              f"{'Conf rate':>10}  {'Mean rank':>10}")
    lines.append(header)
    lines.append("  " + "-" * 84)
    for _, row in df.iterrows():
        lines.append(
            f"  {row['feature']:<38} "
            f"{row['n_confirmed']:>5} {row['n_tentative']:>5} {row['n_rejected']:>5}  "
            f"{row['confirmation_rate']:>10.2f}  {row['mean_rank']:>10.2f}"
        )
    return "\n".join(lines)


def _load_noise_verdicts() -> Dict[str, pd.DataFrame]:
    """
    Try to load the ranked tables from cp_noise_injection.py for side-by-side
    comparison. Returns {} if the noise-injection outputs aren't present.
    """
    out = {}
    nd  = get_noise_injection_dir()
    for label, fname in [('pairwise',   'noise_ranked_pairwise.csv'),
                         ('hypergraph', 'noise_ranked_hypergraph.csv')]:
        path = nd / fname
        if path.exists():
            out[label] = pd.read_csv(path)
    return out


def write_summary(
    pair_agg:  pd.DataFrame,
    hyper_agg: pd.DataFrame,
    output_path: Path,
):
    meta = get_task_meta()
    sep  = "=" * 90

    lines = [
        sep,
        "  BORUTA ROBUSTNESS CHECK — FEATURE CONFIRMATION ACROSS SPLITS",
        sep,
        f"  Task          : {meta['DISPLAY_NAME']}",
        f"  Model         : {CONFIG['MODEL_TYPE']} (impurity importance)",
        f"  Splits        : {pair_agg['n_splits'].iloc[0]}",
        f"  Boruta config : max_iter={CONFIG['BORUTA_MAX_ITER']}, "
        f"perc={CONFIG['BORUTA_PERC']}, alpha={CONFIG['BORUTA_ALPHA']}",
        "",
        "  Boruta verdict (per split): confirmed = importance significantly above",
        "  the max shadow; tentative = ambiguous; rejected = not above shadow.",
        "  Confirmation rate = fraction of splits where the feature was confirmed.",
        "",
        "  Reference: Kursa & Rudnicki (2010), 'Feature Selection with the Boruta",
        "  Package', Journal of Statistical Software 36(11):1-13.",
        "",
        sep,
        "  PAIRWISE FEATURE SET",
        sep,
        _format_boruta_table(pair_agg),
        "",
        sep,
        "  HYPERGRAPH FEATURE SET",
        sep,
        _format_boruta_table(hyper_agg),
        "",
    ]

    # Cross-comparison with noise-injection rank criterion, if available
    noise_tables = _load_noise_verdicts()
    if noise_tables:
        lines += [
            sep,
            "  CROSS-COMPARISON WITH NOISE-INJECTION RANK CRITERION",
            sep,
            "  Below: side-by-side of Boruta confirmation rate vs noise-injection",
            "  hit rate (where available). Strong agreement = features rated",
            "  similarly by two independent methods.",
            "",
        ]
        for label, agg in [('PAIRWISE', pair_agg), ('HYPERGRAPH', hyper_agg)]:
            key = label.lower()
            if key not in noise_tables:
                continue
            noise_df = noise_tables[key]
            noise_df = noise_df[noise_df['type'] == 'real'][
                ['feature', 'rank', 'hit_rate']
            ] if 'hit_rate' in noise_df.columns else noise_df[noise_df['type'] == 'real'][
                ['feature', 'rank']
            ]
            merged = agg.merge(noise_df, on='feature', how='left')

            lines.append(f"  {label}:")
            hdr = (f"  {'Feature':<38} {'Boruta conf':>11}  "
                   f"{'NI rank':>8}  {'NI hit rate':>11}")
            lines.append(hdr)
            lines.append("  " + "-" * 75)
            for _, r in merged.iterrows():
                ni_rank = r.get('rank', float('nan'))
                ni_hit  = r.get('hit_rate', float('nan'))
                rank_s  = f"{int(ni_rank):>8}" if pd.notna(ni_rank) else f"{'—':>8}"
                hit_s   = f"{ni_hit:>11.2f}" if pd.notna(ni_hit)   else f"{'—':>11}"
                lines.append(
                    f"  {r['feature']:<38} {r['confirmation_rate']:>11.2f}  "
                    f"{rank_s}  {hit_s}"
                )
            lines.append("")

    text = "\n".join(lines)
    output_path.write_text(text)
    print(text)


# =======================================================
# MAIN
# =======================================================

if __name__ == '__main__':

    output_dir = get_output_dir()
    meta       = get_task_meta()

    print(f"\n{'='*70}")
    print("  BORUTA ROBUSTNESS CHECK")
    print(f"  Task   : {meta['DISPLAY_NAME']}")
    print(f"  Model  : {CONFIG['MODEL_TYPE']} (impurity importance)")
    print(f"  Output : {output_dir}")
    print(f"{'='*70}\n")

    start = time.time()

    features_df = load_all_features()
    splits_df   = load_splits()
    split_idxs  = sorted(splits_df['split_index'].unique())

    hypergraph_features = [f for f in CONFIG["FEATURES"]["HYPERGRAPH"]
                           if f in features_df.columns]
    pairwise_features   = [f for f in CONFIG["FEATURES"]["PAIRWISE"]
                           if f in features_df.columns]

    print(f"\n3. Running Boruta on {len(split_idxs)} splits "
          f"× 2 feature sets = {len(split_idxs) * 2} Boruta fits...\n")

    pair_records  = []
    hyper_records = []

    for split_idx in split_idxs:
        print(f"   Split {split_idx:>2}/{len(split_idxs)}...", end=' ', flush=True)
        try:
            pair_records.append(
                run_boruta_split(split_idx, features_df, splits_df, pairwise_features)
            )
            hyper_records.append(
                run_boruta_split(split_idx, features_df, splits_df, hypergraph_features)
            )

            pc = (pair_records[-1]['verdict']  == 'confirmed').sum()
            hc = (hyper_records[-1]['verdict'] == 'confirmed').sum()
            print(f"pair confirmed {pc}/{len(pairwise_features)}   "
                  f"hyper confirmed {hc}/{len(hypergraph_features)}")
        except Exception as e:
            print(f"ERROR: {e}")
            raise

    pair_per_split  = pd.concat(pair_records,  ignore_index=True)
    hyper_per_split = pd.concat(hyper_records, ignore_index=True)
    per_split_all   = pd.concat(
        [pair_per_split.assign(feature_set='pairwise'),
         hyper_per_split.assign(feature_set='hypergraph')],
        ignore_index=True,
    )

    print("\n4. Aggregating confirmation rates...")
    pair_agg  = aggregate_confirmations(pair_per_split)
    hyper_agg = aggregate_confirmations(hyper_per_split)

    print("\n5. Saving outputs...")
    per_split_all.to_csv(output_dir / 'boruta_per_split.csv', index=False)
    pd.concat(
        [pair_agg.assign(feature_set='pairwise'),
         hyper_agg.assign(feature_set='hypergraph')],
        ignore_index=True,
    ).to_csv(output_dir / 'boruta_confirmation_rates.csv', index=False)
    print(f"   Saved: boruta_per_split.csv")
    print(f"   Saved: boruta_confirmation_rates.csv")

    print("\n6. Writing summary...\n")
    write_summary(pair_agg, hyper_agg, output_dir / 'boruta_summary.txt')

    elapsed = time.time() - start
    print(f"\n{'='*70}")
    print(f"  COMPLETE — {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'='*70}")