"""
=======================================================================
CHECK: are flag_HasStoich and stoich_MedianRatio proxies for complex size?
=======================================================================

The question this answers
    Under logistic regression, CORUM shows a large stoichiometry gain that
    the Random Forest did not, and in every CORUM hb-graph model the effect
    is carried almost entirely by ONE feature: stoich_MedianRatio (mean
    permutation importance 0.106 HPA / 0.117 ChEMBL, with the other five
    stoichiometry features at or near zero).

    stoich_MedianRatio is also the feature flag_HasStoich is derived from
    (flag = ratio > 0), and it is exactly 0 for 89.5% of CORUM proteins.
    Meanwhile curation coverage falls steeply with complex size: 9.1% of
    dimer incidences down to 0.2% for 11+ subunit complexes.

    So the worry is that these two variables are not measuring copy number
    at all, but acting as a recoding of complex size — one a linear model
    cannot otherwise express (the size relationship is steeply nonlinear)
    but that a tree gets for free from protein_MedComplexNodes. That would
    explain both why LR gains from them and why RF did not.

What it reports
    1. Correlations (Spearman, and Pearson where meaningful) between the two
       stoichiometry variables and the size / degree features.
    2. HOW WELL COMPLEX SIZE ALONE PREDICTS THE FLAG. A univariate logistic
       regression of flag ~ protein_MedComplexNodes, scored by ROC-AUC on
       held-out folds. This is the headline number:
         AUC ~ 0.5  -> the flag is independent of size; Test A is clean
         AUC ~ 0.7  -> substantial overlap, worth reporting
         AUC > 0.8  -> the flag is largely a size indicator, and the
                       annotation-presence term cannot be cleanly separated
                       from the complex-size signal
       Also reported for the graded stoich_MedianRatio via Spearman.
    3. THE GRADED PART, NET OF PRESENCE. Restricting to curated proteins
       only (ratio > 0), does stoich_MedianRatio still track size? If the
       whole relationship lives in the zero/non-zero split, the feature is
       a presence indicator; if it persists among curated proteins, it is
       carrying genuine graded information.
    4. Incremental R^2: how much of stoich_MedianRatio's variance is left
       once the hypergraph size features are regressed out.
    5. The same diagnostics for CP, as a contrast — CP curates ~60% of
       incidences with a much flatter size gradient, so if the size-proxy
       reading is right, CP should show markedly weaker coupling.

USAGE
    Edit the PATHS block, then:  python check_flag_size_confound.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr, pearsonr, mannwhitneyu

# =======================================================
# PATHS — edit these
# =======================================================
BASE = Path("/Users/anitaapplegarth/github/dphil/protein_complexes/"
            "data/lookup_tables")
PATHS = {
    "CP":    BASE / "cp",
    "CORUM": BASE / "corum",
}
HG_FILE   = "hypergraph_features.csv"
PAIR_FILE = "pairwise_features.csv"

FLAG_SOURCE = 'stoich_MedianRatio'
FLAG        = 'flag_HasStoich'

# Size / membership features the flag might be proxying for.
SIZE_FEATURES = [
    'protein_MedComplexNodes',
    'protein_RangeComplexNodes',
    'protein_RangeUniqueRatio',
    'base_Degree',
    'base_UniquePartners',
    'base_TriangleCount',
    'base_AvgNeighbourDegree',
    'base_LocalClustCoeff',
]

# The other stoichiometry features, for context.
OTHER_STOICH = [
    'stoich_WeightedTriangles',
    'stoich_AvgNeighbourDegreeStoich',
    'stoich_RangeComplexSize',
    'stoich_MedComplexSize',
    'stoich_RangeRatio',
]

RANDOM_STATE = 42


def load(db_dir: Path) -> pd.DataFrame:
    hg   = pd.read_csv(db_dir / HG_FILE)
    pair = pd.read_csv(db_dir / PAIR_FILE)
    df = pd.merge(hg, pair, on='ProteinId', how='inner')
    if FLAG_SOURCE not in df.columns:
        raise KeyError(f"'{FLAG_SOURCE}' not in {db_dir / HG_FILE}")
    df[FLAG] = (df[FLAG_SOURCE] > 0).astype(int)
    return df


def cv_auc(X: pd.DataFrame, y: pd.Series) -> float:
    """Out-of-fold ROC-AUC for a standardised univariate/multivariate logit."""
    if y.nunique() < 2:
        return float('nan')
    pipe = Pipeline([("s", StandardScaler()),
                     ("c", LogisticRegression(max_iter=5000,
                                              class_weight='balanced',
                                              random_state=RANDOM_STATE))])
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    proba = cross_val_predict(pipe, X, y, cv=cv, method='predict_proba')[:, 1]
    return roc_auc_score(y, proba)


def analyse(name: str, db_dir: Path):
    print(f"\n{'='*74}")
    print(f"  {name}")
    print(f"{'='*74}")

    df = load(db_dir)
    present = [f for f in SIZE_FEATURES if f in df.columns]
    missing = [f for f in SIZE_FEATURES if f not in df.columns]
    if missing:
        print(f"  NOTE: not found, skipped: {missing}")

    n = len(df)
    n_flag = int(df[FLAG].sum())
    print(f"\n  Proteins: {n:,}    flagged: {n_flag:,} ({100*n_flag/n:.1f}%)")

    # ---------- 1. Correlations ------------------------------------------
    print(f"\n  {'-'*70}")
    print(f"  CORRELATION WITH SIZE / MEMBERSHIP FEATURES  (Spearman rho)")
    print(f"  {'-'*70}")
    print(f"  {'Feature':<32} {'vs flag':>10} {'vs ratio':>10} "
          f"{'vs ratio|cur':>14}")
    print(f"  {'-'*70}")

    cur = df[df[FLAG] == 1]
    for f in present:
        r_flag, _ = spearmanr(df[FLAG], df[f])
        r_ratio, _ = spearmanr(df[FLAG_SOURCE], df[f])
        if len(cur) > 10 and cur[FLAG_SOURCE].nunique() > 1:
            r_cur, _ = spearmanr(cur[FLAG_SOURCE], cur[f])
        else:
            r_cur = float('nan')
        print(f"  {f:<32} {r_flag:>+10.3f} {r_ratio:>+10.3f} {r_cur:>+14.3f}")

    print(f"\n  'vs ratio|cur' restricts to the {len(cur):,} curated proteins, so it")
    print(f"  measures the GRADED part of the feature net of presence. If the first")
    print(f"  two columns are large and the third is near zero, the variable is a")
    print(f"  presence/size indicator rather than a copy-number measurement.")

    # ---------- 2. Can size alone predict the flag? -----------------------
    print(f"\n  {'-'*70}")
    print(f"  HOW WELL DOES COMPLEX SIZE PREDICT THE FLAG?")
    print(f"  {'-'*70}")
    print(f"  (out-of-fold ROC-AUC, 5-fold, standardised logistic regression)")

    y = df[FLAG]
    single = 'protein_MedComplexNodes'
    if single in df.columns:
        auc1 = cv_auc(df[[single]], y)
        print(f"\n  {single} alone        : AUC = {auc1:.3f}")
    aucA = cv_auc(df[present], y)
    print(f"  all {len(present)} hypergraph features   : AUC = {aucA:.3f}")

    print(f"\n  Reading: 0.5 = flag independent of structure (Test A clean);")
    print(f"           0.7 = substantial overlap, worth reporting;")
    print(f"           >0.8 = the flag is largely recoverable from size, so the")
    print(f"                  annotation-presence term cannot be cleanly separated")
    print(f"                  from the complex-size signal already in the hypergraph.")

    # ---------- 3. Size difference between flagged and unflagged ----------
    if single in df.columns:
        a = df.loc[df[FLAG] == 1, single]
        b = df.loc[df[FLAG] == 0, single]
        if len(a) > 1 and len(b) > 1:
            u, p = mannwhitneyu(a, b, alternative='two-sided')
            # rank-biserial effect size
            rb = 1 - 2 * u / (len(a) * len(b))
            print(f"\n  {single}:")
            print(f"    flagged   median = {a.median():.2f}  (mean {a.mean():.2f})")
            print(f"    unflagged median = {b.median():.2f}  (mean {b.mean():.2f})")
            print(f"    Mann-Whitney p = {p:.3g}   rank-biserial = {rb:+.3f}")

    # ---------- 4. Residual variance of the ratio -------------------------
    print(f"\n  {'-'*70}")
    print(f"  HOW MUCH OF stoich_MedianRatio IS *NOT* EXPLAINED BY SIZE?")
    print(f"  {'-'*70}")
    for label, sub in [("all proteins", df), ("curated only", cur)]:
        if len(sub) < 20 or sub[FLAG_SOURCE].nunique() < 2:
            print(f"  {label:<16}: too few / constant, skipped")
            continue
        lr = LinearRegression().fit(sub[present], sub[FLAG_SOURCE])
        r2 = lr.score(sub[present], sub[FLAG_SOURCE])
        print(f"  {label:<16}: R^2 = {r2:.3f}  -> {100*(1-r2):.0f}% of its "
              f"variance is independent of the size features")

    # ---------- 5. Other stoichiometry features, for contrast -------------
    others = [f for f in OTHER_STOICH if f in df.columns]
    if others:
        print(f"\n  {'-'*70}")
        print(f"  OTHER STOICHIOMETRY FEATURES (Spearman vs {single})")
        print(f"  {'-'*70}")
        for f in others:
            r, _ = spearmanr(df[f], df[single])
            r_c = (spearmanr(cur[f], cur[single])[0]
                   if len(cur) > 10 else float('nan'))
            print(f"  {f:<34} all {r:>+7.3f}    curated {r_c:>+7.3f}")

    return {
        'database': name, 'n_proteins': n,
        'pct_flagged': 100 * n_flag / n,
        'auc_size_predicts_flag': aucA,
        'auc_medcomplexnodes_alone': auc1 if single in df.columns else np.nan,
    }


if __name__ == "__main__":
    rows = []
    for name, d in PATHS.items():
        if not (d / HG_FILE).exists():
            print(f"\n  {name}: {d / HG_FILE} not found — skipping.")
            continue
        rows.append(analyse(name, d))

    if len(rows) > 1:
        print(f"\n\n{'='*74}")
        print("  SIDE BY SIDE")
        print(f"{'='*74}\n")
        out = pd.DataFrame(rows).set_index('database').T
        with pd.option_context('display.float_format', '{:,.3f}'.format):
            print(out.to_string())
        out.T.to_csv('flag_size_confound.csv')
        print(f"\n  Saved: flag_size_confound.csv")
        print(f"\n  If the AUC is much higher for CORUM than for CP, the CORUM")
        print(f"  stoichiometry gain under logistic regression is most likely a")
        print(f"  size recoding rather than a copy-number effect.")