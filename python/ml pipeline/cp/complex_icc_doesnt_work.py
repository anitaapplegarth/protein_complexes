# %% [markdown]
# Complex-level ICC for binary labels (essentiality, drug-target)
# -------------------------------------------------------------------
# Question it answers: how much of a binary label's variance sits BETWEEN
# complexes versus WITHIN them? High ICC => the label is largely a property
# of the complex (the hyperedge). Low ICC => it's a property of the individual
# protein (the node).
#
# Two estimators are reported:
#   1. ANOVA-type ICC(1)  - treats the 0/1 label as continuous; fast, robust,
#      what gave the ~0.54 essentiality figure. Can go slightly negative -> read
#      as "≈ 0" (within-complex variance exceeds between-complex).
#   2. Latent-scale logistic ICC - the principled version for a binary label,
#      from a null random-intercept logistic model: sigma_u^2 / (sigma_u^2 + pi^2/3).
#      Slower; needs statsmodels. Set FIT_LOGISTIC_ICC=False to skip.
#
# A permutation null (shuffle labels across proteins, recompute) tells you
# whether the observed ICC is above what complex sizes alone would produce.
#
# ASSUMED INPUT: one CSV per task, one row per (complex, protein) membership
# pair, carrying that protein's binary label. If your labels live in a separate
# file from membership, use the optional join block below.

# %% Config ----------------------------------------------------------------
from pathlib import Path

DATA_DIR = Path("../../../data/lookup_tables/")   # folder containing the CSVs

# task name -> CSV filename (edit to your actual files)
TASKS = {
    "essentiality":        "lu_essentiality_protein.csv",
    "drug_target_hpa":     "cp_drug_target_hpa.csv",
    "drug_target_chembl":  "cp_drug_target_chembl_single.csv",
}

COL_COMPLEX = "complex_id"   # column identifying the complex
COL_PROTEIN = "protein_id"   # column identifying the protein
COL_LABEL   = "label"        # binary 0/1 column

SINGLE_COMPLEX_ONLY = True   # clean first pass: keep only proteins in exactly 1 complex
MIN_GROUP_SIZE      = 2      # drop complexes smaller than this
SIZE_BUCKETS        = [(2, 3), (4, 6), (7, 10), (11, 10_000)]  # inclusive (min, max)
N_PERM              = 500    # permutation-null iterations
RANDOM_SEED         = 0
FIT_LOGISTIC_ICC    = True   # latent-scale logistic ICC (slower, more principled)

# Optional: if labels are in a SEPARATE file from membership, set these and the
# join runs automatically. Leave MEMBERSHIP_FILE = None to use the per-task files above.
MEMBERSHIP_FILE = None       # e.g. "cp_membership.csv" with COL_COMPLEX, COL_PROTEIN

# %% Imports ---------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 12

rng = np.random.default_rng(RANDOM_SEED)

# %% Estimators ------------------------------------------------------------
def anova_icc(df, gcol, ycol):
    """One-way random-effects ICC(1), 0/1 label treated as continuous.
    ICC = (MSB - MSW) / (MSB + (n0 - 1) * MSW)."""
    groups = df.groupby(gcol)[ycol]
    k = groups.ngroups
    if k < 2:
        return np.nan
    ni = groups.size().values
    N = ni.sum()
    grand = df[ycol].mean()
    gmeans = groups.mean()
    ssb = float((ni * (gmeans.values - grand) ** 2).sum())
    msb = ssb / (k - 1)
    ssw = float(((df[ycol].values - df[gcol].map(gmeans).values) ** 2).sum())
    dfw = N - k
    msw = ssw / dfw if dfw > 0 else np.nan
    n0 = (N - (ni ** 2).sum() / N) / (k - 1)   # adjusted average group size
    denom = msb + (n0 - 1) * msw
    return (msb - msw) / denom if denom and denom > 0 else np.nan

def logistic_icc(df, gcol, ycol):
    """Latent-scale ICC from a null random-intercept logistic model."""
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    d = df[[gcol, ycol]].copy()
    d.columns = ["grp", "y"]
    model = BinomialBayesMixedGLM.from_formula("y ~ 1", {"grp": "0 + C(grp)"}, d)
    res = model.fit_vb()
    sd = float(np.exp(res.vcp_mean[0]))   # variance-component params are on log-SD scale
    var_u = sd ** 2
    return var_u / (var_u + np.pi ** 2 / 3)

def perm_null(df, gcol, ycol, n_perm, rng):
    """Shuffle labels across rows; recompute ANOVA ICC each time."""
    y = df[ycol].values.copy()
    tmp = df[[gcol]].copy()
    null = np.empty(n_perm)
    for i in range(n_perm):
        tmp["_y"] = rng.permutation(y)
        null[i] = anova_icc(tmp, gcol, "_y")
    return null

# %% Load + prep -----------------------------------------------------------
def load_task(path):
    df = pd.read_csv(DATA_DIR / path)
    if MEMBERSHIP_FILE is not None:              # join labels onto membership
        mem = pd.read_csv(DATA_DIR / MEMBERSHIP_FILE)[[COL_COMPLEX, COL_PROTEIN]]
        df = mem.merge(df[[COL_PROTEIN, COL_LABEL]], on=COL_PROTEIN, how="inner")
    df = df[[COL_COMPLEX, COL_PROTEIN, COL_LABEL]].dropna()
    df[COL_LABEL] = df[COL_LABEL].astype(int)
    if SINGLE_COMPLEX_ONLY:
        n_cx = df.groupby(COL_PROTEIN)[COL_COMPLEX].nunique()
        df = df[df[COL_PROTEIN].isin(n_cx[n_cx == 1].index)]
    sizes = df.groupby(COL_COMPLEX)[COL_PROTEIN].transform("size")
    return df[sizes >= MIN_GROUP_SIZE].copy()

# %% Run -------------------------------------------------------------------
rows = []
for task, path in TASKS.items():
    df = load_task(path)
    icc = anova_icc(df, COL_COMPLEX, COL_LABEL)
    null = perm_null(df, COL_COMPLEX, COL_LABEL, N_PERM, rng)
    p = (np.sum(null >= icc) + 1) / (N_PERM + 1)     # one-sided
    row = {
        "task": task,
        "n_pairs": len(df),
        "n_complexes": int(df[COL_COMPLEX].nunique()),
        "prevalence": round(float(df[COL_LABEL].mean()), 3),
        "ICC_anova": round(float(icc), 3),
        "perm_p": round(float(p), 4),
        "null_mean": round(float(np.nanmean(null)), 3),
    }
    if FIT_LOGISTIC_ICC:
        try:
            row["ICC_logistic"] = round(float(logistic_icc(df, COL_COMPLEX, COL_LABEL)), 3)
        except Exception as e:
            row["ICC_logistic"] = np.nan
            print(f"[{task}] logistic ICC failed: {e}")
    rows.append(row)

    print(f"\n{task}: ICC={row['ICC_anova']:.3f}  perm_p={row['perm_p']}  "
          f"(complexes={row['n_complexes']}, pairs={row['n_pairs']}, "
          f"prevalence={row['prevalence']})")
    for lo, hi in SIZE_BUCKETS:                        # ICC per size bucket
        sizes = df.groupby(COL_COMPLEX)[COL_PROTEIN].transform("size")
        sub = df[(sizes >= lo) & (sizes <= hi)]
        if sub[COL_COMPLEX].nunique() >= 2:
            print(f"    size {lo:>2}-{hi:<5}: ICC={anova_icc(sub, COL_COMPLEX, COL_LABEL):+.3f} "
                  f"(complexes={sub[COL_COMPLEX].nunique()}, pairs={len(sub)})")

res = pd.DataFrame(rows)
print("\n" + res.to_string(index=False))