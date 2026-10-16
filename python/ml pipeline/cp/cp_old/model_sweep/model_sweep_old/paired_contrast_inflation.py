"""
=======================================================================
PAIRED REPRESENTATION CONTRASTS UNDER TWO SPLIT REGIMES
=======================================================================
Companion to split_inflation_analysis.py. That script contrasts PR-AUC
LEVELS between the group-based and protein-level random split regimes.
This one contrasts the quantity the paper actually reports: the paired
per-split difference between representations.

For each split i within an arm, and each contrast

    H1 : hypergraph - pairwise
    H2 : hb_graph   - hypergraph
    H3 : hb_graph   - pairwise

the difference is computed on the SAME split, so it is properly paired.
Reported per (task, model, contrast, arm):

    mean_delta   mean of the 50 paired differences
    sd_delta     sd of those differences
    dz           mean_delta / sd_delta (paired effect size)
    n_pos, n_neg split counts (positive / negative differences)
    p_sign       two-sided exact sign test on those counts

The arms are then compared:

    delta_inflation = mean_delta(random) - mean_delta(group)
    dz_inflation    = dz(random) - dz(group)

with a percentile bootstrap CI on each, resampling the 50 paired
differences within each arm independently. The arms are NOT paired with
each other (different allocation units), so this is a contrast of two
separately estimated quantities and not a significance test of the
difference; say so in the SI.

WHY THIS MATTERS
----------------
Leakage can leave the representation gap almost unchanged while
shrinking the within-arm variance, which inflates dz and the split
counts without inflating the gap itself. Levels alone do not show that.

INPUT
-----
The same sweep CSVs as split_inflation_analysis.py, one file per arm per
task, with columns: model, tier, split_index, pr_auc.

OUTPUT
------
  paired_contrast_long.csv    one row per (task, model, contrast) with
                              both arms side by side and both inflations
  paired_contrast_table.csv   wide, SI-facing: mean_delta and dz in each
                              arm, per model x contrast, blocked by task

paired_contrast_plots.py reads paired_contrast_long.csv.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:                                   # exact sign test if scipy is present
    from scipy.stats import binomtest
    def _sign_p(n_pos: int, n_tot: int) -> float:
        if n_tot == 0:
            return float("nan")
        return float(binomtest(n_pos, n_tot, 0.5, alternative="two-sided").pvalue)
except Exception:                      # fall back to an exact binomial by hand
    from math import comb
    def _sign_p(n_pos: int, n_tot: int) -> float:
        if n_tot == 0:
            return float("nan")
        k = min(n_pos, n_tot - n_pos)
        tail = sum(comb(n_tot, i) for i in range(0, k + 1)) / 2 ** n_tot
        return float(min(1.0, 2 * tail))


# ======================================================================
# CONFIG  — edit here, then run the whole file in the interactive window.
# ======================================================================
CONFIG = {
    "ROOT":          Path("."),
    "TASKS":         ["ess", "hpa", "chembl"],
    "OUT_DIR":       Path("."),
    "GROUP_NAME":    None,   # e.g. "sweep_results.csv"; None = glob sweep_results*.csv
    "RANDOM_NAME":   "sweep_results_randomprotein.csv",
    "RANDOM_SUBDIR": "randomprotein",   # tried if the file is not in cp_<task>/

    # Explicit file lists override discovery; single task only.
    "GROUP_CSV":  None,
    "RANDOM_CSV": None,

    "N_BOOT": 10000,
    "ALPHA":  0.05,
    "SEED":   42,

    # Splits where any tier is missing or failed are dropped from the paired
    # comparison for that model. Set False to keep partial splits out silently.
    "REPORT_DROPPED": True,
}

MODEL_ORDER = [
    "LogisticRegression", "RBF-SVM", "MLP",
    "RandomForest", "XGBoost", "LightGBM", "TabPFN",
]
TIER_ORDER = ["pairwise", "hypergraph", "hb_graph"]
TASK_LABEL = {"ess": "Gene essentiality",
              "hpa": "Drug target (HPA)",
              "chembl": "Drug target (ChEMBL)"}

# contrast key -> (minuend tier, subtrahend tier, label)
CONTRASTS = {
    "H1": ("hypergraph", "pairwise",   "Hypergraph - pairwise"),
    "H2": ("hb_graph",   "hypergraph", "HB-graph - hypergraph"),
    "H3": ("hb_graph",   "pairwise",   "HB-graph - pairwise"),
}
CONTRAST_ORDER = ["H1", "H2", "H3"]

REQUIRED_COLS = {"model", "tier", "split_index", "pr_auc"}


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def load_arm(paths: List[Path], arm_name: str) -> pd.DataFrame:
    """Read one arm's CSV(s); de-duplicate on (model, tier, split_index)."""
    frames = []
    for p in paths:
        if not p.exists():
            print(f"   [warn] {arm_name}: file not found, skipped: {p}")
            continue
        df = pd.read_csv(p)
        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"{p} is missing required columns: {sorted(missing)}")
        frames.append(df)
        print(f"   [{arm_name}] {p.name}: {len(df)} rows, "
              f"{df['model'].nunique()} models, {df['split_index'].nunique()} splits")
    if not frames:
        raise FileNotFoundError(f"No usable {arm_name} CSVs among: {paths}")

    df = pd.concat(frames, ignore_index=True)
    before = len(df)
    df = df.drop_duplicates(subset=["model", "tier", "split_index"], keep="last")
    if len(df) != before:
        print(f"   [{arm_name}] de-duplicated {before - len(df)} rows")
    return df


def resolve_default_paths(root: Path, task: str, group_name: Optional[str],
                          random_name: str, random_subdir: str) -> Dict[str, List[Path]]:
    task_dir = root / f"cp_{task}"
    stem = Path(random_name).stem
    if group_name:
        group_paths = [task_dir / group_name]
    else:
        group_paths = sorted(task_dir.glob("sweep_results*.csv"))
        group_paths = [p for p in group_paths
                       if stem not in p.name and "unstrat" not in p.name]
    candidates = [task_dir / random_name, task_dir / random_subdir / random_name]
    random_paths = [p for p in candidates if p.exists()][:1] or [candidates[0]]
    return {"group": group_paths, "random": random_paths}


# ----------------------------------------------------------------------
# Pairing
# ----------------------------------------------------------------------
def widen(df: pd.DataFrame, arm_name: str, report_dropped: bool) -> pd.DataFrame:
    """One row per (model, split_index), one column per tier.

    Splits missing any tier, or with a NaN in any tier, are dropped for that
    model: a paired difference needs both members of the pair.
    """
    w = df.pivot_table(index=["model", "split_index"], columns="tier",
                       values="pr_auc", aggfunc="last")
    have = [t for t in TIER_ORDER if t in w.columns]
    missing_tiers = set(TIER_ORDER) - set(have)
    if missing_tiers:
        print(f"   [warn] {arm_name}: tiers absent from the data: "
              f"{sorted(missing_tiers)}")
    w = w[have]
    before = len(w)
    w = w.dropna()
    if report_dropped and len(w) != before:
        print(f"   [{arm_name}] dropped {before - len(w)} (model, split) rows "
              f"with an incomplete or failed tier")
    return w.reset_index()


def paired_stats(delta: np.ndarray) -> Dict[str, float]:
    delta = delta[np.isfinite(delta)]
    n = len(delta)
    if n == 0:
        return {"n_splits": 0, "mean_delta": np.nan, "sd_delta": np.nan,
                "dz": np.nan, "n_pos": 0, "n_neg": 0, "p_sign": np.nan}
    sd = delta.std(ddof=1) if n > 1 else np.nan
    n_pos = int((delta > 0).sum())
    n_neg = int((delta < 0).sum())
    return {
        "n_splits":   n,
        "mean_delta": float(delta.mean()),
        "sd_delta":   float(sd) if np.isfinite(sd) else np.nan,
        "dz":         float(delta.mean() / sd) if (np.isfinite(sd) and sd > 0) else np.nan,
        "n_pos":      n_pos,
        "n_neg":      n_neg,
        "p_sign":     _sign_p(n_pos, n_pos + n_neg),
    }


def bootstrap_two_arm_ci(g: np.ndarray, r: np.ndarray, n_boot: int,
                         alpha: float, rng: np.random.Generator) -> Dict[str, float]:
    """CI on (random - group) for both the mean delta and dz.

    Each arm's paired differences are resampled independently. The arms are not
    paired with each other, and both rest on the same protein set, so the
    interval is mildly conservative.
    """
    g = g[np.isfinite(g)]
    r = r[np.isfinite(r)]
    out = {"ci_delta_lo": np.nan, "ci_delta_hi": np.nan,
           "ci_dz_lo": np.nan, "ci_dz_hi": np.nan}
    if len(g) < 2 or len(r) < 2:
        return out

    gb = g[rng.integers(0, len(g), (n_boot, len(g)))]
    rb = r[rng.integers(0, len(r), (n_boot, len(r)))]

    d_diff = rb.mean(axis=1) - gb.mean(axis=1)
    out["ci_delta_lo"] = float(np.percentile(d_diff, 100 * alpha / 2))
    out["ci_delta_hi"] = float(np.percentile(d_diff, 100 * (1 - alpha / 2)))

    with np.errstate(divide="ignore", invalid="ignore"):
        gz = gb.mean(axis=1) / gb.std(axis=1, ddof=1)
        rz = rb.mean(axis=1) / rb.std(axis=1, ddof=1)
    z_diff = rz - gz
    z_diff = z_diff[np.isfinite(z_diff)]
    if len(z_diff):
        out["ci_dz_lo"] = float(np.percentile(z_diff, 100 * alpha / 2))
        out["ci_dz_hi"] = float(np.percentile(z_diff, 100 * (1 - alpha / 2)))
    return out


def analyse_task(task: str, gw: pd.DataFrame, rw: pd.DataFrame,
                 n_boot: int, alpha: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []

    models = sorted(set(gw["model"]) & set(rw["model"]),
                    key=lambda m: (MODEL_ORDER.index(m)
                                   if m in MODEL_ORDER else len(MODEL_ORDER), m))
    only_g = set(gw["model"]) - set(rw["model"])
    only_r = set(rw["model"]) - set(gw["model"])
    if only_g:
        print(f"   [warn] {task}: models in group arm only, skipped: {sorted(only_g)}")
    if only_r:
        print(f"   [warn] {task}: models in random arm only, skipped: {sorted(only_r)}")

    for model in models:
        g = gw[gw["model"] == model]
        r = rw[rw["model"] == model]
        for key in CONTRAST_ORDER:
            hi_t, lo_t, label = CONTRASTS[key]
            if not {hi_t, lo_t} <= set(g.columns) or not {hi_t, lo_t} <= set(r.columns):
                continue
            gd = (g[hi_t] - g[lo_t]).to_numpy(dtype=float)
            rd = (r[hi_t] - r[lo_t]).to_numpy(dtype=float)
            if len(gd) == 0 or len(rd) == 0:
                continue

            gs = paired_stats(gd)
            rs = paired_stats(rd)
            ci = bootstrap_two_arm_ci(gd, rd, n_boot, alpha, rng)

            rec = {"task": task, "model": model,
                   "contrast": key, "contrast_label": label}
            rec.update({f"{k}_group": v for k, v in gs.items()})
            rec.update({f"{k}_random": v for k, v in rs.items()})
            rec["delta_inflation"] = rs["mean_delta"] - gs["mean_delta"]
            rec["dz_inflation"] = rs["dz"] - gs["dz"]
            rec["sd_ratio"] = (gs["sd_delta"] / rs["sd_delta"]
                               if rs["sd_delta"] else np.nan)
            rec.update(ci)
            records.append(rec)
    return pd.DataFrame.from_records(records)


# ----------------------------------------------------------------------
# Wide table for the SI
# ----------------------------------------------------------------------
def make_wide_table(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task in [t for t in TASK_LABEL if t in set(long_df["task"])]:
        sub = long_df[long_df["task"] == task]
        for model in [m for m in MODEL_ORDER if m in set(sub["model"])]:
            for key in CONTRAST_ORDER:
                cell = sub[(sub["model"] == model) & (sub["contrast"] == key)]
                if not len(cell):
                    continue
                v = cell.iloc[0]
                rows.append({
                    "task":      TASK_LABEL.get(task, task),
                    "model":     model,
                    "contrast":  CONTRASTS[key][2],
                    "delta_group":   f"{v['mean_delta_group']:+.3f}",
                    "delta_random":  f"{v['mean_delta_random']:+.3f}",
                    "delta_change":  f"{v['delta_inflation']:+.3f}",
                    "dz_group":      f"{v['dz_group']:.2f}",
                    "dz_random":     f"{v['dz_random']:.2f}",
                    "dz_change":     f"{v['dz_inflation']:+.2f}",
                    "split_counts_group":  f"{int(v['n_pos_group'])}/{int(v['n_neg_group'])}",
                    "split_counts_random": f"{int(v['n_pos_random'])}/{int(v['n_neg_random'])}",
                })
    return pd.DataFrame(rows)


def main():
    C = CONFIG
    if (C["GROUP_CSV"] or C["RANDOM_CSV"]) and len(C["TASKS"]) > 1:
        raise ValueError("GROUP_CSV/RANDOM_CSV apply to a single task; "
                         "put one entry in CONFIG['TASKS'].")
    C["OUT_DIR"].mkdir(parents=True, exist_ok=True)

    all_long = []
    for task in C["TASKS"]:
        print(f"\n{'='*70}\nTASK: {task}\n{'='*70}")
        if C["GROUP_CSV"] or C["RANDOM_CSV"]:
            paths = {"group": C["GROUP_CSV"] or [], "random": C["RANDOM_CSV"] or []}
        else:
            paths = resolve_default_paths(C["ROOT"], task, C["GROUP_NAME"],
                                          C["RANDOM_NAME"], C["RANDOM_SUBDIR"])
            print(f"   group  CSVs: {[str(p) for p in paths['group']]}")
            print(f"   random CSV : {[str(p) for p in paths['random']]}")

        gw = widen(load_arm(paths["group"],  "group"),  "group",  C["REPORT_DROPPED"])
        rw = widen(load_arm(paths["random"], "random"), "random", C["REPORT_DROPPED"])

        long_df = analyse_task(task, gw, rw, C["N_BOOT"], C["ALPHA"], C["SEED"])
        if long_df.empty:
            print(f"   [warn] no usable (model, contrast) cells for {task}")
            continue
        all_long.append(long_df)

        # Console summary — H3, the headline representation contrast.
        h3 = long_df[long_df["contrast"] == "H3"]
        print(f"\n   H3 (HB-graph - pairwise), paired within arm:")
        print(f"     {'model':<20} {'delta_grp':>9} {'delta_rnd':>9} "
              f"{'change':>8} {'dz_grp':>7} {'dz_rnd':>7} {'n+/n- grp':>10} "
              f"{'n+/n- rnd':>10}")
        for _, r in h3.iterrows():
            print(f"     {r['model']:<20} {r['mean_delta_group']:+9.3f} "
                  f"{r['mean_delta_random']:+9.3f} {r['delta_inflation']:+8.3f} "
                  f"{r['dz_group']:7.2f} {r['dz_random']:7.2f} "
                  f"{int(r['n_pos_group'])}/{int(r['n_neg_group']):<8} "
                  f"{int(r['n_pos_random'])}/{int(r['n_neg_random']):<8}")

    if not all_long:
        print("\nNothing to write.")
        return

    long_df = pd.concat(all_long, ignore_index=True)
    long_path = C["OUT_DIR"] / "paired_contrast_long.csv"
    long_df.to_csv(long_path, index=False)
    print(f"\nWrote {long_path}  ({len(long_df)} rows)")

    wide = make_wide_table(long_df)
    wide_path = C["OUT_DIR"] / "paired_contrast_table.csv"
    wide.to_csv(wide_path, index=False)
    print(f"Wrote {wide_path}  ({len(wide)} rows)")

    # The paper claim: the gap is stable, the effect size is not.
    print(f"\n{'='*70}\nACROSS MODELS: GAP vs EFFECT SIZE, BY TASK x CONTRAST"
          f"\n{'='*70}")
    agg = (long_df.groupby(["task", "contrast"])
                  .agg(delta_group=("mean_delta_group", "mean"),
                       delta_random=("mean_delta_random", "mean"),
                       delta_change=("delta_inflation", "mean"),
                       dz_group=("dz_group", "mean"),
                       dz_random=("dz_random", "mean"),
                       dz_change=("dz_inflation", "mean"),
                       sd_ratio=("sd_ratio", "median"),
                       n_models=("model", "nunique"))
                  .reset_index())
    for task in [t for t in TASK_LABEL if t in set(agg["task"])]:
        print(f"\n  {TASK_LABEL.get(task, task)}")
        sub = agg[agg["task"] == task].set_index("contrast")
        for key in CONTRAST_ORDER:
            if key not in sub.index:
                continue
            r = sub.loc[key]
            print(f"    {CONTRASTS[key][2]:<24} "
                  f"gap {r['delta_group']:+.3f} -> {r['delta_random']:+.3f} "
                  f"({r['delta_change']:+.3f});  "
                  f"dz {r['dz_group']:.2f} -> {r['dz_random']:.2f} "
                  f"({r['dz_change']:+.2f});  "
                  f"sd ratio group/random {r['sd_ratio']:.2f}  "
                  f"[{int(r['n_models'])} models]")

    print("\n[note] Within an arm the contrasts are paired across splits, so dz "
          "and the sign test are valid. Between arms the comparison is of two "
          "separately estimated quantities, not a paired test.")


main()