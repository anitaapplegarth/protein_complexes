"""
=======================================================================
GROUP-vs-RANDOM SPLIT INFLATION  —  analysis
=======================================================================
Quantifies how much protein-level random splitting inflates PR-AUC
relative to the structural-group-based splitting used for the headline
results.

WHICH RANDOM ARM THIS IS
------------------------
The random arm here is produced by cp_random_ess_split_protein.py:
proteins are assigned to train/test INDIVIDUALLY and AT RANDOM, with the
draw stratified by label class, and structural groups ignored entirely.
Roughly 24% of structural groups therefore have members on both sides of
the boundary; that leakage is the thing being measured.

This is NOT the earlier "unstrat" arm. That arm kept groups atomic and
only dropped size-bucket stratification, so it was a bucket ablation and
not a leakage baseline. Those files have been moved out of the way and
are not read by this script.

DESIGN NOTE — why this is an UNPAIRED, mean-vs-mean contrast
-----------------------------------------------------------
The two arms use different allocation units (structural groups versus
individual proteins), so split 7 of the group arm and split 7 of the
random arm are genuinely different partitions and share nothing but a
seed lineage. There is no matched pair in which only the train/test
boundary moved.

Consequently the inflation is reported as a difference of arm MEANS:

    inflation(model, task, tier) = mean_random - mean_group

positive => random splitting inflates performance. A bootstrap CI is
placed around that difference of means by resampling the 50 PR-AUC
values within each arm independently. This is NOT a per-split paired
test; it is an interval on the single averaged number reported in the
table. Both arms sit on the same fixed protein set and feature matrix,
so they are positively correlated and resampling them independently
overstates the variance of the difference; the interval is therefore
mildly conservative. State this in the SI.

PREVALENCE NORMALISATION
------------------------
The baseline of PR-AUC is the test positive rate. The group arm holds
that rate only to within tolerance (1.5pp for the drug-target tasks,
4.5pp for essentiality) whereas the random arm stratifies exactly, so a
raw difference of PR-AUC means mixes leakage with a prevalence
difference. If the sweep CSVs carry a test positive rate column (any of
PREVALENCE_COLS below), this script also reports lift = pr_auc /
prevalence and the inflation on that scale. If they do not, it says so
and reports the raw scale only.

INPUT
-----
Each sweep CSV has (at least) columns:
    model, family, tier, split_index, pr_auc, n_features
One file per arm per task, TabPFN included:
Group arm  : <ROOT>/cp_<task>/<GROUP_NAME>, or sweep_results*.csv if
             GROUP_NAME is None
Random arm : <ROOT>/cp_<task>/sweep_results_randomprotein.csv

OUTPUT
------
  split_inflation_long.csv   — one row per (task, model, tier):
        n_group, n_random, mean_group, mean_random, per-arm sd,
        inflation (= mean_random - mean_group), ci_lo, ci_hi, and, where
        prevalence is available, mean_lift_group / mean_lift_random /
        inflation_lift with its own CI.
  split_inflation_table.csv  — wide, publication-facing: inflation per
        model x tier, one block per task, ready to drop into the SI.

The plotting script (split_inflation_plots.py) reads
split_inflation_long.csv.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ======================================================================
# CONFIG  — edit here, then run the whole file in the interactive window.
# ======================================================================
CONFIG = {
    # Root holding cp_<task>/ sweep output dirs. Per task, files are found at:
    #   group  arm : <ROOT>/cp_<task>/sweep_results*.csv
    #   random arm : <ROOT>/cp_<task>/sweep_results_randomprotein.csv, or the
    #                same filename inside <ROOT>/cp_<task>/<RANDOM_SUBDIR>/
    "ROOT":          Path("."),
    "TASKS":         ["ess", "hpa", "chembl"],
    "OUT_DIR":       Path("."),
    "GROUP_NAME":    "sweep_results_svm.csv",   # e.g. "sweep_results.csv"; None = glob sweep_results*.csv
    "RANDOM_NAME":   "sweep_results_svm_randomprotein.csv",
    "RANDOM_SUBDIR": "randomprotein",   # tried if the file is not in cp_<task>/

    # To bypass discovery and name files explicitly, set these (applies to a
    # SINGLE task — put one entry in TASKS). Leave as None to use discovery.
    "GROUP_CSV":  None,   # e.g. [Path("cp_ess/sweep_results_tabpfn_svm.csv"), ...]
    "RANDOM_CSV": None,   # e.g. [Path("cp_ess/sweep_results_randomprotein.csv")]

    # Bootstrap CI on the difference of arm means (mean_random - mean_group).
    "N_BOOT": 10000,
    "ALPHA":  0.05,       # 1 - ALPHA CI
    "SEED":   42,
}

# ----------------------------------------------------------------------
# Presentation order. Models not listed are appended alphabetically;
# tiers not listed are dropped. Edit here, not downstream.
# ----------------------------------------------------------------------
MODEL_ORDER = [
    "LogisticRegression", "RBF-SVM", "MLP",
    "RandomForest", "XGBoost", "LightGBM", "TabPFN",
]
TIER_ORDER  = ["pairwise", "hypergraph", "hb_graph"]
TIER_LABEL  = {"pairwise": "Pairwise", "hypergraph": "Hypergraph",
               "hb_graph": "HB-graph"}
TASK_LABEL  = {"ess": "Gene essentiality",
               "hpa": "Drug target (HPA)",
               "chembl": "Drug target (ChEMBL)"}

REQUIRED_COLS = {"model", "tier", "split_index", "pr_auc"}

# First match wins. Add your sweep's column name here if it differs.
PREVALENCE_COLS = ["test_pos_rate", "test_prevalence", "prevalence",
                   "pos_rate_test", "test_positive_rate", "baseline_pr_auc"]


def find_prevalence_col(df: pd.DataFrame) -> Optional[str]:
    for c in PREVALENCE_COLS:
        if c in df.columns:
            return c
    return None


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def load_arm(paths: List[Path], arm_name: str) -> pd.DataFrame:
    """Read one arm's CSV(s); de-duplicate on (model, tier, split_index).

    Both arms are normally a single file (TabPFN included), so this is usually
    just a read plus validation. The list form and the de-duplication are kept
    as a safety net: if a cell is ever re-run into a second file, the LAST
    occurrence wins.
    """
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
              f"{df['model'].nunique()} models, "
              f"{df['split_index'].nunique()} splits")
    if not frames:
        raise FileNotFoundError(f"No usable {arm_name} CSVs among: {paths}")

    df = pd.concat(frames, ignore_index=True)
    before = len(df)
    df = df.drop_duplicates(subset=["model", "tier", "split_index"], keep="last")
    if len(df) != before:
        print(f"   [{arm_name}] de-duplicated {before - len(df)} rows "
              f"on (model, tier, split_index)")

    # Drop failed cells (NaN pr_auc) but report how many.
    n_nan = int(df["pr_auc"].isna().sum())
    if n_nan:
        print(f"   [{arm_name}] dropping {n_nan} failed cells (NaN pr_auc)")
        df = df[df["pr_auc"].notna()].copy()

    # Lift = pr_auc / test positive rate, where the rate is recorded.
    pcol = find_prevalence_col(df)
    if pcol is not None:
        prev = pd.to_numeric(df[pcol], errors="coerce")
        # accept either a proportion or a percentage
        if prev.max(skipna=True) is not np.nan and prev.max(skipna=True) > 1.5:
            prev = prev / 100.0
        df["lift"] = df["pr_auc"] / prev.replace(0, np.nan)
        print(f"   [{arm_name}] prevalence column '{pcol}': "
              f"mean test positive rate {prev.mean():.4f}")
    else:
        df["lift"] = np.nan

    df["arm"] = arm_name
    return df


# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------
def bootstrap_diff_ci(group_vals: np.ndarray,
                      random_vals: np.ndarray,
                      n_boot: int,
                      alpha: float,
                      rng: np.random.Generator) -> tuple:
    """Percentile bootstrap CI for (mean_random - mean_group).

    Arms resampled independently (they are not paired). Returns (lo, hi).
    Degenerate arms (n < 2) or all-NaN arms return (nan, nan).
    """
    group_vals = group_vals[np.isfinite(group_vals)]
    random_vals = random_vals[np.isfinite(random_vals)]
    if len(group_vals) < 2 or len(random_vals) < 2:
        return (float("nan"), float("nan"))
    ng, nr = len(group_vals), len(random_vals)
    g_boot = group_vals[rng.integers(0, ng, (n_boot, ng))].mean(axis=1)
    r_boot = random_vals[rng.integers(0, nr, (n_boot, nr))].mean(axis=1)
    diffs = r_boot - g_boot
    lo = float(np.percentile(diffs, 100 * alpha / 2))
    hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return (lo, hi)


def analyse_task(task: str,
                 group_df: pd.DataFrame,
                 random_df: pd.DataFrame,
                 n_boot: int,
                 alpha: float,
                 seed: int) -> pd.DataFrame:
    """One long-format row per (model, tier) for a single task."""
    rng = np.random.default_rng(seed)
    records = []

    models = sorted(set(group_df["model"]) & set(random_df["model"]),
                    key=lambda m: (MODEL_ORDER.index(m)
                                   if m in MODEL_ORDER else len(MODEL_ORDER), m))
    only_g = set(group_df["model"]) - set(random_df["model"])
    only_r = set(random_df["model"]) - set(group_df["model"])
    if only_g:
        print(f"   [warn] {task}: models in group arm only, skipped: {sorted(only_g)}")
    if only_r:
        print(f"   [warn] {task}: models in random arm only, skipped: {sorted(only_r)}")

    for model in models:
        for tier in TIER_ORDER:
            gm = (group_df["model"] == model) & (group_df["tier"] == tier)
            rm = (random_df["model"] == model) & (random_df["tier"] == tier)
            g = group_df.loc[gm, "pr_auc"].to_numpy()
            r = random_df.loc[rm, "pr_auc"].to_numpy()
            if len(g) == 0 or len(r) == 0:
                continue
            mean_g, mean_r = g.mean(), r.mean()
            lo, hi = bootstrap_diff_ci(g, r, n_boot, alpha, rng)

            rec = {
                "task":        task,
                "model":       model,
                "tier":        tier,
                "n_group":     len(g),
                "n_random":    len(r),
                "mean_group":  mean_g,
                "std_group":   g.std(ddof=1) if len(g) > 1 else float("nan"),
                "mean_random": mean_r,
                "std_random":  r.std(ddof=1) if len(r) > 1 else float("nan"),
                "inflation":   mean_r - mean_g,   # random - group
                "ci_lo":       lo,
                "ci_hi":       hi,
            }

            # Prevalence-normalised scale, where available.
            gl = group_df.loc[gm, "lift"].to_numpy(dtype=float)
            rl = random_df.loc[rm, "lift"].to_numpy(dtype=float)
            if np.isfinite(gl).any() and np.isfinite(rl).any():
                l_lo, l_hi = bootstrap_diff_ci(gl, rl, n_boot, alpha, rng)
                rec.update({
                    "mean_lift_group":  np.nanmean(gl),
                    "mean_lift_random": np.nanmean(rl),
                    "inflation_lift":   np.nanmean(rl) - np.nanmean(gl),
                    "ci_lift_lo":       l_lo,
                    "ci_lift_hi":       l_hi,
                })
            else:
                rec.update({
                    "mean_lift_group":  float("nan"),
                    "mean_lift_random": float("nan"),
                    "inflation_lift":   float("nan"),
                    "ci_lift_lo":       float("nan"),
                    "ci_lift_hi":       float("nan"),
                })
            records.append(rec)
    return pd.DataFrame.from_records(records)


# ----------------------------------------------------------------------
# Wide table for the SI
# ----------------------------------------------------------------------
def make_wide_table(long_df: pd.DataFrame) -> pd.DataFrame:
    """Model x tier inflation, blocked by task. Values are mean_random-mean_group."""
    rows = []
    for task in [t for t in TASK_LABEL if t in set(long_df["task"])]:
        sub = long_df[long_df["task"] == task]
        for model in [m for m in MODEL_ORDER if m in set(sub["model"])]:
            row = {"task": TASK_LABEL.get(task, task), "model": model}
            for tier in TIER_ORDER:
                cell = sub[(sub["model"] == model) & (sub["tier"] == tier)]
                if len(cell):
                    v = cell.iloc[0]
                    row[TIER_LABEL[tier]] = f"{v['inflation']:+.3f}"
                else:
                    row[TIER_LABEL[tier]] = ""
            rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Path discovery
# ----------------------------------------------------------------------
def resolve_default_paths(root: Path, task: str, group_name: Optional[str],
                          random_name: str, random_subdir: str) -> Dict[str, List[Path]]:
    """Group arm: the single group_name file, or every sweep_results*.csv in
    cp_<task>/ except the random one if group_name is None.

    Random arm: the single random_name file, looked for in cp_<task>/ first and
    then in cp_<task>/<random_subdir>/.
    """
    task_dir = root / f"cp_{task}"
    stem = Path(random_name).stem

    if group_name:
        group_paths = [task_dir / group_name]
    else:
        group_paths = sorted(task_dir.glob("sweep_results*.csv"))
        # Never let the random-arm file (or any leftover unstrat file) into the
        # group arm, whichever directory layout is in use.
        group_paths = [p for p in group_paths
                       if stem not in p.name and "unstrat" not in p.name]

    candidates = [task_dir / random_name, task_dir / random_subdir / random_name]
    random_paths = [p for p in candidates if p.exists()][:1]
    if not random_paths:
        random_paths = [candidates[0]]   # let load_arm report the miss clearly

    return {"group": group_paths, "random": random_paths}


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
            paths = {"group": C["GROUP_CSV"] or [],
                     "random": C["RANDOM_CSV"] or []}
        else:
            paths = resolve_default_paths(C["ROOT"], task, C["GROUP_NAME"],
                                          C["RANDOM_NAME"], C["RANDOM_SUBDIR"])
            print(f"   group  CSVs: {[str(p) for p in paths['group']]}")
            print(f"   random CSV : {[str(p) for p in paths['random']]}")

        group_df  = load_arm(paths["group"],  "group")
        random_df = load_arm(paths["random"], "random")

        long_df = analyse_task(task, group_df, random_df,
                               C["N_BOOT"], C["ALPHA"], C["SEED"])
        if long_df.empty:
            print(f"   [warn] no overlapping (model, tier) cells for {task}")
            continue
        all_long.append(long_df)

        # Console summary — hb-graph tier, matches the paper's headline number.
        hb = long_df[long_df["tier"] == "hb_graph"].sort_values("inflation")
        print(f"\n   Inflation (mean_random - mean_group), HB-graph tier:")
        for _, r in hb.iterrows():
            print(f"     {r['model']:<20} {r['inflation']:+.3f}  "
                  f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]  "
                  f"(group {r['mean_group']:.3f} -> random {r['mean_random']:.3f})")

    if not all_long:
        print("\nNothing to write.")
        return

    long_df = pd.concat(all_long, ignore_index=True)
    long_path = C["OUT_DIR"] / "split_inflation_long.csv"
    long_df.to_csv(long_path, index=False)
    print(f"\nWrote {long_path}  ({len(long_df)} rows)")

    wide = make_wide_table(long_df)
    wide_path = C["OUT_DIR"] / "split_inflation_table.csv"
    wide.to_csv(wide_path, index=False)
    print(f"Wrote {wide_path}  ({len(wide)} rows)")

    # Cross-model mean inflation per task x tier — the one-line paper claim.
    print(f"\n{'='*70}\nMEAN INFLATION ACROSS MODELS (per task x tier)\n{'='*70}")
    agg = (long_df.groupby(["task", "tier"])["inflation"]
                  .agg(["mean", "std", "count"]).reset_index())
    agg["tier"] = pd.Categorical(agg["tier"], TIER_ORDER, ordered=True)
    for task in [t for t in TASK_LABEL if t in set(agg["task"])]:
        print(f"\n  {TASK_LABEL.get(task, task)}")
        for _, r in agg[agg["task"] == task].sort_values("tier").iterrows():
            print(f"    {TIER_LABEL[r['tier']]:<12} "
                  f"{r['mean']:+.3f}  (sd {r['std']:.3f} over {int(r['count'])} models)")

    # Same, on the prevalence-normalised scale, if the sweep recorded a rate.
    if long_df["inflation_lift"].notna().any():
        print(f"\n{'='*70}\nMEAN INFLATION, PREVALENCE-NORMALISED (lift = PR-AUC / "
              f"test positive rate)\n{'='*70}")
        agg_l = (long_df.dropna(subset=["inflation_lift"])
                        .groupby(["task", "tier"])["inflation_lift"]
                        .agg(["mean", "std", "count"]).reset_index())
        agg_l["tier"] = pd.Categorical(agg_l["tier"], TIER_ORDER, ordered=True)
        for task in [t for t in TASK_LABEL if t in set(agg_l["task"])]:
            print(f"\n  {TASK_LABEL.get(task, task)}")
            for _, r in agg_l[agg_l["task"] == task].sort_values("tier").iterrows():
                print(f"    {TIER_LABEL[r['tier']]:<12} "
                      f"{r['mean']:+.3f}  (sd {r['std']:.3f} over "
                      f"{int(r['count'])} models)")
    else:
        print("\n[note] No test positive rate column found in the sweep CSVs, so "
              "the raw PR-AUC scale is all that is reported. The two arms control "
              "prevalence differently, so add one of "
              f"{PREVALENCE_COLS[:3]} to the sweep output if you want the "
              "normalised figure for the SI.")


main()