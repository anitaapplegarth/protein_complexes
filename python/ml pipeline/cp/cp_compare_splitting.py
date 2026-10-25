"""
=======================================================================
SPLITTING-SCHEME COMPARISON  —  structure-aware vs random, CP database
=======================================================================
Compares the model sweep run under two splitting schemes:

  structure-aware : group-atomic splits, structurally similar proteins
                    grouped and assigned to train or test as a unit
  random          : proteins assigned to train/test at random

The gap between them estimates how much structural leakage inflates
performance. Two questions are asked, and they need different handling:

  (a) ABSOLUTE performance. How much does random splitting inflate
      PR-AUC? Reported per (task, model, tier).

  (b) The REPRESENTATION EFFECT. Does the gain from each step of the
      ladder survive? This is the one that matters for the claim.
      Reported per (task, model, step).

      The hypergraph and hb-graph features are derived from complex
      membership, so under random splitting two proteins in the same
      complex can straddle train and test with near-identical features.
      Pairwise features are far less exposed to this. Leakage should
      therefore inflate the upper tiers more than the pairwise tier, and
      so inflate the step differences too. If the steps are unchanged,
      the representation advantage is not a leakage artefact.

PAIRING. Within a scheme the tiers are paired (same split, same
proteins), so step differences are computed per split as before. ACROSS
schemes there is no pairing: split 7 is a different partition in each.
Cross-scheme comparisons here are therefore UNPAIRED (Welch and
Mann-Whitney).

CAVEAT ON THE p-VALUES. The 50 splits within a scheme share the same
protein set and have overlapping training sets, so they are not
independent draws. Standard errors computed from them understate the
true uncertainty, and the cross-scheme p-values below are optimistic.
They are reported as a descriptive guide, not as strict tests. The
effect sizes and the figure carry the argument.

Outputs (to OUTPUT_DIR):
  fig_splitting_comparison.png   two-panel scatter, see above
  splitting_absolute.csv         per (task, model, tier) optimism
  splitting_steps.csv            per (task, model, step) effect change
  splitting_checks.csv           prevalence / size / coverage checks
  si_table_splitting.tex         both LaTeX tables
=======================================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, ttest_ind

# =======================================================================
# CONFIGURATION
# =======================================================================
CONFIG = {
    # Root holding the per-task directories.
    "SWEEP_ROOT": Path("./model_sweep"),

    # <SWEEP_ROOT>/<dir>/<RESULTS_STRUCTURED>
    # <SWEEP_ROOT>/<dir>/<RANDOM_SUBDIR>/<RESULTS_RANDOM>
    "RESULTS_STRUCTURED": "sweep_results.csv",
    "RESULTS_RANDOM":     "sweep_results_randomprotein.csv",
    "PREDS_STRUCTURED":   "predictions.csv",
    "PREDS_RANDOM":       "predictions_randomprotein.csv",
    "RANDOM_SUBDIR":      "randomprotein",

    "OUTPUT_DIR": Path("./figs/paper"),

    # Skip the prevalence check if the predictions files are large or
    # absent. Without it, class balance between schemes is unverified.
    "CHECK_PREVALENCE": True,

    # Include the total step (hb-graph - pairwise) in the step figure.
    # Its point estimate is the sum of the other two, so it adds points
    # without adding independent information.
    "INCLUDE_TOTAL_IN_FIGURE": False,

    # Force zero into panel (b)'s range. Keeps the sign of each effect
    # visible; set False if every effect sits far from zero and the
    # panel ends up mostly empty space.
    "ZERO_IN_STEP_AXIS": True,
}

TASKS = {
    "essentiality": dict(dir="cp_ess",    label="Gene essentiality"),
    "chembl":       dict(dir="cp_chembl", label="Drug target (ChEMBL)"),
    "hpa":          dict(dir="cp_hpa",    label="Drug target (HPA)"),
}
TASK_ORDER = ["essentiality", "chembl", "hpa"]

TIERS = ["pairwise", "hypergraph", "hb_graph"]
TIER_LABEL = {"pairwise": "Pairwise", "hypergraph": "Hypergraph",
              "hb_graph": "Hb-graph"}

# The two nested steps, plus the total.
STEPS = {
    "hyper_minus_pair": ("hypergraph", "pairwise",  "Hypergraph $-$ Pairwise"),
    "hb_minus_hyper":   ("hb_graph",   "hypergraph", "Hb-graph $-$ Hypergraph"),
    "hb_minus_pair":    ("hb_graph",   "pairwise",  "Hb-graph $-$ Pairwise"),
}
FIGURE_STEPS = (["hyper_minus_pair", "hb_minus_hyper", "hb_minus_pair"]
                if CONFIG["INCLUDE_TOTAL_IN_FIGURE"]
                else ["hyper_minus_pair", "hb_minus_hyper"])

# Tier colours match the other sweep figures.
TIER_COLOUR = {"pairwise": "#999999", "hypergraph": "#87CEEB",
               "hb_graph": "#4682B4"}

STEP_COLOUR = {"hyper_minus_pair": "#6ec6e8",
               "hb_minus_hyper":   "#8ccf7e",
               "hb_minus_pair":    "#5b2c6f"}
TASK_MARKER = {"essentiality": "o", "chembl": "s", "hpa": "^"}

plt.rcParams.update({
    "font.size":        13,
    "axes.titlesize":   15,
    "axes.labelsize":   14,
    "xtick.labelsize":  13,
    "ytick.labelsize":  13,
    "legend.fontsize":  12,
    "figure.titlesize": 16,
})


# =======================================================================
# LOADING
# =======================================================================
def _paths(task_key: str) -> dict:
    d = CONFIG["SWEEP_ROOT"] / TASKS[task_key]["dir"]
    return {
        "structured_results": d / CONFIG["RESULTS_STRUCTURED"],
        "random_results":     d / CONFIG["RANDOM_SUBDIR"] / CONFIG["RESULTS_RANDOM"],
        "structured_preds":   d / CONFIG["PREDS_STRUCTURED"],
        "random_preds":       d / CONFIG["RANDOM_SUBDIR"] / CONFIG["PREDS_RANDOM"],
    }


def load_results(path: Path, task: str, scheme: str) -> pd.DataFrame:
    """One long-format results file. De-duplicates resumed rows."""
    if not path.exists():
        raise SystemExit(f"Results file not found: {path}")
    df = pd.read_csv(path)
    before = len(df)
    df = df.drop_duplicates(subset=["model", "tier", "split_index"],
                            keep="last")
    if len(df) < before:
        print(f"   {task}/{scheme}: dropped {before - len(df)} duplicate "
              f"rows from resumed runs.")
    n_nan = int(df["pr_auc"].isna().sum())
    if n_nan:
        print(f"   {task}/{scheme}: WARNING {n_nan} cells have no PR-AUC "
              f"(failed fits); the affected splits are dropped.")
    df["task"] = task
    df["scheme"] = scheme
    return df


def to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (task, scheme, model, split); one column per tier.

    Tiers are paired by construction within a split, so the step
    differences below are legitimate paired quantities."""
    wide = df.pivot_table(index=["task", "scheme", "model", "family",
                                 "split_index"],
                          columns="tier", values="pr_auc").reset_index()
    missing = [t for t in TIERS if t not in wide.columns]
    if missing:
        raise SystemExit(f"Tiers absent from the results: {missing}")
    wide = wide.dropna(subset=TIERS)
    for name, (hi, lo, _) in STEPS.items():
        wide[name] = wide[hi] - wide[lo]
    return wide


# =======================================================================
# CHECKS — are the two schemes comparable at all?
# =======================================================================
def prevalence_check(task: str) -> pd.DataFrame:
    """Test-set size and positive rate per split, from the predictions.

    PR-AUC depends on class balance, so absolute values are only
    comparable between schemes if the test prevalence matches."""
    rows = []
    for scheme, key in (("structure-aware", "structured_preds"),
                        ("random", "random_preds")):
        path = _paths(task)[key]
        if not path.exists():
            print(f"   {task}/{scheme}: no predictions file at {path}; "
                  f"prevalence not checked.")
            continue
        preds = pd.read_csv(path, usecols=["model", "tier", "split_index",
                                           "true_label"])
        # true_label repeats across models and tiers; keep one combination.
        one_model = sorted(preds["model"].unique())[0]
        preds = preds[(preds["model"] == one_model)
                      & (preds["tier"] == TIERS[0])]
        g = preds.groupby("split_index")["true_label"]
        per_split = pd.DataFrame({"n_test": g.size(), "prevalence": g.mean()})
        rows.append({
            "task": task, "scheme": scheme,
            "n_splits": len(per_split),
            "n_test_mean": per_split["n_test"].mean(),
            "prevalence_mean": per_split["prevalence"].mean(),
            "prevalence_sd": per_split["prevalence"].std(ddof=1),
        })
    return pd.DataFrame(rows)


def coverage_check(wide: pd.DataFrame) -> pd.DataFrame:
    """Which (task, model) cells have complete data under both schemes."""
    g = (wide.groupby(["task", "scheme", "model"])
             .size().rename("n_splits").reset_index())
    piv = g.pivot_table(index=["task", "model"], columns="scheme",
                        values="n_splits").reset_index()
    return piv


# =======================================================================
# COMPARISONS
# =======================================================================
def _unpaired(a: np.ndarray, b: np.ndarray) -> dict:
    """Compare two independent samples (random minus structure-aware).

    Welch for the difference in means with an approximate 95% CI, and
    Mann-Whitney as a distribution-free check. See the caveat in the
    module docstring: the splits within a scheme are not independent, so
    both p-values are optimistic."""
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return dict(diff=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                    p_welch=np.nan, p_mwu=np.nan, n_struct=len(b),
                    n_random=len(a))
    diff = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return dict(
        diff=diff,
        ci_lo=diff - 1.96 * se,
        ci_hi=diff + 1.96 * se,
        p_welch=ttest_ind(a, b, equal_var=False).pvalue,
        p_mwu=mannwhitneyu(a, b, alternative="two-sided").pvalue,
        n_struct=len(b), n_random=len(a),
    )


def compare_absolute(wide: pd.DataFrame) -> pd.DataFrame:
    """(a) How much does random splitting inflate raw PR-AUC?"""
    rows = []
    for (task, model), g in wide.groupby(["task", "model"]):
        s = g[g["scheme"] == "structure-aware"]
        r = g[g["scheme"] == "random"]
        for tier in TIERS:
            res = _unpaired(r[tier].to_numpy(), s[tier].to_numpy())
            rows.append({"task": task, "model": model, "tier": tier,
                         "structured_mean": s[tier].mean(),
                         "structured_sd": s[tier].std(ddof=1),
                         "random_mean": r[tier].mean(),
                         "random_sd": r[tier].std(ddof=1),
                         **res})
    return pd.DataFrame(rows)


def compare_steps(wide: pd.DataFrame) -> pd.DataFrame:
    """(b) Does the gain from each ladder step survive random splitting?"""
    rows = []
    for (task, model), g in wide.groupby(["task", "model"]):
        s = g[g["scheme"] == "structure-aware"]
        r = g[g["scheme"] == "random"]
        for step in STEPS:
            sv, rv = s[step].to_numpy(), r[step].to_numpy()
            res = _unpaired(rv, sv)
            rows.append({
                "task": task, "model": model, "step": step,
                "structured_delta": np.nanmean(sv),
                "structured_pos": int((sv > 0).sum()),
                "structured_neg": int((sv < 0).sum()),
                "random_delta": np.nanmean(rv),
                "random_pos": int((rv > 0).sum()),
                "random_neg": int((rv < 0).sum()),
                **res})
    return pd.DataFrame(rows)


# =======================================================================
# FIGURE
# =======================================================================
def _square(ax, lo, hi, pad=0.03):
    span = hi - lo
    lo, hi = lo - pad * span, hi + pad * span
    ax.plot([lo, hi], [lo, hi], color="0.5", lw=1.3, ls="--", zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")


def make_figure(absolute: pd.DataFrame, steps: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.0))

    # ---- (a) absolute PR-AUC ----
    ax = axes[0]
    for task in TASK_ORDER:
        sub = absolute[absolute["task"] == task]
        if sub.empty:
            continue
        for tier in TIERS:
            s = sub[sub["tier"] == tier]
            ax.scatter(s["structured_mean"], s["random_mean"],
                       marker=TASK_MARKER[task], s=62,
                       color=TIER_COLOUR[tier], edgecolor="0.3",
                       linewidth=0.7, zorder=3)
    vals = np.concatenate([absolute["structured_mean"].to_numpy(),
                           absolute["random_mean"].to_numpy()])
    _square(ax, np.nanmin(vals), np.nanmax(vals))
    ax.set_xlabel("PR-AUC, structure-aware splits")
    ax.set_ylabel("PR-AUC, random splits")
    ax.set_title("(a) Absolute performance\nPoints above the line = "
                 "optimism from leakage", fontsize=14)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Line2D([], [], ls="none", marker=TASK_MARKER[t],
                          markerfacecolor="none", markeredgecolor="#444444",
                          markersize=9, label=TASKS[t]["label"])
               for t in TASK_ORDER]
    handles += [plt.Line2D([], [], ls="none", marker="o",
                           color=TIER_COLOUR[t], markersize=9,
                           label=TIER_LABEL[t]) for t in TIERS]
    ax.legend(handles=handles, frameon=False, loc="upper left", ncol=2,
              columnspacing=1.0, handletextpad=0.3)

    # ---- (b) the ladder steps ----
    ax = axes[1]
    sub_all = steps[steps["step"].isin(FIGURE_STEPS)]
    for task in TASK_ORDER:
        for step in FIGURE_STEPS:
            s = sub_all[(sub_all["task"] == task) & (sub_all["step"] == step)]
            if s.empty:
                continue
            ax.scatter(s["structured_delta"], s["random_delta"],
                       marker=TASK_MARKER[task], s=62,
                       color=STEP_COLOUR[step], edgecolor="0.3",
                       linewidth=0.7, zorder=3)
    vals = [sub_all["structured_delta"].to_numpy(),
            sub_all["random_delta"].to_numpy()]
    if CONFIG["ZERO_IN_STEP_AXIS"]:
        vals.append(np.array([0.0]))
    vals = np.concatenate(vals)
    _square(ax, np.nanmin(vals), np.nanmax(vals))
    ax.axhline(0, color="0.75", lw=1.0, zorder=1)
    ax.axvline(0, color="0.75", lw=1.0, zorder=1)
    ax.set_xlabel(r"$\Delta$ PR-AUC, structure-aware splits")
    ax.set_ylabel(r"$\Delta$ PR-AUC, random splits")
    ax.set_title("(b) Effect of each representation step\nPoints on the "
                 "line = effect unchanged by leakage", fontsize=14)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Line2D([], [], ls="none", marker="o",
                          color=STEP_COLOUR[s], markersize=9,
                          label=STEPS[s][2].replace("$-$", "−"))
               for s in FIGURE_STEPS]
    ax.legend(handles=handles, frameon=False, loc="upper left")

    fig.suptitle("CP database: random vs structure-aware splitting "
                 "(7 models, 3 tasks, 50 splits each)")
    fig.tight_layout()
    path = out / "fig_splitting_comparison.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved: {path}")


# =======================================================================
# LATEX
# =======================================================================
def _esc(s) -> str:
    return str(s).replace("_", r"\_")


def _fmt_p(p) -> str:
    if pd.isna(p):
        return "--"
    if p < 1e-3:
        m, e = f"{p:.1e}".split("e")
        return rf"${m}\times10^{{{int(e)}}}$"
    return f"{p:.3f}"


def latex_tables(absolute: pd.DataFrame, steps: pd.DataFrame,
                 out: Path) -> None:
    # ---- Table A: optimism in absolute PR-AUC ----
    piv = absolute.pivot_table(index=["task", "model"], columns="tier",
                               values="diff")
    body = []
    for task in TASK_ORDER:
        if task not in piv.index.get_level_values("task"):
            continue
        sub = piv.xs(task, level="task")
        body.append(r"\midrule" if body else "")
        body.append(rf"\multicolumn{{5}}{{l}}{{\textit{{{TASKS[task]['label']}}}}} \\")
        for model, r in sub.iterrows():
            body.append(f"  & {_esc(model)} & "
                        + " & ".join(f"{r[t]:+.3f}" for t in TIERS) + r" \\")
    tbl_a = (r"""\begin{table*}[ht]
\centering
\small
\caption{\textbf{Optimism from random splitting, CP database.} Each
value is the mean PR-AUC under random splitting minus the mean under
structure-aware (group-atomic) splitting, across 50 splits of each kind.
Positive values indicate that random splitting inflates performance.
The two sets of splits are different partitions, so these differences
are unpaired; the 50 splits within a scheme also share a protein set and
have overlapping training sets, so associated standard errors understate
the true uncertainty.}
\label{tab:splitting-absolute}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrr@{}}
\toprule
& Model & Pairwise & Hypergraph & Hb-graph \\
""" + "\n".join(x for x in body if x) + r"""
\bottomrule
\end{tabular*}
\end{table*}""")

    # ---- Table B: does each step survive? ----
    body = []
    for task in TASK_ORDER:
        sub = steps[(steps["task"] == task)
                    & (steps["step"] != "hb_minus_pair")]
        if sub.empty:
            continue
        body.append(r"\midrule" if body else "")
        body.append(rf"\multicolumn{{8}}{{l}}{{\textit{{{TASKS[task]['label']}}}}} \\")
        for model in sorted(sub["model"].unique()):
            cells = [f"  & {_esc(model)}"]
            for step in ("hyper_minus_pair", "hb_minus_hyper"):
                r = sub[(sub["model"] == model) & (sub["step"] == step)]
                if r.empty:
                    cells += ["--", "--", "--"]
                    continue
                r = r.iloc[0]
                cells += [f"{r['structured_delta']:+.3f}",
                          f"{r['random_delta']:+.3f}",
                          _fmt_p(r["p_mwu"])]
            body.append(" & ".join(cells) + r" \\")
    tbl_b = (r"""\begin{table*}[ht]
\centering
\small
\caption{\textbf{Effect of each representation step under both splitting
schemes, CP database.} $\Delta$ is the mean paired PR-AUC difference
across the 50 splits of the scheme concerned; the tiers are paired within
a split, so each $\Delta$ is a within-split quantity. The two schemes are
not paired with one another, so $p$ is a two-sided Mann-Whitney test
comparing the two sets of 50 differences. Because splits within a scheme
are not independent, $p$ is optimistic and is given as a descriptive
guide. Values close to each other indicate that the representation
advantage is not an artefact of structural leakage.}
\label{tab:splitting-steps}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}ll rrl rrl@{}}
\toprule
& & \multicolumn{3}{c}{Hypergraph $-$ Pairwise}
  & \multicolumn{3}{c}{Hb-graph $-$ Hypergraph} \\
\cmidrule(lr){3-5} \cmidrule(lr){6-8}
& Model & Struct. & Random & $p$ & Struct. & Random & $p$ \\
""" + "\n".join(x for x in body if x) + r"""
\bottomrule
\end{tabular*}
\end{table*}""")

    path = out / "si_table_splitting.tex"
    path.write_text(tbl_a + "\n\n" + tbl_b + "\n")
    print(f"   Saved: {path}")


# =======================================================================
# MAIN
# =======================================================================
def main():
    out = CONFIG["OUTPUT_DIR"]
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  SPLITTING-SCHEME COMPARISON  —  CP")
    print("=" * 70)

    print("\n1. Loading results")
    frames = []
    for task in TASK_ORDER:
        p = _paths(task)
        frames.append(load_results(p["structured_results"], task,
                                   "structure-aware"))
        frames.append(load_results(p["random_results"], task, "random"))
    results = pd.concat(frames, ignore_index=True)
    wide = to_wide(results)
    print(f"   {len(wide)} complete (task, scheme, model, split) rows")

    print("\n2. Checks")
    cover = coverage_check(wide)
    ragged = cover[cover.get("random") != cover.get("structure-aware")]
    if len(ragged):
        print("   WARNING: unequal split counts between schemes:")
        print(ragged.to_string(index=False))
    else:
        print("   Split counts match between schemes for every "
              "(task, model).")

    checks = pd.DataFrame()
    if CONFIG["CHECK_PREVALENCE"]:
        checks = pd.concat([prevalence_check(t) for t in TASK_ORDER],
                           ignore_index=True)
        if not checks.empty:
            print(checks.to_string(index=False,
                                   float_format=lambda v: f"{v:.4f}"))
            for task in TASK_ORDER:
                c = checks[checks["task"] == task]
                if len(c) == 2:
                    gap = abs(c["prevalence_mean"].diff().iloc[-1])
                    if gap > 0.005:
                        print(f"   WARNING: {task} test prevalence differs "
                              f"by {gap:.4f} between schemes; absolute "
                              f"PR-AUC is not directly comparable.")
            checks.to_csv(out / "splitting_checks.csv", index=False)

    print("\n3. Comparisons")
    absolute = compare_absolute(wide)
    steps = compare_steps(wide)
    absolute.to_csv(out / "splitting_absolute.csv", index=False)
    steps.to_csv(out / "splitting_steps.csv", index=False)
    print(f"   Saved: {out / 'splitting_absolute.csv'}")
    print(f"   Saved: {out / 'splitting_steps.csv'}")

    print("\n4. Figure and tables")
    make_figure(absolute, steps, out)
    latex_tables(absolute, steps, out)

    print("\n5. Summary")
    infl = absolute.groupby("tier")["diff"].mean().reindex(TIERS)
    print("   Mean optimism from random splitting, by tier:")
    for t in TIERS:
        print(f"      {TIER_LABEL[t]:<11}: {infl[t]:+.4f}")
    two = steps[steps["step"] != "hb_minus_pair"]
    print("   Mean change in step size (random minus structure-aware):")
    for step in ("hyper_minus_pair", "hb_minus_hyper"):
        d = two[two["step"] == step]["diff"].mean()
        print(f"      {STEPS[step][2].replace('$-$', '-'):<26}: {d:+.4f}")
    n_sig = int((two["p_mwu"] < 0.05).sum())
    print(f"   Step size differs at p<0.05 in {n_sig}/{len(two)} "
          f"(task, model, step) cells.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()