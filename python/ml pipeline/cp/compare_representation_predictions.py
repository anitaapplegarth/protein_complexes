"""
compare_representation_predictions.py

Compares top-N predicted positives across three graph-representation models
(pairwise, hypergraph, hb-graph) for one or more prediction tasks
(e.g. HPA-derived label, essentiality, ChEMBL druggability).

Each CSV has repeated rows per protein (one per test split_index). Predictions
are first averaged per protein across the splits where it appeared in the
test set, then ranked.

For each top-N tier (expressed as 1%, 5%, 10% of the protein pool, so
tiers are comparable across tasks with different pool sizes / prevalence),
the script reports:
  - Precision@N per tier: of the top-N ranked proteins, what fraction are
    true positives, alongside the pool's baseline prevalence for context
  - Jaccard overlap of the top-N set between every pair of tiers (how much
    the two tiers' top-N sets agree, as a fraction of their union)

Separately, for two small fixed N (10, 20) — used only for eyeballing
individual proteins, not for cross-task comparison — it reports which
proteins are in the richest tier's (hb-graph) top-N but fall outside a
simpler tier's top-N, along with that simpler tier's probability/rank for
the protein.

Note: this is purely a comparison between the three representations (do
they agree, does one surface things the others miss), not an evaluation
of any one model as a deployable predictor — none of these numbers are
meant to be read as "how good is this at finding real targets".
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score

# ---------------------------------------------------------------------
# Point this at wherever your data/ folder actually lives. Default:
# a "data" folder in the same directory as this script.
# ---------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent / "./randomforest/cp_hpa_second_testA/"

# One entry per task. File names are fixed (pairwise_predictions.csv,
# hypergraph_predictions.csv, hb_graph_predictions.csv); only the folder
# changes per task. Add/remove task names as needed: "hpa", "ess", "chembl"
TASK_NAMES = ["hpa"]

FILE_NAMES = {
    "pair": "pairwise_predictions.csv",
    "hyper": "hypergraph_predictions.csv",
    "hbg": "hb_graph_predictions.csv",
}

TASKS = {
    task: {tier: DATA_DIR / fname for tier, fname in FILE_NAMES.items()}
    for task in TASK_NAMES
}

# Tier order, simplest representation first -> richest last.
# "lower tier" in the output always means a tier earlier in this list.
TIER_ORDER = ["pair", "hyper", "hbg"]
# Top-N tiers, expressed as a fraction of the protein pool rather than a
# fixed count, so results are comparable across tasks with different pool
# sizes / different prevalence (e.g. HPA vs ess vs ChEMBL).
TOP_N_PCTS = [0.01, 0.05, 0.10]

# Fixed, small N used only for the qualitative "which proteins does the
# richest tier surface" table below — kept separate from the precision/
# Jaccard reporting since a fixed N isn't comparable across tasks.
EYEBALL_NS = [10, 20]

PROB_COL = {
    "pair": "pair_pred_proba",
    "hyper": "hyper_pred_proba",
    "hbg": "hbg_pred_proba",
}


def load_and_aggregate(path, prob_col):
    """One row per protein: mean/std predicted probability across the
    repeated test splits it appeared in."""
    df = pd.read_csv(path)
    agg = (
        df.groupby("ProteinId")
        .agg(
            true_label=("true_label", "first"),
            mean_proba=(prob_col, "mean"),
            std_proba=(prob_col, "std"),
            n_splits=(prob_col, "size"),
        )
        .reset_index()
    )
    return agg


def load_task(paths):
    """Merge all three tiers for one task into a single wide dataframe."""
    merged = None
    for tier, path in paths.items():
        agg = load_and_aggregate(path, PROB_COL[tier])
        agg = agg.rename(
            columns={
                "mean_proba": f"{tier}_mean_proba",
                "std_proba": f"{tier}_std_proba",
                "n_splits": f"{tier}_n_splits",
            }
        )
        merged = agg if merged is None else merged.merge(
            agg.drop(columns=["true_label"]), on="ProteinId", how="outer"
        )
    return merged


def top_n_set(df, tier, n):
    ranked = df.sort_values(f"{tier}_mean_proba", ascending=False).reset_index(drop=True)
    top = ranked.head(n).copy()
    top["rank"] = range(1, len(top) + 1)
    return top.set_index("ProteinId")


def novel_in_top_tier(df, top_tier, lower_tier, n):
    """Proteins in top_tier's top-N absent from lower_tier's top-N, with
    lower_tier's probability and full rank for context."""
    top = top_n_set(df, top_tier, n)
    full_rank = df.sort_values(f"{lower_tier}_mean_proba", ascending=False).reset_index(drop=True)
    full_rank["rank_full"] = full_rank.index + 1
    full_rank = full_rank.set_index("ProteinId")
    lower_top_ids = set(top_n_set(df, lower_tier, n).index)

    rows = []
    for pid in top.index:
        if pid in lower_top_ids:
            continue
        rows.append(
            {
                "ProteinId": pid,
                "true_label": top.loc[pid, "true_label"],
                f"{top_tier}_rank": int(top.loc[pid, "rank"]),
                f"{top_tier}_proba": round(top.loc[pid, f"{top_tier}_mean_proba"], 3),
                f"{lower_tier}_proba": round(full_rank.loc[pid, f"{lower_tier}_mean_proba"], 3)
                if pid in full_rank.index else np.nan,
                f"{lower_tier}_full_rank": int(full_rank.loc[pid, "rank_full"])
                if pid in full_rank.index else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(f"{top_tier}_rank")


def jaccard(a, b):
    """Fraction of overlap between two top-N sets, as a share of their
    combined (union) size. 1.0 = identical sets, 0.0 = no shared proteins."""
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else np.nan


def precision_at_n(df, tier, n):
    """Of the top-N ranked proteins for this tier, what fraction are
    true positives (true_label == 1)."""
    top = top_n_set(df, tier, n)
    return top["true_label"].sum() / n


def pct_to_n(pct, n_total):
    return max(1, round(pct * n_total))


def run_task(task_name, paths):
    print(f"\n{'=' * 70}\nTASK: {task_name}\n{'=' * 70}")
    df = load_task(paths)
    n_total = len(df)
    n_pos = int(df["true_label"].sum())
    prevalence = n_pos / n_total
    print(f"Proteins: {n_total} | positives: {n_pos} ({prevalence:.1%})")

    print("\n-- Per-tier performance (predictions averaged across splits) --")
    for tier in TIER_ORDER:
        col = f"{tier}_mean_proba"
        sub = df.dropna(subset=[col, "true_label"])
        auroc = roc_auc_score(sub["true_label"], sub[col])
        auprc = average_precision_score(sub["true_label"], sub[col])
        print(f"  {tier:6s}  AUROC={auroc:.3f}  AUPRC={auprc:.3f}  (n={len(sub)})")

    top_tier = TIER_ORDER[-1]  # richest representation, e.g. hb-graph

    # --- Precision@N and Jaccard, N expressed as % of the pool ---
    for pct in TOP_N_PCTS:
        n = pct_to_n(pct, n_total)
        print(f"\n-- Top {pct:.0%} of pool (N={n}): precision@N per tier (baseline prevalence={prevalence:.1%}) --")
        for tier in TIER_ORDER:
            p = precision_at_n(df, tier, n)
            print(f"  {tier:6s}  precision@{n} = {p:.2f}  ({int(round(p * n))}/{n} true positives, "
                  f"{p / prevalence:.1f}x baseline)")

        print(f"\n-- Top {pct:.0%} of pool (N={n}): Jaccard overlap across tiers --")
        for i, t1 in enumerate(TIER_ORDER):
            for t2 in TIER_ORDER[i + 1:]:
                s1 = top_n_set(df, t1, n).index
                s2 = top_n_set(df, t2, n).index
                print(f"  {t1} vs {t2}: Jaccard={jaccard(s1, s2):.2f}  overlap={len(set(s1) & set(s2))}/{n}")

    # --- Qualitative "what does the richest tier surface" table, fixed
    # small N since this is for eyeballing individual proteins, not a
    # cross-task comparison ---
    for n in EYEBALL_NS:
        print(f"\n-- Top-{n}: '{top_tier}' predictions absent from lower tiers --")
        for lower_tier in TIER_ORDER[:-1]:
            missing = novel_in_top_tier(df, top_tier, lower_tier, n)
            print(f"\n  In {top_tier} top-{n} but NOT in {lower_tier} top-{n}  ({len(missing)} proteins):")
            if len(missing):
                print(missing.to_string(index=False))
            else:
                print("    (none - full overlap)")

    return df


if __name__ == "__main__":
    results = {}
    for task_name, paths in TASKS.items():
        try:
            results[task_name] = run_task(task_name, paths)
        except FileNotFoundError as e:
            print(f"\n[skipped '{task_name}': file not found — {e}]")