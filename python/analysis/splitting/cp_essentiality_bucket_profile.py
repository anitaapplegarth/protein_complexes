"""
Check the randomness of group allocation in the large bucket
(>20 proteins; merged from former large + very_large) across
the 15 repeated stratified splits.
"""

import pandas as pd
from collections import Counter

#### Need to add output file here as a CONFIG (cp_essentiality_by_group_size)

# ── Load ──────────────────────────────────────────────────────────────
df = pd.read_csv("../../../data/lookup_tables/cp_ess_merged_splits.csv")
FOCUS_GROUPS = ['grp_0222', 'grp_0064', 'grp_1179', 'grp_0000']

# ── Filter to large test proteins ─────────────────────────────────────
lg_test = df[(df["group_bucket"] == "large") & (df["split"] == "test")]

# ── 1. Which groups land in test per split? ───────────────────────────
print("=" * 60)
print("1. Which large groups are in TEST per split?")
print("=" * 60)
groups_per_split = (
    lg_test.groupby("split_index")["group_id"]
    .apply(lambda x: sorted(x.unique()))
)
for split_idx, grps in groups_per_split.items():
    print(f"  Split {split_idx:>2d}: {', '.join(grps)}")

# ── 2. Summary of all large groups ────────────────────────────────────
print("\n" + "=" * 60)
print("2. All large groups and their sizes")
print("=" * 60)
lg_all = df[df["group_bucket"] == "large"]
for gid, sub in sorted(lg_all.groupby("group_id")):
    size = int(sub["group_size"].iloc[0])
    test_splits = [int(x) for x in sorted(
        lg_test[lg_test["group_id"] == gid]["split_index"].unique()
    )]
    print(f"  {gid}: {size:>4d} proteins  |  in test for splits {test_splits}")

# ── 3. Test set sizes per split ───────────────────────────────────────
print("\n" + "=" * 60)
print("3. large test set size per split")
print("=" * 60)
proteins_per_split = lg_test.groupby("split_index")["UniProt_AC"].apply(set)
for split_idx, prots in proteins_per_split.items():
    print(f"  Split {split_idx:>2d}: {len(prots):>4d} proteins")

# ── 4. Overlap statistics ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. Overlap statistics across all 15 splits")
print("=" * 60)
common = set.intersection(*proteins_per_split.values)
union = set.union(*proteins_per_split.values)
print(f"  Intersection (in ALL 15 test sets): {len(common)}")
print(f"  Union (in ANY test set):            {len(union)}")
if len(union) > 0:
    print(f"  Jaccard similarity:                 {len(common) / len(union):.3f}")

# ── 5. How many splits does each protein appear in test? ──────────────
print("\n" + "=" * 60)
print("5. Per-protein test frequency (how many splits is each protein in test?)")
print("=" * 60)
freq = lg_test.groupby("UniProt_AC")["split_index"].nunique()
print(freq.value_counts().sort_index().to_string())

# ── 6. Per-group test frequency — randomness check ────────────────────
print("\n" + "=" * 60)
print("6. Per-group test frequency — randomness check")
print("=" * 60)
n_splits   = df["split_index"].nunique()
lg_groups  = df[df["group_bucket"] == "large"]["group_id"].unique()
n_groups   = len(lg_groups)
# Expected draw rate: round(n_groups * 0.20) groups chosen per split
n_drawn_per_split = round(n_groups * 0.20)
expected_freq     = n_drawn_per_split * n_splits / n_groups

print(f"  Total large groups:          {n_groups}")
print(f"  Groups drawn to test/split:  {n_drawn_per_split}  "
      f"({100 * n_drawn_per_split / n_groups:.1f}%)")
print(f"  Expected test freq per group over {n_splits} splits: "
      f"{expected_freq:.1f}")
print()
print(f"  {'Group':30s} {'Size':>6}  {'Test count':>10}  {'Note':}")

group_test_counts = (
    lg_test.groupby("group_id")["split_index"].nunique()
    .reindex(lg_groups, fill_value=0)
)

for gid in sorted(lg_groups):
    size  = int(df[df["group_id"] == gid]["group_size"].iloc[0])
    count = int(group_test_counts[gid])
    if count == 0:
        note = "⚠ NEVER in test"
    elif count == n_splits:
        note = "⚠ ALWAYS in test"
    elif abs(count - expected_freq) > 2 * expected_freq:
        note = "⚠ unusually high/low"
    else:
        note = "ok"
    print(f"  {gid:30s} {size:>6}  {count:>10}  {note}")


print()
obs_counts = group_test_counts.values
print(f"  Min test count across groups: {obs_counts.min()}")
print(f"  Max test count across groups: {obs_counts.max()}")
print(f"  Mean test count:              {obs_counts.mean():.2f}")
print(f"  Std dev:                      {obs_counts.std():.2f}")

# Use split 1 data only to get stable per-group label counts
# (label_mask and protein_label are constant across splits)
s1 = df[df["split_index"] == 1]

# ── 8. Average essentiality per group size ────────────────────────────────

print("\n" + "=" * 60)
print("9. Average essentiality per group size")
print("=" * 60)

size_stats = (
    s1[s1["protein_label"].isin(["Essential", "Non-essential"])]
    .assign(is_essential=lambda x: x["protein_label"] == "Essential")
    .groupby("group_size")
    .agg(
        n_proteins=("is_essential", "count"),
        n_essential=("is_essential", "sum"),
        ess_rate=("is_essential", "mean"),
    )
    .sort_index()
)

n_labelled    = (s1["protein_label"].isin(["Essential", "Non-essential"])).sum()
n_unknown     = (s1["protein_label"] == "Unknown").sum()
n_no_group    = s1["group_size"].isna().sum()
n_labelled_no_group = s1[s1["group_size"].isna() & 
                          s1["protein_label"].isin(["Essential", "Non-essential"])].shape[0]
print(f"\n  Coverage note (split 1, one row per protein):")
print(f"    Total proteins in CSV:              {len(s1)}")
print(f"    Labelled (Ess + Non-ess):           {n_labelled}")
print(f"    Unknown label (excluded):           {n_unknown}")
print(f"    No group_size / NaN (excluded):     {n_no_group}")
print(f"      of which labelled (double-excl):  {n_labelled_no_group}")
print(f"    Counted in size_stats:              {n_labelled - n_labelled_no_group}")

print(size_stats.to_string(float_format=lambda x: f"{x*100:.1f}%" if x <= 1 else f"{x:.1f}"))

out = size_stats.copy()
out["ess_rate_pct"] = (out["ess_rate"] * 100).round(2)
out = out.drop(columns="ess_rate").reset_index()
out.to_csv("cp_avg_essentiality_by_group_size.csv", index=False)
print("\n  Saved: cp_avg_essentiality_by_group_size.csv")

# Global essential rate for reference (labelled proteins only)
global_ess   = (s1["protein_label"] == "Essential").sum()
global_noness = (s1["protein_label"] == "Non-essential").sum()
global_rate  = 100 * global_ess / (global_ess + global_noness)
print(f"\n  Global essential rate (all labelled proteins): {global_rate:.1f}%\n")

print(f"  {'Group':12s} {'Size':>6}  {'Essential':>10}  "
      f"{'Non-ess':>8}  {'Unknown':>8}  {'Ess rate':>9}  {'vs global':>10}")
print("  " + "-" * 68)

for gid in FOCUS_GROUPS:
    grp_df    = s1[s1["group_id"] == gid]
    size      = int(grp_df["group_size"].iloc[0]) if len(grp_df) > 0 else 0
    n_ess     = int((grp_df["protein_label"] == "Essential").sum())
    n_noness  = int((grp_df["protein_label"] == "Non-essential").sum())
    n_unknown = int((grp_df["protein_label"] == "Unknown").sum())
    n_lab     = n_ess + n_noness
    ess_rate  = 100 * n_ess / n_lab if n_lab > 0 else float('nan')
    diff      = ess_rate - global_rate if n_lab > 0 else float('nan')
    direction = f"+{diff:.1f}pp" if diff >= 0 else f"{diff:.1f}pp"
    flag      = "  ⚠ high" if diff > 10 else ("  ⚠ low" if diff < -10 else "")
    print(f"  {gid:12s} {size:>6}  {n_ess:>10}  "
          f"{n_noness:>8}  {n_unknown:>8}  {ess_rate:>8.1f}%  "
          f"{direction:>10}{flag}")

# Per-split essential rate in large bucket broken down by which heavy group
# is in test — shows directly whether heavy group drives the variance
print(f"\n  Large-bucket test essential rate by split, "
      f"annotated with heavy group present:")
print(f"  {'Split':>5}  {'Heavy group in test':<30}  {'Heavy grp ess%':>14}  {'Large bucket ess%':>17}")
print("  " + "-" * 72)

for split_idx in sorted(lg_test["split_index"].unique()):
    sp_df = lg_test[lg_test["split_index"] == split_idx]
    groups_in_test = sp_df["group_id"].unique()

    # Identify ALL focus groups in test this split
    heavy_in_test = [g for g in FOCUS_GROUPS if g in groups_in_test]
    heavy_label   = ', '.join(heavy_in_test) if heavy_in_test else "none"

    # Combined essential rate across ALL focus groups present
    if heavy_in_test:
        hdf      = sp_df[sp_df["group_id"].isin(heavy_in_test)]
        h_ess    = (hdf["protein_label"] == "Essential").sum()
        h_noness = (hdf["protein_label"] == "Non-essential").sum()
        h_lab    = h_ess + h_noness
        h_rate   = f"{100 * h_ess / h_lab:.1f}%" if h_lab > 0 else "N/A"
    else:
        h_rate = "N/A"

    # Overall large-bucket essential rate this split
    b_ess    = (sp_df["protein_label"] == "Essential").sum()
    b_noness = (sp_df["protein_label"] == "Non-essential").sum()
    b_lab    = b_ess + b_noness
    b_rate   = f"{100 * b_ess / b_lab:.1f}%" if b_lab > 0 else "N/A"

    print(f"  {int(split_idx):>5}  {heavy_label:<30}  {h_rate:>14}  {b_rate:>17}")