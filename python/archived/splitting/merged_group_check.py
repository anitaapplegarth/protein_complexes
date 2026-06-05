"""
Check the randomness of group allocation in the large bucket
(>20 proteins; merged from former large + very_large) across
the 15 repeated stratified splits.
"""

import pandas as pd
from collections import Counter

# ── Load ──────────────────────────────────────────────────────────────
df = pd.read_csv("../../data/lookup_tables/cp_ess_merged3_splits.csv")

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