"""
Check whether the same proteins end up in the very_large test set
across the 15 repeated stratified splits.
"""

import pandas as pd
from collections import Counter

# ── Load ──────────────────────────────────────────────────────────────
df = pd.read_csv("../../data/lookup_tables/cp_drug_chembl_protein_splits.csv")

# ── Filter to very_large test proteins ────────────────────────────────
vl_test = df[(df["group_bucket"] == "very_large") & (df["split"] == "test")]

# ── 1. Which group lands in test per split? ───────────────────────────
print("=" * 60)
print("1. Which very_large group is in TEST per split?")
print("=" * 60)
groups_per_split = (
    vl_test.groupby("split_index")["group_id"]
    .apply(lambda x: sorted(x.unique()))
)
for split_idx, grps in groups_per_split.items():
    print(f"  Split {split_idx:>2d}: {', '.join(grps)}")

# ── 2. Summary of all very_large groups ───────────────────────────────
print("\n" + "=" * 60)
print("2. All very_large groups and their sizes")
print("=" * 60)
vl_all = df[df["group_bucket"] == "very_large"]
for gid, sub in sorted(vl_all.groupby("group_id")):
    size = int(sub["group_size"].iloc[0])
    test_splits = sorted(
        vl_test[vl_test["group_id"] == gid]["split_index"].unique()
    )
    print(f"  {gid}: {size:>4d} proteins  |  in test for splits {test_splits}")

# ── 3. Test set sizes per split ───────────────────────────────────────
print("\n" + "=" * 60)
print("3. very_large test set size per split")
print("=" * 60)
proteins_per_split = vl_test.groupby("split_index")["UniProt_AC"].apply(set)
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
freq = vl_test.groupby("UniProt_AC")["split_index"].nunique()
print(freq.value_counts().sort_index().to_string())