"""
How much of the structural grouping comes from proteins sharing a PDB entry?

For each structural group, checks whether its members were given the same
PDB entry by the structure-retrieval step. Reports the result per group-size
bucket (the same buckets used for splitting).

Inputs (both already produced by the pipeline):
  - mapping_struct.csv             columns: uniprot_id, group_id, representative
  - structure_coverage_report.csv  columns: uniprot_id, source, structure_id, ...
"""

import pandas as pd

# =======================================================
# CONFIGURATION
# =======================================================
CONFIG = {
    "MAPPING_FILE": "../../data/lookup_tables/cp/mapping_struct.csv",
    "COVERAGE_FILE": "../../data/lookup_tables/cp/structure_coverage_report.csv",
    "OUTPUT_CSV": "shared_pdb_entry_by_bucket.csv",
}

BUCKETS = [(1, 1, "1"), (2, 5, "2-5"), (6, 20, "6-20"), (21, 10**9, ">20")]


def bucket_of(size: int) -> str:
    for lo, hi, name in BUCKETS:
        if lo <= size <= hi:
            return name
    return "?"


# =======================================================
# LOAD
# =======================================================
groups = pd.read_csv(CONFIG["MAPPING_FILE"])
cov = pd.read_csv(CONFIG["COVERAGE_FILE"])[["uniprot_id", "source", "structure_id"]]
df = groups.merge(cov, on="uniprot_id", how="left")

print(f"Proteins: {len(df)}   Groups: {df['group_id'].nunique()}")
print(f"From PDB: {(df['source'] == 'pdb').sum()}   From AlphaFold: {(df['source'] == 'alphafold').sum()}")

# =======================================================
# PER-PROTEIN: does another member of my group use the same PDB entry?
# =======================================================
df["group_size"] = df.groupby("group_id")["uniprot_id"].transform("size")
df["bucket"] = df["group_size"].apply(bucket_of)

is_pdb = df["source"] == "pdb"
# number of group members using each (group, PDB entry) pair
df["n_same_entry"] = 0
df.loc[is_pdb, "n_same_entry"] = (
    df[is_pdb].groupby(["group_id", "structure_id"])["uniprot_id"].transform("size")
)
df["shares_entry"] = df["n_same_entry"] >= 2

# =======================================================
# PER-GROUP: largest share of members that come from one PDB entry
# =======================================================
per_group = df.groupby("group_id").agg(
    group_size=("uniprot_id", "size"),
    bucket=("bucket", "first"),
    max_same_entry=("n_same_entry", "max"),
)
per_group["share_from_one_entry"] = per_group["max_same_entry"] / per_group["group_size"]
per_group["all_one_entry"] = (per_group["group_size"] >= 2) & (
    per_group["max_same_entry"] == per_group["group_size"]
)
per_group["majority_one_entry"] = (per_group["group_size"] >= 2) & (
    per_group["share_from_one_entry"] > 0.5
)

# =======================================================
# SUMMARY BY BUCKET
# =======================================================
order = [name for _, _, name in BUCKETS]
summary = pd.DataFrame({
    "n_groups": per_group.groupby("bucket").size(),
    "n_proteins": df.groupby("bucket").size(),
    "proteins_sharing_entry": df.groupby("bucket")["shares_entry"].sum(),
    "groups_all_one_entry": per_group.groupby("bucket")["all_one_entry"].sum(),
    "groups_majority_one_entry": per_group.groupby("bucket")["majority_one_entry"].sum(),
}).reindex(order)
summary["pct_proteins_sharing_entry"] = (
    100 * summary["proteins_sharing_entry"] / summary["n_proteins"]
).round(1)
summary["pct_groups_majority_one_entry"] = (
    100 * summary["groups_majority_one_entry"] / summary["n_groups"]
).round(1)

print("\nBy group-size bucket:")
print(summary.to_string())

total_multi = df[df["group_size"] >= 2]
print(
    f"\nOf {len(total_multi)} proteins in groups of two or more, "
    f"{int(total_multi['shares_entry'].sum())} "
    f"({100 * total_multi['shares_entry'].mean():.1f}%) share a PDB entry with another member of their group."
)

print("\nTen largest groups:")
top = per_group.sort_values("group_size", ascending=False).head(10)
print(top[["group_size", "max_same_entry", "share_from_one_entry"]].round(2).to_string())

summary.to_csv(CONFIG["OUTPUT_CSV"])
print(f"\nSaved: {CONFIG['OUTPUT_CSV']}")