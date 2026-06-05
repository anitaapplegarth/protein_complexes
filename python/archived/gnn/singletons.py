import pandas as pd
from pathlib import Path

DATA_DIR = Path("../../data/lookup_tables")

df = pd.read_csv(DATA_DIR / "Complex_noimpute_stoich_protein_evidence.csv")
splits = pd.read_csv(DATA_DIR / "protein_splits_all_strat.csv")

# Complexes with only one unique protein
complex_sizes = df.groupby("ComplexId")["ProteinId"].nunique()
singleton_complexes = complex_sizes[complex_sizes == 1].index

singleton_proteins = (
    df[df["ComplexId"].isin(singleton_complexes)][["ProteinId", "Stoichiometry"]]
    .drop_duplicates()
)

print(f"Complexes with one unique protein : {len(singleton_complexes)}")
print(f"Unique proteins in those complexes: {singleton_proteins['ProteinId'].nunique()}")
print(f"\nStoichiometry distribution:")
print(singleton_proteins["Stoichiometry"].value_counts().sort_index())

# Check labels - use one split (split_index==1) as representative
split1 = splits[splits["split_index"] == 1][["UniProt_AC", "protein_label", "label_mask", "split"]]
merged = singleton_proteins.merge(split1, left_on="ProteinId", right_on="UniProt_AC", how="left")

print(f"\nLabel distribution:")
print(merged["protein_label"].value_counts(dropna=False))

print(f"\nTrain/test split:")
print(merged["split"].value_counts(dropna=False))

print(f"\nLabelled (label_mask=True):")
print(merged[merged["label_mask"] == True]["protein_label"].value_counts(dropna=False))