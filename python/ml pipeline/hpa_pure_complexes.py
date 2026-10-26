"""
List the complexes in which every labelled member is an HPA drug target,
and show how many of those proteins are also ChEMBL (single protein) targets.
"""
import pandas as pd
from pathlib import Path

# =======================================================
# CONFIG  (run from the same folder as cp_model_sweep.py)
# =======================================================
DATA_DIR        = Path("../../data/lookup_tables/cp/")
MEMBERSHIP_FILE = "stoich_protein.csv"        # ComplexId, ProteinId, ...
HPA_SPLITS      = "hpa_protein_splits.csv"
CHEMBL_SPLITS   = "chembl_protein_splits.csv"
TARGET_LABEL    = "Drug_target"

# Optional: a file with complex names (set to None if you don't have one).
# Complex Portal IDs can also be looked up at
# https://www.ebi.ac.uk/complexportal/complex/<ComplexId>
NAMES_FILE      = "../../data/complexes/complex_portal_9606.tsv"                       # e.g. "complex_names.csv"
NAME_ID_COL     = "#Complex ac"  # column in NAMES_FILE with the ComplexId
NAME_COL        = "Recommended name"

MIN_LABELLED    = 2    # same threshold as the homogeneity test

# ChEMBL labels including PROTEIN COMPLEX targets (single + complex).
# Set to None to skip. Full path, not relative to DATA_DIR.
MIXED_FILE        = "/home/applegarth/protein_complexes/data/lookup_tables/cp_drug_target_chembl_mixed.csv"
MIXED_PROTEIN_COL = None   # None = auto-detect (UniProt_AC / ProteinId / ...)
MIXED_LABEL_COL   = None   # None = auto-detect (protein_label / label / ...)
MIXED_TARGET_VAL  = "Drug_target"

OUTPUT_FILE = Path("./feature_distributions/by_label/hpa_pure_complexes.csv")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


# =======================================================
# Load
# =======================================================
def load_targets(splits_file):
    df = pd.read_csv(DATA_DIR / splits_file).rename(
        columns={'UniProt_AC': 'ProteinId'})
    df = df[df['label_mask']].drop_duplicates('ProteinId')
    df['is_target'] = (df['protein_label'] == TARGET_LABEL).astype(int)
    return df.set_index('ProteinId')['is_target']


membership = (pd.read_csv(DATA_DIR / MEMBERSHIP_FILE)
              [['ComplexId', 'ProteinId']].drop_duplicates())
hpa    = load_targets(HPA_SPLITS)
chembl = load_targets(CHEMBL_SPLITS)

mem = membership.copy()
mem['hpa']    = mem['ProteinId'].map(hpa)       # NaN = not labelled
mem['chembl'] = mem['ProteinId'].map(chembl)

# =======================================================
# Per-complex summary
# =======================================================
lab = mem.dropna(subset=['hpa'])
summary = lab.groupby('ComplexId').agg(
    n_members_labelled=('ProteinId', 'size'),
    n_hpa_targets=('hpa', 'sum'),
    n_chembl_targets=('chembl', 'sum'),
    proteins=('ProteinId', lambda s: ";".join(sorted(s))),
)
# Which members are ChEMBL targets, and which are HPA-only
chembl_members = (lab[lab['chembl'] == 1].groupby('ComplexId')['ProteinId']
                  .apply(lambda s: ";".join(sorted(s))))
hpa_only = (lab[lab['chembl'] != 1].groupby('ComplexId')['ProteinId']
            .apply(lambda s: ";".join(sorted(s))))
summary['chembl_target_proteins'] = chembl_members.reindex(summary.index).fillna("")
summary['hpa_only_proteins'] = hpa_only.reindex(summary.index).fillna("")
summary['n_members_total'] = membership.groupby('ComplexId').size()
summary = summary[summary['n_members_labelled'] >= MIN_LABELLED]

pure = summary[summary['n_hpa_targets'] == summary['n_members_labelled']].copy()
pure['n_chembl_targets'] = pure['n_chembl_targets'].astype(int)
pure['n_hpa_targets'] = pure['n_hpa_targets'].astype(int)

if NAMES_FILE:
    names = (pd.read_csv(NAMES_FILE, sep="\t", usecols=[NAME_ID_COL, NAME_COL])
             .set_index(NAME_ID_COL)[NAME_COL])
    pure.insert(0, 'name', pure.index.map(names))

pure = pure.sort_values(['n_members_labelled', 'n_chembl_targets'],
                        ascending=[False, True])
pure.to_csv(OUTPUT_FILE)

# =======================================================
# Report
# =======================================================
print(f"Complexes with >= {MIN_LABELLED} labelled members: {len(summary)}")
print(f"Wholly HPA-target complexes: {len(pure)} "
      f"({100 * len(pure) / len(summary):.1f}%)")

prots = set(";".join(pure['proteins']).split(";"))
in_chembl = sum(chembl.get(p, 0) == 1 for p in prots)
print(f"Distinct proteins in them: {len(prots)}; "
      f"also ChEMBL targets: {in_chembl} ({100 * in_chembl / len(prots):.0f}%)")

print("\nWholly HPA-target complexes with no ChEMBL target among members: "
      f"{(pure['n_chembl_targets'] == 0).sum()}")

# Proteins shared between many pure complexes (one target family can
# produce many pure complexes, e.g. receptor subtypes)
counts = (pure['proteins'].str.split(';').explode().value_counts())
print("\nProteins appearing in the most wholly-target complexes:")
print(counts.head(15).to_string())

with pd.option_context('display.max_rows', None, 'display.width', 200,
                       'display.max_colwidth', 60):
    print("\nWholly HPA-target complexes:")
    print(pure.to_string())
print(f"\nSaved: {OUTPUT_FILE}")

# =======================================================
# Per-protein table: every protein in a wholly-target complex
# =======================================================
prot_table = pd.DataFrame({'ProteinId': sorted(prots)})
prot_table['chembl_label'] = prot_table['ProteinId'].map(chembl).map(
    {1: 'target', 0: 'not target'}).fillna('not labelled')
prot_table['n_pure_complexes'] = prot_table['ProteinId'].map(counts)


def _pick(cols, given, candidates, what):
    if given:
        return given
    for c in candidates:
        if c in cols:
            return c
    raise KeyError(f"Could not find the {what} column in MIXED_FILE. "
                   f"Columns are: {list(cols)}. Set it in the config.")


if MIXED_FILE:
    mixed = pd.read_csv(MIXED_FILE, sep=None, engine='python')
    print(f"\nMixed ChEMBL file columns: {list(mixed.columns)}")
    pcol = _pick(mixed.columns, MIXED_PROTEIN_COL,
                 ['UniProt_AC', 'ProteinId', 'uniprot_id', 'UniProt', 'uniprot'],
                 'protein')
    lcol = _pick(mixed.columns, MIXED_LABEL_COL,
                 ['protein_label', 'label', 'Label', 'target', 'is_target'],
                 'label')
    print(f"Using protein column '{pcol}' and label column '{lcol}'")
    if 'label_mask' in mixed.columns:
        mixed = mixed[mixed['label_mask']]
    mixed = mixed.drop_duplicates(pcol)
    lab_vals = mixed[lcol]
    is_tgt = ((lab_vals == MIXED_TARGET_VAL) | (lab_vals == 1)
              | (lab_vals == True))                      # noqa: E712
    mixed_targets = set(mixed.loc[is_tgt, pcol])
    print(f"Mixed ChEMBL targets: {len(mixed_targets)}")

    def mixed_status(row):
        if row['chembl_label'] == 'target':
            return 'single protein target'
        if row['ProteinId'] in mixed_targets:
            return 'complex target only'
        return 'not a ChEMBL target'

    prot_table['chembl_mixed'] = prot_table.apply(mixed_status, axis=1)
    print("\nProteins in wholly HPA-target complexes, against mixed ChEMBL:")
    print(prot_table['chembl_mixed'].value_counts().to_string())

    # Complexes: does mixed ChEMBL also label every member?
    pure['n_chembl_mixed_targets'] = pure['proteins'].str.split(';').apply(
        lambda ps: sum(p in mixed_targets for p in ps))
    pure['wholly_chembl_mixed'] = (pure['n_chembl_mixed_targets']
                                   == pure['n_members_labelled'])
    print(f"\nWholly HPA-target complexes that are also wholly ChEMBL "
          f"targets once complex targets are included: "
          f"{pure['wholly_chembl_mixed'].sum()} of {len(pure)}")
    pure.to_csv(OUTPUT_FILE)

    if 'name' in pure.columns:
        print("\nBy complex (members labelled / mixed ChEMBL targets):")
        with pd.option_context('display.max_rows', None, 'display.width', 200):
            print(pure[['name', 'n_members_labelled', 'n_chembl_targets',
                        'n_chembl_mixed_targets']].to_string())
prot_table = prot_table.sort_values(['chembl_label', 'n_pure_complexes'],
                                    ascending=[True, False])
prot_file = OUTPUT_FILE.with_name("hpa_pure_complex_proteins.csv")
prot_table.to_csv(prot_file, index=False)
print("\nChEMBL label of proteins in wholly HPA-target complexes:")
print(prot_table['chembl_label'].value_counts().to_string())
print(f"Saved: {prot_file}")