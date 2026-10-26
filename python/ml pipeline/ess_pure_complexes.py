"""
Which complexes are wholly or mostly essential?

Lists complexes by the share of their labelled members that are essential,
with names, sizes and stoichiometry status, so the essential complexes behind
the homogeneity result can be inspected.
"""
import pandas as pd
from pathlib import Path

# =======================================================
# CONFIG  (run from the same folder as cp_model_sweep.py)
# =======================================================
DATA_DIR        = Path("../../data/lookup_tables/cp/")
MEMBERSHIP_FILE = "stoich_protein.csv"     # ComplexId, ProteinId, Stoichiometry
SPLITS_FILE     = "ess_protein_splits.csv"
POS_LABEL       = "Essential"
NEG_LABEL       = "Non-essential"

NAMES_FILE      = "../../data/complexes/complex_portal_9606.tsv"   # or None
NAME_ID_COL     = "#Complex ac"
NAME_COL        = "Recommended name"

MIN_LABELLED    = 2      # same threshold as the homogeneity test
MOSTLY_CUTOFF   = 2 / 3  # "mostly essential" = more than this share, below 100%

OUTPUT_DIR = Path("./feature_distributions/by_label")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =======================================================
# Load
# =======================================================
membership = (pd.read_csv(DATA_DIR / MEMBERSHIP_FILE)
              [['ComplexId', 'ProteinId', 'Stoichiometry']]
              .drop_duplicates(['ComplexId', 'ProteinId']))

labels = pd.read_csv(DATA_DIR / SPLITS_FILE).rename(
    columns={'UniProt_AC': 'ProteinId'})
labels = labels[labels['label_mask']].drop_duplicates('ProteinId')
labels = labels[labels['protein_label'].isin([POS_LABEL, NEG_LABEL])]
ess = (labels['protein_label'] == POS_LABEL).astype(int)
ess.index = labels['ProteinId']

mem = membership.copy()
mem['essential'] = mem['ProteinId'].map(ess)          # NaN = not labelled

# =======================================================
# Per-complex summary
# =======================================================
total = mem.groupby('ComplexId').agg(
    n_members_total=('ProteinId', 'size'),
    stoich_recorded=('Stoichiometry', lambda s: 'all' if (s > 0).all()
                     else ('none' if (s == 0).all() else 'mixed')),
)
lab = mem.dropna(subset=['essential'])
summary = lab.groupby('ComplexId').agg(
    n_labelled=('ProteinId', 'size'),
    n_essential=('essential', 'sum'),
    essential_proteins=('ProteinId', lambda s: ";".join(
        sorted(s[lab.loc[s.index, 'essential'] == 1]))),
    non_essential_proteins=('ProteinId', lambda s: ";".join(
        sorted(s[lab.loc[s.index, 'essential'] == 0]))),
)
summary = summary.join(total)
summary['n_essential'] = summary['n_essential'].astype(int)
summary['frac_essential'] = summary['n_essential'] / summary['n_labelled']
summary = summary[summary['n_labelled'] >= MIN_LABELLED]

if NAMES_FILE:
    names = (pd.read_csv(NAMES_FILE, sep='\t')
             .set_index(NAME_ID_COL)[NAME_COL])
    summary.insert(0, 'name', summary.index.map(names))


def group(f):
    if f == 1:
        return 'wholly essential'
    if f > MOSTLY_CUTOFF:
        return 'mostly essential'
    if f == 0:
        return 'wholly non-essential'
    return 'mixed'


summary['group'] = summary['frac_essential'].map(group)
summary = summary.sort_values(['frac_essential', 'n_labelled'],
                              ascending=[False, False])
summary.to_csv(OUTPUT_DIR / 'ess_complexes.csv')

# Proteins in several essential complexes (variants of one complex can
# repeat the same essential subunits)
ess_cx = summary[summary['group'].isin(['wholly essential', 'mostly essential'])]
repeat = (ess_cx['essential_proteins'].str.split(';').explode()
          .value_counts())

# =======================================================
# Report
# =======================================================
print(f"Complexes with >= {MIN_LABELLED} labelled members: {len(summary)}")
print(summary['group'].value_counts()
      .reindex(['wholly essential', 'mostly essential', 'mixed',
                'wholly non-essential']).to_string())

print("\nSize and stoichiometry by group (median labelled / total members):")
print(summary.groupby('group').agg(
    n=('n_labelled', 'size'),
    median_labelled=('n_labelled', 'median'),
    median_total=('n_members_total', 'median'),
    pct_stoich_recorded=('stoich_recorded',
                         lambda s: round(100 * (s == 'all').mean())),
).to_string())

print("\nEssential proteins appearing in the most essential complexes:")
print(repeat.head(15).to_string())

cols = [c for c in ['name', 'n_labelled', 'n_essential', 'n_members_total',
                    'stoich_recorded'] if c in summary.columns]
with pd.option_context('display.max_rows', None, 'display.width', 220,
                       'display.max_colwidth', 70):
    for g in ['wholly essential', 'mostly essential']:
        sub = summary[summary['group'] == g]
        print(f"\n{g.capitalize()} complexes ({len(sub)}):")
        print(sub[cols].to_string())

print(f"\nSaved: {OUTPUT_DIR / 'ess_complexes.csv'}")
