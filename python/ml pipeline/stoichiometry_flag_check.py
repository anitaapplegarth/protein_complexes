"""
=======================================================================
CHECK: does protein-level stoichiometry coverage follow from incidence
       coverage plus membership density?
=======================================================================

The question this answers
    CP  reports ~59.5% of incidences curated and 59.9% of proteins flagged
        — near-identical.
    CORUM reports  3.8% of incidences curated but 10.5% of proteins flagged
        — a threefold gap.

A protein is flagged if ANY of its complexes has curated stoichiometry, so
proteins in many complexes are more likely to be flagged even at a low
incidence rate. CORUM proteins sit in more complexes than CP proteins, so
some gap is expected. The question is whether the OBSERVED gap is fully
explained by that, or whether curation is concentrated in a way that
changes what the annotation-presence flag actually measures.

What it reports
    1. Incidence-level and protein-level coverage, and duplicate incidences.
    2. The membership-density distribution (complexes per protein).
    3. The protein-level coverage EXPECTED if curation were scattered at
       random over incidences at the observed rate. For a protein in k
       incidences, P(flagged) = 1 - (1-p)^k. Summing over proteins gives the
       expected flag rate under that null.
         - observed ~= expected  -> the gap is pure density, nothing to fix
         - observed <  expected  -> curation is CLUSTERED (whole complexes
                                    curated together), so flagged proteins
                                    are concentrated in fewer complexes
         - observed >  expected  -> curation is SPREAD across complexes more
                                    evenly than chance, so it reaches more
                                    proteins than the headline rate suggests
    4. Whether curation is all-or-nothing at the complex level, which is the
       usual cause of clustering.
    5. Coverage by complex size, and the flag rate among proteins that appear
       in only one complex (where density cannot inflate anything).

USAGE
    Edit the PATHS block, then:  python check_stoich_coverage.py
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

# =======================================================
# PATHS — edit these
# =======================================================
PATHS = {
    "CP":    Path("/Users/anitaapplegarth/github/dphil/protein_complexes/"
                  "data/lookup_tables/cp/stoich_protein.csv"),
    "CORUM": Path("/Users/anitaapplegarth/github/dphil/protein_complexes/"
                  "data/lookup_tables/corum/stoich_protein.csv"),
}

# Candidate column names, tried in order. Add yours if they differ.
COMPLEX_COL_CANDIDATES = ['ComplexId', 'ComplexID', 'complex_id', 'Complex',
                          'ComplexAC', 'complex_ac', 'CORUM_ID', 'corum_id']
PROTEIN_COL_CANDIDATES = ['ProteinId', 'ProteinID', 'protein_id',
                          'UniProt_AC', 'UniProtAC', 'uniprot_ac']
STOICH_COL_CANDIDATES  = ['Stoichiometry', 'stoichiometry']


def resolve_column(df: pd.DataFrame, candidates, label: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"Could not find the {label} column. Tried {candidates}.\n"
        f"   Columns present: {list(df.columns)}\n"
        f"   Add the correct name to the *_CANDIDATES list at the top."
    )


def analyse(name: str, path: Path) -> dict:
    print(f"\n{'='*72}")
    print(f"  {name}")
    print(f"  {path}")
    print(f"{'='*72}")

    if not path.exists():
        print(f"  FILE NOT FOUND — skipping.")
        return {}

    df = pd.read_csv(path)
    cx  = resolve_column(df, COMPLEX_COL_CANDIDATES, "complex")
    pid = resolve_column(df, PROTEIN_COL_CANDIDATES, "protein")
    st  = resolve_column(df, STOICH_COL_CANDIDATES,  "stoichiometry")
    print(f"\n  Columns used: complex='{cx}'  protein='{pid}'  stoich='{st}'")

    # --- 1. Deduplication ------------------------------------------------
    n_raw = len(df)
    df = df.drop_duplicates(subset=[cx, pid])
    n_dedup = len(df)
    if n_raw != n_dedup:
        print(f"\n  NOTE: {n_raw - n_dedup} duplicate (complex, protein) rows "
              f"dropped ({n_raw} -> {n_dedup}).")
        print(f"        If the pipeline does NOT dedup, the incidence-level "
              f"rate you quote may differ from the one below.")

    df['curated'] = (df[st].fillna(0) > 0).astype(int)

    n_inc      = len(df)
    n_proteins = df[pid].nunique()
    n_complex  = df[cx].nunique()

    # --- 2. Coverage at each level ---------------------------------------
    inc_rate = df['curated'].mean()
    prot_flag = df.groupby(pid)['curated'].max()
    prot_rate = prot_flag.mean()

    print(f"\n  {'-'*68}")
    print(f"  BASIC COUNTS")
    print(f"  {'-'*68}")
    print(f"  Complexes                     : {n_complex:,}")
    print(f"  Unique proteins               : {n_proteins:,}")
    print(f"  Incidences (deduplicated)     : {n_inc:,}")
    print(f"\n  Incidence-level coverage      : {100*inc_rate:6.2f}%  "
          f"({int(df['curated'].sum()):,} / {n_inc:,})")
    print(f"  Protein-level flag rate       : {100*prot_rate:6.2f}%  "
          f"({int(prot_flag.sum()):,} / {n_proteins:,})")
    print(f"  Ratio (protein / incidence)   : {prot_rate/inc_rate:6.2f}x")

    # --- 3. Membership density -------------------------------------------
    k = df.groupby(pid).size()
    print(f"\n  {'-'*68}")
    print(f"  MEMBERSHIP DENSITY (complexes per protein)")
    print(f"  {'-'*68}")
    print(f"  Mean   : {k.mean():.2f}     Median : {k.median():.0f}     "
          f"Max : {k.max()}")
    print(f"  In exactly 1 complex : {100*(k == 1).mean():5.1f}%  "
          f"({int((k == 1).sum()):,} proteins)")
    print(f"  In 2-5 complexes     : {100*k.between(2, 5).mean():5.1f}%")
    print(f"  In 6+ complexes      : {100*(k >= 6).mean():5.1f}%")

    # --- 4. Expected flag rate under random scattering --------------------
    # P(protein in k incidences is flagged) = 1 - (1-p)^k
    p = inc_rate
    expected_flag = (1 - (1 - p) ** k)
    exp_rate = expected_flag.mean()

    print(f"\n  {'-'*68}")
    print(f"  IS THE GAP EXPLAINED BY DENSITY ALONE?")
    print(f"  {'-'*68}")
    print(f"  Observed protein flag rate    : {100*prot_rate:6.2f}%")
    print(f"  Expected if curation were     : {100*exp_rate:6.2f}%")
    print(f"    scattered at random over incidences at {100*p:.2f}%")
    print(f"  Observed / expected           : {prot_rate/exp_rate:6.2f}x")

    if exp_rate > 0:
        ratio = prot_rate / exp_rate
        if 0.9 <= ratio <= 1.1:
            verdict = ("MATCHES. The protein-level rate is what membership "
                       "density alone predicts.\n     Nothing anomalous; the "
                       "headline incidence figure just isn't the\n     "
                       "quantity the flag measures.")
        elif ratio < 0.9:
            verdict = ("BELOW expectation -> curation is CLUSTERED. Curated "
                       "incidences pile up\n     in the same complexes or the "
                       "same proteins, so the flag reaches fewer\n     "
                       "proteins than chance would. Flagged proteins are a "
                       "narrower, more\n     heavily-studied subset than the "
                       "rate suggests.")
        else:
            verdict = ("ABOVE expectation -> curation is SPREAD across "
                       "complexes more evenly\n     than chance, reaching more "
                       "proteins than the incidence rate implies.\n     The "
                       "flag is better-populated than the headline figure "
                       "suggests.")
        print(f"\n  VERDICT: {verdict}")

    # --- 5. Is curation all-or-nothing per complex? -----------------------
    per_cx = df.groupby(cx)['curated'].agg(['mean', 'size'])
    n_all  = int((per_cx['mean'] == 1).sum())
    n_none = int((per_cx['mean'] == 0).sum())
    n_part = int(((per_cx['mean'] > 0) & (per_cx['mean'] < 1)).sum())

    print(f"\n  {'-'*68}")
    print(f"  CURATION AT THE COMPLEX LEVEL")
    print(f"  {'-'*68}")
    print(f"  Fully curated complexes       : {n_all:,}  "
          f"({100*n_all/n_complex:.1f}%)")
    print(f"  Partially curated             : {n_part:,}  "
          f"({100*n_part/n_complex:.1f}%)")
    print(f"  Not curated at all            : {n_none:,}  "
          f"({100*n_none/n_complex:.1f}%)")
    if n_part == 0:
        print(f"\n  Curation is ALL-OR-NOTHING per complex — it is a property "
              f"of the complex,\n  not of the individual subunit. The flag is "
              f"therefore 'belongs to at least one\n  characterised complex', "
              f"which is worth stating that way in the text.")
    elif n_part / max(n_all + n_part, 1) < 0.1:
        print(f"\n  Curation is close to all-or-nothing per complex "
              f"({100*n_part/max(n_all+n_part,1):.1f}% of\n  curated complexes "
              f"are only partial).")

    # --- 6. Single-complex proteins (density cannot inflate these) --------
    singles = k[k == 1].index
    if len(singles):
        single_rate = prot_flag.loc[singles].mean()
        multi_rate  = prot_flag.loc[k[k > 1].index].mean() if (k > 1).any() else np.nan
        print(f"\n  {'-'*68}")
        print(f"  FLAG RATE BY MEMBERSHIP COUNT")
        print(f"  {'-'*68}")
        print(f"  Proteins in 1 complex         : {100*single_rate:6.2f}% flagged")
        print(f"  Proteins in 2+ complexes      : {100*multi_rate:6.2f}% flagged")
        print(f"  (The first is a density-free estimate of curation reach; it "
              f"should sit\n   close to the incidence-level rate if curation "
              f"is not size-biased.)")

    # --- 7. Coverage by complex size --------------------------------------
    size = df.groupby(cx)[pid].nunique().rename('n_subunits')
    tmp = df.merge(size, left_on=cx, right_index=True)
    bins = [0, 2, 3, 5, 10, np.inf]
    labels = ['2', '3', '4-5', '6-10', '11+']
    tmp['size_band'] = pd.cut(tmp['n_subunits'], bins=bins, labels=labels,
                              right=True, include_lowest=True)
    by_size = tmp.groupby('size_band', observed=True)['curated'].agg(
        ['mean', 'size'])
    print(f"\n  {'-'*68}")
    print(f"  INCIDENCE COVERAGE BY COMPLEX SIZE")
    print(f"  {'-'*68}")
    print(f"  {'Subunits':<10} {'Curated':>9} {'Incidences':>12}")
    for band, row in by_size.iterrows():
        print(f"  {str(band):<10} {100*row['mean']:>8.1f}% {int(row['size']):>12,}")

    return {
        'database': name, 'n_complexes': n_complex, 'n_proteins': n_proteins,
        'n_incidences': n_inc, 'incidence_rate': inc_rate,
        'protein_flag_rate': prot_rate, 'expected_flag_rate': exp_rate,
        'obs_over_exp': prot_rate / exp_rate if exp_rate else np.nan,
        'mean_complexes_per_protein': k.mean(),
        'pct_complexes_partially_curated': 100 * n_part / n_complex,
    }


if __name__ == "__main__":
    rows = [r for name, path in PATHS.items() if (r := analyse(name, path))]

    if len(rows) > 1:
        print(f"\n\n{'='*72}")
        print("  SIDE BY SIDE")
        print(f"{'='*72}\n")
        out = pd.DataFrame(rows).set_index('database').T
        with pd.option_context('display.float_format', '{:,.3f}'.format,
                               'display.width', 100):
            print(out.to_string())
        out.T.to_csv('stoich_coverage_check.csv')
        print(f"\n  Saved: stoich_coverage_check.csv")