#!/usr/bin/env python3
"""
Foldseek Structural Clustering Pipeline v2
============================================
Clusters protein structures into structural groups using Foldseek,
then builds the data splitting logic.

Run this AFTER pdb_structures_v3.py has downloaded all structures.

Changes from v1:
    - No more prepare_structures() step: structures are already saved as
      {uid}.cif by the retrieval script, so Foldseek reads them directly.
    - Coverage report is loaded once and passed through (no redundant I/O).
    - Majority-vote logic simplified (no deprecated groupby().apply()).

Requirements:
    - Foldseek installed (see install instructions below)
    - pip install pandas
"""

import pandas as pd
import subprocess
import shutil
from pathlib import Path
import sys
import re

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Path to your downloaded structures (now contains {uid}.cif files directly)
    'structures_dir': '/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/cp_protein_structures/',

    # Structure coverage report from the retrieval script
    'coverage_report': '/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/cp/structure_coverage_report.csv',

    # Working directory for Foldseek temp files (will be created)
    'foldseek_work_dir': '/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/cp_foldseek_work/',

    # Output: protein -> group mapping
    'output_group_mapping': '/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/cp/mapping_struct.csv',

    # Foldseek clustering parameters
    'min_seq_id': 0.0,      # minimum sequence identity (none)
    'coverage': 0.8,        # minimum alignment coverage (80% of shorter seq)
    'cov_mode': 1,          # 1 = coverage of both query and target
    'cluster_mode': 0,      # 0 = greedy set cover (default)

    # Proteins of interest
    'trace_ids': [
    'O15392',  # CPX-111, CPX-116 (essential)
    'Q53HL2',  # CPX-116
    'Q96GD4',  # CPX-116
    'Q9NQS7',  # CPX-116
    'P07333',  # CPX-25717, CPX-10333
    'P09603',  # CPX-25717
    'Q6ZMJ4',  # CPX-10333
    ],
}


# ============================================================================
# STEP 1: Validate structures directory
# ============================================================================

def validate_structures(report: pd.DataFrame) -> int:
    """
    Quick sanity check: count .cif files in structures_dir and compare
    against the coverage report. 
    """
    structures_dir = Path(CONFIG['structures_dir'])
    cif_files = list(structures_dir.glob('*.cif'))

    expected = report['available'].sum()
    found = len(cif_files)

    print(f"Structures directory: {structures_dir}")
    print(f"  Expected (from report): {expected}")
    print(f"  Found .cif files:       {found}")

    if found == 0:
        print("ERROR: No .cif files found! Check structures_dir path.")
        return 0

    if found < expected:
        # Identify which ones are missing
        expected_ids = set(report.loc[report['available'], 'uniprot_id'])
        found_ids = {f.stem for f in cif_files}
        missing = expected_ids - found_ids
        print(f"  Missing: {len(missing)} files")
        if len(missing) <= 10:
            print(f"  IDs: {sorted(missing)}")

    return found


# ============================================================================
# STEP 2: Run Foldseek clustering
# ============================================================================

def run_foldseek_clustering():
    """
    Run Foldseek easy-cluster directly on structures_dir.

    easy-cluster wraps: createdb -> search -> clust
    """
    structures_dir = Path(CONFIG['structures_dir'])
    work_dir = Path(CONFIG['foldseek_work_dir'])
    tmp_dir = work_dir / 'tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    output_prefix = work_dir / 'cluster_results'

    min_seq_id = CONFIG['min_seq_id']
    coverage = CONFIG['coverage']
    cov_mode = CONFIG['cov_mode']
    cluster_mode = CONFIG['cluster_mode']

    cmd = [
        'foldseek', 'easy-cluster',
        str(structures_dir),          # <-- point directly at structures
        str(output_prefix),
        str(tmp_dir),
        '--min-seq-id', str(min_seq_id),
        '-c', str(coverage),
        '--cov-mode', str(cov_mode),
        '--cluster-mode', str(cluster_mode),
    ]

    print()
    print("Running Foldseek easy-cluster...")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Parameters:")
    print(f"    min-seq-id:   {min_seq_id} (30% = supergroup level)")
    print(f"    coverage:     {coverage} (50% alignment coverage)")
    print(f"    cov-mode:     {cov_mode} (both query and target)")
    print(f"    cluster-mode: {cluster_mode} (greedy set cover)")
    print()

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(work_dir)
    )

    if result.returncode != 0:
        print(f"ERROR: Foldseek failed!")
        print(f"  stdout: {result.stdout[-500:]}")
        print(f"  stderr: {result.stderr[-500:]}")
        return None

    print(f"  stdout: {result.stdout[-300:]}")

    # Check output files (Foldseek uses _clu.tsv or _cluster.tsv)
    cluster_tsv = Path(f"{output_prefix}_clu.tsv")
    if not cluster_tsv.exists():
        cluster_tsv = Path(f"{output_prefix}_cluster.tsv")

    if cluster_tsv.exists():
        print(f"Cluster file: {cluster_tsv}")
    else:
        print("  Files in work dir:")
        for f in work_dir.iterdir():
            if f.is_file():
                print(f"    {f.name}")
        return None

    return cluster_tsv


# ============================================================================
# STEP 3: Parse Foldseek output -> protein group mapping
# ============================================================================

# Regex to extract UniProt accession from Foldseek IDs.
# Foldseek appends chain/model suffixes: _A, _I3, _MODEL_1, _4B, etc.
_UNIPROT_RE = re.compile(
    r'^([A-Z][0-9][A-Z0-9]{3}[0-9](?:[A-Z0-9]{4}[0-9])?)'  # 6 or 10 char accession
    r'(?:_.*)?$'                                               # optional suffix
)


def _clean_id(foldseek_id: str) -> str:
    """Extract UniProt accession from a Foldseek identifier.

    Examples:
        P04637       -> P04637
        O00303_I3    -> O00303
        O00168_MODEL_2 -> O00168
        P25024_4B    -> P25024
    """
    m = _UNIPROT_RE.match(foldseek_id.strip())
    return m.group(1) if m else foldseek_id.strip()


def parse_cluster_results(cluster_tsv: Path) -> pd.DataFrame:
    """
    Parse Foldseek cluster TSV and produce a protein -> group mapping.

    The TSV has two columns: representative_id  member_id
    Multi-chain CIF files produce multiple rows per protein; we use
    majority vote to assign each protein to a single group.
    """
    print(f"\nParsing cluster results from {cluster_tsv}")

    clusters = pd.read_csv(cluster_tsv, sep='\t', header=None,
                            names=['representative', 'member'])

    clusters['rep_clean'] = clusters['representative'].apply(_clean_id)
    clusters['mem_clean'] = clusters['member'].apply(_clean_id)

    # Assign group IDs based on representative
    unique_reps = clusters['rep_clean'].unique()
    rep_to_group = {rep: f"grp_{i:04d}" for i, rep in enumerate(unique_reps)}
    clusters['group_id'] = clusters['rep_clean'].map(rep_to_group)

    # ── Majority vote (one group per protein) ──
    # For each member protein, pick the group that appears most often
    # across its chains. This replaces the old groupby().apply() pattern.
    vote_counts = (
        clusters
        .groupby(['mem_clean', 'group_id'])
        .size()
        .reset_index(name='count')
    )
    # Keep the group with the highest count per protein
    idx_best = vote_counts.groupby('mem_clean')['count'].idxmax()
    best_group = vote_counts.loc[idx_best, ['mem_clean', 'group_id']].copy()

    # Look up the representative for each chosen group
    group_to_rep = {grp: rep for rep, grp in rep_to_group.items()}
    best_group['representative'] = best_group['group_id'].map(group_to_rep)

    protein_group = best_group.rename(columns={'mem_clean': 'uniprot_id'}).reset_index(drop=True)

    # ── Statistics ──
    n_proteins = protein_group['uniprot_id'].nunique()
    n_groups = protein_group['group_id'].nunique()
    group_sizes = protein_group.groupby('group_id').size()

    print()
    print("=" * 60)
    print("STRUCTURAL group CLUSTERING RESULTS")
    print("=" * 60)
    print(f"Proteins clustered:  {n_proteins}")
    print(f"Structural groups: {n_groups}")
    print()
    print(f"Group size distribution:")
    print(f"  Singletons (1 protein):  {(group_sizes == 1).sum()}")
    print(f"  Small (2-5):             {((group_sizes >= 2) & (group_sizes <= 5)).sum()}")
    print(f"  Medium (6-20):           {((group_sizes >= 6) & (group_sizes <= 20)).sum()}")
    print(f"  Large (21-100):          {((group_sizes >= 21) & (group_sizes <= 100)).sum()}")
    print(f"  Very large (>100):       {(group_sizes > 100).sum()}")
    print()
    print(f"  Mean group size:   {group_sizes.mean():.1f}")
    print(f"  Median group size: {group_sizes.median():.1f}")
    print(f"  Largest group:     {group_sizes.max()} proteins")
    print()

    # Top 10 largest groups
    top_groups = group_sizes.nlargest(10)
    print("Top 10 largest groups:")
    for grp_id, size in top_groups.items():
        rep = protein_group.loc[protein_group['group_id'] == grp_id, 'representative'].iloc[0]
        print(f"  {grp_id}: {size} proteins (rep: {rep})")

    print("=" * 60)

    # Save
    output_path = Path(CONFIG['output_group_mapping'])
    protein_group.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

    return protein_group


# ============================================================================
# STEP 4: Validate coverage
# ============================================================================

def validate_coverage(protein_group: pd.DataFrame, report: pd.DataFrame):
    """Check how many of our proteins got assigned to a group.

    Takes the already-loaded report DataFrame to avoid re-reading the CSV.
    """
    all_proteins = set(report['uniprot_id'].unique())
    clustered_proteins = set(protein_group['uniprot_id'].unique())

    covered = all_proteins & clustered_proteins
    uncovered = all_proteins - clustered_proteins

    print(f"\nCoverage validation:")
    print(f"  Total proteins:     {len(all_proteins)}")
    print(f"  With group:        {len(covered)} ({100*len(covered)/len(all_proteins):.1f}%)")
    print(f"  Without group:     {len(uncovered)} ({100*len(uncovered)/len(all_proteins):.1f}%)")

    if uncovered:
        print(f"\n  Uncovered proteins (first 20):")
        for uid in sorted(uncovered)[:20]:
            rows = report[report['uniprot_id'] == uid]
            if len(rows) > 0:
                row = rows.iloc[0]
                print(f"    {uid}: source={row.get('source', 'N/A')}, available={row.get('available', 'N/A')}")
            else:
                print(f"    {uid}: not in coverage report")

    return uncovered


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("Foldseek Structural Clustering Pipeline v2")
    print("=" * 60)

    # Load coverage report once (shared across steps)
    report_path = Path(CONFIG['coverage_report'])
    if not report_path.exists():
        print(f"ERROR: Coverage report not found: {report_path}")
        print("Run pdb_structures_v3.py first.")
        sys.exit(1)
    report = pd.read_csv(report_path)

    # Step 1: Validate structures directory
    print("\n--- Step 1: Validating structures ---")
    n_structures = validate_structures(report)
    if n_structures == 0:
        sys.exit(1)

    # Step 2: Run Foldseek (directly on structures_dir)
    print("\n--- Step 2: Running Foldseek clustering ---")
    cluster_tsv = run_foldseek_clustering()
    if cluster_tsv is None:
        print("ERROR: Foldseek clustering failed!")
        sys.exit(1)

    # Step 3: Parse results
    print("\n--- Step 3: Parsing cluster results ---")
    protein_group = parse_cluster_results(cluster_tsv)

    # Trace proteins of interest
    if CONFIG['trace_ids']:
        trace_proteins(protein_group)

    # Step 4: Validate
    print("\n--- Step 4: Validating coverage ---")
    validate_coverage(protein_group, report)    

    print("\nDone! Next step: use protein_group_mapping.csv for data splitting.")

def trace_proteins(protein_group: pd.DataFrame):
    """Print group assignments for proteins of interest."""
    ids = CONFIG['trace_ids']
    if not ids:
        return
    print("\nTracing proteins of interest:")
    subset = protein_group[protein_group['uniprot_id'].isin(ids)].copy()
    missing = set(ids) - set(subset['uniprot_id'])
    print(subset[['uniprot_id', 'group_id', 'representative']].to_string(index=False))
    if missing:
        print(f"  Not found in clustering output: {sorted(missing)}")

if __name__ == '__main__':
    main()
