#!/usr/bin/env python3
"""
Protein Structure Retrieval v3: PDB First, AlphaFold Fallback
==============================================================
Uses UniProt cross-references API for PDB lookup.

Changes from v2:
    - Saves structure files as {uid}.cif (no _pdb/_alphafold suffix)
      so downstream tools (Foldseek) can use the directory directly.
    - Standardises all downloads to .cif format.
    - Source information is recorded only in the CSV report.

Requirements:
    pip install pandas requests tqdm

Usage:
    Update CONFIG below and run.
"""

import pandas as pd
import requests
import json
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import sys
from typing import Dict, Optional

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'input_file': '/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/corum_master_protein_list.csv',
    'uniprot_column': 'ProteinId',
    'output_report': '/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/corum_structure_coverage_report.csv',
    'missing_file': '/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/corum_missing_structures.csv',

    'download_structures': True,
    'structures_dir': '/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/corum_protein_structures/',

    # PDB quality thresholds
    'pdb_resolution_cutoff': 3.5,     # Angstroms
    'accept_nmr': True,             # Accept NMR structures (no resolution)?

    # Performance
    'max_workers': 5,
    'timeout': 30,
    'retry_attempts': 3,
    'retry_delay': 2,

    # Example proteins
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
# API HELPER
# ============================================================================

def api_get(url, timeout=30, retries=3, delay=2, headers=None):
    """GET request with retries."""
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout, headers=headers or {})
            if r.status_code == 429:
                wait = int(r.headers.get('Retry-After', delay * (attempt + 1)))
                time.sleep(wait)
                continue
            return r
        except requests.exceptions.RequestException:
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
            else:
                return None
    return None

# ============================================================================
# PDB LOOKUP via UniProt Cross-References
# ============================================================================

def parse_resolution(res_str: str) -> Optional[float]:
    """Parse resolution string like '1.50 A' -> 1.50."""
    if not res_str or res_str == '-':
        return None
    match = re.search(r'([\d.]+)', res_str)
    if match:
        return float(match.group(1))
    return None


def check_pdb_via_uniprot(uniprot_id: str,
                           resolution_cutoff: float = 3.5,
                           accept_nmr: bool = True) -> Optional[Dict]:
    """
    Find best PDB structure for a UniProt ID using UniProt cross-references.

    API: https://rest.uniprot.org/uniprotkb/{id}?fields=xref_pdb&format=json
    """
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}?fields=xref_pdb&format=json"
    r = api_get(url, headers={'Accept': 'application/json'})

    if r is None or r.status_code != 200:
        return None

    try:
        data = r.json()
    except (json.JSONDecodeError, ValueError):
        return None

    xrefs = data.get('uniProtKBCrossReferences', [])
    pdb_xrefs = [x for x in xrefs if x.get('database') == 'PDB']

    if not pdb_xrefs:
        return None

    # Score and rank structures
    candidates = []

    for x in pdb_xrefs:
        pdb_id = x.get('id', '')
        props = {p['key']: p['value'] for p in x.get('properties', [])}

        method = props.get('Method', 'Unknown')
        res_str = props.get('Resolution', '-')
        chains = props.get('Chains', '')

        resolution = parse_resolution(res_str)

        # Filter by method and resolution
        is_nmr = 'NMR' in method.upper()

        if is_nmr:
            if not accept_nmr:
                continue
            score = 100.0  # Low priority
        elif resolution is not None:
            if resolution > resolution_cutoff:
                continue
            score = resolution  # Lower is better
        else:
            continue

        # Parse chain coverage to estimate how much of the protein is covered
        coverage_length = 0
        if chains:
            for segment in chains.split(','):
                range_match = re.search(r'(\d+)-(\d+)', segment)
                if range_match:
                    start, end = int(range_match.group(1)), int(range_match.group(2))
                    coverage_length = max(coverage_length, end - start + 1)

        candidates.append({
            'pdb_id': pdb_id,
            'method': method,
            'resolution': resolution,
            'coverage_length': coverage_length,
            'score': score,
            'chains': chains,
        })

    if not candidates:
        return None

    # Sort: best resolution first, then by coverage length (descending)
    candidates.sort(key=lambda c: (c['score'], -c['coverage_length']))
    best = candidates[0]

    # Always provide a .cif URL
    cif_url = f"https://files.rcsb.org/download/{best['pdb_id'].lower()}.cif"

    return {
        'source': 'pdb',
        'structure_id': best['pdb_id'].lower(),
        'resolution': best['resolution'],
        'exp_method': best['method'],
        'chains': best['chains'],
        'url': cif_url,
        'n_pdb_structures': len(pdb_xrefs),
    }


# ============================================================================
# ALPHAFOLD LOOKUP
# ============================================================================

def check_alphafold(uniprot_id: str) -> Optional[Dict]:
    """
    Check AlphaFold DB for predicted structure.
    Uses prediction API -> direct file URL -> 3D-Beacons as fallbacks.
    Always returns a .cif URL when possible.
    """
    # Method 1: Prediction API
    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
    r = api_get(api_url)

    if r is not None and r.status_code == 200:
        try:
            data = r.json()
            if isinstance(data, list) and len(data) > 0:
                entry = data[0]
                entry_id = entry.get('modelEntityId') or entry.get('entryId', f'AF-{uniprot_id}-F1')
                # Prefer CIF over PDB
                cif_url = entry.get('cifUrl')
                pdb_url = entry.get('pdbUrl')

                return {
                    'source': 'alphafold',
                    'structure_id': entry_id,
                    'resolution': None,
                    'exp_method': 'AlphaFold prediction',
                    'chains': None,
                    'url': cif_url or pdb_url or f"https://alphafold.ebi.ac.uk/files/{entry_id}-model_v4.cif",
                    'n_pdb_structures': 0,
                }
        except (json.JSONDecodeError, ValueError, KeyError, IndexError):
            pass

    # Method 2: Direct file URL (CIF only)
    for version in ['v4', 'v2']:
        file_url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_{version}.cif"
        r = api_get(file_url, retries=1, delay=1)
        if r is not None and r.status_code == 200:
            return {
                'source': 'alphafold',
                'structure_id': f'AF-{uniprot_id}-F1',
                'resolution': None,
                'exp_method': 'AlphaFold prediction',
                'chains': None,
                'url': file_url,
                'n_pdb_structures': 0,
            }

    return None


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def process_protein(uniprot_id: str, resolution_cutoff: float, accept_nmr: bool) -> Dict:
    """Process one protein: PDB first, then AlphaFold."""
    result = {
        'uniprot_id': uniprot_id,
        'source': None,
        'structure_id': None,
        'resolution': None,
        'exp_method': None,
        'chains': None,
        'url': None,
        'available': False,
        'n_pdb_structures': 0,
    }

    # Step 1: Try PDB via UniProt
    pdb_result = check_pdb_via_uniprot(uniprot_id, resolution_cutoff, accept_nmr)
    if pdb_result:
        result.update(pdb_result)
        result['available'] = True
        return result

    # Step 2: Try AlphaFold
    af_result = check_alphafold(uniprot_id)
    if af_result:
        result.update(af_result)
        result['available'] = True
        return result

    return result


def main():
    print("=" * 60)
    print("Structure Retrieval v3: PDB (via UniProt) -> AlphaFold")
    print("=" * 60)

    # Load protein list
    input_path = Path(CONFIG['input_file'])
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    df = pd.read_csv(input_path)
    col = CONFIG['uniprot_column']

    if col not in df.columns:
        print(f"ERROR: Column '{col}' not found. Available: {list(df.columns)}")
        sys.exit(1)

    uniprot_ids = df[col].dropna().unique().tolist()
    print(f"Loaded {len(uniprot_ids)} proteins")

    resolution_cutoff = CONFIG['pdb_resolution_cutoff']
    accept_nmr = CONFIG['accept_nmr']
    print(f"PDB resolution cutoff: {resolution_cutoff} A")
    print(f"Accept NMR structures: {accept_nmr}")
    print(f"Checking {len(uniprot_ids)} proteins (PDB first, then AlphaFold)...")
    print()

    # Process with progress bar
    results = []

    with ThreadPoolExecutor(max_workers=CONFIG['max_workers']) as executor:
        futures = {
            executor.submit(process_protein, uid, resolution_cutoff, accept_nmr): uid
            for uid in uniprot_ids
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Querying"):
            try:
                result = future.result(timeout=120)
                results.append(result)
            except Exception as e:
                uid = futures[future]
                results.append({
                    'uniprot_id': uid,
                    'source': None,
                    'structure_id': None,
                    'resolution': None,
                    'exp_method': None,
                    'chains': None,
                    'url': None,
                    'available': False,
                    'n_pdb_structures': 0,
                })

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Statistics
    total = len(results_df)
    available = results_df['available'].sum()
    pdb_count = (results_df['source'] == 'pdb').sum()
    af_count = (results_df['source'] == 'alphafold').sum()
    missing = total - available

    print()
    print("=" * 60)
    print(f"Total:       {total}")
    print(f"Available:   {available:>5}  ({100*available/total:5.1f}%)")
    print(f"  PDB:       {pdb_count:>5}  ({100*pdb_count/total:5.1f}%)")
    print(f"  AlphaF:    {af_count:>5}  ({100*af_count/total:5.1f}%)")
    print(f"Missing:     {missing:>5}  ({100*missing/total:5.1f}%)")

    # PDB stats
    pdb_rows = results_df[results_df['source'] == 'pdb']
    if len(pdb_rows) > 0:
        resolutions = pdb_rows['resolution'].dropna()
        if len(resolutions) > 0:
            print(f"\nPDB Statistics:")
            print(f"  Avg resolution:  {resolutions.mean():.2f} A")
            print(f"  Best:            {resolutions.min():.2f} A")
            print(f"  Worst:           {resolutions.max():.2f} A")
            print(f"  Median:          {resolutions.median():.2f} A")

        methods = pdb_rows['exp_method'].value_counts()
        if len(methods) > 0:
            print(f"  Methods:")
            for method, count in methods.items():
                print(f"    {method}: {count}")

        total_pdb = pdb_rows['n_pdb_structures'].sum()
        print(f"  Total PDB entries across all proteins: {total_pdb}")

    print("=" * 60)

    # Save report
    output_path = Path(CONFIG['output_report'])
    results_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

    # Sample
    print(f"\nSample results (first 10):")
    cols = ['uniprot_id', 'source', 'structure_id', 'resolution', 'exp_method']
    print(results_df[cols].head(10).to_string())

    if pdb_count > 0:
        print(f"\nSample PDB hits:")
        print(pdb_rows[cols].head(10).to_string())

    if CONFIG['trace_ids']:
        print("\nTracing proteins of interest...")
        trace_proteins(CONFIG['trace_ids'])

    # Save missing
    missing_df = results_df[~results_df['available']]
    if len(missing_df) > 0:
        missing_path = Path(CONFIG['missing_file'])
        missing_df[['uniprot_id']].to_csv(missing_path, index=False)
        print(f"\nMissing proteins saved: {missing_path}")

    # Download if requested
    if CONFIG['download_structures']:
        print(f"\nDownloading structures to {CONFIG['structures_dir']}...")
        structures_dir = Path(CONFIG['structures_dir'])
        structures_dir.mkdir(parents=True, exist_ok=True)

        available_results = [r for r in results if r['available']]
        downloaded = 0
        skipped = 0
        failed = 0

        for result in tqdm(available_results, desc="Downloading"):
            url = result.get('url')
            if not url:
                failed += 1
                continue

            uid = result['uniprot_id']

            # ── v3 CHANGE: save as {uid}.cif (no source suffix) ──
            filepath = structures_dir / f"{uid}.cif"

            if filepath.exists():
                skipped += 1
                downloaded += 1
                continue

            r = api_get(url)
            if r and r.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(r.content)
                downloaded += 1
            else:
                failed += 1
            time.sleep(0.2)

        print(f"Downloaded: {downloaded} (skipped existing: {skipped}), Failed: {failed}")

    return results_df

# ============================================================================
# DEBUG: TRACE SPECIFIC PROTEINS
# ============================================================================

def trace_proteins(uniprot_ids: list,
                   resolution_cutoff: float = None,
                   accept_nmr: bool = None) -> pd.DataFrame:
    """
    Process a short list of UniProt IDs and print results in the same
    format as the 'Sample results' table in main().

    Can be called independently of main(), e.g.:
        from pdb_structures_v3 import trace_proteins
        trace_proteins(['O15392', 'Q53HL2', 'P07333'])
    """
    res_cutoff = resolution_cutoff if resolution_cutoff is not None else CONFIG['pdb_resolution_cutoff']
    nmr        = accept_nmr      if accept_nmr      is not None else CONFIG['accept_nmr']

    print(f"Tracing {len(uniprot_ids)} proteins "
          f"(resolution ≤ {res_cutoff} Å, NMR={'yes' if nmr else 'no'})...")

    results = []
    for uid in tqdm(uniprot_ids, desc="Querying"):
        results.append(process_protein(uid, res_cutoff, nmr))

    df = pd.DataFrame(results)

    cols = ['uniprot_id', 'source', 'structure_id', 'resolution', 'exp_method']
    print("\nResults:")
    print(df[cols].to_string(index=False))

    return df

if __name__ == '__main__':
    main()
