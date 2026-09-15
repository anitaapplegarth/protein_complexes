"""
Repeated random structural-group-level splits -- essentiality labels.

All logic lives in split_common.py; this file is configuration only.

LABELS: OGEE v3 + DepMap. Core -> Essential; Cancer and Non-essential ->
Non-essential; anything else -> dropped to Unknown and excluded from the
analysis population. This is the binarisation used throughout the paper.

NOTE: this is the RANDOM structural-group-atomic split. The complex-atomic
leak-free variant used for the supplementary robustness paragraph lives in
cp_complexatomic_ess_split.py and is a different scheme -- do not confuse the
output files.
"""

from split_common import run_splits

BASE = '/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables'


def binarise_essentiality(raw):
    """Core -> Essential; Cancer & Non-essential -> Non-essential; else drop."""
    if raw == 'Core':
        return 'Essential'
    if raw in ('Cancer', 'Non-essential'):
        return 'Non-essential'
    return None


CONFIG = {
    'task_description': 'Essentiality Labels (OGEE v3 + DepMap)',

    # ---- Input files -------------------------------------------------------
    'group_mapping':      f'{BASE}/cp/mapping_struct.csv',
    'complex_membership': f'{BASE}/cp/stoich_protein.csv',
    'label_file':         f'{BASE}/lu_essentiality_protein.csv',

    # ---- Input column names ------------------------------------------------
    # NOTE: the essentiality table keys on 'Protein', not 'ProteinId' as the
    # drug-target tables do. This asymmetry is the reason the column names are
    # configurable rather than hardcoded.
    'col_group_protein':   'uniprot_id',
    'col_group_id':        'group_id',
    'col_complex_id':      'ComplexId',
    'col_complex_protein': 'ProteinId',
    'col_label_protein':   'Protein',
    'col_label_value':     'essential_category',

    # ---- Label scheme ------------------------------------------------------
    'label_scheme': {
        'positive': 'Essential',
        'negative': 'Non-essential',
        'noun':     'essential',
        'binarise': binarise_essentiality,
    },

    # ---- Output files ------------------------------------------------------
    'output_all_splits':   f'{BASE}/cp/ess_protein_splits.csv',
    'output_balance':      f'{BASE}/cp/ess_split_balance.csv',
    'output_summary':      f'{BASE}/cp/ess_split_summary.txt',
    'output_rate_by_size': f'{BASE}/cp/avg_essentiality_by_group_size.csv',

    # ---- Split parameters --------------------------------------------------
    'n_splits':    50,
    'train_ratio': 0.80,
    'test_ratio':  0.20,
    'base_seed':   42,

    'max_size_deviation': 5.0,   # pp -- acceptable range [15%, 25%]

    # Looser than the drug-target tasks (4.5pp vs 1.5pp) because the global
    # essential rate is far higher, so the same absolute pp tolerance is
    # proportionally much tighter. Renamed from 'max_ess_rate_deviation' to
    # match the shared engine.
    'max_label_rate_deviation': 4.5,

    'max_attempts': 25,

    # ---- Validation tuning -------------------------------------------------
    'validate_abs_tol':            0.15,
    'validate_min_groups':         3,
    'min_test_groups_per_bucket':  1,
    'strict_duplicate_check':      False,
    'require_full_n_splits':       True,

    # ---- Bucket ablation ---------------------------------------------------
    'stratify_by_bucket': True,
    'unstrat_suffix':     '_unstrat',
}


if __name__ == '__main__':
    run_splits(CONFIG)