"""
Repeated random structural-group-level splits -- ChEMBL drug-target labels.

All logic lives in split_common.py; this file is configuration only.

LABELS: single-protein ChEMBL targets only (primary analysis). Complex-level
(PROTEIN COMPLEX) targets are NOT in this set, so no positive label is shared
across the train/test boundary. To split the mixed set for the Supplementary,
point 'label_file' at the _mixed CSV and change the output paths.

NOTE: corum_drug_targets_chembl.py currently writes to the rotations/gesine
repository with a _single / _mixed suffix. 'label_file' below must point at
whichever file that script actually produced.
"""

from split_common import run_splits

BASE = '/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables'

CONFIG = {
    'task_description': 'Drug Target Labels (ChEMBL, single-protein targets only)',

    # ---- Input files -------------------------------------------------------
    'group_mapping':      f'{BASE}/corum/mapping_struct.csv',
    'complex_membership': f'{BASE}/corum/stoich_protein.csv',
    'label_file':         f'{BASE}/corum_drug_target_chembl_single.csv',

    # ---- Input column names ------------------------------------------------
    'col_group_protein':   'uniprot_id',
    'col_group_id':        'group_id',
    'col_complex_id':      'ComplexId',
    'col_complex_protein': 'ProteinId',
    'col_label_protein':   'ProteinId',
    'col_label_value':     'target',

    # ---- Label scheme ------------------------------------------------------
    'label_scheme': {
        'positive': 'Drug_target',
        'negative': 'Non_target',
        'noun':     'drug target',
        # 1 -> positive, anything else -> negative. Never returns None:
        # every protein in the master list carries an explicit 0 or 1.
        'binarise': lambda v: 'Drug_target' if v == 1 else 'Non_target',
    },

    # ---- Output files ------------------------------------------------------
    'output_all_splits':   f'{BASE}/corum/chembl_protein_splits.csv',
    'output_balance':      f'{BASE}/corum/chembl_split_balance.csv',
    'output_summary':      f'{BASE}/corum/chembl_split_summary.txt',
    'output_rate_by_size': f'{BASE}/corum/avg_chembl_by_group_size.csv',

    # ---- Split parameters --------------------------------------------------
    'n_splits':    50,
    'train_ratio': 0.80,
    'test_ratio':  0.20,
    'base_seed':   42,

    # Flag a split if test set size deviates more than this from 20%
    'max_size_deviation': 5.0,   # pp -- acceptable range [15%, 25%]

    # Flag a split if the test positive rate deviates more than this from the
    # global rate. Tight because the global rate is low. If the printed global
    # rate moves materially, revisit -- too tight and every attempt fails and
    # falls through to the flagged-best-candidate path.
    'max_label_rate_deviation': 1.5,

    'max_attempts': 25,

    # ---- Validation tuning -------------------------------------------------
    'validate_abs_tol':            0.15,
    'validate_min_groups':         3,
    'min_test_groups_per_bucket':  1,
    'strict_duplicate_check':      False,
    'require_full_n_splits':       True,

    # ---- Bucket ablation ---------------------------------------------------
    # False pools all groups and ignores size buckets; output paths gain the
    # '_unstrat' suffix so the stratified files are never overwritten.
    'stratify_by_bucket': True,
    'unstrat_suffix':     '_unstrat',
}


if __name__ == '__main__':
    run_splits(CONFIG)