"""
Repeated random structural-group-level splits -- HPA drug-target labels.

All logic lives in split_common.py; this file is configuration only.

LABELS: Human Protein Atlas drug targets, kept strictly separate from the
ChEMBL set (Jaccard ~31%). Nothing is shared between the two label
derivations; they are independent prediction tasks.
"""

from split_common import run_splits

BASE = '/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables'

CONFIG = {
    'task_description': 'Drug Target Labels (HPA)',

    # ---- Input files -------------------------------------------------------
    'group_mapping':      f'{BASE}/corum/mapping_struct.csv',
    'complex_membership': f'{BASE}/corum/stoich_protein.csv',
    'label_file':         f'{BASE}/corum_drug_target_hpa.csv',

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
        'binarise': lambda v: 'Drug_target' if v == 1 else 'Non_target',
    },

    # ---- Output files ------------------------------------------------------
    'output_all_splits':   f'{BASE}/corum/hpa_protein_splits.csv',
    'output_balance':      f'{BASE}/corum/hpa_split_balance.csv',
    'output_summary':      f'{BASE}/corum/hpa_split_summary.txt',
    'output_rate_by_size': f'{BASE}/corum/avg_hpa_by_group_size.csv',

    # ---- Split parameters --------------------------------------------------
    'n_splits':    50,
    'train_ratio': 0.80,
    'test_ratio':  0.20,
    'base_seed':   42,

    'max_size_deviation': 5.0,   # pp -- acceptable range [15%, 25%]

    # Check the printed global rate on the first run. If HPA's positive rate
    # differs materially from ChEMBL's, 1.5pp is a different level of
    # strictness and should be retuned -- roughly a tenth of the global rate
    # is a reasonable starting point.
    'max_label_rate_deviation': 1.5,

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