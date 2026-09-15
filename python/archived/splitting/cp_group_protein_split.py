"""
Naive random PROTEIN-level splits -- leakage-comparison ablation (CP only).

Purpose
-------
Baseline arm for the split-regime comparison. Every protein is drawn
independently, exactly as a standard ML train/test split would do. Structural
groups and complexes are therefore SPLIT ACROSS the train/test boundary --
this is the leakage the group-atomic scheme removes, and quantifying it is
the point of this script.

This is deliberately NOT the 'unstratified' arm in cp_random_*_split_unstrat.py.
That arm still keeps structural groups atomic and only removes size-bucket
allocation. Three regimes:

    random-protein   (this file)  -- nothing atomic; groups and complexes leak
    group-atomic, unstratified    -- groups atomic; no bucket balancing
    group-atomic, stratified      -- groups atomic; bucket balancing (main)

Output schema is byte-compatible with split_common.py -- same nine columns,
same dtypes, same master protein list (all proteins in the complex membership
file), same 'Unknown' string for unlabelled proteins, same boolean label_mask.
The existing Test A / modelling scripts consume it unchanged.

Modelling code is NOT touched. Only the splits file changes.

Run: set TASK below, then run directly in VS Code.
"""

import numpy as np
import pandas as pd

# size_bucket is imported rather than redefined so the bucket boundaries can
# never drift apart from the group-atomic engine.
from split_common import size_bucket


# ============================================================================
# TASK SELECTION
# ============================================================================
TASK = 'hpa'          # 'chembl' or 'ess'

BASE = '/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables'


# ============================================================================
# CONFIG
# ============================================================================
CONFIG_CHEMBL = {
    'task_description': 'Drug Target Labels (ChEMBL, single-protein targets only)',
    'task_key':         'chembl',

    'group_mapping':      f'{BASE}/cp/mapping_struct.csv',
    'complex_membership': f'{BASE}/cp/stoich_protein.csv',
    'label_file':         f'{BASE}/cp_drug_target_chembl_single.csv',

    'col_group_protein':   'uniprot_id',
    'col_group_id':        'group_id',
    'col_complex_id':      'ComplexId',
    'col_complex_protein': 'ProteinId',
    'col_label_protein':   'ProteinId',
    'col_label_value':     'target',

    'label_scheme': {
        'positive': 'Drug_target',
        'negative': 'Non_target',
        'noun':     'drug target',
        'binarise': lambda v: 'Drug_target' if v == 1 else 'Non_target',
    },

    # Group-atomic splits file, read only for the schema audit and the
    # side-by-side leakage diagnostic. Set to None to skip.
    'reference_splits': f'{BASE}/cp/chembl_protein_splits.csv',
}

CONFIG_HPA = {
    'task_description': 'Drug Target Labels (HPA)',
    'task_key':         'hpa',

    'group_mapping':      f'{BASE}/cp/mapping_struct.csv',
    'complex_membership': f'{BASE}/cp/stoich_protein.csv',
    'label_file':         f'{BASE}/cp_drug_target_hpa.csv',

    'col_group_protein':   'uniprot_id',
    'col_group_id':        'group_id',
    'col_complex_id':      'ComplexId',
    'col_complex_protein': 'ProteinId',
    'col_label_protein':   'ProteinId',
    'col_label_value':     'target',

    'label_scheme': {
        'positive': 'Drug_target',
        'negative': 'Non_target',
        'noun':     'drug target',
        'binarise': lambda v: 'Drug_target' if v == 1 else 'Non_target',
    },

    'reference_splits': f'{BASE}/cp/hpa_protein_splits.csv',
}

CONFIG_ESS = {
    'task_description': 'Gene Essentiality Labels (OGEE v3 + DepMap)',
    'task_key':         'ess',

    'group_mapping':      f'{BASE}/cp/mapping_struct.csv',
    'complex_membership': f'{BASE}/cp/stoich_protein.csv',

    # <<< CONFIRM: copy these four lines from cp_random_ess_split.py >>>
    'label_file':         f'{BASE}/lu_essentiality_protein.csv',
    'col_group_protein':   'uniprot_id',
    'col_group_id':        'group_id',
    'col_complex_id':      'ComplexId',
    'col_complex_protein': 'ProteinId',
    'col_label_protein':   'Protein',
    'col_label_value':     'essential_category',

    # <<< CONFIRM: copy the label_scheme block verbatim from that file too.
    # If the raw values differ from the guess below the Unknown count will not
    # come out at 83 and the analysis population will not be 3306 -- that is
    # the check that this mapping is right. >>>
    'label_scheme': {
        'positive': 'Essential',
        'negative': 'Non-essential',
        'noun':     'essential',
        'binarise': lambda v: (
            'Essential'     if v == 'Core' else
            'Non-essential' if v in ('Cancer', 'Non-essential') else
            None
        ),
    },

    'reference_splits': f'{BASE}/cp/ess_protein_splits.csv',
}

COMMON = {
    'n_splits':    50,
    'train_ratio': 0.80,
    'test_ratio':  0.20,
    'base_seed':   42,        # same seed family as the group-atomic runs

    # Standard ML practice is train_test_split(..., stratify=y). Keeping this
    # True holds the per-split positive COUNT essentially constant, so any
    # PR-AUC difference against the group-atomic arms is attributable to
    # leakage rather than to positive-count variance. It also makes the naive
    # baseline as strong as it reasonably can be, which is the point -- a
    # weakened baseline would invite the obvious objection.
    'stratify_by_label': True,

    'output_dir': f'{BASE}/cp',
    'suffix':     '_randprot',
}


# ============================================================================
# LOADING  (mirrors split_common.load_data)
# ============================================================================
def load_inputs(config):
    """Build the master protein table exactly as split_common does.

    Master list = every protein in the complex membership file. Labels are
    merged on; anything missing or dropped by binarise() becomes the string
    'Unknown'. Analysis population = labelled AND structured.
    """
    scheme  = config['label_scheme']
    pos_lbl = scheme['positive']
    neg_lbl = scheme['negative']
    noun    = scheme['noun']

    groups    = pd.read_csv(config['group_mapping'])
    complexes = pd.read_csv(config['complex_membership'])
    labels    = pd.read_csv(config['label_file'])

    g_pid, g_gid = config['col_group_protein'],  config['col_group_id']
    x_cid, x_pid = config['col_complex_id'],     config['col_complex_protein']
    l_pid, l_val = config['col_label_protein'],  config['col_label_value']

    for df, cols, name in ((groups,    (g_pid, g_gid), config['group_mapping']),
                           (complexes, (x_cid, x_pid), config['complex_membership']),
                           (labels,    (l_pid, l_val), config['label_file'])):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"{name}: missing column(s) {missing}. "
                           f"Present: {list(df.columns)}")

    print(f"  Groups:    {len(groups)} rows, {groups[g_gid].nunique()} unique groups")
    print(f"  Complexes: {complexes[x_cid].nunique()} complexes, "
          f"{complexes[x_pid].nunique()} unique proteins")
    print(f"  Labels:    {len(labels)} rows from {config['label_file']}")

    complex_proteins = set(complexes[x_pid].unique())

    g_sub = groups[groups[g_pid].isin(complex_proteins)]
    protein_to_group = dict(zip(g_sub[g_pid], g_sub[g_gid]))

    group_sizes = {}
    for grp in protein_to_group.values():
        group_sizes[grp] = group_sizes.get(grp, 0) + 1

    binarise = scheme['binarise']
    protein_to_label = {}
    for pid, raw in zip(labels[l_pid], labels[l_val]):
        bl = binarise(raw)
        if bl is not None:
            protein_to_label[pid] = bl

    all_proteins    = sorted(complex_proteins)
    structured_pids = set(protein_to_group.keys())
    analysis_pids = {
        p for p in all_proteins
        if p in structured_pids and protein_to_label.get(p) in (pos_lbl, neg_lbl)
    }

    n_pos_global  = sum(1 for p in analysis_pids if protein_to_label[p] == pos_lbl)
    n_structured  = len(analysis_pids)
    n_unknown_all = sum(1 for p in all_proteins
                        if protein_to_label.get(p, 'Unknown') == 'Unknown')
    n_no_group    = len(complex_proteins - structured_pids)
    n_pos_all = sum(1 for p in all_proteins if protein_to_label.get(p) == pos_lbl)
    n_neg_all = sum(1 for p in all_proteins if protein_to_label.get(p) == neg_lbl)

    print(f"\n  Label distribution ({len(all_proteins)} total complex proteins):")
    print(f"    {pos_lbl:<15s}: {n_pos_all:5d}")
    print(f"    {neg_lbl:<15s}: {n_neg_all:5d}")
    print(f"    {'Unknown':<15s}: {n_unknown_all:5d}  (no usable label -- excluded)")
    print(f"    {'No-group':<15s}: {n_no_group:5d}  (no structural assignment -- excluded)")
    print(f"  Analysis population (labelled + structured): {n_structured}")
    print(f"  Global {noun} rate (analysis population): "
          f"{100 * n_pos_global / n_structured:.1f}%  "
          f"({n_pos_global} / {n_structured})")

    # Complex membership as sets, for the leakage diagnostic only.
    complex_of = complexes.groupby(x_pid)[x_cid].apply(set).to_dict()

    return {
        'all_proteins':     all_proteins,
        'protein_to_group': protein_to_group,
        'group_sizes':      group_sizes,
        'protein_to_label': protein_to_label,
        'analysis_pids':    analysis_pids,
        'complex_of':       complex_of,
        'n_structured':     n_structured,
        'n_pos_global':     n_pos_global,
    }


# ============================================================================
# SPLITTING
# ============================================================================
def assign_one_split(D, seed, config, common):
    """Draw one naive split. Returns dict[pid -> 'train'|'test'].

    Every protein is drawn independently -- nothing is held atomic. Proteins
    outside the analysis population are drawn too (split_common assigns them
    by stratified random sampling as well) so that row counts match; they are
    masked out downstream.
    """
    scheme  = config['label_scheme']
    pos_lbl = scheme['positive']
    rng = np.random.default_rng(seed)

    all_proteins  = D['all_proteins']
    analysis_pids = D['analysis_pids']
    ptl           = D['protein_to_label']

    if common['stratify_by_label']:
        strata = {'pos': [], 'neg': [], 'unusable': []}
        for pid in all_proteins:
            if pid not in analysis_pids:
                strata['unusable'].append(pid)
            elif ptl[pid] == pos_lbl:
                strata['pos'].append(pid)
            else:
                strata['neg'].append(pid)
    else:
        strata = {'all': list(all_proteins)}

    protein_to_split = {}
    for members in strata.values():
        if not members:
            continue
        arr = np.array(sorted(members))      # canonical order before the draw
        rng.shuffle(arr)
        n_test = int(round(len(arr) * common['test_ratio']))
        for pid in arr[:n_test]:
            protein_to_split[pid] = 'test'
        for pid in arr[n_test:]:
            protein_to_split[pid] = 'train'

    return protein_to_split


def generate_splits(D, config, common):
    """Generate all splits; return the long splits frame and balance records."""
    scheme  = config['label_scheme']
    pos_lbl = scheme['positive']
    analysis_pids    = D['analysis_pids']
    protein_to_group = D['protein_to_group']
    group_sizes      = D['group_sizes']
    ptl              = D['protein_to_label']

    all_rows, balance_records = [], []

    print(f"\n--- Generating {common['n_splits']} naive random "
          f"protein-level splits ---")
    if common['stratify_by_label']:
        print("    label-stratified (standard sklearn practice); "
              "no redraw loop -- every draw is valid by construction")

    for i in range(common['n_splits']):
        seed = common['base_seed'] + i
        pts = assign_one_split(D, seed, config, common)

        ap_train = [p for p in analysis_pids if pts[p] == 'train']
        ap_test  = [p for p in analysis_pids if pts[p] == 'test']
        n_tr, n_te = len(ap_train), len(ap_test)
        tr_pos = sum(1 for p in ap_train if ptl[p] == pos_lbl)
        te_pos = sum(1 for p in ap_test  if ptl[p] == pos_lbl)

        print(f"  Split {i + 1:2d} | train={n_tr:4d} "
              f"({100 * n_tr / (n_tr + n_te):.1f}%) "
              f"pos={100 * tr_pos / n_tr:.1f}% | "
              f"test={n_te:4d} ({100 * n_te / (n_tr + n_te):.1f}%) "
              f"pos={100 * te_pos / n_te:.1f}% | test pos n={te_pos}")

        balance_records.append({
            'split_index':       i + 1,
            'seed_accepted':     seed,
            'train_n':           n_tr,
            'test_n':            n_te,
            'train_pct':         100 * n_tr / (n_tr + n_te),
            'test_pct':          100 * n_te / (n_tr + n_te),
            'train_n_positive':  tr_pos,
            'test_n_positive':   te_pos,
            'train_n_negative':  n_tr - tr_pos,
            'test_n_negative':   n_te - te_pos,
            'train_pos_rate_pct': 100 * tr_pos / n_tr,
            'test_pos_rate_pct':  100 * te_pos / n_te,
        })

        for pid in D['all_proteins']:
            grp    = protein_to_group.get(pid)
            grp_sz = group_sizes.get(grp) if grp else None
            all_rows.append({
                'split_index':   i + 1,
                'seed_accepted': seed,
                'UniProt_AC':    pid,
                'group_id':      grp,
                'group_size':    grp_sz,
                'group_bucket':  size_bucket(grp_sz) if grp_sz else None,
                'group_status':  'constrained' if grp else 'no_group',
                'split':         pts[pid],
                'protein_label': ptl.get(pid, 'Unknown'),
                'label_mask':    pid in analysis_pids,
            })

    return pd.DataFrame(all_rows), pd.DataFrame(balance_records)


# ============================================================================
# LEAKAGE DIAGNOSTIC
# ============================================================================
def leakage_diagnostic(splits_df, D, regime_name):
    """Per split, the fraction of analysis-population TEST proteins sharing a
    structural group -- and separately a complex -- with at least one TRAIN
    protein.

    This states the mechanism as a number rather than asserting it:
        random-protein  -> both near 1.0
        group-atomic    -> group exactly 0.000, complex still > 0
    """
    protein_to_group = D['protein_to_group']
    complex_of       = D['complex_of']

    rows = []
    for idx, block in splits_df.groupby('split_index'):
        ap = block[block['label_mask'].astype(bool)]
        train_ids = list(ap.loc[ap['split'] == 'train', 'UniProt_AC'])
        test_ids  = list(ap.loc[ap['split'] == 'test',  'UniProt_AC'])
        if not test_ids:
            continue

        train_groups = {protein_to_group.get(p) for p in train_ids}
        train_groups.discard(None)

        train_complexes = set()
        for p in train_ids:
            train_complexes |= complex_of.get(p, set())

        grp_hits = sum(1 for p in test_ids
                       if protein_to_group.get(p) in train_groups)
        cx_hits = sum(1 for p in test_ids
                      if complex_of.get(p, set()) & train_complexes)

        rows.append({
            'regime':      regime_name,
            'split_index': idx,
            'n_test':      len(test_ids),
            'frac_test_with_train_groupmate':   grp_hits / len(test_ids),
            'frac_test_with_train_complexmate': cx_hits / len(test_ids),
        })

    diag = pd.DataFrame(rows)
    g = diag['frac_test_with_train_groupmate']
    c = diag['frac_test_with_train_complexmate']
    print(f"\n--- Leakage diagnostic: {regime_name} ---")
    print(f"  Test proteins sharing a structural group with train: "
          f"{g.mean():.3f}  (min {g.min():.3f}, max {g.max():.3f})")
    print(f"  Test proteins sharing a complex with train:          "
          f"{c.mean():.3f}  (min {c.min():.3f}, max {c.max():.3f})")
    return diag


def audit_reference(config, splits_df, D):
    """Confirm schema compatibility with the group-atomic file and run the
    same diagnostic on it, so both regimes appear in one place."""
    path = config.get('reference_splits')
    if not path:
        return None
    try:
        ref = pd.read_csv(path)
    except FileNotFoundError:
        print(f"\n--- Schema audit: reference not found, skipped ---\n  {path}")
        return None

    same_cols = list(ref.columns) == list(splits_df.columns)
    same_rows = len(ref) == len(splits_df)
    print("\n--- Schema audit against group-atomic splits ---")
    print(f"  Reference: {path}")
    print(f"  Columns    {'MATCH' if same_cols else 'MISMATCH'}")
    if not same_cols:
        print(f"    ref={list(ref.columns)}")
        print(f"    new={list(splits_df.columns)}")
    print(f"  Rows       ref={len(ref):,}  new={len(splits_df):,}  "
          f"{'MATCH' if same_rows else 'MISMATCH -- investigate'}")
    print(f"  label_mask ref={ref['label_mask'].value_counts().to_dict()}")
    print(f"             new={splits_df['label_mask'].value_counts().to_dict()}")

    ref_diag = leakage_diagnostic(ref, D, 'group-atomic (reference)')
    return ref_diag


# ============================================================================
# MAIN
# ============================================================================
def run(config, common):
    print("=" * 74)
    print(f"Naive random PROTEIN-level splitting, N={common['n_splits']}")
    print(config['task_description'])
    print("=" * 74)

    print("\n--- Loading data ---")
    D = load_inputs(config)

    splits_df, balance_df = generate_splits(D, config, common)

    diag = leakage_diagnostic(splits_df, D, 'random-protein')
    ref_diag = audit_reference(config, splits_df, D)
    if ref_diag is not None:
        diag = pd.concat([diag, ref_diag], ignore_index=True)

    out, sfx, key = common['output_dir'], common['suffix'], config['task_key']
    splits_path  = f'{out}/{key}_protein_splits{sfx}.csv'
    balance_path = f'{out}/{key}_split_balance{sfx}.csv'
    diag_path    = f'{out}/{key}_leakage_diagnostic{sfx}.csv'

    print("\n--- Saving outputs ---")
    splits_df.to_csv(splits_path, index=False)
    print(f"  Saved: {splits_path}  ({len(splits_df):,} rows)")
    balance_df.to_csv(balance_path, index=False)
    print(f"  Saved: {balance_path}")
    diag.to_csv(diag_path, index=False)
    print(f"  Saved: {diag_path}")
    print("\nDone!")


if __name__ == '__main__':
    CONFIGS = {
        'chembl': CONFIG_CHEMBL,
        'ess':    CONFIG_ESS,
        'hpa':    CONFIG_HPA,
    }
    if TASK not in CONFIGS:
        raise ValueError(f"TASK must be one of {sorted(CONFIGS)}, got {TASK!r}")
    run(CONFIGS[TASK], COMMON)