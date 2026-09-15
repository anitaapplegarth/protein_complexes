"""
=======================================================================
PROTEIN-LEVEL RANDOM, LABEL-STRATIFIED SPLITS  (the leakage comparison)
=======================================================================
This is the naive machine-learning baseline referred to in the paper's
"Random-based splitting comparison": proteins are assigned to train/test
INDIVIDUALLY and AT RANDOM, ignoring structural groups entirely, with the
draw stratified by label class so the positive rate is held constant.

Because groups are ignored, members of the same structural group land on
both sides of the split. That is the whole point: this is the leakage the
group-based scheme prevents, and the group-vs-random PR-AUC gap is the
inflation it produces.

RELATION TO THE OTHER SPLITTERS
-------------------------------
  cp_random_ess_split.py            group-atomic, size-STRATIFIED  (headline)
  cp_random_ess_split_unstrat.*     group-atomic, size-UNstratified (bucket
                                    ablation -- groups still atomic!)
  THIS FILE                         protein-level random, groups IGNORED
                                    (the leakage baseline)

The first two never break a group across the boundary; only this one does.
The unstrat file is NOT the leakage baseline -- do not use it as one.

DESIGN
------
Reuses split_common's loaders, balance statistics and output schema, so the
resulting CSV is drop-in readable by cp_model_sweep_unstrat.py. Only the
allocation step differs:

  * label-stratified random assignment of proteins (pos / neg / Unknown each
    split test_ratio into test), groups ignored;
  * leakage is COUNTED and reported, never used to reject a split (rejecting
    on leakage here would loop forever, since leakage is guaranteed);
  * label-rate and size tolerances remain accept criteria, with the same
    seed = base_seed + i + attempts*1000 scheme as the group splitters, so
    the arms share a seed lineage.

Output columns match split_common exactly:
  split_index, seed_accepted, UniProt_AC, group_id, group_size,
  group_bucket, group_status, split, protein_label, label_mask
plus a balance CSV; group_id etc. are still recorded (for reference) even
though they play no part in the allocation.
"""

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from split_common import load_inputs, balance_stats, size_bucket

# ======================================================================
# CONFIG  — edit here, then run the whole file in the interactive window.
# ======================================================================
BASE = '/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables'


def binarise_essentiality(raw):
    """Core -> Essential; Cancer & Non-essential -> Non-essential; else drop."""
    if raw == 'Core':
        return 'Essential'
    if raw in ('Cancer', 'Non-essential'):
        return 'Non-essential'
    return None


CONFIG = {
    'task_description': 'Essentiality Labels (OGEE v3 + DepMap) — PROTEIN-LEVEL RANDOM',

    # ---- Input files -------------------------------------------------------
    'group_mapping':      f'{BASE}/cp/mapping_struct.csv',
    'complex_membership': f'{BASE}/cp/stoich_protein.csv',
    'label_file':         f'{BASE}/lu_essentiality_protein.csv',

    # ---- Input column names ------------------------------------------------
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
    # NOTE distinct filenames so this cannot overwrite the group-based or the
    # bucket-ablation splits.
    'output_all_splits': f'{BASE}/cp/ess_protein_splits_randomprotein.csv',
    'output_balance':    f'{BASE}/cp/ess_split_balance_randomprotein.csv',

    # ---- Split parameters --------------------------------------------------
    'n_splits':    50,
    'train_ratio': 0.80,
    'test_ratio':  0.20,
    'base_seed':   42,

    'max_size_deviation':       5.0,   # pp
    'max_label_rate_deviation': 4.5,   # pp (matches the essentiality group split)
    'max_attempts':             25,

    'require_full_n_splits': True,
}


# ----------------------------------------------------------------------
# Allocation: label-stratified random over PROTEINS, groups ignored.
# ----------------------------------------------------------------------
def attempt_protein_split(all_proteins, protein_to_label, analysis_pids,
                          test_ratio, rng):
    """One protein-level, label-stratified random assignment.

    Every protein is placed independently; group membership plays no part.
    Stratification is by label class (positive / negative / Unknown) so the
    test positive rate tracks the global rate by construction.

    Returns (protein_to_split, test_pos_rate, test_size_frac).
    """
    scheme  = CONFIG['label_scheme']
    pos_lbl = scheme['positive']
    neg_lbl = scheme['negative']

    protein_to_split = {}
    for lbl in (pos_lbl, neg_lbl, 'Unknown'):
        members = sorted(p for p in all_proteins
                         if protein_to_label.get(p, 'Unknown') == lbl)
        rng.shuffle(members)
        n_test = round(len(members) * test_ratio)
        for i, pid in enumerate(members):
            protein_to_split[pid] = 'test' if i < n_test else 'train'

    test_ap   = [p for p, s in protein_to_split.items()
                 if s == 'test' and p in analysis_pids]
    ap_counts = Counter(protein_to_label[p] for p in test_ap)
    ap_lab    = ap_counts[pos_lbl] + ap_counts[neg_lbl]
    test_pos_rate = ap_counts[pos_lbl] / ap_lab if ap_lab > 0 else 0.5

    total = len(protein_to_split)
    test_size_frac = (sum(1 for s in protein_to_split.values() if s == 'test')
                      / total if total else 0.0)
    return protein_to_split, test_pos_rate, test_size_frac


def count_leaking_groups(protein_to_split, protein_to_group):
    """How many structural groups straddle the train/test boundary.

    Reported, never used to reject a split -- leakage is the expected
    behaviour of this scheme. Returns (n_leaking, n_groups_present).
    """
    grp_sides = {}
    for pid, split in protein_to_split.items():
        grp = protein_to_group.get(pid)
        if grp is None:
            continue
        grp_sides.setdefault(grp, set()).add(split)
    n_leaking = sum(1 for sides in grp_sides.values() if len(sides) > 1)
    return n_leaking, len(grp_sides)


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
def run_protein_random_splits(config):
    C      = dict(config)
    scheme = C['label_scheme']
    noun   = scheme['noun']

    print("=" * 70)
    print(f"PROTEIN-LEVEL RANDOM (label-stratified) splitting, N={C['n_splits']}")
    print(C['task_description'])
    print("=" * 70)

    D = load_inputs(C)
    protein_to_group = D['protein_to_group']
    protein_to_label = D['protein_to_label']
    group_sizes      = D['group_sizes']
    all_proteins     = D['all_proteins']
    analysis_pids    = D['analysis_pids']
    global_pos_ratio = D['global_pos_ratio']
    global_rate_pct  = D['global_rate_pct']

    print(f"\n--- Allocation mode: PROTEIN-LEVEL RANDOM (groups ignored) ---")
    print(f"--- Generating {C['n_splits']} splits "
          f"(tolerance +/-{C['max_label_rate_deviation']}pp label, "
          f"+/-{C['max_size_deviation']}pp size, "
          f"max {C['max_attempts']} attempts each) ---")

    all_rows        = []
    balance_records = []
    n_forced        = 0

    for i in range(C['n_splits']):
        best, best_key, accepted, attempts = None, None, False, 0

        while attempts < C['max_attempts']:
            seed = C['base_seed'] + i + attempts * 1000
            rng  = np.random.default_rng(seed)
            p2s, pos_rate, sz_frac = attempt_protein_split(
                all_proteins, protein_to_label, analysis_pids,
                C['test_ratio'], rng)
            attempts += 1

            sz_dev  = abs(100 * sz_frac  - 100 * C['test_ratio'])
            dev     = abs(100 * pos_rate - global_rate_pct)
            size_ok = sz_dev <= C['max_size_deviation']
            lbl_ok  = dev    <= C['max_label_rate_deviation']

            cand = {'p2s': p2s, 'seed': seed, 'sz_dev': sz_dev, 'dev': dev}

            if size_ok and lbl_ok:
                accepted, best = True, cand
                break
            # no leakage veto here -- rank by label then size deviation only
            key = (not size_ok, dev, sz_dev)
            if best_key is None or key < best_key:
                best_key, best = key, cand

        protein_to_split = best['p2s']
        if not accepted:
            n_forced += 1
            print(f"  !! Split {i+1}: no candidate within tolerance in "
                  f"{attempts} attempts; emitting best "
                  f"(label_dev={best['dev']:.2f}pp, size_dev={best['sz_dev']:.2f}pp).")

        bstats = balance_stats(protein_to_split, protein_to_label,
                               global_pos_ratio, analysis_pids, C)
        n_leak, n_grp = count_leaking_groups(protein_to_split, protein_to_group)
        leak_pct = 100 * n_leak / n_grp if n_grp else 0.0

        balance_records.append({
            'split_index':            i + 1,
            'seed_accepted':          best['seed'],
            'attempts':               attempts,
            'accepted':               accepted,
            'train_n':                bstats['train']['n_proteins'],
            'train_pct':              bstats['train']['size_pct'],
            'test_n':                 bstats['test']['n_proteins'],
            'test_pct':               bstats['test']['size_pct'],
            'test_size_deviation_pp': bstats['test_size_deviation_pp'],
            'size_warning':           bstats['size_warning'],
            'train_n_positive':       bstats['train']['n_positive'],
            'test_n_positive':        bstats['test']['n_positive'],
            'train_n_negative':       bstats['train']['n_negative'],
            'test_n_negative':        bstats['test']['n_negative'],
            'train_pos_rate_pct':     bstats['train']['pos_rate_pct'],
            'test_pos_rate_pct':      bstats['test']['pos_rate_pct'],
            'test_deviation_pp':      bstats['test_deviation_pp'],
            'warning':                bstats['warning'],
            'n_leaking_groups':       n_leak,
            'n_groups_present':       n_grp,
            'leaking_group_pct':      round(leak_pct, 1),
        })

        retry = f' (accepted on attempt {attempts})' if attempts > 1 else ''
        flag  = ' !! label warning' if bstats['warning']      else ''
        sflag = ' !! size warning'  if bstats['size_warning'] else ''
        print(f"  Split {i+1:2d} | "
              f"train={bstats['train']['n_proteins']} ({bstats['train']['size_pct']:.1f}%) "
              f"pos={bstats['train']['pos_rate_pct']:.1f}% | "
              f"test={bstats['test']['n_proteins']} ({bstats['test']['size_pct']:.1f}%) "
              f"pos={bstats['test']['pos_rate_pct']:.1f}% | "
              f"leak={n_leak}/{n_grp} groups ({leak_pct:.0f}%)"
              f"{retry}{sflag}{flag}")

        for pid in all_proteins:
            grp    = protein_to_group.get(pid)
            grp_sz = group_sizes.get(grp) if grp else None
            all_rows.append({
                'split_index':   i + 1,
                'seed_accepted': best['seed'],
                'UniProt_AC':    pid,
                'group_id':      grp,
                'group_size':    grp_sz,
                'group_bucket':  size_bucket(grp_sz) if grp_sz else None,
                'group_status':  'constrained' if grp else 'no_group',
                'split':         protein_to_split[pid],
                'protein_label': protein_to_label.get(pid, 'Unknown'),
                'label_mask':    pid in analysis_pids,
            })

    # -- Integrity gate -------------------------------------------------------
    n_emitted = len(balance_records)
    assert n_emitted == C['n_splits'], \
        f"expected {C['n_splits']} splits, built {n_emitted}"
    if C.get('require_full_n_splits', True) and n_forced > 0:
        raise RuntimeError(
            f"{n_forced} split(s) never met the label/size tolerance. "
            f"Loosen max_label_rate_deviation / max_size_deviation, raise "
            f"max_attempts, or set require_full_n_splits=False to save anyway.")

    splits_df = pd.DataFrame(all_rows)
    splits_df.to_csv(C['output_all_splits'], index=False)
    pd.DataFrame(balance_records).to_csv(C['output_balance'], index=False)

    leak = pd.DataFrame(balance_records)['leaking_group_pct']
    print(f"\nWrote {C['output_all_splits']}  ({len(splits_df)} rows)")
    print(f"Wrote {C['output_balance']}")
    print(f"\nLeaking groups across {C['n_splits']} splits: "
          f"mean {leak.mean():.1f}%  (min {leak.min():.1f}%, max {leak.max():.1f}%)")
    print("This is the leakage the group-based scheme removes; the PR-AUC gap "
          "between the two arms is the inflation it produces.")


run_protein_random_splits(CONFIG)