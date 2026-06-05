"""
Repeated Random group-Level Splitting — Size-Stratified, N=15
==============================================================
Generates N=15 independent train/test splits where each Foldseek structural
group is allocated atomically (never split across train and test).

ALGORITHM — per split
---------------------
groups are divided into five size buckets (largest first):

  very_large : >50 proteins   (6 groups)
  large      : 21–50          (17 groups)
  medium     : 6–20           (97 groups)
  small      : 2–5            (274 groups)
  singleton  : 1 protein      (592 groups)

Within each bucket, exactly round(n_groups * 0.20) groups are drawn at
random for test; the remainder go to train. This is a pure random draw —
no essentiality information is used during the draw itself.

After all buckets are processed, overall label balance is checked.  If the
test essential rate deviates from the global rate by more than
`max_ess_rate_deviation` percentage points, the entire split is redrawn
(up to `max_attempts` tries). All 15 accepted splits are saved regardless
of whether they used retries.

The 6 proteins with no group assignment are allocated by stratified random
sampling (Essential / Non-essential / Unknown groups independently).

OUTPUTS
-------
  protein_splits_all_stratified.csv    : one row per (protein, split_index)
  split_balance_summary_stratified.csv : per-split label balance statistics
  split_summary_stratified.txt         : human-readable summary

"""

import pandas as pd
import numpy as np
from collections import defaultdict, Counter


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Input files
    'group_mapping':     '/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/cp_mapping_struct.csv',
    'complex_membership': '/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/cp_stoich_protein.csv',
    'essentiality':       '/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/lu_essentiality_protein.csv',


    # Output files
    'output_all_splits': '/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/cp_ess_protein_splits.csv',
    'output_balance':    '/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/cp_ess_split.csv',
    'output_summary':    '/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/cp_ess_split_summary.txt',

    # Number of independent splits to produce
    'n_splits': 15,

    # Train/test ratios
    'train_ratio': 0.80,
    'test_ratio':  0.20,

    # Flag a split if test essential rate deviates more than this from global
    # (percentage points). Flagged splits are redrawn up to max_attempts.
    'max_ess_rate_deviation': 3.0,

    # Maximum redraw attempts per split before accepting best available
    'max_attempts': 10,

    # Base seed — split i uses seed = base_seed + i for first attempt;
    # retries use base_seed + i + attempt * 1000
    'base_seed': 42,
}


# ============================================================================
# HELPERS
# ============================================================================

def binarise_label(raw_label):
    """Core -> Essential; Cancer & Non-essential -> Non-essential."""
    if raw_label == 'Core':
        return 'Essential'
    elif raw_label in ('Cancer', 'Non-essential'):
        return 'Non-essential'
    return None


def essential_ratio(counts):
    """Essential / labelled. Returns 0.5 if no labelled proteins."""
    labelled = counts.get('Essential', 0) + counts.get('Non-essential', 0)
    return counts.get('Essential', 0) / labelled if labelled > 0 else 0.5


def size_bucket(n_proteins):
    """Map group size to bucket name."""
    if n_proteins == 1:    return 'singleton'
    elif n_proteins <= 5:  return 'small'
    elif n_proteins <= 20: return 'medium'
    elif n_proteins <= 50: return 'large'
    else:                  return 'very_large'


# ============================================================================
# CORE SPLIT LOGIC — one attempt
# ============================================================================

def attempt_split(buckets, group_sizes, group_protein_map,
                  no_group_proteins, protein_to_label, config, rng):
    """
    Make one attempt at a train/test split.

    For each size bucket (largest first), randomly draw round(n * 0.20)
    groups into test; the rest go to train.  No-group proteins are
    split by stratified random sampling.

    Returns
    -------
    protein_to_split : dict[protein_id -> 'train'|'test']
    group_to_split  : dict[group_id  -> 'train'|'test']
    test_ess_rate    : float  (essential rate in test among labelled proteins)
    deviation_pp     : float  (abs deviation from global rate, percentage pts)
    """
    test_ratio = config['test_ratio']
    group_to_split = {}

    # --- Bucket-level random draw, largest bucket first ---
    for bucket_grps in buckets.values():
        if not bucket_grps:
            continue
        n_grps     = len(bucket_grps)
        n_test     = round(n_grps * test_ratio)   # e.g. round(6*0.2) = 1
        n_test     = max(n_test, 0)

        grps_list  = list(bucket_grps)
        rng.shuffle(grps_list)

        for i, grp in enumerate(grps_list):
            group_to_split[grp] = 'test' if i < n_test else 'train'

    # --- Assign proteins from their group's split ---
    protein_to_split = {}
    for grp, split in group_to_split.items():
        for pid in group_protein_map.get(grp, set()):
            protein_to_split[pid] = split

    # --- No-group proteins: stratified random sampling ---
    for group_label in ('Essential', 'Non-essential', 'Unknown'):
        group = sorted(
            p for p in no_group_proteins
            if protein_to_label.get(p, 'Unknown') == group_label
        )
        rng.shuffle(group)
        n_test_grp = round(len(group) * test_ratio)
        for i, pid in enumerate(group):
            protein_to_split[pid] = 'test' if i < n_test_grp else 'train'

    # --- Compute test label balance ---
    test_prots  = [p for p, s in protein_to_split.items() if s == 'test']
    test_counts = Counter(protein_to_label.get(p, 'Unknown') for p in test_prots)
    test_ess    = test_counts['Essential']
    test_lab    = test_ess + test_counts['Non-essential']
    test_ess_rate = test_ess / test_lab if test_lab > 0 else 0.5

    return protein_to_split, group_to_split, test_ess_rate


# ============================================================================
# VALIDATION
# ============================================================================

# def validate_split(protein_to_split, protein_to_group):
#     """Return dict of groups that span both splits (empty = clean)."""
#     group_splits = defaultdict(set)
#     for pid, split in protein_to_split.items():
#         grp = protein_to_group.get(pid)
#         if grp is not None:
#             group_splits[grp].add(split)


def validate_split(protein_to_split, protein_to_group, group_sizes, config):
    """
    Validate a proposed split.

    Args:
        protein_to_split: dict mapping protein_id -> 'train'/'test'
        protein_to_group: dict mapping protein_id -> group_id (or None)
        group_sizes: dict mapping group_id -> size (int)
        config: dict with at least 'test_fraction' (float, e.g. 0.2)

    Returns:
        offending_groups: set of group_ids that were split across train/test (leakage)
        bucket_stats: dict {bucket_name: {'train': int, 'test': int, 'total': int, 'test_frac': float}}
        problems: list of string messages describing issues found (empty if ok)
    """
    # Existing leakage check: ensure no group has proteins in both splits
    grp_splits = defaultdict(set)   # grp_id -> set of splits seen {'train','test'}
    protein_to_group_local = protein_to_group  # name used in calling code; keep consistent

    for prot, split in protein_to_split.items():
        grp = protein_to_group_local.get(prot)
        if grp is None:
            continue
        grp_splits[grp].add(split)

    offending_groups = {grp for grp, splits in grp_splits.items() if len(splits) > 1}

    # Now compute bucket-level stats
    # bucket_name -> {'train':count, 'test':count, 'total':count}
    bucket_counts = defaultdict(lambda: {'train': 0, 'test': 0, 'total': 0})

    # Build group -> representative split (should be consistent because validate_split is called
    # after the "no group split" constraint is enforced; but if leakage exists we still assign
    # the group to the majority split for reporting)
    group_to_proteins = defaultdict(list)
    for prot, grp in protein_to_group_local.items():
        if grp is None:
            continue
        group_to_proteins[grp].append(prot)

    for grp, proteins in group_to_proteins.items():
        # determine the split for the group. If group is leaked, pick majority split for reporting.
        splits = [protein_to_split.get(p) for p in proteins if p in protein_to_split]
        splits = [s for s in splits if s is not None]
        if not splits:
            # group has no proteins in mapping (weird) — skip
            continue
        # choose the majority split if needed
        train_count = splits.count('train')
        test_count = splits.count('test')
        if train_count >= test_count:
            grp_split = 'train'
        else:
            grp_split = 'test'

        size = group_sizes.get(grp, len(proteins))
        bucket = size_bucket(size)

        bucket_counts[bucket][grp_split] += 1
        bucket_counts[bucket]['total'] += 1

    # Summarize and flag problems
    bucket_stats = {}
    problems = []
    expected_test_frac = float(config.get('test_fraction', 0.2))
    # absolute tolerance (how far test fraction may deviate); and minimum groups to consider
    abs_tol = float(config.get('validate_abs_tol', 0.15))   # e.g., 0.15 means +/-15% absolute
    min_groups_to_warn = int(config.get('validate_min_groups', 3))  # small buckets may be noisy

    # Print header for quick human inspection
    print("\ngroup-size bucket distribution after split (groups counted):")
    print("{:15s} {:>6s} {:>6s} {:>6s} {:>9s}".format("BUCKET", "TRAIN", "TEST", "TOTAL", "TEST_FRAC"))
    for bucket, counts in sorted(bucket_counts.items(), key=lambda x: x[0]):
        train_c = counts['train']
        test_c = counts['test']
        total_c = counts['total']
        test_frac = (test_c / total_c) if total_c > 0 else 0.0
        bucket_stats[bucket] = {
            'train': train_c,
            'test': test_c,
            'total': total_c,
            'test_frac': test_frac
        }
        print("{:15s} {:6d} {:6d} {:6d} {:9.3f}".format(bucket, train_c, test_c, total_c, test_frac))

        # If bucket has groups but none ended up in test (or train), flag it
        if total_c > 0 and test_c == 0:
            problems.append(f"Bucket '{bucket}' has {total_c} groups but 0 in TEST.")
        if total_c > 0 and train_c == 0:
            problems.append(f"Bucket '{bucket}' has {total_c} groups but 0 in TRAIN.")

        # If bucket has enough groups, check expected fraction within tolerance
        if total_c >= min_groups_to_warn:
            if abs(test_frac - expected_test_frac) > abs_tol:
                problems.append(
                    f"Bucket '{bucket}' test fraction {test_frac:.3f} "
                    f"differs from expected {expected_test_frac:.3f} by > {abs_tol:.3f}."
                )

    # If there were offending groups (leakage), include a problem message
    if offending_groups:
        problems.append(f"{len(offending_groups)} group(ies) are split across train/test (leakage).")

    lines = []
    lines.append("group-size bucket distribution after split (groups counted):")
    lines.append("{:15s} {:>6s} {:>6s} {:>6s} {:>9s}".format(
        "BUCKET", "TRAIN", "TEST", "TOTAL", "TEST_FRAC"
    ))

    for bucket, counts in sorted(bucket_counts.items()):
        train_c = counts['train']
        test_c = counts['test']
        total_c = counts['total']
        test_frac = (test_c / total_c) if total_c > 0 else 0.0

        lines.append("{:15s} {:6d} {:6d} {:6d} {:9.3f}".format(
            bucket, train_c, test_c, total_c, test_frac
        ))

    report_text = "\n".join(lines)

    return offending_groups, bucket_stats, problems, report_text

    # # Summary print
    # if problems:
    #     print("\nVALIDATION PROBLEMS FOUND:")
    #     for p in problems:
    #         print(" -", p)
    # else:
    #     print("\nValidation OK: no bucket-level problems detected (within tolerances).")

    # return offending_groups, bucket_stats, problems

# ============================================================================
# BALANCE STATS
# ============================================================================

def balance_stats(protein_to_split, protein_to_label, global_ess_ratio, config):
    stats = {}
    for sp in ('train', 'test'):
        prots    = [p for p, s in protein_to_split.items() if s == sp]
        counts   = Counter(protein_to_label.get(p, 'Unknown') for p in prots)
        ess      = counts['Essential']
        noness   = counts['Non-essential']
        unk      = counts['Unknown']
        labelled = ess + noness
        rate_pct = 100 * ess / labelled if labelled > 0 else float('nan')
        stats[sp] = {
            'n_proteins':   len(prots),
            'n_essential':  ess,
            'n_noness':     noness,
            'n_unknown':    unk,
            'ess_rate_pct': round(rate_pct, 2),
        }
    global_rate_pct = 100 * global_ess_ratio
    dev = abs(stats['test']['ess_rate_pct'] - global_rate_pct)
    stats['test_deviation_pp'] = round(dev, 2)
    stats['warning'] = dev > config['max_ess_rate_deviation']
    return stats

def report_group_size_distribution(protein_to_split, group_sizes, protein_to_group):
    from collections import defaultdict

    split_bucket_counts = {
        "train": defaultdict(int),
        "test": defaultdict(int)
    }

    for grp, size in group_sizes.items():
        bucket = size_bucket(size)

        # get one protein from this group to determine split
        example_protein = next(iter(
            [p for p, f in protein_to_group.items() if f == grp]
        ))

        split = protein_to_split[example_protein]
        split_bucket_counts[split][bucket] += 1

    print("\nGroup Size Distribution:")
    for split in ["train", "test"]:
        print(f"\n{split.upper()}:")
        for bucket, count in split_bucket_counts[split].items():
            print(f"  {bucket}: {count}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("Repeated Random group-Level Splitting — Size-Stratified, N=15")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    print("\n--- Loading data ---")
    groups  = pd.read_csv(CONFIG['group_mapping'])
    complexes = pd.read_csv(CONFIG['complex_membership'])
    targets   = pd.read_csv(CONFIG['essentiality'])

    print(f"  Groups:  {len(groups)} rows, "
          f"{groups['group_id'].nunique()} unique groups")
    print(f"  Complexes: {complexes['ComplexId'].nunique()} complexes, "
          f"{complexes['ProteinId'].nunique()} unique proteins")
    print(f"  Targets:   {len(targets)} labelled proteins")

    # -------------------------------------------------------------------------
    # Build lookups
    # -------------------------------------------------------------------------
    complex_proteins  = set(complexes['ProteinId'].unique())

    protein_to_group = {
        row['uniprot_id']: row['group_id']
        for _, row in groups.iterrows()
        if row['uniprot_id'] in complex_proteins
    }

    group_sizes = (
        groups[groups['uniprot_id'].isin(complex_proteins)]
        .groupby('group_id')['uniprot_id'].nunique().to_dict()
    )

    protein_to_label = {}
    for _, row in targets.iterrows():
        bl = binarise_label(row['essential_category'])
        if bl is not None:
            protein_to_label[row['Protein']] = bl

    all_proteins  = sorted(complex_proteins)
    global_counts = Counter(protein_to_label.get(p, 'Unknown') for p in all_proteins)
    global_ess_ratio = essential_ratio(global_counts)
    global_rate_pct  = 100 * global_ess_ratio

    print(f"\n  Global essential rate (labelled): {global_rate_pct:.1f}%")
    print(f"  Label distribution ({len(all_proteins)} proteins):")
    for label in ('Essential', 'Non-essential', 'Unknown'):
        n = global_counts[label]
        print(f"    {label:15s}: {n:4d} ({100*n/len(all_proteins):.1f}%)")

    # -------------------------------------------------------------------------
    # Build group -> protein map and size buckets
    # -------------------------------------------------------------------------
    group_protein_map = defaultdict(set)
    for pid, grp in protein_to_group.items():
        group_protein_map[grp].add(pid)

    no_group_proteins = complex_proteins - set(protein_to_group.keys())

    # Buckets ordered largest-first
    buckets = {'very_large': [], 'large': [], 'medium': [], 'small': [], 'singleton': []}
    for grp, n in group_sizes.items():
        buckets[size_bucket(n)].append(grp)

    print(f"\n--- Group size buckets (largest first) ---")
    bucket_label = {
        'very_large': '>50 proteins',
        'large':      '21–50',
        'medium':     '6–20',
        'small':      '2–5',
        'singleton':  '1 protein',
    }
    for bname, bgrps in buckets.items():
        n_grp   = len(bgrps)
        n_prot  = sum(group_sizes[f] for f in bgrps)
        n_test  = round(n_grp * CONFIG['test_ratio'])
        print(f"  {bname:12s} ({bucket_label[bname]:13s}): "
              f"{n_grp:4d} groups, {n_prot:5d} proteins — "
              f"{n_test} grp{'' if n_test==1 else 's'} drawn to test each split")
    print(f"  No-group proteins: {len(no_group_proteins)}")

    # -------------------------------------------------------------------------
    # Generate N splits with reject-and-redraw for label balance
    # -------------------------------------------------------------------------
    print(f"\n--- Generating {CONFIG['n_splits']} splits "
          f"(tolerance ±{CONFIG['max_ess_rate_deviation']}pp, "
          f"max {CONFIG['max_attempts']} attempts each) ---")

    all_rows        = []
    balance_records = []

    for i in range(CONFIG['n_splits']):
        best_split      = None
        best_grp_split  = None
        best_deviation  = float('inf')
        attempts        = 0
        accepted        = False

        while attempts < CONFIG['max_attempts']:
            seed = CONFIG['base_seed'] + i + attempts * 1000
            rng  = np.random.default_rng(seed)

            protein_to_split, group_to_split, test_ess_rate = attempt_split(
                buckets            = buckets,
                group_sizes       = group_sizes,
                group_protein_map = group_protein_map,
                no_group_proteins = sorted(no_group_proteins),
                protein_to_label   = protein_to_label,
                config             = CONFIG,
                rng                = rng,
            )
            attempts += 1

            deviation = abs(100 * test_ess_rate - global_rate_pct)

            # Keep track of best attempt in case we exhaust the budget
            if deviation < best_deviation:
                best_deviation = deviation
                best_split     = protein_to_split
                best_grp_split = group_to_split
                best_seed      = seed

            if deviation <= CONFIG['max_ess_rate_deviation']:
                accepted = True
                break

        # Use best available if budget exhausted
        protein_to_split = best_split
        group_to_split  = best_grp_split

        # # Validate leakage
        # violations = validate_split(protein_to_split, protein_to_group)
        violations, bucket_stats, problems, report_text = validate_split(protein_to_split, protein_to_group, group_sizes, CONFIG)
        if violations:
            # existing behavior: reject the split attempt
            continue
        if problems:
            # option A: reject split attempts that have any problem
            print("Rejecting split due to validation problems.")
            continue
        with open("split_summary.txt", "a") as f:
            f.write("\n")
            f.write(report_text)
            f.write("\n")

        # otherwise accept split

        # Final balance stats
        bstats = balance_stats(
            protein_to_split, protein_to_label, global_ess_ratio, CONFIG
        )

        bal_row = {
            'split_index':        i + 1,
            'seed_accepted':      best_seed,
            'attempts':           attempts,
            'accepted':           accepted,
            'train_n':            bstats['train']['n_proteins'],
            'test_n':             bstats['test']['n_proteins'],
            'train_essential':    bstats['train']['n_essential'],
            'test_essential':     bstats['test']['n_essential'],
            'train_noness':       bstats['train']['n_noness'],
            'test_noness':        bstats['test']['n_noness'],
            'train_unknown':      bstats['train']['n_unknown'],
            'test_unknown':       bstats['test']['n_unknown'],
            'train_ess_rate_pct': bstats['train']['ess_rate_pct'],
            'test_ess_rate_pct':  bstats['test']['ess_rate_pct'],
            'test_deviation_pp':  bstats['test_deviation_pp'],
            'warning':            bstats['warning'],
            'leakage_violations': len(violations),
        }
        balance_records.append(bal_row)

        flag    = ' ⚠ balance warning' if bstats['warning'] else ''
        lk      = f' ⚠ {len(violations)} leakage' if violations else ''
        retry   = f' (accepted on attempt {attempts})' if attempts > 1 else ''
        print(f"  Split {i+1:2d} | "
              f"train={bstats['train']['n_proteins']} "
              f"(ess={bstats['train']['ess_rate_pct']:.1f}%) | "
              f"test={bstats['test']['n_proteins']} "
              f"(ess={bstats['test']['ess_rate_pct']:.1f}%) | "
              f"dev={bstats['test_deviation_pp']:.1f}pp"
              f"{retry}{flag}{lk}")

        # Collect rows for output CSV
        for pid in all_proteins:
            grp    = protein_to_group.get(pid)
            grp_sz = group_sizes.get(grp) if grp else None
            all_rows.append({
                'split_index':   i + 1,
                'seed_accepted': best_seed,
                'UniProt_AC':    pid,
                'group_id':     grp,
                'group_size':   grp_sz,
                'group_bucket': size_bucket(grp_sz) if grp_sz else None,
                'group_status': 'constrained' if grp else 'no_group',
                'split':         protein_to_split[pid],
                'protein_label': protein_to_label.get(pid, 'Unknown'),
                'label_mask':    protein_to_label.get(pid, 'Unknown') != 'Unknown',
            })

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------
    print("\n--- Saving outputs ---")

    pd.DataFrame(all_rows).to_csv(CONFIG['output_all_splits'], index=False)
    print(f"  Saved: {CONFIG['output_all_splits']}  ({len(all_rows):,} rows)")

    balance_df = pd.DataFrame(balance_records)
    balance_df.to_csv(CONFIG['output_balance'], index=False)
    print(f"  Saved: {CONFIG['output_balance']}")

    # -------------------------------------------------------------------------
    # Summary report
    # -------------------------------------------------------------------------
    n_warnings  = sum(1 for r in balance_records if r['warning'])
    n_leakage   = sum(1 for r in balance_records if r['leakage_violations'] > 0)
    n_retried   = sum(1 for r in balance_records if r['attempts'] > 1)
    test_rates  = [r['test_ess_rate_pct']  for r in balance_records]
    train_rates = [r['train_ess_rate_pct'] for r in balance_records]

    lines = []
    lines.append("=" * 70)
    lines.append("SPLIT SUMMARY — Size-stratified repeated random group-Level splits")
    lines.append("=" * 70)
    lines.append(f"Total unique proteins:            {len(all_proteins)}")
    lines.append(f"Total complexes:                  {complexes['ComplexId'].nunique()}")
    lines.append(f"Total groups (all atomic):      {len(group_sizes)}")
    lines.append(f"  Very large (>50):               {len(buckets['very_large'])}")
    lines.append(f"  Large      (21–50):             {len(buckets['large'])}")
    lines.append(f"  Medium     (6–20):              {len(buckets['medium'])}")
    lines.append(f"  Small      (2–5):               {len(buckets['small'])}")
    lines.append(f"  Singleton  (1 protein):         {len(buckets['singleton'])}")
    lines.append(f"Global essential rate (labelled): {global_rate_pct:.1f}%")
    lines.append(f"Number of splits:                 {CONFIG['n_splits']}")
    lines.append(f"Train/test ratio:                 "
                 f"{CONFIG['train_ratio']:.0%} / {CONFIG['test_ratio']:.0%}")
    lines.append(f"Balance tolerance:                "
                 f"±{CONFIG['max_ess_rate_deviation']}pp from global rate")
    lines.append(f"Max redraw attempts per split:    {CONFIG['max_attempts']}")
    lines.append(f"Splits requiring redraw:          {n_retried} / {CONFIG['n_splits']}")
    lines.append(f"Splits with balance warnings:     {n_warnings} / {CONFIG['n_splits']}")
    lines.append(f"Splits with leakage violations:   {n_leakage} / {CONFIG['n_splits']}")
    lines.append("")

    lines.append(f"{'Split':>6} {'Attempts':>8}  "
                 f"{'Train N':>8} {'Train ess%':>11}  "
                 f"{'Test N':>7} {'Test ess%':>10}  "
                 f"{'Dev pp':>7}  {'Flag':>4}")
    lines.append("-" * 70)
    for r in balance_records:
        flag = '⚠' if r['warning'] or r['leakage_violations'] > 0 else ''
        lines.append(
            f"{r['split_index']:>6} {r['attempts']:>8}  "
            f"{r['train_n']:>8} {r['train_ess_rate_pct']:>10.1f}%  "
            f"{r['test_n']:>7} {r['test_ess_rate_pct']:>9.1f}%  "
            f"{r['test_deviation_pp']:>6.1f}pp  {flag:>4}"
        )
    lines.append("-" * 70)
    lines.append(
        f"{'Mean':>6} {'':>8}  {'':>8} {sum(train_rates)/len(train_rates):>10.1f}%  "
        f"{'':>7} {sum(test_rates)/len(test_rates):>9.1f}%"
    )
    lines.append(
        f"{'Std':>6} {'':>8}  {'':>8} {pd.Series(train_rates).std():>10.2f}%  "
        f"{'':>7} {pd.Series(test_rates).std():>9.2f}%"
    )
    lines.append(
        f"{'Min':>6} {'':>8}  {'':>8} {min(train_rates):>10.1f}%  "
        f"{'':>7} {min(test_rates):>9.1f}%"
    )
    lines.append(
        f"{'Max':>6} {'':>8}  {'':>8} {max(train_rates):>10.1f}%  "
        f"{'':>7} {max(test_rates):>9.1f}%"
    )
    lines.append("")
    lines.append("Leakage guarantee: all groups are atomic — no group spans")
    lines.append("train and test within any single split.")

    summary = '\n'.join(lines)
    print(f"\n{summary}")
    with open(CONFIG['output_summary'], 'w') as f:
        f.write(summary + '\n')
    print(f"\n  Saved: {CONFIG['output_summary']}")
    print("\nDone!")


if __name__ == '__main__':
    main()