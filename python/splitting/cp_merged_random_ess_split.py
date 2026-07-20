"""
Repeated Random group-Level Splitting — N=50
===========================================
Generates N=50 independent train/test splits where each Foldseek structural
group is allocated atomically (never split across train and test).

ALLOCATION MODE — CONFIG['stratify_by_bucket']
---------------------------------------------
  True  (default) : size-stratified draw. The original scheme, described below.
  False (ablation): all groups are pooled and round(N_total * 0.20) are drawn
                    to test uniformly at random. Size buckets are ignored.
                    Output paths gain the '_unstrat' suffix so the stratified
                    split files are never overwritten.

ALGORITHM — per split
---------------------
groups are divided into four size buckets (largest first):

  large      : >20 proteins   (formerly large + very_large)
  medium     : 6–20           (97 groups)
  small      : 2–5            (274 groups)
  singleton  : 1 protein      (592 groups)

Within each bucket, exactly round(n_groups * 0.20) groups are drawn at
random for test; the remainder go to train. This is a pure random draw —
no essentiality information is used during the draw itself.

After all buckets are processed, overall label balance is checked.  If the
test essential rate deviates from the global rate by more than
`max_ess_rate_deviation` percentage points, the entire split is redrawn
(up to `max_attempts` tries). All accepted splits are saved regardless
of whether they used retries.

Proteins with no group assignment are allocated by stratified random
sampling (Essential / Non-essential / Unknown groups independently).

GLOBAL RATE NOTE
----------------
The global essential rate used for balance checking and deviation reporting
is computed only over proteins that are BOTH labelled (Essential or
Non-essential) AND have a structural group assignment (i.e. group_size is
not NaN). Proteins lacking labels or lacking structure are excluded from
this denominator, consistent with the analysis population used in modelling.

OUTPUTS
-------
  cp_ess_merged_splits.csv         : one row per (protein, split_index)
  cp_ess_merged_groups.csv         : per-split label balance statistics
  cp_ess_merged_split_summary.txt  : human-readable summary + bucket profile
  cp_avg_essentiality_by_group_size.csv : average essentiality per group size

"""

import pandas as pd
import numpy as np
from collections import defaultdict, Counter


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Input files
    'group_mapping':      '/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/cp/mapping_struct.csv',
    'complex_membership': '/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/cp/stoich_protein.csv',
    'essentiality':       '/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/lu_essentiality_protein.csv',

    # Output files
    'output_all_splits':  '/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/cp/ess_protein_merged_splits.csv',
    'output_balance':     '/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/cp/ess_merged_groups.csv',
    'output_summary':     '/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/cp/ess_merged_split_summary.txt',
    'output_ess_by_size': '/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/cp/avg_essentiality_by_group_size.csv',

    # Large groups to highlight in the bucket profile section
    'focus_groups': ['grp_0222', 'grp_0064', 'grp_1179', 'grp_0000'],

    # Number of independent splits to produce
    'n_splits': 50,        # CHANGED: was 15

    # Train/test ratios
    'train_ratio': 0.80,
    'test_ratio':  0.20,

    # Flag a split if the test set size deviates from exactly 20% of total
    # proteins by more than this many percentage points (i.e. test must be
    # between 15% and 25% of total proteins). Checked before label balance.
    'max_size_deviation': 5.0,   # pp — so acceptable range is [15%, 25%]

    # Flag a split if test essential rate deviates more than this from global
    # (percentage points). Flagged splits are redrawn up to max_attempts.
    'max_ess_rate_deviation': 4.5,

    # Maximum redraw attempts per split before accepting best available
    'max_attempts': 10,

    # Base seed — split i uses seed = base_seed + i for first attempt;
    # retries use base_seed + i + attempt * 1000
    'base_seed': 42,

    # ---- BUCKET ABLATION ---------------------------------------------------
    # True  : original behaviour. Groups are stratified into the four size
    #         buckets and round(n_bucket * 0.20) groups are drawn to test from
    #         each bucket independently.
    # False : ABLATION. All groups are pooled and round(N_total * 0.20) are
    #         drawn to test uniformly at random, ignoring size entirely.
    #
    # Setting this to False has two automatic consequences:
    #   (a) output paths gain the '_unstrat' suffix, so the stratified split
    #       files are never overwritten;
    #   (b) the per-bucket composition checks in validate_split() are demoted
    #       from vetoes to printed diagnostics. Those checks exist solely to
    #       police the stratification, so under the ablation they would reject
    #       the very splits we are trying to produce. Leakage remains a hard
    #       veto in both modes.
    'stratify_by_bucket': True,
    'unstrat_suffix': '_unstrat',
}


# --- Ablation output routing (no effect when stratify_by_bucket is True) -----
if not CONFIG['stratify_by_bucket']:
    for _key in ('output_all_splits', 'output_balance',
                 'output_summary', 'output_ess_by_size'):
        _stem, _dot, _ext = CONFIG[_key].rpartition('.')
        CONFIG[_key] = f"{_stem}{CONFIG['unstrat_suffix']}{_dot}{_ext}"


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
    """Map group size to bucket name (4 buckets: large/medium/small/singleton).
    large covers all groups >20 proteins (formerly separate large and very_large)."""
    if n_proteins == 1:    return 'singleton'
    elif n_proteins <= 5:  return 'small'
    elif n_proteins <= 20: return 'medium'
    else:                  return 'large'


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
    test_ess_rate    : float  (essential rate in test among labelled+structured proteins)
    test_size_frac   : float  (fraction of all proteins that went to test)
    """
    test_ratio = config['test_ratio']
    group_to_split = {}

    if config.get('stratify_by_bucket', True):
        # --- Bucket-level random draw, largest bucket first ---
        for bucket_grps in buckets.values():
            if not bucket_grps:
                continue
            n_grps    = len(bucket_grps)
            n_test    = round(n_grps * test_ratio)
            n_test    = max(n_test, 0)

            grps_list = list(bucket_grps)
            rng.shuffle(grps_list)

            for i, grp in enumerate(grps_list):
                group_to_split[grp] = 'test' if i < n_test else 'train'
    else:
        # --- ABLATION: pooled random draw, size buckets ignored ---
        grps_list = [g for bucket_grps in buckets.values() for g in bucket_grps]
        n_test    = max(round(len(grps_list) * test_ratio), 0)

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

    # --- Compute test label balance (labelled + structured proteins only) ---
    # Only proteins that have a group assignment (i.e. are in group_protein_map)
    # and are labelled are counted for the essentiality rate check.
    grouped_pids = set(p for grp in group_protein_map.values() for p in grp)
    test_prots_structured = [
        p for p, s in protein_to_split.items()
        if s == 'test' and p in grouped_pids
    ]
    test_counts = Counter(protein_to_label.get(p, 'Unknown') for p in test_prots_structured)
    test_ess    = test_counts['Essential']
    test_lab    = test_ess + test_counts['Non-essential']
    test_ess_rate = test_ess / test_lab if test_lab > 0 else 0.5

    # --- Compute test set size fraction (vs total proteins) ---
    total_prots    = len(protein_to_split)
    test_size_frac = sum(1 for s in protein_to_split.values() if s == 'test') / total_prots \
                     if total_prots > 0 else 0.0

    return protein_to_split, group_to_split, test_ess_rate, test_size_frac


# ============================================================================
# VALIDATION
# ============================================================================

def validate_split(protein_to_split, protein_to_group, group_sizes, config):
    """
    Validate a proposed split.

    Returns:
        offending_groups : set of group_ids split across train/test (leakage)
        bucket_stats     : dict {bucket_name: {train, test, total, test_frac}}
        problems         : list of problem messages (empty = clean)
        report_text      : str formatted bucket table for logging
    """
    grp_splits = defaultdict(set)
    protein_to_group_local = protein_to_group

    for prot, split in protein_to_split.items():
        grp = protein_to_group_local.get(prot)
        if grp is None:
            continue
        grp_splits[grp].add(split)

    offending_groups = {grp for grp, splits in grp_splits.items() if len(splits) > 1}

    bucket_counts   = defaultdict(lambda: {'train': 0, 'test': 0, 'total': 0})
    group_to_proteins = defaultdict(list)
    for prot, grp in protein_to_group_local.items():
        if grp is None:
            continue
        group_to_proteins[grp].append(prot)

    for grp, proteins in group_to_proteins.items():
        splits = [protein_to_split.get(p) for p in proteins if p in protein_to_split]
        splits = [s for s in splits if s is not None]
        if not splits:
            continue
        grp_split = 'train' if splits.count('train') >= splits.count('test') else 'test'
        size   = group_sizes.get(grp, len(proteins))
        bucket = size_bucket(size)
        bucket_counts[bucket][grp_split] += 1
        bucket_counts[bucket]['total']   += 1

    bucket_stats = {}
    problems    = []
    diagnostics = []                                    # non-vetoing (ablation)
    stratified  = bool(config.get('stratify_by_bucket', True))
    expected_test_frac = float(config.get('test_fraction', 0.2))
    abs_tol            = float(config.get('validate_abs_tol', 0.15))
    min_groups_to_warn = int(config.get('validate_min_groups', 3))

    print("\ngroup-size bucket distribution after split (groups counted):")
    print("{:15s} {:>6s} {:>6s} {:>6s} {:>9s}".format("BUCKET", "TRAIN", "TEST", "TOTAL", "TEST_FRAC"))
    for bucket, counts in sorted(bucket_counts.items(), key=lambda x: x[0]):
        train_c   = counts['train']
        test_c    = counts['test']
        total_c   = counts['total']
        test_frac = (test_c / total_c) if total_c > 0 else 0.0
        bucket_stats[bucket] = {'train': train_c, 'test': test_c,
                                 'total': total_c, 'test_frac': test_frac}
        print("{:15s} {:6d} {:6d} {:6d} {:9.3f}".format(bucket, train_c, test_c, total_c, test_frac))
        # Bucket-composition checks. Under the ablation the size composition of
        # the test set is deliberately uncontrolled, so these are reported but
        # must not veto the split. Leakage vetoes in both modes.
        _sink = problems if stratified else diagnostics
        if total_c > 0 and test_c == 0:
            _sink.append(f"Bucket '{bucket}' has {total_c} groups but 0 in TEST.")
        if total_c > 0 and train_c == 0:
            _sink.append(f"Bucket '{bucket}' has {total_c} groups but 0 in TRAIN.")
        if total_c >= min_groups_to_warn:
            if abs(test_frac - expected_test_frac) > abs_tol:
                _sink.append(
                    f"Bucket '{bucket}' test fraction {test_frac:.3f} "
                    f"differs from expected {expected_test_frac:.3f} by > {abs_tol:.3f}."
                )
    if offending_groups:
        problems.append(f"{len(offending_groups)} group(ies) are split across train/test (leakage).")

    if diagnostics:
        print("  [ablation] bucket-composition notes (reported, not vetoing):")
        for _d in diagnostics:
            print(f"    - {_d}")

    lines = []
    lines.append("group-size bucket distribution after split (groups counted):")
    lines.append("{:15s} {:>6s} {:>6s} {:>6s} {:>9s}".format(
        "BUCKET", "TRAIN", "TEST", "TOTAL", "TEST_FRAC"))
    for bucket, counts in sorted(bucket_counts.items()):
        train_c   = counts['train']
        test_c    = counts['test']
        total_c   = counts['total']
        test_frac = (test_c / total_c) if total_c > 0 else 0.0
        lines.append("{:15s} {:6d} {:6d} {:6d} {:9.3f}".format(
            bucket, train_c, test_c, total_c, test_frac))
    report_text = "\n".join(lines)
    return offending_groups, bucket_stats, problems, report_text


# ============================================================================
# BALANCE STATS
# ============================================================================

def balance_stats(protein_to_split, protein_to_label, global_ess_ratio,
                  group_protein_map, config):
    """
    Compute per-split balance statistics.

    Essentiality rates are computed over labelled+structured proteins only
    (consistent with global_ess_ratio denominator).  Size percentages use
    the full protein set so train% + test% = 100%.
    """
    stats   = {}
    total_n = len(protein_to_split)
    grouped_pids = set(p for grp in group_protein_map.values() for p in grp)

    for sp in ('train', 'test'):
        prots    = [p for p, s in protein_to_split.items() if s == sp]
        # Essentiality counts — labelled+structured only
        prots_ls = [p for p in prots if p in grouped_pids]
        counts   = Counter(protein_to_label.get(p, 'Unknown') for p in prots_ls)
        ess      = counts['Essential']
        noness   = counts['Non-essential']
        unk_ls   = counts['Unknown']
        labelled = ess + noness
        rate_pct = 100 * ess / labelled if labelled > 0 else float('nan')
        # Size % — all proteins
        size_pct = 100 * len(prots) / total_n if total_n > 0 else float('nan')
        # Unknown count across ALL proteins in this split partition
        all_counts = Counter(protein_to_label.get(p, 'Unknown') for p in prots)
        stats[sp] = {
            'n_proteins':   len(prots),
            'size_pct':     round(size_pct, 2),
            'n_essential':  ess,
            'n_noness':     noness,
            'n_unknown':    all_counts['Unknown'],
            'ess_rate_pct': round(rate_pct, 2),
        }
    global_rate_pct = 100 * global_ess_ratio
    dev = abs(stats['test']['ess_rate_pct'] - global_rate_pct)
    stats['test_deviation_pp']      = round(dev, 2)
    stats['warning']                = dev > config['max_ess_rate_deviation']
    target_test_pct                 = 100 * config['test_ratio']
    size_dev                        = abs(stats['test']['size_pct'] - target_test_pct)
    stats['test_size_deviation_pp'] = round(size_dev, 2)
    stats['size_warning']           = size_dev > config['max_size_deviation']
    return stats


def report_group_size_distribution(protein_to_split, group_sizes, protein_to_group):
    split_bucket_counts = {"train": defaultdict(int), "test": defaultdict(int)}
    for grp, size in group_sizes.items():
        bucket = size_bucket(size)
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
# BUCKET PROFILE ANALYSIS
# ============================================================================

def bucket_profile_analysis(splits_df, global_rate_pct, focus_groups, config):
    """
    Reproduce the analysis from cp_essentiality_bucket_profile.py, returning
    a list of text lines suitable for appending to the summary report, and
    saving the essentiality-by-group-size CSV.

    Parameters
    ----------
    splits_df       : the full output DataFrame (all splits, all proteins)
    global_rate_pct : global essential rate computed over labelled+structured
                      proteins (the corrected 3306-denominator rate)
    focus_groups    : list of large group IDs to highlight
    config          : CONFIG dict (for output path)
    """
    lines = []
    lines.append("")
    lines.append("=" * 90)
    lines.append("LARGE-BUCKET PROFILE ANALYSIS")
    lines.append("=" * 90)

    lg_test = splits_df[(splits_df["group_bucket"] == "large") &
                        (splits_df["split"] == "test")]

    # ── 1. Which large groups are in TEST per split? ──────────────────
    lines.append("")
    lines.append("1. Which large groups are in TEST per split?")
    lines.append("-" * 60)
    groups_per_split = (
        lg_test.groupby("split_index")["group_id"]
        .apply(lambda x: sorted(x.unique()))
    )
    for split_idx, grps in groups_per_split.items():
        lines.append(f"  Split {int(split_idx):>2d}: {', '.join(grps)}")

    # ── 2. All large groups and their sizes ───────────────────────────
    lines.append("")
    lines.append("2. All large groups and their sizes")
    lines.append("-" * 60)
    lg_all    = splits_df[splits_df["group_bucket"] == "large"]
    for gid, sub in sorted(lg_all.groupby("group_id")):
        size = int(sub["group_size"].iloc[0])
        test_splits = [int(x) for x in sorted(
            lg_test[lg_test["group_id"] == gid]["split_index"].unique()
        )]
        lines.append(f"  {gid}: {size:>4d} proteins  |  in test for splits {test_splits}")

    # ── 3. Test set sizes per split ───────────────────────────────────
    lines.append("")
    lines.append("3. Large test set size per split")
    lines.append("-" * 60)
    proteins_per_split = lg_test.groupby("split_index")["UniProt_AC"].apply(set)
    for split_idx, prots in proteins_per_split.items():
        lines.append(f"  Split {int(split_idx):>2d}: {len(prots):>4d} proteins")

    # ── 4. Overlap statistics ─────────────────────────────────────────
    lines.append("")
    lines.append("4. Overlap statistics across all splits")
    lines.append("-" * 60)
    common = set.intersection(*proteins_per_split.values)
    union  = set.union(*proteins_per_split.values)
    lines.append(f"  Intersection (in ALL test sets): {len(common)}")
    lines.append(f"  Union (in ANY test set):         {len(union)}")
    if len(union) > 0:
        lines.append(f"  Jaccard similarity:              {len(common) / len(union):.3f}")

    # ── 5. Per-protein test frequency ─────────────────────────────────
    lines.append("")
    lines.append("5. Per-protein test frequency (how many splits is each protein in test?)")
    lines.append("-" * 60)
    freq = lg_test.groupby("UniProt_AC")["split_index"].nunique()
    for cnt, n in freq.value_counts().sort_index().items():
        lines.append(f"  {cnt} split(s): {n} proteins")

    # ── 6. Per-group test frequency — randomness check ────────────────
    lines.append("")
    lines.append("6. Per-group test frequency — randomness check")
    lines.append("-" * 60)
    n_splits          = splits_df["split_index"].nunique()
    lg_groups         = splits_df[splits_df["group_bucket"] == "large"]["group_id"].unique()
    n_groups          = len(lg_groups)
    n_drawn_per_split = round(n_groups * 0.20)
    expected_freq     = n_drawn_per_split * n_splits / n_groups

    lines.append(f"  Total large groups:          {n_groups}")
    lines.append(f"  Groups drawn to test/split:  {n_drawn_per_split}  "
                 f"({100 * n_drawn_per_split / n_groups:.1f}%)")
    lines.append(f"  Expected test freq per group over {n_splits} splits: "
                 f"{expected_freq:.1f}")
    lines.append("")
    lines.append(f"  {'Group':30s} {'Size':>6}  {'Test count':>10}  Note")

    group_test_counts = (
        lg_test.groupby("group_id")["split_index"].nunique()
        .reindex(lg_groups, fill_value=0)
    )
    for gid in sorted(lg_groups):
        size  = int(splits_df[splits_df["group_id"] == gid]["group_size"].iloc[0])
        count = int(group_test_counts[gid])
        if count == 0:
            note = "⚠ NEVER in test"
        elif count == n_splits:
            note = "⚠ ALWAYS in test"
        elif abs(count - expected_freq) > 2 * expected_freq:
            note = "⚠ unusually high/low"
        else:
            note = "ok"
        lines.append(f"  {gid:30s} {size:>6}  {count:>10}  {note}")

    obs_counts = group_test_counts.values
    lines.append("")
    lines.append(f"  Min test count across groups: {obs_counts.min()}")
    lines.append(f"  Max test count across groups: {obs_counts.max()}")
    lines.append(f"  Mean test count:              {obs_counts.mean():.2f}")
    lines.append(f"  Std dev:                      {obs_counts.std():.2f}")

    # ── 7. Average essentiality per group size ────────────────────────
    # Use split 1 only (labels are constant across splits)
    s1 = splits_df[splits_df["split_index"] == 1]

    lines.append("")
    lines.append("=" * 90)
    lines.append("7. Average essentiality per group size")
    lines.append("   (labelled + structured proteins only; consistent with global rate denominator)")
    lines.append("=" * 90)

    # Coverage note
    n_total         = len(s1)
    n_labelled      = s1["protein_label"].isin(["Essential", "Non-essential"]).sum()
    n_unknown       = (s1["protein_label"] == "Unknown").sum()
    n_no_group      = s1["group_size"].isna().sum()
    n_lab_no_group  = s1[s1["group_size"].isna() &
                         s1["protein_label"].isin(["Essential", "Non-essential"])].shape[0]
    n_counted       = n_labelled - n_lab_no_group

    lines.append("")
    lines.append("  Coverage note (split 1, one row per protein):")
    lines.append(f"    Total proteins in CSV:              {n_total}")
    lines.append(f"    Labelled (Ess + Non-ess):           {n_labelled}")
    lines.append(f"    Unknown label (excluded):           {n_unknown}")
    lines.append(f"    No group_size / NaN (excluded):     {n_no_group}")
    lines.append(f"      of which labelled (double-excl):  {n_lab_no_group}")
    lines.append(f"    Counted in size_stats:              {n_counted}")
    lines.append(f"    Global essential rate (this pop):   {global_rate_pct:.1f}%")

    size_stats = (
        s1[s1["protein_label"].isin(["Essential", "Non-essential"])]
        .dropna(subset=["group_size"])
        .assign(is_essential=lambda x: x["protein_label"] == "Essential")
        .groupby("group_size")
        .agg(
            n_proteins=("is_essential", "count"),
            n_essential=("is_essential", "sum"),
            ess_rate=("is_essential", "mean"),
        )
        .sort_index()
    )

    lines.append("")
    lines.append(f"  {'group_size':>12s}  {'n_proteins':>12s}  {'n_essential':>12s}  {'ess_rate':>10s}")
    lines.append("  " + "-" * 52)
    for gs, row in size_stats.iterrows():
        lines.append(f"  {gs:>12.1f}  {int(row['n_proteins']):>12d}  "
                     f"{int(row['n_essential']):>12d}  {row['ess_rate']*100:>9.1f}%")

    # Save CSV
    out = size_stats.copy()
    out["ess_rate_pct"] = (out["ess_rate"] * 100).round(2)
    out = out.drop(columns="ess_rate").reset_index()
    out.to_csv(config['output_ess_by_size'], index=False)
    lines.append(f"\n  Saved: {config['output_ess_by_size']}")

    # ── 8. Focus group essential rate profile ─────────────────────────
    lines.append("")
    lines.append("=" * 90)
    lines.append("8. Essential rate profile — focus groups")
    lines.append("=" * 90)
    lines.append(f"  Global essential rate (labelled + structured): {global_rate_pct:.1f}%")
    lines.append("")
    lines.append(f"  {'Group':12s} {'Size':>6}  {'Essential':>10}  "
                 f"{'Non-ess':>8}  {'Unknown':>8}  {'Ess rate':>9}  {'vs global':>10}")
    lines.append("  " + "-" * 68)

    for gid in focus_groups:
        grp_df    = s1[s1["group_id"] == gid]
        size      = int(grp_df["group_size"].iloc[0]) if len(grp_df) > 0 else 0
        n_ess     = int((grp_df["protein_label"] == "Essential").sum())
        n_noness  = int((grp_df["protein_label"] == "Non-essential").sum())
        n_unknown = int((grp_df["protein_label"] == "Unknown").sum())
        n_lab     = n_ess + n_noness
        ess_rate  = 100 * n_ess / n_lab if n_lab > 0 else float('nan')
        diff      = ess_rate - global_rate_pct if n_lab > 0 else float('nan')
        direction = f"+{diff:.1f}pp" if diff >= 0 else f"{diff:.1f}pp"
        flag      = "  ⚠ high" if diff > 10 else ("  ⚠ low" if diff < -10 else "")
        lines.append(f"  {gid:12s} {size:>6}  {n_ess:>10}  "
                     f"{n_noness:>8}  {n_unknown:>8}  {ess_rate:>8.1f}%  "
                     f"{direction:>10}{flag}")

    # ── 9. Per-split large-bucket ess% annotated with heavy groups ────
    lines.append("")
    lines.append("  Large-bucket test essential rate by split, annotated with focus groups present:")
    lines.append(f"  {'Split':>5}  {'Focus groups in test':<32}  "
                 f"{'Focus grp ess%':>14}  {'Large bucket ess%':>17}")
    lines.append("  " + "-" * 74)

    for split_idx in sorted(lg_test["split_index"].unique()):
        sp_df          = lg_test[lg_test["split_index"] == split_idx]
        groups_in_test = sp_df["group_id"].unique()
        heavy_in_test  = [g for g in focus_groups if g in groups_in_test]
        heavy_label    = ', '.join(heavy_in_test) if heavy_in_test else "none"

        if heavy_in_test:
            hdf      = sp_df[sp_df["group_id"].isin(heavy_in_test)]
            h_ess    = (hdf["protein_label"] == "Essential").sum()
            h_noness = (hdf["protein_label"] == "Non-essential").sum()
            h_lab    = h_ess + h_noness
            h_rate   = f"{100 * h_ess / h_lab:.1f}%" if h_lab > 0 else "N/A"
        else:
            h_rate = "N/A"

        b_ess    = (sp_df["protein_label"] == "Essential").sum()
        b_noness = (sp_df["protein_label"] == "Non-essential").sum()
        b_lab    = b_ess + b_noness
        b_rate   = f"{100 * b_ess / b_lab:.1f}%" if b_lab > 0 else "N/A"

        lines.append(f"  {int(split_idx):>5}  {heavy_label:<32}  "
                     f"{h_rate:>14}  {b_rate:>17}")

    return lines


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print(f"Repeated Random group-Level Splitting — "
          f"{'Size-Stratified' if CONFIG['stratify_by_bucket'] else 'UNSTRATIFIED'}, "
          f"N={CONFIG['n_splits']}")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    print("\n--- Loading data ---")
    groups    = pd.read_csv(CONFIG['group_mapping'])
    complexes = pd.read_csv(CONFIG['complex_membership'])
    targets   = pd.read_csv(CONFIG['essentiality'])

    print(f"  Groups:    {len(groups)} rows, "
          f"{groups['group_id'].nunique()} unique groups")
    print(f"  Complexes: {complexes['ComplexId'].nunique()} complexes, "
          f"{complexes['ProteinId'].nunique()} unique proteins")
    print(f"  Targets:   {len(targets)} labelled proteins")

    # -------------------------------------------------------------------------
    # Build lookups
    # -------------------------------------------------------------------------
    complex_proteins = set(complexes['ProteinId'].unique())

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

    all_proteins = sorted(complex_proteins)

    # -------------------------------------------------------------------------
    # Compute global essential rate — labelled + structured proteins only
    # (excludes the ~83 proteins with no structural group and the ~6 no-group
    # proteins, giving the 3306-protein denominator consistent with modelling)
    # -------------------------------------------------------------------------
    structured_pids  = set(protein_to_group.keys())   # proteins with a group
    labelled_structured = [
        p for p in all_proteins
        if p in structured_pids and protein_to_label.get(p) in ('Essential', 'Non-essential')
    ]
    n_ess_global    = sum(1 for p in labelled_structured if protein_to_label[p] == 'Essential')
    n_noness_global = sum(1 for p in labelled_structured if protein_to_label[p] == 'Non-essential')
    n_structured    = len(labelled_structured)
    global_ess_ratio = n_ess_global / n_structured if n_structured > 0 else 0.5
    global_rate_pct  = 100 * global_ess_ratio

    # Also compute the old all-labelled counts for reporting
    global_counts_all = Counter(protein_to_label.get(p, 'Unknown') for p in all_proteins)
    n_unknown_all     = global_counts_all['Unknown']
    n_no_group        = len(complex_proteins - structured_pids)

    print(f"\n  Label distribution ({len(all_proteins)} total complex proteins):")
    print(f"    Essential      : {n_ess_global + sum(1 for p in all_proteins if protein_to_label.get(p) == 'Essential' and p not in structured_pids):4d}")
    print(f"    Non-essential  : {n_noness_global + sum(1 for p in all_proteins if protein_to_label.get(p) == 'Non-essential' and p not in structured_pids):4d}")
    print(f"    Unknown        : {n_unknown_all:4d}  (no evidence in DepMap/OGEE — excluded)")
    print(f"    No-group       : {n_no_group:4d}  (no structural assignment — excluded)")
    print(f"  Analysis population (labelled + structured): {n_structured}")
    print(f"  Global essential rate (analysis population): {global_rate_pct:.1f}%")

    # -------------------------------------------------------------------------
    # Build group -> protein map and size buckets
    # -------------------------------------------------------------------------
    group_protein_map = defaultdict(set)
    for pid, grp in protein_to_group.items():
        group_protein_map[grp].add(pid)

    no_group_proteins = complex_proteins - set(protein_to_group.keys())

    buckets = {'large': [], 'medium': [], 'small': [], 'singleton': []}
    for grp, n in group_sizes.items():
        buckets[size_bucket(n)].append(grp)

    print(f"\n--- Group size buckets (largest first) ---")
    bucket_label = {
        'large':     '>20 proteins (merged large + very large)',
        'medium':    '6–20',
        'small':     '2–5',
        'singleton': '1 protein',
    }
    for bname, bgrps in buckets.items():
        n_grp  = len(bgrps)
        n_prot = sum(group_sizes[f] for f in bgrps)
        n_test = round(n_grp * CONFIG['test_ratio'])
        print(f"  {bname:12s} ({bucket_label[bname]:40s}): "
              f"{n_grp:4d} groups, {n_prot:5d} proteins — "
              f"{n_test} grp{'' if n_test == 1 else 's'} drawn to test each split")
    print(f"  No-group proteins: {len(no_group_proteins)}")

    # -------------------------------------------------------------------------
    # Generate N splits
    # -------------------------------------------------------------------------
    print(f"\n--- Allocation mode: "
          f"{'SIZE-STRATIFIED' if CONFIG['stratify_by_bucket'] else 'UNSTRATIFIED (BUCKET ABLATION)'} ---")
    print(f"--- Generating {CONFIG['n_splits']} splits "
          f"(tolerance ±{CONFIG['max_ess_rate_deviation']}pp ess, "
          f"±{CONFIG['max_size_deviation']}pp size, "
          f"max {CONFIG['max_attempts']} attempts each) ---")

    all_rows        = []
    balance_records = []

    for i in range(CONFIG['n_splits']):
        best_split          = None
        best_grp_split      = None
        best_seed           = None
        best_key            = None
        best_deviation      = float('inf')
        best_size_deviation = float('inf')
        attempts            = 0
        accepted            = False

        while attempts < CONFIG['max_attempts']:
            seed = CONFIG['base_seed'] + i + attempts * 1000
            rng  = np.random.default_rng(seed)

            protein_to_split, group_to_split, test_ess_rate, test_size_frac = attempt_split(
                buckets           = buckets,
                group_sizes       = group_sizes,
                group_protein_map = group_protein_map,
                no_group_proteins = sorted(no_group_proteins),
                protein_to_label  = protein_to_label,
                config            = CONFIG,
                rng               = rng,
            )
            attempts += 1

            size_deviation = abs(100 * test_size_frac - 100 * CONFIG['test_ratio'])
            deviation      = abs(100 * test_ess_rate  - global_rate_pct)

            size_ok = size_deviation <= CONFIG['max_size_deviation']
            ess_ok  = deviation      <= CONFIG['max_ess_rate_deviation']

            # ACCEPTED -> keep exactly this split, then stop.
            #
            # This assignment is the fix. Previously the loop broke here but the
            # split written out afterwards was `best_split`, which tracked the
            # smallest SIZE deviation across attempts. An earlier attempt that
            # had FAILED the balance check was therefore frequently saved in
            # place of the one just accepted -- producing splits flagged for
            # balance despite `accepted == True` and `attempts < max_attempts`.
            if size_ok and ess_ok:
                accepted            = True
                best_split          = protein_to_split
                best_grp_split      = group_to_split
                best_seed           = seed
                best_size_deviation = size_deviation
                best_deviation      = deviation
                break

            # NOT accepted -> retain the best candidate so far, in case every
            # attempt fails. Rank: passes the size check, then smallest LABEL
            # deviation, then smallest size deviation. (Previously: size alone,
            # which ignored label balance entirely.)
            key = (not size_ok, deviation, size_deviation)
            if best_key is None or key < best_key:
                best_key            = key
                best_size_deviation = size_deviation
                best_deviation      = deviation
                best_split          = protein_to_split
                best_grp_split      = group_to_split
                best_seed           = seed

        protein_to_split = best_split
        group_to_split   = best_grp_split

        violations, bucket_stats, problems, report_text = validate_split(
            protein_to_split, protein_to_group, group_sizes, CONFIG
        )
        if violations:
            continue
        if problems:
            print("Rejecting split due to validation problems.")
            continue
        with open("split_summary.txt", "a") as f:
            f.write("\n")
            f.write(report_text)
            f.write("\n")

        bstats = balance_stats(
            protein_to_split, protein_to_label, global_ess_ratio,
            group_protein_map, CONFIG
        )

        # Per-bucket protein breakdown
        bucket_breakdown = {}
        for bname in ('large', 'medium', 'small', 'singleton'):
            bgrps   = buckets.get(bname, [])
            train_n = sum(len(group_protein_map.get(g, set()))
                          for g in bgrps if group_to_split.get(g) == 'train')
            test_n  = sum(len(group_protein_map.get(g, set()))
                          for g in bgrps if group_to_split.get(g) == 'test')
            total_n = train_n + test_n
            test_prots_bkt = [
                p for g in bgrps if group_to_split.get(g) == 'test'
                for p in group_protein_map.get(g, set())
            ]
            bc  = Counter(protein_to_label.get(p, 'Unknown') for p in test_prots_bkt)
            lab = bc['Essential'] + bc['Non-essential']
            ess_rate = 100 * bc['Essential'] / lab if lab > 0 else float('nan')
            bucket_breakdown[bname] = {
                'train_n':      train_n,
                'test_n':       test_n,
                'total_n':      total_n,
                'test_pct':     100 * test_n / total_n if total_n > 0 else 0.0,
                'test_ess_pct': round(ess_rate, 1),
            }

        bal_row = {
            'split_index':            i + 1,
            'seed_accepted':          best_seed,
            'attempts':               attempts,
            'accepted':               accepted,
            'train_n':                bstats['train']['n_proteins'],
            'train_pct':              bstats['train']['size_pct'],
            'test_n':                 bstats['test']['n_proteins'],
            'test_pct':               bstats['test']['size_pct'],
            'test_size_deviation_pp': bstats['test_size_deviation_pp'],
            'size_warning':           bstats['size_warning'],
            'train_essential':        bstats['train']['n_essential'],
            'test_essential':         bstats['test']['n_essential'],
            'train_noness':           bstats['train']['n_noness'],
            'test_noness':            bstats['test']['n_noness'],
            'train_unknown':          bstats['train']['n_unknown'],
            'test_unknown':           bstats['test']['n_unknown'],
            'train_ess_rate_pct':     bstats['train']['ess_rate_pct'],
            'test_ess_rate_pct':      bstats['test']['ess_rate_pct'],
            'test_deviation_pp':      bstats['test_deviation_pp'],
            'warning':                bstats['warning'],
            'leakage_violations':     len(violations),
            'large_test_n':           bucket_breakdown['large']['test_n'],
            'large_test_pct':         round(bucket_breakdown['large']['test_pct'], 1),
            'large_test_ess_pct':     bucket_breakdown['large']['test_ess_pct'],
            'medium_test_n':          bucket_breakdown['medium']['test_n'],
            'medium_test_pct':        round(bucket_breakdown['medium']['test_pct'], 1),
            'medium_test_ess_pct':    bucket_breakdown['medium']['test_ess_pct'],
            'small_test_n':           bucket_breakdown['small']['test_n'],
            'small_test_pct':         round(bucket_breakdown['small']['test_pct'], 1),
            'small_test_ess_pct':     bucket_breakdown['small']['test_ess_pct'],
            'singleton_test_n':       bucket_breakdown['singleton']['test_n'],
            'singleton_test_pct':     round(bucket_breakdown['singleton']['test_pct'], 1),
            'singleton_test_ess_pct': bucket_breakdown['singleton']['test_ess_pct'],
        }
        balance_records.append(bal_row)

        flag   = ' ⚠ balance warning' if bstats['warning']     else ''
        szflag = ' ⚠ size warning'    if bstats['size_warning'] else ''
        lk     = f' ⚠ {len(violations)} leakage' if violations  else ''
        retry  = f' (accepted on attempt {attempts})' if attempts > 1 else ''
        print(f"  Split {i+1:2d} | "
              f"train={bstats['train']['n_proteins']} ({bstats['train']['size_pct']:.1f}%) "
              f"ess={bstats['train']['ess_rate_pct']:.1f}% | "
              f"test={bstats['test']['n_proteins']} ({bstats['test']['size_pct']:.1f}%) "
              f"ess={bstats['test']['ess_rate_pct']:.1f}% | "
              f"size_dev={bstats['test_size_deviation_pp']:.1f}pp "
              f"ess_dev={bstats['test_deviation_pp']:.1f}pp"
              f"{retry}{szflag}{flag}{lk}")

        for pid in all_proteins:
            grp    = protein_to_group.get(pid)
            grp_sz = group_sizes.get(grp) if grp else None
            all_rows.append({
                'split_index':   i + 1,
                'seed_accepted': best_seed,
                'UniProt_AC':    pid,
                'group_id':      grp,
                'group_size':    grp_sz,
                'group_bucket':  size_bucket(grp_sz) if grp_sz else None,
                'group_status':  'constrained' if grp else 'no_group',
                'split':         protein_to_split[pid],
                'protein_label': protein_to_label.get(pid, 'Unknown'),
                'label_mask':    protein_to_label.get(pid, 'Unknown') != 'Unknown',
            })

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------
    print("\n--- Saving outputs ---")

    splits_df = pd.DataFrame(all_rows)
    splits_df.to_csv(CONFIG['output_all_splits'], index=False)
    print(f"  Saved: {CONFIG['output_all_splits']}  ({len(all_rows):,} rows)")

    pd.DataFrame(balance_records).to_csv(CONFIG['output_balance'], index=False)
    print(f"  Saved: {CONFIG['output_balance']}")

    # -------------------------------------------------------------------------
    # Summary report
    # -------------------------------------------------------------------------
    n_warnings = sum(1 for r in balance_records if r['warning'])
    n_sz_warn  = sum(1 for r in balance_records if r['size_warning'])
    n_leakage  = sum(1 for r in balance_records if r['leakage_violations'] > 0)
    n_retried  = sum(1 for r in balance_records if r['attempts'] > 1)
    test_rates  = [r['test_ess_rate_pct']      for r in balance_records]
    train_rates = [r['train_ess_rate_pct']     for r in balance_records]
    test_sz     = [r['test_pct']               for r in balance_records]
    train_sz    = [r['train_pct']              for r in balance_records]
    sz_devs     = [r['test_size_deviation_pp'] for r in balance_records]

    lines = []
    lines.append("=" * 90)
    lines.append("SPLIT SUMMARY — Size-stratified repeated random group-Level splits")
    lines.append("=" * 90)
    lines.append(f"Total unique proteins:            {len(all_proteins)}")
    lines.append(f"Total complexes:                  {complexes['ComplexId'].nunique()}")
    lines.append(f"Total groups (all atomic):        {len(group_sizes)}")
    lines.append(f"  Large      (>20, merged):       {len(buckets['large'])}")
    lines.append(f"  Medium     (6–20):              {len(buckets['medium'])}")
    lines.append(f"  Small      (2–5):               {len(buckets['small'])}")
    lines.append(f"  Singleton  (1 protein):         {len(buckets['singleton'])}")
    lines.append(f"Analysis population:              {n_structured}  "
                 f"(labelled + structured; excludes {n_unknown_all} Unknown, "
                 f"{n_no_group} no-group)")
    lines.append(f"Global essential rate:            {global_rate_pct:.1f}%  "
                 f"({n_ess_global} / {n_structured})")
    lines.append(f"Allocation mode:                  "
                 f"{'size-stratified' if CONFIG['stratify_by_bucket'] else 'UNSTRATIFIED (bucket ablation)'}")
    lines.append(f"Number of splits:                 {CONFIG['n_splits']}")
    lines.append(f"Train/test ratio:                 "
                 f"{CONFIG['train_ratio']:.0%} / {CONFIG['test_ratio']:.0%}")
    lines.append(f"Size tolerance:                   "
                 f"±{CONFIG['max_size_deviation']}pp from 20% "
                 f"(acceptable range: "
                 f"{100*CONFIG['test_ratio'] - CONFIG['max_size_deviation']:.0f}%–"
                 f"{100*CONFIG['test_ratio'] + CONFIG['max_size_deviation']:.0f}%)")
    lines.append(f"Balance tolerance:                "
                 f"±{CONFIG['max_ess_rate_deviation']}pp from global rate")
    lines.append(f"Max redraw attempts per split:    {CONFIG['max_attempts']}")
    lines.append(f"Splits requiring redraw:          {n_retried} / {CONFIG['n_splits']}")
    lines.append(f"Splits with size warnings:        {n_sz_warn} / {CONFIG['n_splits']}")
    lines.append(f"Splits with balance warnings:     {n_warnings} / {CONFIG['n_splits']}")
    lines.append(f"Splits with leakage violations:   {n_leakage} / {CONFIG['n_splits']}")
    lines.append("")

    lines.append(
        f"{'Split':>6} {'Att':>4}  "
        f"{'Train N':>8} {'Train%':>7} {'Train ess%':>11}  "
        f"{'Test N':>7} {'Test%':>6} {'Test ess%':>10}  "
        f"{'SzDev':>6} {'EssDev':>7}  {'Flag':>4}"
    )
    lines.append("-" * 90)
    for r in balance_records:
        flag = ''
        if r['size_warning']:           flag += 'Sz'
        if r['warning']:                flag += 'Es'
        if r['leakage_violations'] > 0: flag += 'Lk'
        flag = ('⚠ ' + flag) if flag else ''
        lines.append(
            f"{r['split_index']:>6} {r['attempts']:>4}  "
            f"{r['train_n']:>8} {r['train_pct']:>6.1f}% {r['train_ess_rate_pct']:>10.1f}%  "
            f"{r['test_n']:>7} {r['test_pct']:>5.1f}% {r['test_ess_rate_pct']:>9.1f}%  "
            f"{r['test_size_deviation_pp']:>5.1f}pp {r['test_deviation_pp']:>6.1f}pp  {flag:>6}"
        )
    lines.append("-" * 90)
    lines.append(
        f"{'Mean':>6} {'':>4}  "
        f"{'':>8} {sum(train_sz)/len(train_sz):>6.1f}% {sum(train_rates)/len(train_rates):>10.1f}%  "
        f"{'':>7} {sum(test_sz)/len(test_sz):>5.1f}% {sum(test_rates)/len(test_rates):>9.1f}%  "
        f"{sum(sz_devs)/len(sz_devs):>5.1f}pp"
    )
    lines.append(
        f"{'Std':>6} {'':>4}  "
        f"{'':>8} {pd.Series(train_sz).std():>6.2f}% {pd.Series(train_rates).std():>10.2f}%  "
        f"{'':>7} {pd.Series(test_sz).std():>5.2f}% {pd.Series(test_rates).std():>9.2f}%  "
        f"{pd.Series(sz_devs).std():>5.2f}pp"
    )
    lines.append(
        f"{'Min':>6} {'':>4}  "
        f"{'':>8} {min(train_sz):>6.1f}% {min(train_rates):>10.1f}%  "
        f"{'':>7} {min(test_sz):>5.1f}% {min(test_rates):>9.1f}%  "
        f"{min(sz_devs):>5.1f}pp"
    )
    lines.append(
        f"{'Max':>6} {'':>4}  "
        f"{'':>8} {max(train_sz):>6.1f}% {max(train_rates):>10.1f}%  "
        f"{'':>7} {max(test_sz):>5.1f}% {max(test_rates):>9.1f}%  "
        f"{max(sz_devs):>5.1f}pp"
    )
    lines.append("")
    lines.append("Leakage guarantee: all groups are atomic — no group spans")
    lines.append("train and test within any single split.")
    lines.append("")
    lines.append("Flag key: Sz = size out of range, Es = essentiality imbalance, Lk = leakage")

    # Table 2: per-split bucket breakdown
    lines.append("")
    lines.append("=" * 90)
    lines.append("TEST SET BREAKDOWN BY STRUCTURAL GROUP SIZE BUCKET (proteins in test)")
    lines.append("=" * 90)
    lines.append(
        f"{'Split':>6}  "
        f"{'Large N':>8} {'L%':>5} {'Ess%':>6}  "
        f"{'Medium N':>9} {'M%':>5} {'Ess%':>6}  "
        f"{'Small N':>8} {'S%':>5} {'Ess%':>6}  "
        f"{'Singleton N':>12} {'Sg%':>5} {'Ess%':>6}"
    )
    lines.append("-" * 90)
    for r in balance_records:
        lines.append(
            f"{r['split_index']:>6}  "
            f"{r['large_test_n']:>8} {r['large_test_pct']:>4.1f}% "
            f"{r['large_test_ess_pct']:>5.1f}%  "
            f"{r['medium_test_n']:>9} {r['medium_test_pct']:>4.1f}% "
            f"{r['medium_test_ess_pct']:>5.1f}%  "
            f"{r['small_test_n']:>8} {r['small_test_pct']:>4.1f}% "
            f"{r['small_test_ess_pct']:>5.1f}%  "
            f"{r['singleton_test_n']:>12} {r['singleton_test_pct']:>4.1f}% "
            f"{r['singleton_test_ess_pct']:>5.1f}%"
        )
    lines.append("-" * 90)

    def _col_mean(key):
        vals = [r[key] for r in balance_records if r[key] == r[key]]
        return sum(vals) / len(vals) if vals else float('nan')

    lines.append(
        f"{'Mean':>6}  "
        f"{_col_mean('large_test_n'):>8.1f} {_col_mean('large_test_pct'):>4.1f}% "
        f"{_col_mean('large_test_ess_pct'):>5.1f}%  "
        f"{_col_mean('medium_test_n'):>9.1f} {_col_mean('medium_test_pct'):>4.1f}% "
        f"{_col_mean('medium_test_ess_pct'):>5.1f}%  "
        f"{_col_mean('small_test_n'):>8.1f} {_col_mean('small_test_pct'):>4.1f}% "
        f"{_col_mean('small_test_ess_pct'):>5.1f}%  "
        f"{_col_mean('singleton_test_n'):>12.1f} {_col_mean('singleton_test_pct'):>4.1f}% "
        f"{_col_mean('singleton_test_ess_pct'):>5.1f}%"
    )
    lines.append(
        f"{'Std':>6}  "
        f"{pd.Series([r['large_test_n'] for r in balance_records]).std():>8.1f} "
        f"{pd.Series([r['large_test_pct'] for r in balance_records]).std():>4.2f}% "
        f"{pd.Series([r['large_test_ess_pct'] for r in balance_records]).std():>5.2f}%  "
        f"{pd.Series([r['medium_test_n'] for r in balance_records]).std():>9.1f} "
        f"{pd.Series([r['medium_test_pct'] for r in balance_records]).std():>4.2f}% "
        f"{pd.Series([r['medium_test_ess_pct'] for r in balance_records]).std():>5.2f}%  "
        f"{pd.Series([r['small_test_n'] for r in balance_records]).std():>8.1f} "
        f"{pd.Series([r['small_test_pct'] for r in balance_records]).std():>4.2f}% "
        f"{pd.Series([r['small_test_ess_pct'] for r in balance_records]).std():>5.2f}%  "
        f"{pd.Series([r['singleton_test_n'] for r in balance_records]).std():>12.1f} "
        f"{pd.Series([r['singleton_test_pct'] for r in balance_records]).std():>4.2f}% "
        f"{pd.Series([r['singleton_test_ess_pct'] for r in balance_records]).std():>5.2f}%"
    )
    lines.append("")
    lines.append("  N% = proteins from that bucket that went to test (of all proteins in that bucket).")
    lines.append("  Ess% = essential rate among labelled test proteins from that bucket.")

    # Bucket profile analysis (appended to same summary file)
    profile_lines = bucket_profile_analysis(
        splits_df       = splits_df,
        global_rate_pct = global_rate_pct,
        focus_groups    = CONFIG['focus_groups'],
        config          = CONFIG,
    )
    lines.extend(profile_lines)

    summary = '\n'.join(lines)
    print(f"\n{summary}")
    with open(CONFIG['output_summary'], 'w') as f:
        f.write(summary + '\n')
    print(f"\n  Saved: {CONFIG['output_summary']}")
    print("\nDone!")


if __name__ == '__main__':
    main()