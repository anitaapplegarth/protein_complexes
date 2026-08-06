"""
Shared engine for repeated random structural-group-level splitting.
==================================================================
Used by *_random_chembl_split.py, *_random_ess_split.py and
*_random_hpa_split.py. Those scripts hold nothing but a CONFIG block
and a call to run_splits(); all logic lives here, so a fix applied once
applies to all three tasks.

Generates N independent train/test splits where each Foldseek structural
group is allocated atomically (never split across train and test).

LEAKAGE SCOPE
-------------
Atomicity is enforced at the level of the Foldseek structural group, not the
complex. Proteins belonging to the same complex may therefore fall on opposite
sides of the split. This is intentional: global complex-level features are
dropped and the remaining features are banded into first- and second-order
terms based on the locality of the protein and its neighbours, so the shared
information is bounded and local. Labels are not shared across the boundary.

ALLOCATION MODE -- CONFIG['stratify_by_bucket']
----------------------------------------------
  True  (default) : size-stratified draw, described below.
  False (ablation): all groups are pooled and round(N_total * 0.20) are drawn
                    to test uniformly at random. Size buckets are ignored.
                    Output paths gain the '_unstrat' suffix so the stratified
                    split files are never overwritten.

ALGORITHM -- per split
----------------------
Groups are divided into four size buckets (largest first):

  large      : >20 proteins   (merged large + very_large)
  medium     : 6-20
  small      : 2-5
  singleton  : 1 protein

Within each bucket, exactly round(n_groups * 0.20) groups are drawn at random
for test; the remainder go to train. This is a pure random draw -- no label
information is used during the draw itself.

ACCEPTANCE
----------
A candidate split is accepted only if it passes ALL THREE checks:

  1. test size deviation  <= max_size_deviation
  2. test label deviation <= max_label_rate_deviation
  3. validate_split() returns no leakage and no bucket-composition problems

Check 3 used to be applied AFTER the redraw loop finished, and a failing split
was silently dropped with `continue` -- so a run could emit fewer than n_splits
splits, with non-contiguous split_index values, while the summary still
reported "/ 50" as the denominator. In the degenerate case (a size bucket too
small to contribute a test group) every split was rejected and an EMPTY splits
CSV was written over the previous good one.

Validation now sits inside the redraw loop, so a failing candidate triggers a
redraw rather than a silent drop. If every attempt fails, the best candidate is
still emitted, loudly flagged, and recorded with validation_ok=False -- never
dropped. A final integrity gate guarantees exactly n_splits contiguous splits
reach the output files, or nothing is written at all.

Proteins with no group assignment are allocated by stratified random sampling
(positive / negative / Unknown classes independently).

ANALYSIS POPULATION
-------------------
Proteins that are BOTH labelled (positive or negative) AND structured (present
in mapping_struct.csv, i.e. have a group_id). All rates and deviations use this
population as the denominator. Proteins that are Unknown or have no structural
group are carried through the split CSV for completeness but excluded from all
rate calculations.

LABEL SCHEME
------------
CONFIG['label_scheme'] adapts the engine to each task:

  positive  : label string for the positive class  ('Drug_target', 'Essential')
  negative  : label string for the negative class  ('Non_target', 'Non-essential')
  noun      : phrase used in printed output        ('drug target', 'essential')
  binarise  : callable mapping a raw cell value to positive / negative / None.
              Returning None drops the protein to Unknown.

BALANCE CSV COLUMN NAMES
------------------------
Positive/negative counts use task-neutral names (train_n_positive,
test_n_positive, train_n_negative, test_n_negative) so one downstream reader
serves all three tasks. The older task-specific names (train_drug_target,
train_essential, ...) are gone.
"""

import math
from collections import defaultdict, Counter

import numpy as np
import pandas as pd


# ============================================================================
# HELPERS
# ============================================================================

def size_bucket(n):
    """Map group size to bucket name."""
    if n == 1:    return 'singleton'
    elif n <= 5:  return 'small'
    elif n <= 20: return 'medium'
    else:         return 'large'


def n_test_for_bucket(n_groups, test_ratio, min_test):
    """
    How many groups of a bucket go to test.

    Pure round(n * ratio), then clamped so that a non-empty bucket with at
    least two groups always contributes at least `min_test` to test and at
    least one to train. A bucket holding a single group cannot satisfy both
    and is left to round() -- validate_split() treats that case as a
    diagnostic rather than a veto.
    """
    n_test = round(n_groups * test_ratio)
    if n_groups >= 2 and min_test > 0:
        n_test = max(n_test, min_test)
        n_test = min(n_test, n_groups - 1)
    return int(max(n_test, 0))


def resolve_output_paths(config):
    """Apply the '_unstrat' suffix when running the bucket ablation."""
    if not config.get('stratify_by_bucket', True):
        for key in ('output_all_splits', 'output_balance',
                    'output_summary', 'output_rate_by_size'):
            stem, dot, ext = config[key].rpartition('.')
            config[key] = f"{stem}{config['unstrat_suffix']}{dot}{ext}"
    return config


# ============================================================================
# CORE SPLIT LOGIC
# ============================================================================

def attempt_split(buckets, group_protein_map, no_group_proteins,
                  protein_to_label, analysis_pids, config, rng):
    """
    Make one attempt at a train/test split.

    Returns
    -------
    protein_to_split : dict[pid -> 'train'|'test']
    group_to_split   : dict[gid -> 'train'|'test']
    test_pos_rate    : float  -- positive rate in test, analysis population
    test_size_frac   : float  -- fraction of ALL proteins assigned to test
    """
    scheme     = config['label_scheme']
    pos_lbl    = scheme['positive']
    neg_lbl    = scheme['negative']
    test_ratio = config['test_ratio']
    min_test   = config.get('min_test_groups_per_bucket', 1)

    group_to_split = {}

    if config.get('stratify_by_bucket', True):
        # --- Bucket-level random draw ---
        for bucket_grps in buckets.values():
            if not bucket_grps:
                continue
            n_test    = n_test_for_bucket(len(bucket_grps), test_ratio, min_test)
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

    protein_to_split = {}
    for grp, split in group_to_split.items():
        for pid in group_protein_map.get(grp, set()):
            protein_to_split[pid] = split

    # No-group proteins: stratified by label class
    for lbl in (pos_lbl, neg_lbl, 'Unknown'):
        grp = sorted(p for p in no_group_proteins
                     if protein_to_label.get(p, 'Unknown') == lbl)
        rng.shuffle(grp)
        n_test_grp = round(len(grp) * test_ratio)
        for i, pid in enumerate(grp):
            protein_to_split[pid] = 'test' if i < n_test_grp else 'train'

    # Positive rate -- analysis population (labelled + structured) only
    test_ap   = [p for p, s in protein_to_split.items()
                 if s == 'test' and p in analysis_pids]
    ap_counts = Counter(protein_to_label[p] for p in test_ap)
    ap_lab    = ap_counts[pos_lbl] + ap_counts[neg_lbl]
    test_pos_rate = ap_counts[pos_lbl] / ap_lab if ap_lab > 0 else 0.5

    # Size fraction -- all proteins
    total = len(protein_to_split)
    test_size_frac = sum(1 for s in protein_to_split.values() if s == 'test') \
                     / total if total > 0 else 0.0

    return protein_to_split, group_to_split, test_pos_rate, test_size_frac


# ============================================================================
# VALIDATION
# ============================================================================

def validate_split(protein_to_split, protein_to_group, group_sizes, config,
                   verbose=False):
    """
    Check for leakage and bucket-level balance problems.

    Called once per redraw attempt, so printing is off by default and is
    switched on only for the candidate that is finally kept.

    Returns (offending_groups, bucket_stats, problems, report_text).
    """
    grp_splits = defaultdict(set)
    for pid, split in protein_to_split.items():
        grp = protein_to_group.get(pid)
        if grp:
            grp_splits[grp].add(split)
    offending_groups = {g for g, s in grp_splits.items() if len(s) > 1}

    grp_to_pids = defaultdict(list)
    for pid, grp in protein_to_group.items():
        if grp:
            grp_to_pids[grp].append(pid)

    bucket_counts = defaultdict(lambda: {'train': 0, 'test': 0, 'total': 0})
    for grp, pids in grp_to_pids.items():
        splits = [protein_to_split[p] for p in pids if p in protein_to_split]
        if not splits:
            continue
        gs = 'train' if splits.count('train') >= splits.count('test') else 'test'
        bkt = size_bucket(group_sizes.get(grp, len(pids)))
        bucket_counts[bkt][gs]      += 1
        bucket_counts[bkt]['total'] += 1

    problems    = []
    diagnostics = []                                    # non-vetoing
    stratified  = bool(config.get('stratify_by_bucket', True))
    abs_tol          = float(config.get('validate_abs_tol', 0.15))
    min_grps_to_warn = int(config.get('validate_min_groups', 3))
    bucket_stats     = {}

    for bkt, cts in sorted(bucket_counts.items()):
        tr, te, tot = cts['train'], cts['test'], cts['total']
        frac = te / tot if tot > 0 else 0.0
        bucket_stats[bkt] = {'train': tr, 'test': te, 'total': tot,
                             'test_frac': frac}
        # Bucket-composition checks. Under the ablation the size composition of
        # the test set is deliberately uncontrolled, so these are reported but
        # must not veto the split. Leakage vetoes in both modes.
        #
        # A bucket holding exactly one group cannot be represented on both
        # sides; that is a structural property of the data, not a bad draw, so
        # redrawing would never fix it. Demoted to a diagnostic.
        sink = problems if stratified else diagnostics
        if tot == 1:
            diagnostics.append(
                f"Bucket '{bkt}' holds a single group -- it cannot appear in "
                f"both train and test. Composition check skipped.")
        else:
            if tot > 0 and te == 0:
                sink.append(f"Bucket '{bkt}' has {tot} groups but 0 in TEST.")
            if tot > 0 and tr == 0:
                sink.append(f"Bucket '{bkt}' has {tot} groups but 0 in TRAIN.")
            if tot >= min_grps_to_warn and abs(frac - 0.2) > abs_tol:
                sink.append(f"Bucket '{bkt}' test fraction {frac:.3f} "
                            f"differs from expected 0.200 by > {abs_tol:.3f}.")

    if offending_groups:
        problems.append(
            f"{len(offending_groups)} group(ies) split across train/test (leakage).")

    lines = ["group-size bucket distribution after split (groups counted):",
             "{:15s} {:>6s} {:>6s} {:>6s} {:>9s}".format(
                 "BUCKET", "TRAIN", "TEST", "TOTAL", "TEST_FRAC")]
    for bkt, cts in sorted(bucket_counts.items()):
        tr, te, tot = cts['train'], cts['test'], cts['total']
        frac = te / tot if tot > 0 else 0.0
        lines.append("{:15s} {:6d} {:6d} {:6d} {:9.3f}".format(
            bkt, tr, te, tot, frac))
    report_text = "\n".join(lines)

    if verbose:
        print("\n" + report_text)
        if diagnostics:
            print("  bucket-composition notes (reported, not vetoing):")
            for d in diagnostics:
                print(f"    - {d}")

    return offending_groups, bucket_stats, problems, report_text


# ============================================================================
# BALANCE STATS
# ============================================================================

def balance_stats(protein_to_split, protein_to_label, global_pos_ratio,
                  analysis_pids, config):
    """
    Per-split balance statistics.

    Positive rates use the analysis population (labelled + structured).
    Size percentages use all proteins so train% + test% = 100%.
    """
    scheme  = config['label_scheme']
    pos_lbl = scheme['positive']
    neg_lbl = scheme['negative']

    stats   = {}
    total_n = len(protein_to_split)
    for sp in ('train', 'test'):
        prots    = [p for p, s in protein_to_split.items() if s == sp]
        ap_prots = [p for p in prots if p in analysis_pids]
        cts      = Counter(protein_to_label[p] for p in ap_prots)
        pos, neg = cts[pos_lbl], cts[neg_lbl]
        lab      = pos + neg
        all_cts  = Counter(protein_to_label.get(p, 'Unknown') for p in prots)
        stats[sp] = {
            'n_proteins':   len(prots),
            'size_pct':     round(100 * len(prots) / total_n, 2) if total_n else float('nan'),
            'n_positive':   pos,
            'n_negative':   neg,
            'n_unknown':    all_cts['Unknown'],
            'pos_rate_pct': round(100 * pos / lab, 2) if lab > 0 else float('nan'),
        }
    global_rate_pct = 100 * global_pos_ratio
    dev  = abs(stats['test']['pos_rate_pct'] - global_rate_pct)
    sdev = abs(stats['test']['size_pct'] - 100 * config['test_ratio'])
    stats['test_deviation_pp']      = round(dev, 2)
    stats['warning']                = dev  > config['max_label_rate_deviation']
    stats['test_size_deviation_pp'] = round(sdev, 2)
    stats['size_warning']           = sdev > config['max_size_deviation']
    return stats


# ============================================================================
# BUCKET PROFILE ANALYSIS
# ============================================================================

def bucket_profile_analysis(splits_df, analysis_pids, global_rate_pct, config):
    """
    Large-bucket randomness check and positive rate by group size analysis.

    All group sizes and positive rates are computed over the analysis
    population (labelled + structured proteins) only, so every number is
    consistent with the global rate denominator.

    Returns summary lines; also saves the rate-by-size CSV.
    """
    scheme   = config['label_scheme']
    pos_lbl  = scheme['positive']
    neg_lbl  = scheme['negative']
    noun     = scheme['noun']

    lines = []
    lines.append("")
    lines.append("=" * 90)
    lines.append("LARGE-BUCKET PROFILE ANALYSIS")
    lines.append("=" * 90)

    # -- Build analysis-population view of split 1 ---------------------------
    s1 = splits_df[splits_df["split_index"] == 1].copy()
    s1_ap = s1[s1["UniProt_AC"].isin(analysis_pids)].copy()

    ap_group_sizes = (
        s1_ap.dropna(subset=["group_id"])
        .groupby("group_id")["UniProt_AC"]
        .nunique()
    )

    lg_test = splits_df[
        (splits_df["group_bucket"] == "large") &
        (splits_df["split"] == "test")
    ]

    if lg_test.empty:
        lines.append("")
        lines.append("  No 'large' (>20 protein) groups reached any test set -- "
                     "profile analysis skipped.")
        return lines

    # -- 1. Which large groups are in TEST per split? ------------------------
    lines.append("")
    lines.append("1. Which large groups are in TEST per split?")
    lines.append("-" * 60)
    for split_idx, grps in (
        lg_test.groupby("split_index")["group_id"]
        .apply(lambda x: sorted(x.unique())).items()
    ):
        lines.append(f"  Split {int(split_idx):>2d}: {', '.join(grps)}")

    # -- 2. All large groups and their sizes ---------------------------------
    lines.append("")
    lines.append("2. All large groups and their sizes (analysis population proteins)")
    lines.append("-" * 60)
    lg_all = splits_df[splits_df["group_bucket"] == "large"]
    for gid, _ in sorted(lg_all.groupby("group_id")):
        size = int(ap_group_sizes.get(gid, 0))
        test_splits = [int(x) for x in sorted(
            lg_test[lg_test["group_id"] == gid]["split_index"].unique())]
        lines.append(f"  {gid}: {size:>4d} proteins  |  in test for splits {test_splits}")

    # -- 3. Test set sizes per split -----------------------------------------
    lines.append("")
    lines.append("3. Large test set size per split (all proteins)")
    lines.append("-" * 60)
    proteins_per_split = lg_test.groupby("split_index")["UniProt_AC"].apply(set)
    for split_idx, prots in proteins_per_split.items():
        lines.append(f"  Split {int(split_idx):>2d}: {len(prots):>4d} proteins")

    # -- 4. Overlap statistics ------------------------------------------------
    lines.append("")
    lines.append("4. Overlap statistics across all splits")
    lines.append("-" * 60)
    common = set.intersection(*proteins_per_split.values)
    union  = set.union(*proteins_per_split.values)
    lines.append(f"  Intersection (in ALL test sets): {len(common)}")
    lines.append(f"  Union (in ANY test set):         {len(union)}")
    if union:
        lines.append(f"  Jaccard similarity:              {len(common)/len(union):.3f}")

    # -- 5. Per-protein test frequency ---------------------------------------
    lines.append("")
    lines.append("5. Per-protein test frequency (how many splits each protein is in test)")
    lines.append("-" * 60)
    freq = lg_test.groupby("UniProt_AC")["split_index"].nunique()
    for cnt, n in freq.value_counts().sort_index().items():
        lines.append(f"  {cnt} split(s): {n} proteins")

    # -- 6. Per-group test frequency -- randomness check ---------------------
    #
    # The band is +/- 2 binomial SDs. It used to be
    # `abs(count - expected_freq) > 2 * expected_freq`, which with
    # expected_freq ~ 9 only fired above 27 or below -9 -- so it could never
    # flag a low count and was about seven times too loose on the high side.
    # Every group therefore always read "ok" regardless of the draw.
    lines.append("")
    lines.append("6. Per-group test frequency -- randomness check")
    lines.append("-" * 60)
    n_splits          = splits_df["split_index"].nunique()
    lg_groups         = splits_df[splits_df["group_bucket"] == "large"]["group_id"].unique()
    n_groups          = len(lg_groups)
    n_drawn_per_split = n_test_for_bucket(
        n_groups, config['test_ratio'],
        config.get('min_test_groups_per_bucket', 1))

    p_test        = n_drawn_per_split / n_groups if n_groups else float('nan')
    expected_freq = n_splits * p_test
    sd_freq       = math.sqrt(n_splits * p_test * (1 - p_test)) if n_groups else float('nan')
    lo, hi        = expected_freq - 2 * sd_freq, expected_freq + 2 * sd_freq

    lines.append(f"  Total large groups:          {n_groups}")
    lines.append(f"  Groups drawn to test/split:  {n_drawn_per_split}  "
                 f"({100 * p_test:.1f}%)")
    lines.append(f"  Expected test freq per group over {n_splits} splits: "
                 f"{expected_freq:.1f}")
    lines.append(f"  Binomial SD: {sd_freq:.2f}   "
                 f"2 SD band: [{lo:.1f}, {hi:.1f}]")
    lines.append("  With this many groups, roughly one falling outside the band "
                 "by chance is expected.")
    lines.append("")
    lines.append(f"  {'Group':30s} {'AP size':>8}  {'Test count':>10}  Note")

    group_test_counts = (
        lg_test.groupby("group_id")["split_index"].nunique()
        .reindex(lg_groups, fill_value=0)
    )
    n_outside = 0
    for gid in sorted(lg_groups):
        size  = int(ap_group_sizes.get(gid, 0))
        count = int(group_test_counts[gid])
        if count == 0:
            note = "!! NEVER in test"
            n_outside += 1
        elif count == n_splits:
            note = "!! ALWAYS in test"
            n_outside += 1
        elif count < lo or count > hi:
            note = "!! outside 2 SD"
            n_outside += 1
        else:
            note = "ok"
        lines.append(f"  {gid:30s} {size:>8}  {count:>10}  {note}")

    obs = group_test_counts.values
    lines.append(f"\n  Min test count: {obs.min()}  Max: {obs.max()}  "
                 f"Mean: {obs.mean():.2f}  Std: {obs.std():.2f}")
    lines.append(f"  Groups outside the 2 SD band: {n_outside} / {n_groups}  "
                 f"(~1 expected by chance)")
    if n_groups and obs.std() > 0:
        lines.append(f"  Observed SD vs binomial SD: {obs.std():.2f} vs "
                     f"{sd_freq:.2f}  -- "
                     f"{'tighter than' if obs.std() < sd_freq else 'wider than'} "
                     f"a fair draw would give.")

    # -- 7. Positive rate per group size -------------------------------------
    lines.append("")
    lines.append("=" * 90)
    lines.append(f"7. {noun.capitalize()} rate per group size  "
                 f"(analysis population only)")
    lines.append("=" * 90)
    lines.append(f"  Analysis population = labelled ({pos_lbl}/{neg_lbl}) + structured")
    lines.append(f"  Global {noun} rate: {global_rate_pct:.1f}%")
    lines.append(f"  n = {len(s1_ap)}")
    lines.append("")

    s1_ap = s1_ap.copy()
    s1_ap["ap_group_size"] = s1_ap["group_id"].map(ap_group_sizes)

    size_stats = (
        s1_ap
        .assign(is_positive=lambda x: x["protein_label"] == pos_lbl)
        .groupby("ap_group_size")
        .agg(
            n_groups   =("group_id",    "nunique"),
            n_proteins =("is_positive", "count"),
            n_positive =("is_positive", "sum"),
            pos_rate   =("is_positive", "mean"),
        )
        .sort_index()
    )
    size_stats.index.name = "group_size"

    lines.append(f"  {'group_size':>12s}  {'n_groups':>9s}  {'n_proteins':>12s}  "
                 f"{'n_positive':>12s}  {'pos_rate':>12s}")
    lines.append("  " + "-" * 68)
    for gs, row in size_stats.iterrows():
        lines.append(f"  {gs:>12.0f}  {int(row['n_groups']):>9d}  "
                     f"{int(row['n_proteins']):>12d}  "
                     f"{int(row['n_positive']):>12d}  "
                     f"{row['pos_rate']*100:>11.1f}%")

    out = size_stats.copy()
    out["pos_rate_pct"] = (out["pos_rate"] * 100).round(2)
    out = out.drop(columns="pos_rate").reset_index()
    out = out[["group_size", "n_groups", "n_proteins", "n_positive",
               "pos_rate_pct"]]
    out.to_csv(config['output_rate_by_size'], index=False)
    lines.append(f"\n  Saved: {config['output_rate_by_size']}")

    # -- 8. Per-split large-bucket positive rate -----------------------------
    lines.append("")
    lines.append("=" * 90)
    lines.append(f"8. Large-bucket test {noun} rate by split")
    lines.append("=" * 90)
    lines.append(f"  Global {noun} rate: {global_rate_pct:.1f}%")
    lines.append("")
    lines.append(f"  {'Split':>5}  {'Groups in test':>15}  {'Test proteins':>14}  "
                 f"{'Bucket rate':>11}")
    lines.append("  " + "-" * 54)

    for split_idx in sorted(lg_test["split_index"].unique()):
        sp_df  = lg_test[lg_test["split_index"] == split_idx]
        n_grp  = sp_df["group_id"].nunique()
        b_df   = sp_df[sp_df["UniProt_AC"].isin(analysis_pids)]
        b_cts  = Counter(b_df["protein_label"])
        b_lab  = b_cts[pos_lbl] + b_cts[neg_lbl]
        b_rate = f"{100*b_cts[pos_lbl]/b_lab:.1f}%" if b_lab else "N/A"
        lines.append(f"  {int(split_idx):>5}  {n_grp:>15}  "
                     f"{len(sp_df):>14}  {b_rate:>11}")

    return lines


# ============================================================================
# DATA LOADING
# ============================================================================

def load_inputs(config):
    """
    Read the three input tables, validate columns, and build every lookup the
    splitter needs. Returns a dict.
    """
    scheme  = config['label_scheme']
    pos_lbl = scheme['positive']
    neg_lbl = scheme['negative']
    noun    = scheme['noun']

    print("\n--- Loading data ---")
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

    # -- Duplicate group assignments -----------------------------------------
    # A protein appearing on two rows with different group_ids would be
    # silently collapsed to whichever row came last, quietly discarding the
    # other. Not a leakage risk, but it would corrupt group_sizes and every
    # bucket count downstream, so surface it explicitly.
    dup_map = (groups[[g_pid, g_gid]].drop_duplicates()
               .groupby(g_pid)[g_gid].nunique())
    n_multi = int((dup_map > 1).sum())
    if n_multi:
        offenders = dup_map[dup_map > 1].index.tolist()[:10]
        msg = (f"{n_multi} protein(s) in {config['group_mapping']} are assigned "
               f"to more than one group_id (e.g. {offenders}). Only the last "
               f"assignment is kept.")
        if config.get('strict_duplicate_check', False):
            raise ValueError(msg)
        print(f"  !! WARNING: {msg}")
    else:
        print("  Duplicate group assignments: none")

    # -- Lookups --------------------------------------------------------------
    complex_proteins = set(complexes[x_pid].unique())

    g_sub = groups[groups[g_pid].isin(complex_proteins)]
    protein_to_group = dict(zip(g_sub[g_pid], g_sub[g_gid]))

    group_sizes = defaultdict(int)
    for pid, grp in protein_to_group.items():
        group_sizes[grp] += 1
    group_sizes = dict(group_sizes)

    # Binarise labels. A binarise() returning None drops the protein to
    # Unknown, which is how essentiality discards its 'Unknown' category.
    binarise = scheme['binarise']
    protein_to_label = {}
    n_raw_dropped = 0
    for pid, raw in zip(labels[l_pid], labels[l_val]):
        bl = binarise(raw)
        if bl is None:
            n_raw_dropped += 1
        else:
            protein_to_label[pid] = bl
    if n_raw_dropped:
        print(f"  Label rows dropped as Unknown by binarise(): {n_raw_dropped}")

    all_proteins    = sorted(complex_proteins)
    structured_pids = set(protein_to_group.keys())

    analysis_pids = {
        p for p in all_proteins
        if p in structured_pids and protein_to_label.get(p) in (pos_lbl, neg_lbl)
    }
    n_pos_global = sum(1 for p in analysis_pids if protein_to_label[p] == pos_lbl)
    n_structured = len(analysis_pids)
    global_pos_ratio = n_pos_global / n_structured if n_structured > 0 else 0.5
    global_rate_pct  = 100 * global_pos_ratio

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
    print(f"  Global {noun} rate (analysis population): {global_rate_pct:.1f}%"
          f"  ({n_pos_global} / {n_structured})")

    # -- Groups and buckets ---------------------------------------------------
    group_protein_map = defaultdict(set)
    for pid, grp in protein_to_group.items():
        group_protein_map[grp].add(pid)

    no_group_proteins = complex_proteins - structured_pids

    buckets = {'large': [], 'medium': [], 'small': [], 'singleton': []}
    for grp, n in group_sizes.items():
        buckets[size_bucket(n)].append(grp)
    # Deterministic bucket ordering -- rng.shuffle then fully controls the draw
    for bname in buckets:
        buckets[bname] = sorted(buckets[bname])

    print("\n--- Group size buckets (largest first) ---")
    bucket_label = {
        'large':     '>20 proteins (merged large + very large)',
        'medium':    '6-20',
        'small':     '2-5',
        'singleton': '1 protein',
    }
    preflight = []
    for bname, bgrps in buckets.items():
        n_grp  = len(bgrps)
        n_prot = sum(group_sizes[g] for g in bgrps)
        n_test = n_test_for_bucket(n_grp, config['test_ratio'],
                                   config.get('min_test_groups_per_bucket', 1))
        print(f"  {bname:12s} ({bucket_label[bname]:40s}): "
              f"{n_grp:4d} groups, {n_prot:5d} proteins -- "
              f"{n_test} grp{'' if n_test == 1 else 's'} drawn to test each split")
        if n_grp == 1:
            preflight.append(
                f"Bucket '{bname}' holds exactly 1 group -- it cannot be "
                f"represented in both train and test. Composition check will "
                f"be skipped for this bucket.")
        elif n_grp > 1 and n_test == 0:
            preflight.append(
                f"Bucket '{bname}' has {n_grp} groups but 0 drawn to test. "
                f"Every split will fail validation. Raise "
                f"min_test_groups_per_bucket or merge this bucket.")
    print(f"  No-group proteins: {len(no_group_proteins)}")
    if preflight:
        print("\n  !! PRE-FLIGHT NOTES:")
        for p in preflight:
            print(f"    - {p}")

    return {
        'groups': groups, 'complexes': complexes,
        'protein_to_group': protein_to_group,
        'protein_to_label': protein_to_label,
        'group_sizes': group_sizes,
        'group_protein_map': group_protein_map,
        'no_group_proteins': no_group_proteins,
        'all_proteins': all_proteins,
        'structured_pids': structured_pids,
        'analysis_pids': analysis_pids,
        'buckets': buckets,
        'n_pos_global': n_pos_global,
        'n_structured': n_structured,
        'global_pos_ratio': global_pos_ratio,
        'global_rate_pct': global_rate_pct,
        'n_unknown_all': n_unknown_all,
        'n_no_group': n_no_group,
        'n_complexes': complexes[x_cid].nunique(),
    }


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def run_splits(config):
    """Generate, validate and save the splits described by `config`."""
    C      = resolve_output_paths(dict(config))
    scheme = C['label_scheme']
    noun   = scheme['noun']

    print("=" * 70)
    print(f"Repeated Random group-Level Splitting -- "
          f"{'Size-Stratified' if C['stratify_by_bucket'] else 'UNSTRATIFIED'}, "
          f"N={C['n_splits']}")
    print(C['task_description'])
    print("=" * 70)

    D = load_inputs(C)

    protein_to_group  = D['protein_to_group']
    protein_to_label  = D['protein_to_label']
    group_sizes       = D['group_sizes']
    group_protein_map = D['group_protein_map']
    no_group_proteins = D['no_group_proteins']
    all_proteins      = D['all_proteins']
    analysis_pids     = D['analysis_pids']
    buckets           = D['buckets']
    global_pos_ratio  = D['global_pos_ratio']
    global_rate_pct   = D['global_rate_pct']

    # -- Generate splits ------------------------------------------------------
    print(f"\n--- Allocation mode: "
          f"{'SIZE-STRATIFIED' if C['stratify_by_bucket'] else 'UNSTRATIFIED (BUCKET ABLATION)'} ---")
    print(f"--- Generating {C['n_splits']} splits "
          f"(tolerance +/-{C['max_label_rate_deviation']}pp label, "
          f"+/-{C['max_size_deviation']}pp size, "
          f"max {C['max_attempts']} attempts each) ---")

    all_rows        = []
    balance_records = []
    n_forced        = 0
    protein_to_split = group_to_split = None

    for i in range(C['n_splits']):
        best     = None
        best_key = None
        attempts = 0
        accepted = False

        while attempts < C['max_attempts']:
            seed = C['base_seed'] + i + attempts * 1000
            rng  = np.random.default_rng(seed)
            p2s, g2s, pos_rate, sz_frac = attempt_split(
                buckets, group_protein_map, sorted(no_group_proteins),
                protein_to_label, analysis_pids, C, rng,
            )
            attempts += 1
            sz_dev  = abs(100 * sz_frac  - 100 * C['test_ratio'])
            dev     = abs(100 * pos_rate - global_rate_pct)
            size_ok = sz_dev <= C['max_size_deviation']
            lbl_ok  = dev    <= C['max_label_rate_deviation']

            # Validation is part of the accept test rather than a post-hoc
            # filter. Previously a split that passed size and label but failed
            # validation was dropped with `continue` -- silently producing
            # fewer than n_splits splits with gaps in split_index.
            violations, _, problems, report_text = validate_split(
                p2s, protein_to_group, group_sizes, C, verbose=False)
            valid_ok = not problems

            cand = {
                'p2s': p2s, 'g2s': g2s, 'seed': seed,
                'violations': violations, 'problems': problems,
                'report': report_text,
                'sz_dev': sz_dev, 'dev': dev, 'valid_ok': valid_ok,
            }

            # ACCEPTED -> keep exactly this split, then stop.
            #
            # The explicit assignment here is deliberate. An earlier version
            # broke out of the loop but wrote `best_split`, which tracked the
            # smallest SIZE deviation across attempts -- so a previously FAILED
            # attempt was frequently saved in place of the accepted one.
            if size_ok and lbl_ok and valid_ok:
                accepted = True
                best = cand
                break

            # NOT accepted -> retain the best candidate so far, in case every
            # attempt fails. Rank: validation first (a leaking or malformed
            # split must never outrank a merely imbalanced one), then the size
            # check, then smallest LABEL deviation, then smallest size
            # deviation.
            key = (not valid_ok, not size_ok, dev, sz_dev)
            if best_key is None or key < best_key:
                best_key = key
                best = cand

        protein_to_split = best['p2s']
        group_to_split   = best['g2s']
        violations       = best['violations']
        problems         = best['problems']

        # NO SILENT DROP. If nothing validated in max_attempts tries we still
        # emit the best candidate, flagged, so the run always yields exactly
        # n_splits contiguous splits. The flag propagates to the balance CSV
        # and the summary, and require_full_n_splits aborts the save if any
        # split is unusable.
        if not accepted:
            n_forced += 1
            print(f"  !! Split {i+1}: no candidate satisfied all checks in "
                  f"{attempts} attempts. Emitting best available "
                  f"(validation_ok={best['valid_ok']}, "
                  f"label_dev={best['dev']:.2f}pp, size_dev={best['sz_dev']:.2f}pp).")
            for p in problems:
                print(f"       - {p}")

        bstats = balance_stats(protein_to_split, protein_to_label,
                               global_pos_ratio, analysis_pids, C)

        # Per-bucket protein breakdown
        pos_lbl, neg_lbl = scheme['positive'], scheme['negative']
        bucket_breakdown = {}
        for bname in ('large', 'medium', 'small', 'singleton'):
            bgrps  = buckets.get(bname, [])
            tr_n   = sum(len(group_protein_map.get(g, set()))
                         for g in bgrps if group_to_split.get(g) == 'train')
            te_n   = sum(len(group_protein_map.get(g, set()))
                         for g in bgrps if group_to_split.get(g) == 'test')
            tot_n  = tr_n + te_n
            te_pids = [p for g in bgrps if group_to_split.get(g) == 'test'
                       for p in group_protein_map.get(g, set())]
            ap_te  = [p for p in te_pids if p in analysis_pids]
            bc     = Counter(protein_to_label.get(p) for p in ap_te)
            lab    = bc[pos_lbl] + bc[neg_lbl]
            bucket_breakdown[bname] = {
                'test_n':        te_n,
                'test_pct':      100 * te_n / tot_n if tot_n > 0 else 0.0,
                'test_rate_pct': round(100 * bc[pos_lbl] / lab, 1) if lab > 0
                                 else float('nan'),
            }

        bal_row = {
            'split_index':            i + 1,
            'seed_accepted':          best['seed'],
            'attempts':               attempts,
            'accepted':               accepted,
            'validation_ok':          best['valid_ok'],
            'validation_problems':    '; '.join(problems) if problems else '',
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
            'train_unknown':          bstats['train']['n_unknown'],
            'test_unknown':           bstats['test']['n_unknown'],
            'train_pos_rate_pct':     bstats['train']['pos_rate_pct'],
            'test_pos_rate_pct':      bstats['test']['pos_rate_pct'],
            'test_deviation_pp':      bstats['test_deviation_pp'],
            'warning':                bstats['warning'],
            'leakage_violations':     len(violations),
            **{f'{b}_test_n':        bucket_breakdown[b]['test_n']
               for b in ('large', 'medium', 'small', 'singleton')},
            **{f'{b}_test_pct':      round(bucket_breakdown[b]['test_pct'], 1)
               for b in ('large', 'medium', 'small', 'singleton')},
            **{f'{b}_test_rate_pct': bucket_breakdown[b]['test_rate_pct']
               for b in ('large', 'medium', 'small', 'singleton')},
        }
        balance_records.append(bal_row)

        flag   = ' !! balance warning' if bstats['warning']      else ''
        szflag = ' !! size warning'    if bstats['size_warning'] else ''
        vflag  = ' !! VALIDATION FAIL' if not best['valid_ok']   else ''
        retry  = f' (accepted on attempt {attempts})' if attempts > 1 else ''
        print(f"  Split {i+1:2d} | "
              f"train={bstats['train']['n_proteins']} ({bstats['train']['size_pct']:.1f}%) "
              f"pos={bstats['train']['pos_rate_pct']:.1f}% | "
              f"test={bstats['test']['n_proteins']} ({bstats['test']['size_pct']:.1f}%) "
              f"pos={bstats['test']['pos_rate_pct']:.1f}% | "
              f"size_dev={bstats['test_size_deviation_pp']:.1f}pp "
              f"label_dev={bstats['test_deviation_pp']:.1f}pp"
              f"{retry}{szflag}{flag}{vflag}")

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

    # Bucket distribution for the final split, for the record
    _, _, _, final_report = validate_split(
        protein_to_split, protein_to_group, group_sizes, C, verbose=True)

    # -- Integrity gate -------------------------------------------------------
    # Everything above guarantees n_splits rows; these assertions make that a
    # hard contract rather than an assumption, so a truncated or empty CSV can
    # never overwrite a good one.
    n_emitted = len(balance_records)
    idx_seen  = sorted({r['split_index'] for r in balance_records})
    n_leak_tot = sum(r['leakage_violations'] for r in balance_records)
    n_badval   = sum(1 for r in balance_records if not r['validation_ok'])

    print("\n--- Integrity check ---")
    print(f"  Splits emitted:            {n_emitted} / {C['n_splits']}")
    print(f"  split_index contiguous:    "
          f"{idx_seen == list(range(1, C['n_splits'] + 1))}")
    print(f"  Total leakage violations:  {n_leak_tot}")
    print(f"  Splits failing validation: {n_badval}")

    if n_emitted != C['n_splits'] or idx_seen != list(range(1, C['n_splits'] + 1)):
        raise RuntimeError(
            f"Expected {C['n_splits']} contiguous splits, got {n_emitted} "
            f"({idx_seen[:5]}...). Nothing written.")
    if n_leak_tot > 0:
        raise RuntimeError(
            f"{n_leak_tot} leakage violation(s) across splits. Nothing written.")
    if C.get('require_full_n_splits', True) and n_badval > 0:
        raise RuntimeError(
            f"{n_badval} split(s) failed validation and could not be redrawn "
            f"in {C['max_attempts']} attempts. Nothing written. Inspect the "
            f"pre-flight notes above, then either loosen validate_abs_tol, "
            f"raise max_attempts, or set require_full_n_splits=False to "
            f"accept flagged splits.")

    # -- Save outputs ---------------------------------------------------------
    print("\n--- Saving outputs ---")
    splits_df = pd.DataFrame(all_rows)
    splits_df.to_csv(C['output_all_splits'], index=False)
    print(f"  Saved: {C['output_all_splits']}  ({len(all_rows):,} rows)")
    pd.DataFrame(balance_records).to_csv(C['output_balance'], index=False)
    print(f"  Saved: {C['output_balance']}")

    # -- Summary report -------------------------------------------------------
    summary = build_summary(
        C, D, balance_records, splits_df, final_report, n_emitted, n_forced)
    print(f"\n{summary}")
    with open(C['output_summary'], 'w') as f:
        f.write(summary + '\n')
    print(f"\n  Saved: {C['output_summary']}")
    print("\nDone!")

    return splits_df, pd.DataFrame(balance_records)


# ============================================================================
# SUMMARY REPORT
# ============================================================================

def build_summary(C, D, balance_records, splits_df, final_report,
                  n_emitted, n_forced):
    """Assemble the human-readable summary report."""
    scheme  = C['label_scheme']
    noun    = scheme['noun']
    buckets = D['buckets']

    n_warnings = sum(1 for r in balance_records if r['warning'])
    n_sz_warn  = sum(1 for r in balance_records if r['size_warning'])
    n_leakage  = sum(1 for r in balance_records if r['leakage_violations'] > 0)
    n_retried  = sum(1 for r in balance_records if r['attempts'] > 1)
    test_rates  = [r['test_pos_rate_pct']      for r in balance_records]
    train_rates = [r['train_pos_rate_pct']     for r in balance_records]
    test_sz     = [r['test_pct']               for r in balance_records]
    train_sz    = [r['train_pct']              for r in balance_records]
    sz_devs     = [r['test_size_deviation_pp'] for r in balance_records]
    test_pos_n  = [r['test_n_positive']        for r in balance_records]

    lines = []
    lines.append("=" * 90)
    lines.append("SPLIT SUMMARY -- Size-stratified repeated random group-level splits")
    lines.append(C['task_description'])
    lines.append("=" * 90)
    lines.append(f"Total unique proteins:            {len(D['all_proteins'])}")
    lines.append(f"Total complexes:                  {D['n_complexes']}")
    lines.append(f"Total groups:                     {len(D['group_sizes'])}")
    lines.append(f"  Large      (>20, merged):       {len(buckets['large'])}")
    lines.append(f"  Medium     (6-20):              {len(buckets['medium'])}")
    lines.append(f"  Small      (2-5):               {len(buckets['small'])}")
    lines.append(f"  Singleton  (1 protein):         {len(buckets['singleton'])}")
    lines.append(f"Analysis population:              {D['n_structured']}  "
                 f"(labelled + structured; excludes {D['n_unknown_all']} Unknown, "
                 f"{D['n_no_group']} no-group)")
    lines.append(f"Global {noun} rate:{' ' * max(1, 25 - len(noun))}"
                 f"{D['global_rate_pct']:.1f}%  "
                 f"({D['n_pos_global']} / {D['n_structured']})")
    lines.append(f"Allocation mode:                  "
                 f"{'size-stratified' if C['stratify_by_bucket'] else 'UNSTRATIFIED (bucket ablation)'}")
    req_note = '' if n_emitted == C['n_splits'] else f"  (REQUESTED {C['n_splits']})"
    lines.append(f"Number of splits:                 {n_emitted}{req_note}")
    lines.append(f"Train/test ratio:                 "
                 f"{C['train_ratio']:.0%} / {C['test_ratio']:.0%}")
    lines.append(f"Size tolerance:                   +/-{C['max_size_deviation']}pp  "
                 f"(acceptable range: "
                 f"{100*C['test_ratio']-C['max_size_deviation']:.0f}%-"
                 f"{100*C['test_ratio']+C['max_size_deviation']:.0f}%)")
    lines.append(f"Balance tolerance:                +/-{C['max_label_rate_deviation']}pp")
    lines.append(f"Max redraw attempts per split:    {C['max_attempts']}")
    lines.append(f"Splits requiring redraw:          {n_retried} / {n_emitted}")
    lines.append(f"Splits with size warnings:        {n_sz_warn} / {n_emitted}")
    lines.append(f"Splits with balance warnings:     {n_warnings} / {n_emitted}")
    lines.append(f"Splits with leakage violations:   {n_leakage} / {n_emitted}")
    lines.append(f"Splits emitted without full accept: {n_forced} / {n_emitted}")
    lines.append("")
    lines.append(f"Test positive COUNT across splits: min {min(test_pos_n)}, "
                 f"median {int(pd.Series(test_pos_n).median())}, "
                 f"max {max(test_pos_n)}. On an imbalanced task the per-split "
                 f"positive count drives PR-AUC variance more than the rate does.")
    lines.append("")
    lines.append("Denominators above are the number of splits actually emitted, "
                 "not the requested n_splits.")
    lines.append("")

    # Table 1: per-split overall balance
    lines.append(
        f"{'Split':>6} {'Att':>4}  "
        f"{'Train N':>8} {'Train%':>7} {'Train pos%':>11}  "
        f"{'Test N':>7} {'Test%':>6} {'Test pos%':>10} {'Test pos n':>11}  "
        f"{'SzDev':>6} {'LblDev':>7}  Flag")
    lines.append("-" * 100)
    for r in balance_records:
        flag = ('!! ' + ('Sz' if r['size_warning'] else '') +
                ('Lb' if r['warning'] else '') +
                ('Lk' if r['leakage_violations'] > 0 else '') +
                ('Vd' if not r['validation_ok'] else ''))
        flag = flag if flag != '!! ' else ''
        lines.append(
            f"{r['split_index']:>6} {r['attempts']:>4}  "
            f"{r['train_n']:>8} {r['train_pct']:>6.1f}% {r['train_pos_rate_pct']:>10.1f}%  "
            f"{r['test_n']:>7} {r['test_pct']:>5.1f}% {r['test_pos_rate_pct']:>9.1f}% "
            f"{r['test_n_positive']:>11}  "
            f"{r['test_size_deviation_pp']:>5.1f}pp {r['test_deviation_pp']:>6.1f}pp  {flag}")
    lines.append("-" * 100)
    lines.append(
        f"{'Mean':>6} {'':>4}  {'':>8} {sum(train_sz)/len(train_sz):>6.1f}% "
        f"{sum(train_rates)/len(train_rates):>10.1f}%  {'':>7} "
        f"{sum(test_sz)/len(test_sz):>5.1f}% {sum(test_rates)/len(test_rates):>9.1f}% "
        f"{sum(test_pos_n)/len(test_pos_n):>11.1f}  "
        f"{sum(sz_devs)/len(sz_devs):>5.1f}pp")
    lines.append(
        f"{'Std':>6} {'':>4}  {'':>8} {pd.Series(train_sz).std():>6.2f}% "
        f"{pd.Series(train_rates).std():>10.2f}%  {'':>7} "
        f"{pd.Series(test_sz).std():>5.2f}% {pd.Series(test_rates).std():>9.2f}% "
        f"{pd.Series(test_pos_n).std():>11.2f}  "
        f"{pd.Series(sz_devs).std():>5.2f}pp")
    lines.append(
        f"{'Min':>6} {'':>4}  {'':>8} {min(train_sz):>6.1f}% {min(train_rates):>10.1f}%  "
        f"{'':>7} {min(test_sz):>5.1f}% {min(test_rates):>9.1f}% {min(test_pos_n):>11}  "
        f"{min(sz_devs):>5.1f}pp")
    lines.append(
        f"{'Max':>6} {'':>4}  {'':>8} {max(train_sz):>6.1f}% {max(train_rates):>10.1f}%  "
        f"{'':>7} {max(test_sz):>5.1f}% {max(test_rates):>9.1f}% {max(test_pos_n):>11}  "
        f"{max(sz_devs):>5.1f}pp")
    lines.append("")
    lines.append("Leakage guarantee: all structural groups are atomic -- no group "
                 "spans train and test. Complexes are NOT atomic by design; see "
                 "the split_common module docstring.")
    lines.append("Flag key: Sz = size out of range, Lb = label imbalance, "
                 "Lk = leakage, Vd = failed validation")

    lines.append("")
    lines.append("Bucket distribution (final split):")
    lines.append(final_report)

    # Table 2: per-split bucket breakdown
    lines.append("")
    lines.append("=" * 90)
    lines.append("TEST SET BREAKDOWN BY STRUCTURAL GROUP SIZE BUCKET (proteins in test)")
    lines.append("=" * 90)
    lines.append(
        f"{'Split':>6}  {'Large N':>8} {'L%':>5} {'Pos%':>6}  "
        f"{'Medium N':>9} {'M%':>5} {'Pos%':>6}  "
        f"{'Small N':>8} {'S%':>5} {'Pos%':>6}  "
        f"{'Singleton N':>12} {'Sg%':>5} {'Pos%':>6}")
    lines.append("-" * 90)
    for r in balance_records:
        lines.append(
            f"{r['split_index']:>6}  "
            f"{r['large_test_n']:>8} {r['large_test_pct']:>4.1f}% {r['large_test_rate_pct']:>5.1f}%  "
            f"{r['medium_test_n']:>9} {r['medium_test_pct']:>4.1f}% {r['medium_test_rate_pct']:>5.1f}%  "
            f"{r['small_test_n']:>8} {r['small_test_pct']:>4.1f}% {r['small_test_rate_pct']:>5.1f}%  "
            f"{r['singleton_test_n']:>12} {r['singleton_test_pct']:>4.1f}% "
            f"{r['singleton_test_rate_pct']:>5.1f}%")
    lines.append("-" * 90)

    def _mean(key):
        v = [r[key] for r in balance_records if r[key] == r[key]]
        return sum(v) / len(v) if v else float('nan')

    def _std(key):
        return pd.Series([r[key] for r in balance_records]).std()

    lines.append(
        f"{'Mean':>6}  "
        f"{_mean('large_test_n'):>8.1f} {_mean('large_test_pct'):>4.1f}% "
        f"{_mean('large_test_rate_pct'):>5.1f}%  "
        f"{_mean('medium_test_n'):>9.1f} {_mean('medium_test_pct'):>4.1f}% "
        f"{_mean('medium_test_rate_pct'):>5.1f}%  "
        f"{_mean('small_test_n'):>8.1f} {_mean('small_test_pct'):>4.1f}% "
        f"{_mean('small_test_rate_pct'):>5.1f}%  "
        f"{_mean('singleton_test_n'):>12.1f} {_mean('singleton_test_pct'):>4.1f}% "
        f"{_mean('singleton_test_rate_pct'):>5.1f}%")
    lines.append(
        f"{'Std':>6}  "
        f"{_std('large_test_n'):>8.1f} {_std('large_test_pct'):>4.2f}% "
        f"{_std('large_test_rate_pct'):>5.2f}%  "
        f"{_std('medium_test_n'):>9.1f} {_std('medium_test_pct'):>4.2f}% "
        f"{_std('medium_test_rate_pct'):>5.2f}%  "
        f"{_std('small_test_n'):>8.1f} {_std('small_test_pct'):>4.2f}% "
        f"{_std('small_test_rate_pct'):>5.2f}%  "
        f"{_std('singleton_test_n'):>12.1f} {_std('singleton_test_pct'):>4.2f}% "
        f"{_std('singleton_test_rate_pct'):>5.2f}%")
    lines.append("")
    lines.append("  N% = % of that bucket's proteins going to test.")
    lines.append(f"  Pos% = {noun} rate among analysis-population test proteins "
                 f"in that bucket.")

    lines.extend(bucket_profile_analysis(
        splits_df       = splits_df,
        analysis_pids   = D['analysis_pids'],
        global_rate_pct = D['global_rate_pct'],
        config          = C,
    ))

    return '\n'.join(lines)