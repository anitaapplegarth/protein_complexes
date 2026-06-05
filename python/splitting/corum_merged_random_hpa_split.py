"""
Repeated Random group-Level Splitting — Size-Stratified, N=15
==============================================================
Generates N=15 independent train/test splits where each Foldseek structural
group is allocated atomically (never split across train and test).

ALGORITHM — per split
---------------------
Groups are divided into four size buckets (largest first):

  large      : >20 proteins   (merged large + very_large)
  medium     : 6–20
  small      : 2–5
  singleton  : 1 protein

Within each bucket, exactly round(n_groups * 0.20) groups are drawn at
random for test; the remainder go to train. This is a pure random draw —
no drug target information is used during the draw itself.

After all buckets are processed, overall label balance is checked. If the
test drug target rate deviates from the global rate by more than
`max_label_rate_deviation` percentage points, the entire split is redrawn
(up to `max_attempts` tries). Tolerance is set to 1.5pp (tighter than
essentiality's 4.5pp) to compensate for the lower global positive rate (~6%).

Proteins with no group assignment are allocated by stratified random
sampling (Drug_target / Non_target / Unknown groups independently).

ANALYSIS POPULATION
-------------------
The analysis population is proteins that are BOTH:
  - labelled (Drug_target or Non_target in corum_drug_target_hpa.csv)
  - structured (present in corum_mapping_struct.csv, i.e. have a group_id)

All rates and deviations use this population as the denominator. Proteins
that are Unknown or have no structural group are carried through the split
CSV for completeness but excluded from all rate calculations.

OUTPUTS
-------
  corum_drug_hpa_protein_splits.csv       : one row per (protein, split_index)
  corum_drug_hpa_split.csv                : per-split label balance statistics
  corum_drug_hpa_split_summary.txt        : human-readable summary + profile
  corum_avg_drug_target_by_group_size.csv : drug target rate per group size
                                         (analysis population only)
"""

import pandas as pd
import numpy as np
from collections import defaultdict, Counter


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Input files
    'group_mapping':      '/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/corum/mapping_struct.csv',
    'complex_membership': '/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/corum/stoich_protein.csv',
    'drug_target':        '/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/corum_drug_target_hpa.csv',

    # Output files
    'output_all_splits':    '/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/corum/hpa_protein_merged_splits.csv',
    'output_balance':       '/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/corum/hpa_merged_groups.csv',
    'output_summary':       '/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/corum/hpa_split_merged_summary.txt',
    'output_drug_by_size':  '/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/corum/avg_hpa_by_group__merged_size.csv',

    # Large groups to highlight in the focus-group profile (section 8)
    'focus_groups': ['grp_0222', 'grp_0064', 'grp_1179', 'grp_0000'],

    # Number of independent splits to produce
    'n_splits': 15,

    # Train/test ratios
    'train_ratio': 0.80,
    'test_ratio':  0.20,

    # Flag a split if test set size deviates more than this from 20%
    'max_size_deviation': 5.0,   # pp — acceptable range [15%, 25%]

    # Flag a split if test drug target rate deviates more than this from global
    # (percentage points). Tighter than essentiality (1.5pp vs 4.5pp) to
    # compensate for the lower global positive rate (~6%).
    'max_label_rate_deviation': 1.5,

    # Maximum redraw attempts per split before accepting best available
    'max_attempts': 10,

    # Base seed
    'base_seed': 42,
}


# ============================================================================
# HELPERS
# ============================================================================

def size_bucket(n):
    """Map group size to bucket name."""
    if n == 1:    return 'singleton'
    elif n <= 5:  return 'small'
    elif n <= 20: return 'medium'
    else:         return 'large'


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
    test_pos_rate    : float  — drug target rate in test, analysis population only
    test_size_frac   : float  — fraction of ALL proteins assigned to test
    """
    test_ratio     = config['test_ratio']
    group_to_split = {}

    for bucket_grps in buckets.values():
        if not bucket_grps:
            continue
        n_test    = max(round(len(bucket_grps) * test_ratio), 0)
        grps_list = list(bucket_grps)
        rng.shuffle(grps_list)
        for i, grp in enumerate(grps_list):
            group_to_split[grp] = 'test' if i < n_test else 'train'

    protein_to_split = {}
    for grp, split in group_to_split.items():
        for pid in group_protein_map.get(grp, set()):
            protein_to_split[pid] = split

    # No-group proteins: stratified by label class
    for lbl in ('Drug_target', 'Non_target', 'Unknown'):
        grp = sorted(p for p in no_group_proteins
                     if protein_to_label.get(p, 'Unknown') == lbl)
        rng.shuffle(grp)
        n_test_grp = round(len(grp) * test_ratio)
        for i, pid in enumerate(grp):
            protein_to_split[pid] = 'test' if i < n_test_grp else 'train'

    # Drug target rate — analysis population (labelled + structured) only
    test_ap   = [p for p, s in protein_to_split.items()
                 if s == 'test' and p in analysis_pids]
    ap_counts = Counter(protein_to_label[p] for p in test_ap)
    ap_lab    = ap_counts['Drug_target'] + ap_counts['Non_target']
    test_pos_rate = ap_counts['Drug_target'] / ap_lab if ap_lab > 0 else 0.5

    # Size fraction — all proteins
    total = len(protein_to_split)
    test_size_frac = sum(1 for s in protein_to_split.values() if s == 'test') \
                     / total if total > 0 else 0.0

    return protein_to_split, group_to_split, test_pos_rate, test_size_frac


# ============================================================================
# VALIDATION
# ============================================================================

def validate_split(protein_to_split, protein_to_group, group_sizes, config):
    """
    Check for leakage and bucket-level balance problems.

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

    problems = []
    abs_tol          = float(config.get('validate_abs_tol', 0.15))
    min_grps_to_warn = int(config.get('validate_min_groups', 3))
    bucket_stats     = {}

    print("\ngroup-size bucket distribution after split (groups counted):")
    print("{:15s} {:>6s} {:>6s} {:>6s} {:>9s}".format(
        "BUCKET", "TRAIN", "TEST", "TOTAL", "TEST_FRAC"))
    for bkt, cts in sorted(bucket_counts.items()):
        tr, te, tot = cts['train'], cts['test'], cts['total']
        frac = te / tot if tot > 0 else 0.0
        bucket_stats[bkt] = {'train': tr, 'test': te, 'total': tot, 'test_frac': frac}
        print("{:15s} {:6d} {:6d} {:6d} {:9.3f}".format(bkt, tr, te, tot, frac))
        if tot > 0 and te == 0:
            problems.append(f"Bucket '{bkt}' has {tot} groups but 0 in TEST.")
        if tot > 0 and tr == 0:
            problems.append(f"Bucket '{bkt}' has {tot} groups but 0 in TRAIN.")
        if tot >= min_grps_to_warn and abs(frac - 0.2) > abs_tol:
            problems.append(f"Bucket '{bkt}' test fraction {frac:.3f} "
                            f"differs from expected 0.200 by > {abs_tol:.3f}.")
    if offending_groups:
        problems.append(f"{len(offending_groups)} group(ies) split across train/test (leakage).")

    lines = ["group-size bucket distribution after split (groups counted):",
             "{:15s} {:>6s} {:>6s} {:>6s} {:>9s}".format(
                 "BUCKET", "TRAIN", "TEST", "TOTAL", "TEST_FRAC")]
    for bkt, cts in sorted(bucket_counts.items()):
        tr, te, tot = cts['train'], cts['test'], cts['total']
        frac = te / tot if tot > 0 else 0.0
        lines.append("{:15s} {:6d} {:6d} {:6d} {:9.3f}".format(bkt, tr, te, tot, frac))
    return offending_groups, bucket_stats, problems, "\n".join(lines)


# ============================================================================
# BALANCE STATS
# ============================================================================

def balance_stats(protein_to_split, protein_to_label, global_pos_ratio,
                  analysis_pids, config):
    """
    Per-split balance statistics.

    Drug target rates use the analysis population (labelled+structured).
    Size percentages use all proteins so train% + test% = 100%.
    """
    stats   = {}
    total_n = len(protein_to_split)
    for sp in ('train', 'test'):
        prots    = [p for p, s in protein_to_split.items() if s == sp]
        ap_prots = [p for p in prots if p in analysis_pids]
        cts      = Counter(protein_to_label[p] for p in ap_prots)
        pos, neg = cts['Drug_target'], cts['Non_target']
        lab      = pos + neg
        all_cts  = Counter(protein_to_label.get(p, 'Unknown') for p in prots)
        stats[sp] = {
            'n_proteins':    len(prots),
            'size_pct':      round(100 * len(prots) / total_n, 2) if total_n else float('nan'),
            'n_drug_target': pos,
            'n_non_target':  neg,
            'n_unknown':     all_cts['Unknown'],
            'pos_rate_pct':  round(100 * pos / lab, 2) if lab > 0 else float('nan'),
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

def bucket_profile_analysis(splits_df, analysis_pids, protein_to_label_map,
                             global_rate_pct, focus_groups, config):
    """
    Large-bucket randomness check and drug target rate by group size analysis.

    All group sizes and drug target rates are computed over the analysis
    population (labelled + structured proteins) only, so every number is
    consistent with the global rate denominator.

    Appends to the summary report and saves the CSV.
    """
    lines = []
    lines.append("")
    lines.append("=" * 90)
    lines.append("LARGE-BUCKET PROFILE ANALYSIS")
    lines.append("=" * 90)

    # ── Build analysis-population view of split 1 ────────────────────────────
    s1 = splits_df[splits_df["split_index"] == 1].copy()
    s1_ap = s1[s1["UniProt_AC"].isin(analysis_pids)].copy()

    # Group sizes from analysis population (one UniProt_AC per row in split 1)
    ap_group_sizes = (
        s1_ap.dropna(subset=["group_id"])
        .groupby("group_id")["UniProt_AC"]
        .nunique()
    )

    lg_test = splits_df[
        (splits_df["group_bucket"] == "large") &
        (splits_df["split"] == "test")
    ]

    # ── 1. Which large groups are in TEST per split? ──────────────────────────
    lines.append("")
    lines.append("1. Which large groups are in TEST per split?")
    lines.append("-" * 60)
    for split_idx, grps in (
        lg_test.groupby("split_index")["group_id"]
        .apply(lambda x: sorted(x.unique())).items()
    ):
        lines.append(f"  Split {int(split_idx):>2d}: {', '.join(grps)}")

    # ── 2. All large groups and their sizes (analysis population) ─────────────
    lines.append("")
    lines.append("2. All large groups and their sizes (analysis population proteins)")
    lines.append("-" * 60)
    lg_all = splits_df[splits_df["group_bucket"] == "large"]
    for gid, _ in sorted(lg_all.groupby("group_id")):
        size = int(ap_group_sizes.get(gid, 0))
        test_splits = [int(x) for x in sorted(
            lg_test[lg_test["group_id"] == gid]["split_index"].unique())]
        lines.append(f"  {gid}: {size:>4d} proteins  |  in test for splits {test_splits}")

    # ── 3. Test set sizes per split ───────────────────────────────────────────
    lines.append("")
    lines.append("3. Large test set size per split (all proteins)")
    lines.append("-" * 60)
    proteins_per_split = lg_test.groupby("split_index")["UniProt_AC"].apply(set)
    for split_idx, prots in proteins_per_split.items():
        lines.append(f"  Split {int(split_idx):>2d}: {len(prots):>4d} proteins")

    # ── 4. Overlap statistics ─────────────────────────────────────────────────
    lines.append("")
    lines.append("4. Overlap statistics across all splits")
    lines.append("-" * 60)
    common = set.intersection(*proteins_per_split.values)
    union  = set.union(*proteins_per_split.values)
    lines.append(f"  Intersection (in ALL test sets): {len(common)}")
    lines.append(f"  Union (in ANY test set):         {len(union)}")
    if union:
        lines.append(f"  Jaccard similarity:              {len(common)/len(union):.3f}")

    # ── 5. Per-protein test frequency ─────────────────────────────────────────
    lines.append("")
    lines.append("5. Per-protein test frequency (how many splits each protein is in test)")
    lines.append("-" * 60)
    freq = lg_test.groupby("UniProt_AC")["split_index"].nunique()
    for cnt, n in freq.value_counts().sort_index().items():
        lines.append(f"  {cnt} split(s): {n} proteins")

    # ── 6. Per-group test frequency ───────────────────────────────────────────
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
    lines.append(f"  Expected test freq per group over {n_splits} splits: {expected_freq:.1f}")
    lines.append("")
    lines.append(f"  {'Group':30s} {'AP size':>8}  {'Test count':>10}  Note")

    group_test_counts = (
        lg_test.groupby("group_id")["split_index"].nunique()
        .reindex(lg_groups, fill_value=0)
    )
    for gid in sorted(lg_groups):
        size  = int(ap_group_sizes.get(gid, 0))
        count = int(group_test_counts[gid])
        if count == 0:
            note = "⚠ NEVER in test"
        elif count == n_splits:
            note = "⚠ ALWAYS in test"
        elif abs(count - expected_freq) > 2 * expected_freq:
            note = "⚠ unusually high/low"
        else:
            note = "ok"
        lines.append(f"  {gid:30s} {size:>8}  {count:>10}  {note}")

    obs = group_test_counts.values
    lines.append(f"\n  Min test count: {obs.min()}  Max: {obs.max()}  "
                 f"Mean: {obs.mean():.2f}  Std: {obs.std():.2f}")

    # ── 7. Drug target rate per group size (analysis population) ──────────────
    lines.append("")
    lines.append("=" * 90)
    lines.append("7. Drug target rate per group size  (analysis population only)")
    lines.append("=" * 90)
    lines.append(f"  Analysis population = labelled (Drug_target/Non_target) + structured")
    lines.append(f"  Global drug target rate: {global_rate_pct:.1f}%")
    lines.append(f"  n = {len(s1_ap)}")
    lines.append("")

    s1_ap = s1_ap.copy()
    s1_ap["ap_group_size"] = s1_ap["group_id"].map(ap_group_sizes)

    size_stats = (
        s1_ap
        .assign(is_target=lambda x: x["protein_label"] == "Drug_target")
        .groupby("ap_group_size")
        .agg(
            n_groups    =("group_id",  "nunique"),
            n_proteins  =("is_target", "count"),
            n_drug_target=("is_target", "sum"),
            target_rate =("is_target", "mean"),
        )
        .sort_index()
    )
    size_stats.index.name = "group_size"

    lines.append(f"  {'group_size':>12s}  {'n_groups':>9s}  {'n_proteins':>12s}  "
                 f"{'n_drug_target':>14s}  {'target_rate':>12s}")
    lines.append("  " + "-" * 68)
    for gs, row in size_stats.iterrows():
        lines.append(f"  {gs:>12.0f}  {int(row['n_groups']):>9d}  "
                     f"{int(row['n_proteins']):>12d}  "
                     f"{int(row['n_drug_target']):>14d}  "
                     f"{row['target_rate']*100:>11.1f}%")

    # Save CSV
    out = size_stats.copy()
    out["target_rate_pct"] = (out["target_rate"] * 100).round(2)
    out = out.drop(columns="target_rate").reset_index()
    out = out[["group_size", "n_groups", "n_proteins", "n_drug_target", "target_rate_pct"]]
    out.to_csv(config['output_drug_by_size'], index=False)
    lines.append(f"\n  Saved: {config['output_drug_by_size']}")

    # ── 8. Focus group drug target rate profile ───────────────────────────────
    lines.append("")
    lines.append("=" * 90)
    lines.append("8. Drug target rate profile — focus groups  (analysis population only)")
    lines.append("=" * 90)
    lines.append(f"  Global drug target rate: {global_rate_pct:.1f}%")
    lines.append("")
    lines.append(f"  {'Group':12s} {'AP size':>8}  {'Drug_target':>12}  "
                 f"{'Non_target':>10}  {'vs global':>10}")
    lines.append("  " + "-" * 58)

    for gid in focus_groups:
        grp_ap = s1_ap[s1_ap["group_id"] == gid]
        size   = int(ap_group_sizes.get(gid, 0))
        n_pos  = int((grp_ap["protein_label"] == "Drug_target").sum())
        n_neg  = int((grp_ap["protein_label"] == "Non_target").sum())
        n_lab  = n_pos + n_neg
        rate   = 100 * n_pos / n_lab if n_lab > 0 else float('nan')
        diff   = rate - global_rate_pct if n_lab > 0 else float('nan')
        dirn   = f"+{diff:.1f}pp" if diff >= 0 else f"{diff:.1f}pp"
        flag   = "  ⚠ high" if diff > 5 else ("  ⚠ low" if diff < -5 else "")
        lines.append(f"  {gid:12s} {size:>8}  {n_pos:>12}  {n_neg:>10}  "
                     f"{dirn:>10}{flag}")

    # ── 9. Per-split large-bucket rate annotated with focus groups ────────────
    lines.append("")
    lines.append("  Large-bucket test drug target rate by split, annotated with focus groups:")
    lines.append(f"  {'Split':>5}  {'Focus groups in test':<34}  "
                 f"{'Focus rate':>10}  {'Bucket rate':>11}")
    lines.append("  " + "-" * 68)

    for split_idx in sorted(lg_test["split_index"].unique()):
        sp_df       = lg_test[lg_test["split_index"] == split_idx]
        in_test     = set(sp_df["group_id"].unique())
        heavy       = [g for g in focus_groups if g in in_test]
        heavy_label = ', '.join(heavy) if heavy else "none"

        if heavy:
            hdf   = sp_df[sp_df["group_id"].isin(heavy) &
                          sp_df["UniProt_AC"].isin(analysis_pids)]
            h_cts = Counter(hdf["protein_label"])
            h_lab = h_cts["Drug_target"] + h_cts["Non_target"]
            h_rate = f"{100*h_cts['Drug_target']/h_lab:.1f}%" if h_lab else "N/A"
        else:
            h_rate = "N/A"

        b_df   = sp_df[sp_df["UniProt_AC"].isin(analysis_pids)]
        b_cts  = Counter(b_df["protein_label"])
        b_lab  = b_cts["Drug_target"] + b_cts["Non_target"]
        b_rate = f"{100*b_cts['Drug_target']/b_lab:.1f}%" if b_lab else "N/A"

        lines.append(f"  {int(split_idx):>5}  {heavy_label:<34}  "
                     f"{h_rate:>10}  {b_rate:>11}")

    return lines


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("Repeated Random group-Level Splitting — Size-Stratified, N=15")
    print("Drug Target Labels (HPA)")
    print("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n--- Loading data ---")
    groups    = pd.read_csv(CONFIG['group_mapping'])
    complexes = pd.read_csv(CONFIG['complex_membership'])
    targets   = pd.read_csv(CONFIG['drug_target'])

    print(f"  Groups:    {len(groups)} rows, {groups['group_id'].nunique()} unique groups")
    print(f"  Complexes: {complexes['ComplexId'].nunique()} complexes, "
          f"{complexes['ProteinId'].nunique()} unique proteins")
    print(f"  Drug targets: {len(targets)} labelled proteins "
          f"({targets['target'].sum()} positive)")

    # ── Build lookups ──────────────────────────────────────────────────────────
    complex_proteins = set(complexes['ProteinId'].unique())

    # protein_to_group: one entry per protein (last mapping wins if duplicates)
    protein_to_group = {
        row['uniprot_id']: row['group_id']
        for _, row in groups.iterrows()
        if row['uniprot_id'] in complex_proteins
    }

    # group_sizes: count from protein_to_group so it matches group_protein_map
    group_sizes = defaultdict(int)
    for pid, grp in protein_to_group.items():
        group_sizes[grp] += 1
    group_sizes = dict(group_sizes)

    # Drug target labels: 1 -> Drug_target, 0 -> Non_target
    protein_to_label = {}
    for _, row in targets.iterrows():
        label = 'Drug_target' if row['target'] == 1 else 'Non_target'
        protein_to_label[row['ProteinId']] = label

    all_proteins    = sorted(complex_proteins)
    structured_pids = set(protein_to_group.keys())

    # ── Analysis population ────────────────────────────────────────────────────
    # Proteins that are BOTH labelled AND have a structural group.
    # This is the consistent denominator for all rates throughout the script.
    analysis_pids = {
        p for p in all_proteins
        if p in structured_pids and protein_to_label.get(p) in ('Drug_target', 'Non_target')
    }
    n_pos_global = sum(1 for p in analysis_pids if protein_to_label[p] == 'Drug_target')
    n_neg_global = len(analysis_pids) - n_pos_global
    n_structured = len(analysis_pids)
    global_pos_ratio = n_pos_global / n_structured if n_structured > 0 else 0.5
    global_rate_pct  = 100 * global_pos_ratio

    # Counts for reporting
    n_unknown_all = sum(1 for p in all_proteins
                        if protein_to_label.get(p, 'Unknown') == 'Unknown')
    n_no_group    = len(complex_proteins - structured_pids)

    print(f"\n  Label distribution ({len(all_proteins)} total complex proteins):")
    n_pos_all = sum(1 for p in all_proteins if protein_to_label.get(p) == 'Drug_target')
    n_neg_all = sum(1 for p in all_proteins if protein_to_label.get(p) == 'Non_target')
    print(f"    Drug_target    : {n_pos_all:4d}")
    print(f"    Non_target     : {n_neg_all:4d}")
    print(f"    Unknown        : {n_unknown_all:4d}  (no label — excluded)")
    print(f"    No-group       : {n_no_group:4d}  (no structural assignment — excluded)")
    print(f"  Analysis population (labelled + structured): {n_structured}")
    print(f"  Global drug target rate (analysis population): {global_rate_pct:.1f}%"
          f"  ({n_pos_global} / {n_structured})")

    # ── Build group -> protein map and size buckets ────────────────────────────
    group_protein_map = defaultdict(set)
    for pid, grp in protein_to_group.items():
        group_protein_map[grp].add(pid)

    no_group_proteins = complex_proteins - structured_pids

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
        n_prot = sum(group_sizes[g] for g in bgrps)
        n_test = round(n_grp * CONFIG['test_ratio'])
        print(f"  {bname:12s} ({bucket_label[bname]:40s}): "
              f"{n_grp:4d} groups, {n_prot:5d} proteins — "
              f"{n_test} grp{'' if n_test == 1 else 's'} drawn to test each split")
    print(f"  No-group proteins: {len(no_group_proteins)}")

    # ── Generate splits ────────────────────────────────────────────────────────
    print(f"\n--- Generating {CONFIG['n_splits']} splits "
          f"(tolerance ±{CONFIG['max_label_rate_deviation']}pp label, "
          f"±{CONFIG['max_size_deviation']}pp size, "
          f"max {CONFIG['max_attempts']} attempts each) ---")

    all_rows        = []
    balance_records = []

    for i in range(CONFIG['n_splits']):
        best_split = best_grp_split = None
        best_dev   = best_sz_dev   = float('inf')
        best_seed  = None
        attempts   = 0
        accepted   = False

        while attempts < CONFIG['max_attempts']:
            seed = CONFIG['base_seed'] + i + attempts * 1000
            rng  = np.random.default_rng(seed)
            p2s, g2s, pos_rate, sz_frac = attempt_split(
                buckets, group_protein_map, sorted(no_group_proteins),
                protein_to_label, analysis_pids, CONFIG, rng,
            )
            attempts += 1
            sz_dev = abs(100 * sz_frac  - 100 * CONFIG['test_ratio'])
            dev    = abs(100 * pos_rate - global_rate_pct)
            if (sz_dev, dev) < (best_sz_dev, best_dev):
                best_sz_dev, best_dev = sz_dev, dev
                best_split, best_grp_split, best_seed = p2s, g2s, seed
            if sz_dev <= CONFIG['max_size_deviation'] and \
               dev    <= CONFIG['max_label_rate_deviation']:
                accepted = True
                break

        protein_to_split = best_split
        group_to_split   = best_grp_split

        violations, _, problems, report_text = validate_split(
            protein_to_split, protein_to_group, group_sizes, CONFIG)
        if violations or problems:
            if problems:
                print("Rejecting split due to validation problems.")
            continue

        with open("split_summary.txt", "a") as f:
            f.write(f"\n{report_text}\n")

        bstats = balance_stats(
            protein_to_split, protein_to_label, global_pos_ratio,
            analysis_pids, CONFIG)

        # Per-bucket protein breakdown
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
            # Drug target rate for bucket: analysis population only
            ap_te  = [p for p in te_pids if p in analysis_pids]
            bc     = Counter(protein_to_label.get(p) for p in ap_te)
            lab    = bc['Drug_target'] + bc['Non_target']
            bucket_breakdown[bname] = {
                'test_n':        te_n,
                'test_pct':      100 * te_n / tot_n if tot_n > 0 else 0.0,
                'test_rate_pct': round(100 * bc['Drug_target'] / lab, 1) if lab > 0
                                 else float('nan'),
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
            'train_drug_target':      bstats['train']['n_drug_target'],
            'test_drug_target':       bstats['test']['n_drug_target'],
            'train_non_target':       bstats['train']['n_non_target'],
            'test_non_target':        bstats['test']['n_non_target'],
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

        flag   = ' ⚠ balance warning' if bstats['warning']     else ''
        szflag = ' ⚠ size warning'    if bstats['size_warning'] else ''
        retry  = f' (accepted on attempt {attempts})' if attempts > 1 else ''
        print(f"  Split {i+1:2d} | "
              f"train={bstats['train']['n_proteins']} ({bstats['train']['size_pct']:.1f}%) "
              f"pos={bstats['train']['pos_rate_pct']:.1f}% | "
              f"test={bstats['test']['n_proteins']} ({bstats['test']['size_pct']:.1f}%) "
              f"pos={bstats['test']['pos_rate_pct']:.1f}% | "
              f"size_dev={bstats['test_size_deviation_pp']:.1f}pp "
              f"label_dev={bstats['test_deviation_pp']:.1f}pp"
              f"{retry}{szflag}{flag}")

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
                'label_mask':    pid in analysis_pids,
            })

    # ── Save outputs ──────────────────────────────────────────────────────────
    print("\n--- Saving outputs ---")
    splits_df = pd.DataFrame(all_rows)
    splits_df.to_csv(CONFIG['output_all_splits'], index=False)
    print(f"  Saved: {CONFIG['output_all_splits']}  ({len(all_rows):,} rows)")
    pd.DataFrame(balance_records).to_csv(CONFIG['output_balance'], index=False)
    print(f"  Saved: {CONFIG['output_balance']}")

    # ── Summary report ────────────────────────────────────────────────────────
    n_warnings = sum(1 for r in balance_records if r['warning'])
    n_sz_warn  = sum(1 for r in balance_records if r['size_warning'])
    n_leakage  = sum(1 for r in balance_records if r['leakage_violations'] > 0)
    n_retried  = sum(1 for r in balance_records if r['attempts'] > 1)
    test_rates  = [r['test_pos_rate_pct']      for r in balance_records]
    train_rates = [r['train_pos_rate_pct']     for r in balance_records]
    test_sz     = [r['test_pct']               for r in balance_records]
    train_sz    = [r['train_pct']              for r in balance_records]
    sz_devs     = [r['test_size_deviation_pp'] for r in balance_records]

    lines = []
    lines.append("=" * 90)
    lines.append("SPLIT SUMMARY — Size-stratified repeated random group-level splits")
    lines.append("Drug Target Labels (HPA)")
    lines.append("=" * 90)
    lines.append(f"Total unique proteins:            {len(all_proteins)}")
    lines.append(f"Total complexes:                  {complexes['ComplexId'].nunique()}")
    lines.append(f"Total groups:                     {len(group_sizes)}")
    lines.append(f"  Large      (>20, merged):       {len(buckets['large'])}")
    lines.append(f"  Medium     (6–20):              {len(buckets['medium'])}")
    lines.append(f"  Small      (2–5):               {len(buckets['small'])}")
    lines.append(f"  Singleton  (1 protein):         {len(buckets['singleton'])}")
    lines.append(f"Analysis population:              {n_structured}  "
                 f"(labelled + structured; excludes {n_unknown_all} Unknown, "
                 f"{n_no_group} no-group)")
    lines.append(f"Global drug target rate:          {global_rate_pct:.1f}%  "
                 f"({n_pos_global} / {n_structured})")
    lines.append(f"Number of splits:                 {CONFIG['n_splits']}")
    lines.append(f"Train/test ratio:                 "
                 f"{CONFIG['train_ratio']:.0%} / {CONFIG['test_ratio']:.0%}")
    lines.append(f"Size tolerance:                   ±{CONFIG['max_size_deviation']}pp  "
                 f"(acceptable range: "
                 f"{100*CONFIG['test_ratio']-CONFIG['max_size_deviation']:.0f}%–"
                 f"{100*CONFIG['test_ratio']+CONFIG['max_size_deviation']:.0f}%)")
    lines.append(f"Balance tolerance:                ±{CONFIG['max_label_rate_deviation']}pp")
    lines.append(f"Max redraw attempts per split:    {CONFIG['max_attempts']}")
    lines.append(f"Splits requiring redraw:          {n_retried} / {CONFIG['n_splits']}")
    lines.append(f"Splits with size warnings:        {n_sz_warn} / {CONFIG['n_splits']}")
    lines.append(f"Splits with balance warnings:     {n_warnings} / {CONFIG['n_splits']}")
    lines.append(f"Splits with leakage violations:   {n_leakage} / {CONFIG['n_splits']}")
    lines.append("")

    # Table 1: per-split overall balance
    lines.append(
        f"{'Split':>6} {'Att':>4}  "
        f"{'Train N':>8} {'Train%':>7} {'Train pos%':>11}  "
        f"{'Test N':>7} {'Test%':>6} {'Test pos%':>10}  "
        f"{'SzDev':>6} {'LblDev':>7}  Flag")
    lines.append("-" * 90)
    for r in balance_records:
        flag = ('⚠ ' + ('Sz' if r['size_warning'] else '') +
                ('Lb' if r['warning'] else '') +
                ('Lk' if r['leakage_violations'] > 0 else ''))
        flag = flag if flag != '⚠ ' else ''
        lines.append(
            f"{r['split_index']:>6} {r['attempts']:>4}  "
            f"{r['train_n']:>8} {r['train_pct']:>6.1f}% {r['train_pos_rate_pct']:>10.1f}%  "
            f"{r['test_n']:>7} {r['test_pct']:>5.1f}% {r['test_pos_rate_pct']:>9.1f}%  "
            f"{r['test_size_deviation_pp']:>5.1f}pp {r['test_deviation_pp']:>6.1f}pp  {flag}")
    lines.append("-" * 90)
    lines.append(
        f"{'Mean':>6} {'':>4}  {'':>8} {sum(train_sz)/len(train_sz):>6.1f}% "
        f"{sum(train_rates)/len(train_rates):>10.1f}%  {'':>7} "
        f"{sum(test_sz)/len(test_sz):>5.1f}% {sum(test_rates)/len(test_rates):>9.1f}%  "
        f"{sum(sz_devs)/len(sz_devs):>5.1f}pp")
    lines.append(
        f"{'Std':>6} {'':>4}  {'':>8} {pd.Series(train_sz).std():>6.2f}% "
        f"{pd.Series(train_rates).std():>10.2f}%  {'':>7} "
        f"{pd.Series(test_sz).std():>5.2f}% {pd.Series(test_rates).std():>9.2f}%  "
        f"{pd.Series(sz_devs).std():>5.2f}pp")
    lines.append(
        f"{'Min':>6} {'':>4}  {'':>8} {min(train_sz):>6.1f}% {min(train_rates):>10.1f}%  "
        f"{'':>7} {min(test_sz):>5.1f}% {min(test_rates):>9.1f}%  {min(sz_devs):>5.1f}pp")
    lines.append(
        f"{'Max':>6} {'':>4}  {'':>8} {max(train_sz):>6.1f}% {max(train_rates):>10.1f}%  "
        f"{'':>7} {max(test_sz):>5.1f}% {max(test_rates):>9.1f}%  {max(sz_devs):>5.1f}pp")
    lines.append("")
    lines.append("Leakage guarantee: all groups are atomic — no group spans train and test.")
    lines.append("Flag key: Sz = size out of range, Lb = label imbalance, Lk = leakage")

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
    lines.append("  Pos% = drug target rate among analysis-population test proteins in that bucket.")

    # Bucket profile analysis
    profile_lines = bucket_profile_analysis(
        splits_df            = splits_df,
        analysis_pids        = analysis_pids,
        protein_to_label_map = protein_to_label,
        global_rate_pct      = global_rate_pct,
        focus_groups         = CONFIG['focus_groups'],
        config               = CONFIG,
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