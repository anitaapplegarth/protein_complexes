# """
# Repeated Random group-Level Splitting — Size-Stratified, N=15
# ==============================================================
# Generates N=15 independent train/test splits where each Foldseek structural
# group is allocated atomically (never split across train and test).

# ALGORITHM — per split
# ---------------------
# groups are divided into four size buckets (largest first):

#   large      : >20 proteins   (formerly large + very_large)
#   medium     : 6–20           (97 groups)
#   small      : 2–5            (274 groups)
#   singleton  : 1 protein      (592 groups)

# Within each bucket, exactly round(n_groups * 0.20) groups are drawn at
# random for test; the remainder go to train. This is a pure random draw —
# no essentiality information is used during the draw itself.

# After all buckets are processed, overall label balance is checked.  If the
# test essential rate deviates from the global rate by more than
# `max_ess_rate_deviation` percentage points, the entire split is redrawn
# (up to `max_attempts` tries). All 15 accepted splits are saved regardless
# of whether they used retries.

# The 6 proteins with no group assignment are allocated by stratified random
# sampling (Essential / Non-essential / Unknown groups independently).

# OUTPUTS
# -------
#   protein_splits_all_stratified.csv    : one row per (protein, split_index)
#   split_balance_summary_stratified.csv : per-split label balance statistics
#   split_summary_stratified.txt         : human-readable summary

# """

# import pandas as pd
# import numpy as np
# from collections import defaultdict, Counter


# # ============================================================================
# # CONFIGURATION
# # ============================================================================

# CONFIG = {
#     # Input files
#     'group_mapping':     '/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/corum_mapping_struct.csv',
#     'complex_membership': '/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/corum_stoich_protein.csv',
#     'essentiality':       '/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/lu_essentiality_protein.csv',


#     # Output files
#     'output_all_splits': '/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/corum_ess_merged2_splits.csv',
#     'output_balance':    '/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/corum_ess_merged2_groups.csv',
#     'output_summary':    '/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/corum_ess_merged2_split_summary.txt',

#     # Number of independent splits to produce
#     'n_splits': 15,

#     # Train/test ratios
#     'train_ratio': 0.80,
#     'test_ratio':  0.20,

#     # Flag a split if the test set size deviates from exactly 20% of total
#     # proteins by more than this many percentage points (i.e. test must be
#     # between 15% and 25% of total proteins). Checked before label balance.
#     'max_size_deviation': 5.0,   # pp — so acceptable range is [15%, 25%]

#     # Flag a split if test essential rate deviates more than this from global
#     # (percentage points). Flagged splits are redrawn up to max_attempts.
#     'max_ess_rate_deviation': 2.5,

#     # Maximum redraw attempts per split before accepting best available
#     'max_attempts': 10,

#     # Base seed — split i uses seed = base_seed + i for first attempt;
#     # retries use base_seed + i + attempt * 1000
#     'base_seed': 42,
# }


# # ============================================================================
# # HELPERS
# # ============================================================================

# def binarise_label(raw_label):
#     """Core -> Essential; Cancer & Non-essential -> Non-essential."""
#     if raw_label == 'Core':
#         return 'Essential'
#     elif raw_label in ('Cancer', 'Non-essential'):
#         return 'Non-essential'
#     return None


# def essential_ratio(counts):
#     """Essential / labelled. Returns 0.5 if no labelled proteins."""
#     labelled = counts.get('Essential', 0) + counts.get('Non-essential', 0)
#     return counts.get('Essential', 0) / labelled if labelled > 0 else 0.5


# # ============================================================================
# # LARGE-GROUP PRE-SCHEDULER
# # ============================================================================

# def schedule_large_groups(large_groups, n_splits, draw_per_split, rng):
#     """
#     Pre-assign which large groups go to TEST in each split so that:
#       - every large group appears in test at least once, and
#       - no large group appears more than ceil(total_slots / n_groups) times,
#       - each split receives exactly draw_per_split distinct large groups.

#     Algorithm: build a deck where each group appears base or base+1 times
#     (total_slots = n_splits * draw_per_split distributed evenly), shuffle,
#     then greedily fill splits avoiding duplicates within the same split.

#     Returns
#     -------
#     list of length n_splits, each element a list of draw_per_split group IDs.
#     """
#     groups    = sorted(large_groups)
#     n_groups  = len(groups)
#     total_slots = n_splits * draw_per_split

#     base      = total_slots // n_groups
#     remainder = total_slots  % n_groups

#     # Shuffle before assigning extras to avoid systematic bias.
#     shuffled = list(groups)
#     rng.shuffle(shuffled)
#     counts = {grp: (base + 1 if idx < remainder else base)
#               for idx, grp in enumerate(shuffled)}

#     deck = []
#     for grp, cnt in counts.items():
#         deck.extend([grp] * cnt)
#     rng.shuffle(deck)

#     # Greedy assignment.
#     assignment = [[] for _ in range(n_splits)]
#     unplaced   = []
#     for grp in deck:
#         placed = False
#         for s in range(n_splits):
#             if len(assignment[s]) < draw_per_split and grp not in assignment[s]:
#                 assignment[s].append(grp)
#                 placed = True
#                 break
#         if not placed:
#             unplaced.append(grp)

#     for grp in unplaced:
#         for s in range(n_splits):
#             if len(assignment[s]) < draw_per_split and grp not in assignment[s]:
#                 assignment[s].append(grp)
#                 break

#     for s, grps in enumerate(assignment):
#         assert len(grps) == draw_per_split, (
#             f"Split {s+1}: expected {draw_per_split} large groups, got {len(grps)}")
#         assert len(set(grps)) == len(grps), (
#             f"Split {s+1}: duplicate large group in test assignment")

#     return assignment


# def size_bucket(n_proteins):
#     """Map group size to bucket name (4 buckets: large/medium/small/singleton).
#     large covers all groups >20 proteins (formerly separate large and very_large)."""
#     if n_proteins == 1:    return 'singleton'
#     elif n_proteins <= 5:  return 'small'
#     elif n_proteins <= 20: return 'medium'
#     else:                  return 'large'


# # ============================================================================
# # CORE SPLIT LOGIC — one attempt
# # ============================================================================

# def attempt_split(buckets, group_sizes, group_protein_map,
#                   no_group_proteins, protein_to_label, config, rng,
#                   forced_large_test=None):
#     """
#     Make one attempt at a train/test split.

#     For the large bucket, if `forced_large_test` is provided (a list of group
#     IDs pre-assigned to test by the scheduler), those groups go directly to test
#     and all remaining large groups go to train — no random draw for this bucket.

#     For medium/small/singleton buckets, exactly round(n * 0.20) groups are
#     drawn at random for test.  No-group proteins are split by stratified random
#     sampling.

#     Returns
#     -------
#     protein_to_split : dict[protein_id -> 'train'|'test']
#     group_to_split  : dict[group_id  -> 'train'|'test']
#     test_ess_rate    : float
#     test_size_frac   : float
#     """
#     test_ratio = config['test_ratio']
#     group_to_split = {}

#     for bucket_name, bucket_grps in buckets.items():
#         if not bucket_grps:
#             continue

#         if bucket_name == 'large' and forced_large_test is not None:
#             # Use pre-scheduled assignment — no random draw.
#             forced_set = set(forced_large_test)
#             for grp in bucket_grps:
#                 group_to_split[grp] = 'test' if grp in forced_set else 'train'
#         else:
#             n_grps    = len(bucket_grps)
#             n_test    = round(n_grps * test_ratio)
#             n_test    = max(n_test, 0)
#             grps_list = list(bucket_grps)
#             rng.shuffle(grps_list)
#             for i, grp in enumerate(grps_list):
#                 group_to_split[grp] = 'test' if i < n_test else 'train'

#     # --- Assign proteins from their group's split ---
#     protein_to_split = {}
#     for grp, split in group_to_split.items():
#         for pid in group_protein_map.get(grp, set()):
#             protein_to_split[pid] = split

#     # --- No-group proteins: stratified random sampling ---
#     for group_label in ('Essential', 'Non-essential', 'Unknown'):
#         group = sorted(
#             p for p in no_group_proteins
#             if protein_to_label.get(p, 'Unknown') == group_label
#         )
#         rng.shuffle(group)
#         n_test_grp = round(len(group) * test_ratio)
#         for i, pid in enumerate(group):
#             protein_to_split[pid] = 'test' if i < n_test_grp else 'train'

#     # --- Compute test label balance ---
#     test_prots  = [p for p, s in protein_to_split.items() if s == 'test']
#     test_counts = Counter(protein_to_label.get(p, 'Unknown') for p in test_prots)
#     test_ess    = test_counts['Essential']
#     test_lab    = test_ess + test_counts['Non-essential']
#     test_ess_rate = test_ess / test_lab if test_lab > 0 else 0.5

#     # --- Compute test set size fraction (vs total proteins) ---
#     total_prots   = len(protein_to_split)
#     test_size_frac = len(test_prots) / total_prots if total_prots > 0 else 0.0

#     return protein_to_split, group_to_split, test_ess_rate, test_size_frac


# # ============================================================================
# # VALIDATION
# # ============================================================================


# def validate_split(protein_to_split, protein_to_group, group_sizes, config):
#     """
#     Validate a proposed split.

#     Args:
#         protein_to_split: dict mapping protein_id -> 'train'/'test'
#         protein_to_group: dict mapping protein_id -> group_id (or None)
#         group_sizes: dict mapping group_id -> size (int)
#         config: dict with at least 'test_fraction' (float, e.g. 0.2)

#     Returns:
#         offending_groups: set of group_ids that were split across train/test (leakage)
#         bucket_stats: dict {bucket_name: {'train': int, 'test': int, 'total': int, 'test_frac': float}}
#         problems: list of string messages describing issues found (empty if ok)
#     """
#     # Existing leakage check: ensure no group has proteins in both splits
#     grp_splits = defaultdict(set)   # grp_id -> set of splits seen {'train','test'}
#     protein_to_group_local = protein_to_group  # name used in calling code; keep consistent

#     for prot, split in protein_to_split.items():
#         grp = protein_to_group_local.get(prot)
#         if grp is None:
#             continue
#         grp_splits[grp].add(split)

#     offending_groups = {grp for grp, splits in grp_splits.items() if len(splits) > 1}

#     # Now compute bucket-level stats
#     # bucket_name -> {'train':count, 'test':count, 'total':count}
#     bucket_counts = defaultdict(lambda: {'train': 0, 'test': 0, 'total': 0})

#     # Build group -> representative split (should be consistent because validate_split is called
#     # after the "no group split" constraint is enforced; but if leakage exists we still assign
#     # the group to the majority split for reporting)
#     group_to_proteins = defaultdict(list)
#     for prot, grp in protein_to_group_local.items():
#         if grp is None:
#             continue
#         group_to_proteins[grp].append(prot)

#     for grp, proteins in group_to_proteins.items():
#         # determine the split for the group. If group is leaked, pick majority split for reporting.
#         splits = [protein_to_split.get(p) for p in proteins if p in protein_to_split]
#         splits = [s for s in splits if s is not None]
#         if not splits:
#             # group has no proteins in mapping (weird) — skip
#             continue
#         # choose the majority split if needed
#         train_count = splits.count('train')
#         test_count = splits.count('test')
#         if train_count >= test_count:
#             grp_split = 'train'
#         else:
#             grp_split = 'test'

#         size = group_sizes.get(grp, len(proteins))
#         bucket = size_bucket(size)

#         bucket_counts[bucket][grp_split] += 1
#         bucket_counts[bucket]['total'] += 1

#     # Summarize and flag problems
#     bucket_stats = {}
#     problems = []
#     expected_test_frac = float(config.get('test_fraction', 0.2))
#     # absolute tolerance (how far test fraction may deviate); and minimum groups to consider
#     abs_tol = float(config.get('validate_abs_tol', 0.15))   # e.g., 0.15 means +/-15% absolute
#     min_groups_to_warn = int(config.get('validate_min_groups', 3))  # small buckets may be noisy

#     # Print header for quick human inspection
#     print("\ngroup-size bucket distribution after split (groups counted):")
#     print("{:15s} {:>6s} {:>6s} {:>6s} {:>9s}".format("BUCKET", "TRAIN", "TEST", "TOTAL", "TEST_FRAC"))
#     for bucket, counts in sorted(bucket_counts.items(), key=lambda x: x[0]):
#         train_c = counts['train']
#         test_c = counts['test']
#         total_c = counts['total']
#         test_frac = (test_c / total_c) if total_c > 0 else 0.0
#         bucket_stats[bucket] = {
#             'train': train_c,
#             'test': test_c,
#             'total': total_c,
#             'test_frac': test_frac
#         }
#         print("{:15s} {:6d} {:6d} {:6d} {:9.3f}".format(bucket, train_c, test_c, total_c, test_frac))

#         # If bucket has groups but none ended up in test (or train), flag it
#         if total_c > 0 and test_c == 0:
#             problems.append(f"Bucket '{bucket}' has {total_c} groups but 0 in TEST.")
#         if total_c > 0 and train_c == 0:
#             problems.append(f"Bucket '{bucket}' has {total_c} groups but 0 in TRAIN.")

#         # If bucket has enough groups, check expected fraction within tolerance
#         if total_c >= min_groups_to_warn:
#             if abs(test_frac - expected_test_frac) > abs_tol:
#                 problems.append(
#                     f"Bucket '{bucket}' test fraction {test_frac:.3f} "
#                     f"differs from expected {expected_test_frac:.3f} by > {abs_tol:.3f}."
#                 )

#     # If there were offending groups (leakage), include a problem message
#     if offending_groups:
#         problems.append(f"{len(offending_groups)} group(ies) are split across train/test (leakage).")

#     lines = []
#     lines.append("group-size bucket distribution after split (groups counted):")
#     lines.append("{:15s} {:>6s} {:>6s} {:>6s} {:>9s}".format(
#         "BUCKET", "TRAIN", "TEST", "TOTAL", "TEST_FRAC"
#     ))

#     for bucket, counts in sorted(bucket_counts.items()):
#         train_c = counts['train']
#         test_c = counts['test']
#         total_c = counts['total']
#         test_frac = (test_c / total_c) if total_c > 0 else 0.0

#         lines.append("{:15s} {:6d} {:6d} {:6d} {:9.3f}".format(
#             bucket, train_c, test_c, total_c, test_frac
#         ))

#     report_text = "\n".join(lines)

#     return offending_groups, bucket_stats, problems, report_text


# # ============================================================================
# # BALANCE STATS
# # ============================================================================

# def balance_stats(protein_to_split, protein_to_label, global_ess_ratio, config):
#     stats = {}
#     total_n = len(protein_to_split)
#     for sp in ('train', 'test'):
#         prots    = [p for p, s in protein_to_split.items() if s == sp]
#         counts   = Counter(protein_to_label.get(p, 'Unknown') for p in prots)
#         ess      = counts['Essential']
#         noness   = counts['Non-essential']
#         unk      = counts['Unknown']
#         labelled = ess + noness
#         rate_pct = 100 * ess / labelled if labelled > 0 else float('nan')
#         size_pct = 100 * len(prots) / total_n if total_n > 0 else float('nan')
#         stats[sp] = {
#             'n_proteins':   len(prots),
#             'size_pct':     round(size_pct, 2),
#             'n_essential':  ess,
#             'n_noness':     noness,
#             'n_unknown':    unk,
#             'ess_rate_pct': round(rate_pct, 2),
#         }
#     global_rate_pct = 100 * global_ess_ratio
#     dev = abs(stats['test']['ess_rate_pct'] - global_rate_pct)
#     stats['test_deviation_pp'] = round(dev, 2)
#     stats['warning'] = dev > config['max_ess_rate_deviation']

#     # Size deviation: how far test % is from the target 20%
#     target_test_pct = 100 * config['test_ratio']
#     size_dev = abs(stats['test']['size_pct'] - target_test_pct)
#     stats['test_size_deviation_pp'] = round(size_dev, 2)
#     stats['size_warning'] = size_dev > config['max_size_deviation']
#     return stats

# def report_group_size_distribution(protein_to_split, group_sizes, protein_to_group):
#     from collections import defaultdict

#     split_bucket_counts = {
#         "train": defaultdict(int),
#         "test": defaultdict(int)
#     }

#     for grp, size in group_sizes.items():
#         bucket = size_bucket(size)

#         # get one protein from this group to determine split
#         example_protein = next(iter(
#             [p for p, f in protein_to_group.items() if f == grp]
#         ))

#         split = protein_to_split[example_protein]
#         split_bucket_counts[split][bucket] += 1

#     print("\nGroup Size Distribution:")
#     for split in ["train", "test"]:
#         print(f"\n{split.upper()}:")
#         for bucket, count in split_bucket_counts[split].items():
#             print(f"  {bucket}: {count}")

# # ============================================================================
# # MAIN
# # ============================================================================

# def main():
#     print("=" * 70)
#     print("Repeated Random group-Level Splitting — Size-Stratified, N=15")
#     print("=" * 70)

#     # -------------------------------------------------------------------------
#     # Load data
#     # -------------------------------------------------------------------------
#     print("\n--- Loading data ---")
#     groups  = pd.read_csv(CONFIG['group_mapping'])
#     complexes = pd.read_csv(CONFIG['complex_membership'])
#     targets   = pd.read_csv(CONFIG['essentiality'])

#     print(f"  Groups:  {len(groups)} rows, "
#           f"{groups['group_id'].nunique()} unique groups")
#     print(f"  Complexes: {complexes['ComplexId'].nunique()} complexes, "
#           f"{complexes['ProteinId'].nunique()} unique proteins")
#     print(f"  Targets:   {len(targets)} labelled proteins")

#     # -------------------------------------------------------------------------
#     # Build lookups
#     # -------------------------------------------------------------------------
#     complex_proteins  = set(complexes['ProteinId'].unique())

#     protein_to_group = {
#         row['uniprot_id']: row['group_id']
#         for _, row in groups.iterrows()
#         if row['uniprot_id'] in complex_proteins
#     }

#     group_sizes = (
#         groups[groups['uniprot_id'].isin(complex_proteins)]
#         .groupby('group_id')['uniprot_id'].nunique().to_dict()
#     )

#     protein_to_label = {}
#     for _, row in targets.iterrows():
#         bl = binarise_label(row['essential_category'])
#         if bl is not None:
#             protein_to_label[row['Protein']] = bl

#     all_proteins  = sorted(complex_proteins)
#     global_counts = Counter(protein_to_label.get(p, 'Unknown') for p in all_proteins)
#     global_ess_ratio = essential_ratio(global_counts)
#     global_rate_pct  = 100 * global_ess_ratio

#     print(f"\n  Global essential rate (labelled): {global_rate_pct:.1f}%")
#     print(f"  Label distribution ({len(all_proteins)} proteins):")
#     for label in ('Essential', 'Non-essential', 'Unknown'):
#         n = global_counts[label]
#         print(f"    {label:15s}: {n:4d} ({100*n/len(all_proteins):.1f}%)")

#     # -------------------------------------------------------------------------
#     # Build group -> protein map and size buckets
#     # -------------------------------------------------------------------------
#     group_protein_map = defaultdict(set)
#     for pid, grp in protein_to_group.items():
#         group_protein_map[grp].add(pid)

#     no_group_proteins = complex_proteins - set(protein_to_group.keys())

#     # Buckets ordered largest-first (4 buckets; large covers >20)
#     buckets = {'large': [], 'medium': [], 'small': [], 'singleton': []}
#     for grp, n in group_sizes.items():
#         buckets[size_bucket(n)].append(grp)

#     print(f"\n--- Group size buckets (largest first) ---")
#     bucket_label = {
#         'large':      '>20 proteins (merged large + very large)',
#         'medium':     '6–20',
#         'small':      '2–5',
#         'singleton':  '1 protein',
#     }
#     for bname, bgrps in buckets.items():
#         n_grp   = len(bgrps)
#         n_prot  = sum(group_sizes[f] for f in bgrps)
#         n_test  = round(n_grp * CONFIG['test_ratio'])
#         print(f"  {bname:12s} ({bucket_label[bname]:13s}): "
#               f"{n_grp:4d} groups, {n_prot:5d} proteins — "
#               f"{n_test} grp{'' if n_test==1 else 's'} drawn to test each split")
#     print(f"  No-group proteins: {len(no_group_proteins)}")

#     # -------------------------------------------------------------------------
#     # Pre-schedule large group test assignments across all splits
#     # -------------------------------------------------------------------------
#     # Use a separate RNG seeded from base_seed so the schedule is reproducible
#     # and independent of per-split attempt seeds.
#     schedule_rng = np.random.default_rng(CONFIG['base_seed'])
#     large_draw   = round(len(buckets['large']) * CONFIG['test_ratio'])
#     large_schedule = schedule_large_groups(
#         large_groups   = buckets['large'],
#         n_splits       = CONFIG['n_splits'],
#         draw_per_split = large_draw,
#         rng            = schedule_rng,
#     )

#     # Report the schedule
#     test_counts_by_group = Counter(
#         grp for split_grps in large_schedule for grp in split_grps
#     )
#     print(f"\n--- Large-group pre-schedule ({len(buckets['large'])} groups, "
#           f"{large_draw} drawn/split, {CONFIG['n_splits']} splits) ---")
#     print(f"  Appearance counts: min={min(test_counts_by_group.values())}  "
#           f"max={max(test_counts_by_group.values())}  "
#           f"mean={sum(test_counts_by_group.values())/len(test_counts_by_group):.2f}")
#     print(f"  (every large group will appear in test exactly "
#           f"{min(test_counts_by_group.values())}–{max(test_counts_by_group.values())} times)")

#     # -------------------------------------------------------------------------
#     # Generate N splits with reject-and-redraw for label balance
#     # -------------------------------------------------------------------------
#     print(f"\n--- Generating {CONFIG['n_splits']} splits "
#           f"(tolerance ±{CONFIG['max_ess_rate_deviation']}pp ess, "
#           f"±{CONFIG['max_size_deviation']}pp size, "
#           f"max {CONFIG['max_attempts']} attempts each) ---")

#     all_rows        = []
#     balance_records = []

#     for i in range(CONFIG['n_splits']):
#         best_split      = None
#         best_grp_split  = None
#         best_deviation  = float('inf')
#         best_size_deviation = float('inf')
#         attempts        = 0
#         accepted        = False

#         while attempts < CONFIG['max_attempts']:
#             seed = CONFIG['base_seed'] + i + attempts * 1000
#             rng  = np.random.default_rng(seed)

#             protein_to_split, group_to_split, test_ess_rate, test_size_frac = attempt_split(
#                 buckets            = buckets,
#                 group_sizes       = group_sizes,
#                 group_protein_map = group_protein_map,
#                 no_group_proteins = sorted(no_group_proteins),
#                 protein_to_label   = protein_to_label,
#                 config             = CONFIG,
#                 rng                = rng,
#                 forced_large_test  = large_schedule[i],
#             )
#             attempts += 1

#             # --- Check 1: test set size (must be within 15–25% of total proteins) ---
#             size_deviation = abs(100 * test_size_frac - 100 * CONFIG['test_ratio'])

#             # --- Check 2: essentiality balance ---
#             deviation = abs(100 * test_ess_rate - global_rate_pct)

#             # Keep track of best attempt in case we exhaust the budget
#             # Primary sort: size deviation; secondary: essentiality deviation
#             combined = (size_deviation, deviation)
#             if combined < (best_size_deviation, best_deviation):
#                 best_size_deviation = size_deviation
#                 best_deviation      = deviation
#                 best_split          = protein_to_split
#                 best_grp_split      = group_to_split
#                 best_seed           = seed

#             size_ok = size_deviation <= CONFIG['max_size_deviation']
#             ess_ok  = deviation       <= CONFIG['max_ess_rate_deviation']
#             if size_ok and ess_ok:
#                 accepted = True
#                 break

#         # Use best available if budget exhausted
#         protein_to_split = best_split
#         group_to_split  = best_grp_split

#         # # Validate leakage
#         # violations = validate_split(protein_to_split, protein_to_group)
#         violations, bucket_stats, problems, report_text = validate_split(protein_to_split, protein_to_group, group_sizes, CONFIG)
#         if violations:
#             # existing behavior: reject the split attempt
#             continue
#         if problems:
#             # option A: reject split attempts that have any problem
#             print("Rejecting split due to validation problems.")
#             continue
#         with open("split_summary.txt", "a") as f:
#             f.write("\n")
#             f.write(report_text)
#             f.write("\n")

#         # otherwise accept split

#         # Final balance stats
#         bstats = balance_stats(
#             protein_to_split, protein_to_label, global_ess_ratio, CONFIG
#         )

#         # Per-bucket protein breakdown for the summary table
#         bucket_breakdown = {}
#         for bname in ('large', 'medium', 'small', 'singleton'):
#             bgrps = buckets.get(bname, [])
#             train_n = sum(
#                 len(group_protein_map.get(g, set()))
#                 for g in bgrps if group_to_split.get(g) == 'train'
#             )
#             test_n = sum(
#                 len(group_protein_map.get(g, set()))
#                 for g in bgrps if group_to_split.get(g) == 'test'
#             )
#             total_n = train_n + test_n
#             # Essential rate in test for this bucket
#             test_prots_bkt = [
#                 p for g in bgrps if group_to_split.get(g) == 'test'
#                 for p in group_protein_map.get(g, set())
#             ]
#             bc = Counter(protein_to_label.get(p, 'Unknown') for p in test_prots_bkt)
#             lab = bc['Essential'] + bc['Non-essential']
#             ess_rate = 100 * bc['Essential'] / lab if lab > 0 else float('nan')
#             bucket_breakdown[bname] = {
#                 'train_n':   train_n,
#                 'test_n':    test_n,
#                 'total_n':   total_n,
#                 'test_pct':  100 * test_n / total_n if total_n > 0 else 0.0,
#                 'test_ess_pct': round(ess_rate, 1),
#             }

#         bal_row = {
#             'split_index':           i + 1,
#             'seed_accepted':         best_seed,
#             'attempts':              attempts,
#             'accepted':              accepted,
#             'train_n':               bstats['train']['n_proteins'],
#             'train_pct':             bstats['train']['size_pct'],
#             'test_n':                bstats['test']['n_proteins'],
#             'test_pct':              bstats['test']['size_pct'],
#             'test_size_deviation_pp': bstats['test_size_deviation_pp'],
#             'size_warning':          bstats['size_warning'],
#             'train_essential':       bstats['train']['n_essential'],
#             'test_essential':        bstats['test']['n_essential'],
#             'train_noness':          bstats['train']['n_noness'],
#             'test_noness':           bstats['test']['n_noness'],
#             'train_unknown':         bstats['train']['n_unknown'],
#             'test_unknown':          bstats['test']['n_unknown'],
#             'train_ess_rate_pct':    bstats['train']['ess_rate_pct'],
#             'test_ess_rate_pct':     bstats['test']['ess_rate_pct'],
#             'test_deviation_pp':     bstats['test_deviation_pp'],
#             'warning':               bstats['warning'],
#             'leakage_violations':    len(violations),
#             # Per-bucket test counts
#             'large_test_n':      bucket_breakdown['large']['test_n'],
#             'large_test_pct':    round(bucket_breakdown['large']['test_pct'], 1),
#             'large_test_ess_pct': bucket_breakdown['large']['test_ess_pct'],
#             'medium_test_n':     bucket_breakdown['medium']['test_n'],
#             'medium_test_pct':   round(bucket_breakdown['medium']['test_pct'], 1),
#             'medium_test_ess_pct': bucket_breakdown['medium']['test_ess_pct'],
#             'small_test_n':      bucket_breakdown['small']['test_n'],
#             'small_test_pct':    round(bucket_breakdown['small']['test_pct'], 1),
#             'small_test_ess_pct': bucket_breakdown['small']['test_ess_pct'],
#             'singleton_test_n':  bucket_breakdown['singleton']['test_n'],
#             'singleton_test_pct': round(bucket_breakdown['singleton']['test_pct'], 1),
#             'singleton_test_ess_pct': bucket_breakdown['singleton']['test_ess_pct'],
#         }
#         balance_records.append(bal_row)

#         flag    = ' ⚠ balance warning' if bstats['warning'] else ''
#         szflag  = ' ⚠ size warning'   if bstats['size_warning'] else ''
#         lk      = f' ⚠ {len(violations)} leakage' if violations else ''
#         retry   = f' (accepted on attempt {attempts})' if attempts > 1 else ''
#         print(f"  Split {i+1:2d} | "
#               f"train={bstats['train']['n_proteins']} ({bstats['train']['size_pct']:.1f}%) "
#               f"ess={bstats['train']['ess_rate_pct']:.1f}% | "
#               f"test={bstats['test']['n_proteins']} ({bstats['test']['size_pct']:.1f}%) "
#               f"ess={bstats['test']['ess_rate_pct']:.1f}% | "
#               f"size_dev={bstats['test_size_deviation_pp']:.1f}pp "
#               f"ess_dev={bstats['test_deviation_pp']:.1f}pp"
#               f"{retry}{szflag}{flag}{lk}")

#         # Collect rows for output CSV
#         for pid in all_proteins:
#             grp    = protein_to_group.get(pid)
#             grp_sz = group_sizes.get(grp) if grp else None
#             all_rows.append({
#                 'split_index':   i + 1,
#                 'seed_accepted': best_seed,
#                 'UniProt_AC':    pid,
#                 'group_id':     grp,
#                 'group_size':   grp_sz,
#                 'group_bucket': size_bucket(grp_sz) if grp_sz else None,
#                 'group_status': 'constrained' if grp else 'no_group',
#                 'split':         protein_to_split[pid],
#                 'protein_label': protein_to_label.get(pid, 'Unknown'),
#                 'label_mask':    protein_to_label.get(pid, 'Unknown') != 'Unknown',
#             })

#     # -------------------------------------------------------------------------
#     # Save outputs
#     # -------------------------------------------------------------------------
#     print("\n--- Saving outputs ---")

#     pd.DataFrame(all_rows).to_csv(CONFIG['output_all_splits'], index=False)
#     print(f"  Saved: {CONFIG['output_all_splits']}  ({len(all_rows):,} rows)")

#     balance_df = pd.DataFrame(balance_records)
#     balance_df.to_csv(CONFIG['output_balance'], index=False)
#     print(f"  Saved: {CONFIG['output_balance']}")

#     # -------------------------------------------------------------------------
#     # Summary report
#     # -------------------------------------------------------------------------
#     n_warnings  = sum(1 for r in balance_records if r['warning'])
#     n_sz_warn   = sum(1 for r in balance_records if r['size_warning'])
#     n_leakage   = sum(1 for r in balance_records if r['leakage_violations'] > 0)
#     n_retried   = sum(1 for r in balance_records if r['attempts'] > 1)
#     test_rates  = [r['test_ess_rate_pct']      for r in balance_records]
#     train_rates = [r['train_ess_rate_pct']     for r in balance_records]
#     test_sz     = [r['test_pct']               for r in balance_records]
#     train_sz    = [r['train_pct']              for r in balance_records]
#     sz_devs     = [r['test_size_deviation_pp'] for r in balance_records]

#     lines = []
#     lines.append("=" * 90)
#     lines.append("SPLIT SUMMARY — Size-stratified repeated random group-Level splits")
#     lines.append("=" * 90)
#     lines.append(f"Total unique proteins:            {len(all_proteins)}")
#     lines.append(f"Total complexes:                  {complexes['ComplexId'].nunique()}")
#     lines.append(f"Total groups (all atomic):      {len(group_sizes)}")
#     lines.append(f"  Large      (>20, merged):         {len(buckets['large'])}")
#     lines.append(f"  Medium     (6–20):              {len(buckets['medium'])}")
#     lines.append(f"  Small      (2–5):               {len(buckets['small'])}")
#     lines.append(f"  Singleton  (1 protein):         {len(buckets['singleton'])}")
#     lines.append(f"Global essential rate (labelled): {global_rate_pct:.1f}%")
#     lines.append(f"Number of splits:                 {CONFIG['n_splits']}")
#     lines.append(f"Train/test ratio:                 "
#                  f"{CONFIG['train_ratio']:.0%} / {CONFIG['test_ratio']:.0%}")
#     lines.append(f"Size tolerance:                   "
#                  f"±{CONFIG['max_size_deviation']}pp from 20% "
#                  f"(acceptable test range: "
#                  f"{100*CONFIG['test_ratio'] - CONFIG['max_size_deviation']:.0f}%–"
#                  f"{100*CONFIG['test_ratio'] + CONFIG['max_size_deviation']:.0f}%)")
#     lines.append(f"Balance tolerance:                "
#                  f"±{CONFIG['max_ess_rate_deviation']}pp from global rate")
#     lines.append(f"Max redraw attempts per split:    {CONFIG['max_attempts']}")
#     lines.append(f"Splits requiring redraw:          {n_retried} / {CONFIG['n_splits']}")
#     lines.append(f"Splits with size warnings:        {n_sz_warn} / {CONFIG['n_splits']}")
#     lines.append(f"Splits with balance warnings:     {n_warnings} / {CONFIG['n_splits']}")
#     lines.append(f"Splits with leakage violations:   {n_leakage} / {CONFIG['n_splits']}")
#     lines.append("")

#     # Table header — two extra columns: train%, test%, size_dev
#     lines.append(
#         f"{'Split':>6} {'Att':>4}  "
#         f"{'Train N':>8} {'Train%':>7} {'Train ess%':>11}  "
#         f"{'Test N':>7} {'Test%':>6} {'Test ess%':>10}  "
#         f"{'SzDev':>6} {'EssDev':>7}  {'Flag':>4}"
#     )
#     lines.append("-" * 90)
#     for r in balance_records:
#         flag = ''
#         if r['size_warning']:
#             flag += 'Sz'
#         if r['warning']:
#             flag += 'Es'
#         if r['leakage_violations'] > 0:
#             flag += 'Lk'
#         flag = ('⚠ ' + flag) if flag else ''
#         lines.append(
#             f"{r['split_index']:>6} {r['attempts']:>4}  "
#             f"{r['train_n']:>8} {r['train_pct']:>6.1f}% {r['train_ess_rate_pct']:>10.1f}%  "
#             f"{r['test_n']:>7} {r['test_pct']:>5.1f}% {r['test_ess_rate_pct']:>9.1f}%  "
#             f"{r['test_size_deviation_pp']:>5.1f}pp {r['test_deviation_pp']:>6.1f}pp  {flag:>6}"
#         )
#     lines.append("-" * 90)
#     lines.append(
#         f"{'Mean':>6} {'':>4}  "
#         f"{'':>8} {sum(train_sz)/len(train_sz):>6.1f}% {sum(train_rates)/len(train_rates):>10.1f}%  "
#         f"{'':>7} {sum(test_sz)/len(test_sz):>5.1f}% {sum(test_rates)/len(test_rates):>9.1f}%  "
#         f"{sum(sz_devs)/len(sz_devs):>5.1f}pp"
#     )
#     lines.append(
#         f"{'Std':>6} {'':>4}  "
#         f"{'':>8} {pd.Series(train_sz).std():>6.2f}% {pd.Series(train_rates).std():>10.2f}%  "
#         f"{'':>7} {pd.Series(test_sz).std():>5.2f}% {pd.Series(test_rates).std():>9.2f}%  "
#         f"{pd.Series(sz_devs).std():>5.2f}pp"
#     )
#     lines.append(
#         f"{'Min':>6} {'':>4}  "
#         f"{'':>8} {min(train_sz):>6.1f}% {min(train_rates):>10.1f}%  "
#         f"{'':>7} {min(test_sz):>5.1f}% {min(test_rates):>9.1f}%  "
#         f"{min(sz_devs):>5.1f}pp"
#     )
#     lines.append(
#         f"{'Max':>6} {'':>4}  "
#         f"{'':>8} {max(train_sz):>6.1f}% {max(train_rates):>10.1f}%  "
#         f"{'':>7} {max(test_sz):>5.1f}% {max(test_rates):>9.1f}%  "
#         f"{max(sz_devs):>5.1f}pp"
#     )
#     lines.append("")
#     lines.append("Leakage guarantee: all groups are atomic — no group spans")
#     lines.append("train and test within any single split.")
#     lines.append("")
#     lines.append("Flag key: Sz = size out of range, Es = essentiality imbalance, Lk = leakage")

#     # -------------------------------------------------------------------------
#     # Table 2: Per-split breakdown by structural group size bucket (proteins)
#     # -------------------------------------------------------------------------
#     lines.append("")
#     lines.append("=" * 90)
#     lines.append("TEST SET BREAKDOWN BY STRUCTURAL GROUP SIZE BUCKET (proteins in test)")
#     lines.append("=" * 90)
#     lines.append(
#         f"{'Split':>6}  "
#         f"{'Large N':>8} {'L%':>5} {'Ess%':>6}  "
#         f"{'Medium N':>9} {'M%':>5} {'Ess%':>6}  "
#         f"{'Small N':>8} {'S%':>5} {'Ess%':>6}  "
#         f"{'Singleton N':>12} {'Sg%':>5} {'Ess%':>6}"
#     )
#     lines.append("-" * 90)
#     for r in balance_records:
#         def _fmt(key_n, key_pct, key_ess, width_n, width_pct):
#             ess = r[key_ess]
#             ess_str = f"{ess:5.1f}%" if ess == ess else "   N/A"  # nan check
#             return (f"{r[key_n]:>{width_n}} {r[key_pct]:>{width_pct}.1f}% {ess_str}")

#         lines.append(
#             f"{r['split_index']:>6}  "
#             f"{r['large_test_n']:>8} {r['large_test_pct']:>4.1f}% "
#             f"{r['large_test_ess_pct']:>5.1f}%  "
#             f"{r['medium_test_n']:>9} {r['medium_test_pct']:>4.1f}% "
#             f"{r['medium_test_ess_pct']:>5.1f}%  "
#             f"{r['small_test_n']:>8} {r['small_test_pct']:>4.1f}% "
#             f"{r['small_test_ess_pct']:>5.1f}%  "
#             f"{r['singleton_test_n']:>12} {r['singleton_test_pct']:>4.1f}% "
#             f"{r['singleton_test_ess_pct']:>5.1f}%"
#         )
#     lines.append("-" * 90)
#     # Column means
#     def _col_mean(key):
#         vals = [r[key] for r in balance_records if r[key] == r[key]]  # exclude nan
#         return sum(vals) / len(vals) if vals else float('nan')
#     lines.append(
#         f"{'Mean':>6}  "
#         f"{_col_mean('large_test_n'):>8.1f} {_col_mean('large_test_pct'):>4.1f}% "
#         f"{_col_mean('large_test_ess_pct'):>5.1f}%  "
#         f"{_col_mean('medium_test_n'):>9.1f} {_col_mean('medium_test_pct'):>4.1f}% "
#         f"{_col_mean('medium_test_ess_pct'):>5.1f}%  "
#         f"{_col_mean('small_test_n'):>8.1f} {_col_mean('small_test_pct'):>4.1f}% "
#         f"{_col_mean('small_test_ess_pct'):>5.1f}%  "
#         f"{_col_mean('singleton_test_n'):>12.1f} {_col_mean('singleton_test_pct'):>4.1f}% "
#         f"{_col_mean('singleton_test_ess_pct'):>5.1f}%"
#     )
#     lines.append(
#         f"{'Std':>6}  "
#         f"{pd.Series([r['large_test_n'] for r in balance_records]).std():>8.1f} "
#         f"{pd.Series([r['large_test_pct'] for r in balance_records]).std():>4.2f}% "
#         f"{pd.Series([r['large_test_ess_pct'] for r in balance_records]).std():>5.2f}%  "
#         f"{pd.Series([r['medium_test_n'] for r in balance_records]).std():>9.1f} "
#         f"{pd.Series([r['medium_test_pct'] for r in balance_records]).std():>4.2f}% "
#         f"{pd.Series([r['medium_test_ess_pct'] for r in balance_records]).std():>5.2f}%  "
#         f"{pd.Series([r['small_test_n'] for r in balance_records]).std():>8.1f} "
#         f"{pd.Series([r['small_test_pct'] for r in balance_records]).std():>4.2f}% "
#         f"{pd.Series([r['small_test_ess_pct'] for r in balance_records]).std():>5.2f}%  "
#         f"{pd.Series([r['singleton_test_n'] for r in balance_records]).std():>12.1f} "
#         f"{pd.Series([r['singleton_test_pct'] for r in balance_records]).std():>4.2f}% "
#         f"{pd.Series([r['singleton_test_ess_pct'] for r in balance_records]).std():>5.2f}%"
#     )
#     lines.append("")
#     lines.append("  N% = proteins from that bucket that went to test (of all proteins in that bucket).")
#     lines.append("  Ess% = essential rate among labelled test proteins from that bucket.")

#     # -------------------------------------------------------------------------
#     # Table 3: Large-group test frequency across all splits
#     # -------------------------------------------------------------------------
#     all_large_groups = sorted(buckets['large'])
#     # Build per-group test-count and per-group protein count
#     large_freq = Counter(
#         grp for split_grps in large_schedule for grp in split_grps
#     )
#     lines.append("")
#     lines.append("=" * 70)
#     lines.append("LARGE-GROUP TEST-FREQUENCY TABLE (pre-scheduled, guaranteed coverage)")
#     lines.append("=" * 70)
#     lines.append(f"  {'Group':<12} {'Size':>6}  {'Test count':>10}  {'Splits in test'}")
#     lines.append("  " + "-" * 65)
#     for grp in all_large_groups:
#         sz   = group_sizes[grp]
#         cnt  = large_freq.get(grp, 0)
#         split_list = [str(s + 1) for s, sg in enumerate(large_schedule) if grp in sg]
#         lines.append(f"  {grp:<12} {sz:>6}  {cnt:>10}  [{', '.join(split_list)}]")
#     lines.append("  " + "-" * 65)
#     freq_vals = [large_freq.get(g, 0) for g in all_large_groups]
#     lines.append(f"  {'Min':>18}: {min(freq_vals)}")
#     lines.append(f"  {'Max':>18}: {max(freq_vals)}")
#     lines.append(f"  {'Mean':>18}: {sum(freq_vals)/len(freq_vals):.2f}")

#     summary = '\n'.join(lines)
#     print(f"\n{summary}")
#     with open(CONFIG['output_summary'], 'w') as f:
#         f.write(summary + '\n')
#     print(f"\n  Saved: {CONFIG['output_summary']}")
#     print("\nDone!")


# if __name__ == '__main__':
#     main()

"""
Repeated Random group-Level Splitting — Size-Stratified, N=15
==============================================================
Generates N=15 independent train/test splits where each Foldseek structural
group is allocated atomically (never split across train and test).

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
    'group_mapping':     '/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/corum_mapping_struct.csv',
    'complex_membership': '/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/corum_stoich_protein.csv',
    'essentiality':       '/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/lu_essentiality_protein.csv',


    # Output files
    'output_all_splits':  '/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/corum_ess_merged2_splits.csv',
    'output_balance':     '/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/corum_ess_merged2_groups.csv',
    'output_summary':     '/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/corum_ess_merged2_split_summary.txt',
    'output_ess_by_size': '/Users/anitaapplegarth/github/dphil/protein_complexes/data/lookup_tables/corum_avg_essentiality_by_group_size2.csv',

    # Large groups to highlight in the focus-group profile (section 8)
    'focus_groups': [
        'grp_0038', 'grp_0152', 'grp_0514', 'grp_0761', 'grp_1154',
        'grp_1184', 'grp_1262', 'grp_1727', 'grp_1764', 'grp_1900',
        'grp_1932', 'grp_2065',
    ],

    # Number of independent splits to produce
    'n_splits': 15,

    # Train/test ratios
    'train_ratio': 0.80,
    'test_ratio':  0.20,

    # Flag a split if the test set size deviates from exactly 20% of total
    # proteins by more than this many percentage points (i.e. test must be
    # between 15% and 25% of total proteins). Checked before label balance.
    'max_size_deviation': 5.0,   # pp — so acceptable range is [15%, 25%]

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
    """Core -> Essential; Cancer/Non-essential -> Non-essential; else None."""
    if raw_label == 'Core':
        return 'Essential'
    elif raw_label in ('Cancer', 'Non-essential'):
        return 'Non-essential'
    return None


# ============================================================================
# LARGE-GROUP PRE-SCHEDULER
# ============================================================================

def schedule_large_groups(large_groups, n_splits, draw_per_split, rng):
    """
    Pre-assign which large groups go to TEST in each split so that:
      - every large group appears in test at least once,
      - no large group appears more than ceil(total_slots / n_groups) times,
      - each split receives exactly draw_per_split distinct large groups.

    Algorithm: build a pool where each group appears base or base+1 times,
    shuffle it, then consume it sequentially — skipping any group already
    chosen for the current split.  This guarantees no within-split duplicates
    and works for any n_groups >= draw_per_split.

    Returns
    -------
    list of length n_splits, each element a list of draw_per_split group IDs.
    """
    groups      = sorted(large_groups)
    n_groups    = len(groups)
    total_slots = n_splits * draw_per_split

    assert n_groups >= draw_per_split, (
        f"Cannot draw {draw_per_split} distinct groups from {n_groups} large groups.")

    # Build target frequency per group (base or base+1)
    base      = total_slots // n_groups
    remainder = total_slots % n_groups
    shuffled  = list(groups)
    rng.shuffle(shuffled)
    freq_target = {g: base + (1 if i < remainder else 0)
                   for i, g in enumerate(shuffled)}

    # Pool ordered highest-frequency first then shuffled, so scarce groups
    # are placed early when there are more open slots to land in.
    pool = []
    for g, cnt in sorted(freq_target.items(), key=lambda x: -x[1]):
        pool.extend([g] * cnt)
    rng.shuffle(pool)

    # Fill each split: consume pool in order, skipping duplicates for the
    # current split and putting skipped items back for later splits.
    assignment = [[] for _ in range(n_splits)]
    remaining  = list(pool)

    for s in range(n_splits):
        chosen  = []
        skipped = []
        for g in remaining:
            if len(chosen) == draw_per_split:
                skipped.append(g)
            elif g not in chosen:
                chosen.append(g)
            else:
                skipped.append(g)
        # Edge-case fallback: if pool exhausted before split is full,
        # top up with any group not yet chosen for this split.
        if len(chosen) < draw_per_split:
            extras = list(groups)
            rng.shuffle(extras)
            for g in extras:
                if g not in chosen:
                    chosen.append(g)
                if len(chosen) == draw_per_split:
                    break
        assignment[s] = chosen
        remaining = skipped

    for s, grps in enumerate(assignment):
        assert len(grps) == draw_per_split, (
            f"Split {s+1}: expected {draw_per_split} large groups, got {len(grps)}")
        assert len(set(grps)) == len(grps), (
            f"Split {s+1}: duplicate large group in test assignment")

    return assignment


def size_bucket(n):
    """Map group size to bucket name."""
    if n == 1:    return 'singleton'
    elif n <= 5:  return 'small'
    elif n <= 20: return 'medium'
    else:         return 'large'


# ============================================================================
# CORE SPLIT LOGIC — one attempt
# ============================================================================

def attempt_split(buckets, group_sizes, group_protein_map,
                  no_group_proteins, protein_to_label, analysis_pids,
                  config, rng, forced_large_test=None):
    """
    Make one attempt at a train/test split.

    For the large bucket, if `forced_large_test` is provided (a list of group
    IDs pre-assigned to test by the scheduler), those groups go directly to test
    and all remaining large groups go to train — no random draw for this bucket.

    For medium/small/singleton buckets, exactly round(n * 0.20) groups are
    drawn at random for test.  No-group proteins are split by stratified random
    sampling.

    Returns
    -------
    protein_to_split : dict[protein_id -> 'train'|'test']
    group_to_split   : dict[group_id  -> 'train'|'test']
    test_ess_rate    : float  — essential rate in test, analysis population only
    test_size_frac   : float  — fraction of ALL proteins assigned to test
    """
    test_ratio = config['test_ratio']
    group_to_split = {}

    for bucket_name, bucket_grps in buckets.items():
        if not bucket_grps:
            continue

        if bucket_name == 'large' and forced_large_test is not None:
            # Use pre-scheduled assignment — no random draw.
            forced_set = set(forced_large_test)
            for grp in bucket_grps:
                group_to_split[grp] = 'test' if grp in forced_set else 'train'
        else:
            n_grps    = len(bucket_grps)
            n_test    = round(n_grps * test_ratio)
            n_test    = max(n_test, 0)
            grps_list = list(bucket_grps)
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

    # Essential rate — analysis population (labelled + structured) only
    test_ap = [p for p, s in protein_to_split.items()
               if s == 'test' and p in analysis_pids]
    ap_counts  = Counter(protein_to_label[p] for p in test_ap)
    ap_lab     = ap_counts['Essential'] + ap_counts['Non-essential']
    test_ess_rate = ap_counts['Essential'] / ap_lab if ap_lab > 0 else 0.5

    # Size fraction — all proteins
    total = len(protein_to_split)
    test_size_frac = sum(1 for s in protein_to_split.values() if s == 'test') \
                     / total if total > 0 else 0.0

    return protein_to_split, group_to_split, test_ess_rate, test_size_frac


# ============================================================================
# VALIDATION
# ============================================================================


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


# ============================================================================
# BALANCE STATS
# ============================================================================

def balance_stats(protein_to_split, protein_to_label, global_ess_ratio,
                  analysis_pids, config):
    """
    Per-split balance statistics.

    Essentiality rates use the analysis population (labelled+structured).
    Size percentages use all proteins so train% + test% = 100%.
    """
    stats   = {}
    total_n = len(protein_to_split)
    for sp in ('train', 'test'):
        prots    = [p for p, s in protein_to_split.items() if s == sp]
        ap_prots = [p for p in prots if p in analysis_pids]
        cts      = Counter(protein_to_label[p] for p in ap_prots)
        ess, noness = cts['Essential'], cts['Non-essential']
        lab      = ess + noness
        all_cts  = Counter(protein_to_label.get(p, 'Unknown') for p in prots)
        stats[sp] = {
            'n_proteins':   len(prots),
            'size_pct':     round(100 * len(prots) / total_n, 2) if total_n else float('nan'),
            'n_essential':  ess,
            'n_noness':     noness,
            'n_unknown':    all_cts['Unknown'],
            'ess_rate_pct': round(100 * ess / lab, 2) if lab > 0 else float('nan'),
        }
    global_rate_pct = 100 * global_ess_ratio
    dev  = abs(stats['test']['ess_rate_pct'] - global_rate_pct)
    sdev = abs(stats['test']['size_pct'] - 100 * config['test_ratio'])
    stats['test_deviation_pp']      = round(dev, 2)
    stats['warning']                = dev  > config['max_ess_rate_deviation']
    stats['test_size_deviation_pp'] = round(sdev, 2)
    stats['size_warning']           = sdev > config['max_size_deviation']
    return stats

def bucket_profile_analysis(splits_df, analysis_pids, protein_to_label_map,
                             global_rate_pct, focus_groups, config):
    """
    Large-bucket randomness check and essentiality-by-group-size analysis.

    All group sizes and essentiality rates are computed over the analysis
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
    # All subsequent calculations use only analysis_pids, so group sizes,
    # protein counts and essentiality rates are all on the same denominator.
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

    # ── 7. Average essentiality per group size (analysis population) ──────────
    lines.append("")
    lines.append("=" * 90)
    lines.append("7. Average essentiality per group size  (analysis population only)")
    lines.append("=" * 90)
    lines.append(f"  Analysis population = labelled (Essential/Non-essential) + structured")
    lines.append(f"  Global essential rate: {global_rate_pct:.1f}%")
    lines.append(f"  n = {len(s1_ap)}")
    lines.append("")

    # group_size for each analysis protein = ap_group_sizes mapped back
    s1_ap = s1_ap.copy()
    s1_ap["ap_group_size"] = s1_ap["group_id"].map(ap_group_sizes)

    size_stats = (
        s1_ap
        .assign(is_essential=lambda x: x["protein_label"] == "Essential")
        .groupby("ap_group_size")
        .agg(
            n_groups   =("group_id",      "nunique"),
            n_proteins =("is_essential",  "count"),
            n_essential=("is_essential",  "sum"),
            ess_rate   =("is_essential",  "mean"),
        )
        .sort_index()
    )
    size_stats.index.name = "group_size"

    lines.append(f"  {'group_size':>12s}  {'n_groups':>9s}  {'n_proteins':>12s}  "
                 f"{'n_essential':>12s}  {'ess_rate':>10s}")
    lines.append("  " + "-" * 65)
    for gs, row in size_stats.iterrows():
        lines.append(f"  {gs:>12.0f}  {int(row['n_groups']):>9d}  "
                     f"{int(row['n_proteins']):>12d}  "
                     f"{int(row['n_essential']):>12d}  "
                     f"{row['ess_rate']*100:>9.1f}%")

    # Save CSV
    out = size_stats.copy()
    out["ess_rate_pct"] = (out["ess_rate"] * 100).round(2)
    out = out.drop(columns="ess_rate").reset_index()
    out = out[["group_size", "n_groups", "n_proteins", "n_essential", "ess_rate_pct"]]
    out.to_csv(config['output_ess_by_size'], index=False)
    lines.append(f"\n  Saved: {config['output_ess_by_size']}")

    # ── 8. Focus group essential rate profile ─────────────────────────────────
    lines.append("")
    lines.append("=" * 90)
    lines.append("8. Essential rate profile — focus groups  (analysis population only)")
    lines.append("=" * 90)
    lines.append(f"  Global essential rate: {global_rate_pct:.1f}%")
    lines.append("")
    lines.append(f"  {'Group':12s} {'AP size':>8}  {'Essential':>10}  "
                 f"{'Non-ess':>8}  {'vs global':>10}")
    lines.append("  " + "-" * 56)

    for gid in focus_groups:
        grp_ap = s1_ap[s1_ap["group_id"] == gid]
        size   = int(ap_group_sizes.get(gid, 0))
        n_ess  = int((grp_ap["protein_label"] == "Essential").sum())
        n_ne   = int((grp_ap["protein_label"] == "Non-essential").sum())
        n_lab  = n_ess + n_ne
        rate   = 100 * n_ess / n_lab if n_lab > 0 else float('nan')
        diff   = rate - global_rate_pct if n_lab > 0 else float('nan')
        dirn   = f"+{diff:.1f}pp" if diff >= 0 else f"{diff:.1f}pp"
        flag   = "  ⚠ high" if diff > 10 else ("  ⚠ low" if diff < -10 else "")
        lines.append(f"  {gid:12s} {size:>8}  {n_ess:>10}  {n_ne:>8}  "
                     f"{dirn:>10}{flag}")

    # ── 9. Per-split large-bucket ess% annotated with focus groups ────────────
    lines.append("")
    lines.append("  Large-bucket test essential rate by split, annotated with focus groups:")
    lines.append(f"  {'Split':>5}  {'Focus groups in test':<34}  "
                 f"{'Focus ess%':>10}  {'Bucket ess%':>11}")
    lines.append("  " + "-" * 68)

    for split_idx in sorted(lg_test["split_index"].unique()):
        sp_df         = lg_test[lg_test["split_index"] == split_idx]
        in_test       = set(sp_df["group_id"].unique())
        heavy         = [g for g in focus_groups if g in in_test]
        heavy_label   = ', '.join(heavy) if heavy else "none"

        if heavy:
            hdf    = sp_df[sp_df["group_id"].isin(heavy) &
                           sp_df["UniProt_AC"].isin(analysis_pids)]
            h_cts  = Counter(hdf["protein_label"])
            h_lab  = h_cts["Essential"] + h_cts["Non-essential"]
            h_rate = f"{100*h_cts['Essential']/h_lab:.1f}%" if h_lab else "N/A"
        else:
            h_rate = "N/A"

        b_df   = sp_df[sp_df["UniProt_AC"].isin(analysis_pids)]
        b_cts  = Counter(b_df["protein_label"])
        b_lab  = b_cts["Essential"] + b_cts["Non-essential"]
        b_rate = f"{100*b_cts['Essential']/b_lab:.1f}%" if b_lab else "N/A"

        lines.append(f"  {int(split_idx):>5}  {heavy_label:<34}  "
                     f"{h_rate:>10}  {b_rate:>11}")

    return lines


# ============================================================================
# MAIN
# ============================================================================


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

    # group_sizes: count from protein_to_group so it matches group_protein_map
    group_sizes = defaultdict(int)
    for pid, grp in protein_to_group.items():
        group_sizes[grp] += 1
    group_sizes = dict(group_sizes)

    protein_to_label = {}
    for _, row in targets.iterrows():
        bl = binarise_label(row['essential_category'])
        if bl is not None:
            protein_to_label[row['Protein']] = bl

    all_proteins    = sorted(complex_proteins)
    structured_pids = set(protein_to_group.keys())

    # ── Analysis population ────────────────────────────────────────────────────
    # Proteins that are BOTH labelled AND have a structural group.
    # This is the consistent denominator for all rates throughout the script.
    analysis_pids = {
        p for p in all_proteins
        if p in structured_pids and protein_to_label.get(p) in ('Essential', 'Non-essential')
    }
    n_ess_global    = sum(1 for p in analysis_pids if protein_to_label[p] == 'Essential')
    n_noness_global = len(analysis_pids) - n_ess_global
    n_structured    = len(analysis_pids)
    global_ess_ratio = n_ess_global / n_structured if n_structured > 0 else 0.5
    global_rate_pct  = 100 * global_ess_ratio

    # Counts for reporting
    n_unknown_all = sum(1 for p in all_proteins
                        if protein_to_label.get(p, 'Unknown') == 'Unknown')
    n_no_group    = len(complex_proteins - structured_pids)

    n_ess_all = sum(1 for p in all_proteins if protein_to_label.get(p) == 'Essential')
    n_ne_all  = sum(1 for p in all_proteins if protein_to_label.get(p) == 'Non-essential')
    print(f"\n  Label distribution ({len(all_proteins)} total complex proteins):")
    print(f"    Essential      : {n_ess_all:4d}")
    print(f"    Non-essential  : {n_ne_all:4d}")
    print(f"    Unknown        : {n_unknown_all:4d}  (no DepMap/OGEE evidence — excluded)")
    print(f"    No-group       : {n_no_group:4d}  (no structural assignment — excluded)")
    print(f"  Analysis population (labelled + structured): {n_structured}")
    print(f"  Global essential rate (analysis population): {global_rate_pct:.1f}%"
          f"  ({n_ess_global} / {n_structured})")

    # -------------------------------------------------------------------------
    # Build group -> protein map and size buckets
    # -------------------------------------------------------------------------
    group_protein_map = defaultdict(set)
    for pid, grp in protein_to_group.items():
        group_protein_map[grp].add(pid)

    no_group_proteins = complex_proteins - structured_pids

    # Buckets ordered largest-first (4 buckets; large covers >20)
    buckets = {'large': [], 'medium': [], 'small': [], 'singleton': []}
    for grp, n in group_sizes.items():
        buckets[size_bucket(n)].append(grp)

    print(f"\n--- Group size buckets (largest first) ---")
    bucket_label = {
        'large':      '>20 proteins (merged large + very large)',
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
    # Pre-schedule large group test assignments across all splits
    # -------------------------------------------------------------------------
    # Use a separate RNG seeded from base_seed so the schedule is reproducible
    # and independent of per-split attempt seeds.
    schedule_rng = np.random.default_rng(CONFIG['base_seed'])
    large_draw   = round(len(buckets['large']) * CONFIG['test_ratio'])
    large_schedule = schedule_large_groups(
        large_groups   = buckets['large'],
        n_splits       = CONFIG['n_splits'],
        draw_per_split = large_draw,
        rng            = schedule_rng,
    )

    # Report the schedule
    test_counts_by_group = Counter(
        grp for split_grps in large_schedule for grp in split_grps
    )
    print(f"\n--- Large-group pre-schedule ({len(buckets['large'])} groups, "
          f"{large_draw} drawn/split, {CONFIG['n_splits']} splits) ---")
    print(f"  Appearance counts: min={min(test_counts_by_group.values())}  "
          f"max={max(test_counts_by_group.values())}  "
          f"mean={sum(test_counts_by_group.values())/len(test_counts_by_group):.2f}")
    print(f"  (every large group will appear in test exactly "
          f"{min(test_counts_by_group.values())}–{max(test_counts_by_group.values())} times)")

    # -------------------------------------------------------------------------
    # Generate N splits with reject-and-redraw for label balance
    # -------------------------------------------------------------------------
    print(f"\n--- Generating {CONFIG['n_splits']} splits "
          f"(tolerance ±{CONFIG['max_ess_rate_deviation']}pp ess, "
          f"±{CONFIG['max_size_deviation']}pp size, "
          f"max {CONFIG['max_attempts']} attempts each) ---")

    all_rows        = []
    balance_records = []

    for i in range(CONFIG['n_splits']):
        best_split      = None
        best_grp_split  = None
        best_deviation  = float('inf')
        best_size_deviation = float('inf')
        attempts        = 0
        accepted        = False

        while attempts < CONFIG['max_attempts']:
            seed = CONFIG['base_seed'] + i + attempts * 1000
            rng  = np.random.default_rng(seed)

            protein_to_split, group_to_split, test_ess_rate, test_size_frac = attempt_split(
                buckets           = buckets,
                group_sizes       = group_sizes,
                group_protein_map = group_protein_map,
                no_group_proteins = sorted(no_group_proteins),
                protein_to_label  = protein_to_label,
                analysis_pids     = analysis_pids,
                config            = CONFIG,
                rng               = rng,
                forced_large_test = large_schedule[i],
            )
            attempts += 1

            # --- Check 1: test set size (must be within 15–25% of total proteins) ---
            size_deviation = abs(100 * test_size_frac - 100 * CONFIG['test_ratio'])

            # --- Check 2: essentiality balance ---
            deviation = abs(100 * test_ess_rate - global_rate_pct)

            # Keep track of best attempt in case we exhaust the budget
            # Primary sort: size deviation; secondary: essentiality deviation
            combined = (size_deviation, deviation)
            if combined < (best_size_deviation, best_deviation):
                best_size_deviation = size_deviation
                best_deviation      = deviation
                best_split          = protein_to_split
                best_grp_split      = group_to_split
                best_seed           = seed

            size_ok = size_deviation <= CONFIG['max_size_deviation']
            ess_ok  = deviation       <= CONFIG['max_ess_rate_deviation']
            if size_ok and ess_ok:
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
            protein_to_split, protein_to_label, global_ess_ratio,
            analysis_pids, CONFIG)

        # Per-bucket protein breakdown for the summary table
        bucket_breakdown = {}
        for bname in ('large', 'medium', 'small', 'singleton'):
            bgrps = buckets.get(bname, [])
            train_n = sum(
                len(group_protein_map.get(g, set()))
                for g in bgrps if group_to_split.get(g) == 'train'
            )
            test_n = sum(
                len(group_protein_map.get(g, set()))
                for g in bgrps if group_to_split.get(g) == 'test'
            )
            total_n = train_n + test_n
            # Essential rate in test for this bucket
            test_prots_bkt = [
                p for g in bgrps if group_to_split.get(g) == 'test'
                for p in group_protein_map.get(g, set())
            ]
            bc = Counter(protein_to_label.get(p, 'Unknown') for p in test_prots_bkt)
            lab = bc['Essential'] + bc['Non-essential']
            ess_rate = 100 * bc['Essential'] / lab if lab > 0 else float('nan')
            bucket_breakdown[bname] = {
                'train_n':   train_n,
                'test_n':    test_n,
                'total_n':   total_n,
                'test_pct':  100 * test_n / total_n if total_n > 0 else 0.0,
                'test_ess_pct': round(ess_rate, 1),
            }

        bal_row = {
            'split_index':           i + 1,
            'seed_accepted':         best_seed,
            'attempts':              attempts,
            'accepted':              accepted,
            'train_n':               bstats['train']['n_proteins'],
            'train_pct':             bstats['train']['size_pct'],
            'test_n':                bstats['test']['n_proteins'],
            'test_pct':              bstats['test']['size_pct'],
            'test_size_deviation_pp': bstats['test_size_deviation_pp'],
            'size_warning':          bstats['size_warning'],
            'train_essential':       bstats['train']['n_essential'],
            'test_essential':        bstats['test']['n_essential'],
            'train_noness':          bstats['train']['n_noness'],
            'test_noness':           bstats['test']['n_noness'],
            'train_unknown':         bstats['train']['n_unknown'],
            'test_unknown':          bstats['test']['n_unknown'],
            'train_ess_rate_pct':    bstats['train']['ess_rate_pct'],
            'test_ess_rate_pct':     bstats['test']['ess_rate_pct'],
            'test_deviation_pp':     bstats['test_deviation_pp'],
            'warning':               bstats['warning'],
            'leakage_violations':    len(violations),
            # Per-bucket test counts
            'large_test_n':      bucket_breakdown['large']['test_n'],
            'large_test_pct':    round(bucket_breakdown['large']['test_pct'], 1),
            'large_test_ess_pct': bucket_breakdown['large']['test_ess_pct'],
            'medium_test_n':     bucket_breakdown['medium']['test_n'],
            'medium_test_pct':   round(bucket_breakdown['medium']['test_pct'], 1),
            'medium_test_ess_pct': bucket_breakdown['medium']['test_ess_pct'],
            'small_test_n':      bucket_breakdown['small']['test_n'],
            'small_test_pct':    round(bucket_breakdown['small']['test_pct'], 1),
            'small_test_ess_pct': bucket_breakdown['small']['test_ess_pct'],
            'singleton_test_n':  bucket_breakdown['singleton']['test_n'],
            'singleton_test_pct': round(bucket_breakdown['singleton']['test_pct'], 1),
            'singleton_test_ess_pct': bucket_breakdown['singleton']['test_ess_pct'],
        }
        balance_records.append(bal_row)

        flag    = ' ⚠ balance warning' if bstats['warning'] else ''
        szflag  = ' ⚠ size warning'   if bstats['size_warning'] else ''
        lk      = f' ⚠ {len(violations)} leakage' if violations else ''
        retry   = f' (accepted on attempt {attempts})' if attempts > 1 else ''
        print(f"  Split {i+1:2d} | "
              f"train={bstats['train']['n_proteins']} ({bstats['train']['size_pct']:.1f}%) "
              f"ess={bstats['train']['ess_rate_pct']:.1f}% | "
              f"test={bstats['test']['n_proteins']} ({bstats['test']['size_pct']:.1f}%) "
              f"ess={bstats['test']['ess_rate_pct']:.1f}% | "
              f"size_dev={bstats['test_size_deviation_pp']:.1f}pp "
              f"ess_dev={bstats['test_deviation_pp']:.1f}pp"
              f"{retry}{szflag}{flag}{lk}")

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
                'label_mask':    pid in analysis_pids,
            })

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------
    print("\n--- Saving outputs ---")

    splits_df = pd.DataFrame(all_rows)
    splits_df.to_csv(CONFIG['output_all_splits'], index=False)
    print(f"  Saved: {CONFIG['output_all_splits']}  ({len(all_rows):,} rows)")

    balance_df = pd.DataFrame(balance_records)
    balance_df.to_csv(CONFIG['output_balance'], index=False)
    print(f"  Saved: {CONFIG['output_balance']}")

    # -------------------------------------------------------------------------
    # Summary report
    # -------------------------------------------------------------------------
    n_warnings  = sum(1 for r in balance_records if r['warning'])
    n_sz_warn   = sum(1 for r in balance_records if r['size_warning'])
    n_leakage   = sum(1 for r in balance_records if r['leakage_violations'] > 0)
    n_retried   = sum(1 for r in balance_records if r['attempts'] > 1)
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
    lines.append(f"Total groups (all atomic):      {len(group_sizes)}")
    lines.append(f"  Large      (>20, merged):         {len(buckets['large'])}")
    lines.append(f"  Medium     (6–20):              {len(buckets['medium'])}")
    lines.append(f"  Small      (2–5):               {len(buckets['small'])}")
    lines.append(f"  Singleton  (1 protein):         {len(buckets['singleton'])}")
    lines.append(f"Analysis population:              {n_structured}  "
                 f"(labelled + structured; excludes {n_unknown_all} Unknown, "
                 f"{n_no_group} no-group)")
    lines.append(f"Global essential rate:            {global_rate_pct:.1f}%  "
                 f"({n_ess_global} / {n_structured})")
    lines.append(f"Number of splits:                 {CONFIG['n_splits']}")
    lines.append(f"Train/test ratio:                 "
                 f"{CONFIG['train_ratio']:.0%} / {CONFIG['test_ratio']:.0%}")
    lines.append(f"Size tolerance:                   "
                 f"±{CONFIG['max_size_deviation']}pp from 20% "
                 f"(acceptable test range: "
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

    # Table header — two extra columns: train%, test%, size_dev
    lines.append(
        f"{'Split':>6} {'Att':>4}  "
        f"{'Train N':>8} {'Train%':>7} {'Train ess%':>11}  "
        f"{'Test N':>7} {'Test%':>6} {'Test ess%':>10}  "
        f"{'SzDev':>6} {'EssDev':>7}  {'Flag':>4}"
    )
    lines.append("-" * 90)
    for r in balance_records:
        flag = ''
        if r['size_warning']:
            flag += 'Sz'
        if r['warning']:
            flag += 'Es'
        if r['leakage_violations'] > 0:
            flag += 'Lk'
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

    # -------------------------------------------------------------------------
    # Table 2: Per-split breakdown by structural group size bucket (proteins)
    # -------------------------------------------------------------------------
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
        def _fmt(key_n, key_pct, key_ess, width_n, width_pct):
            ess = r[key_ess]
            ess_str = f"{ess:5.1f}%" if ess == ess else "   N/A"  # nan check
            return (f"{r[key_n]:>{width_n}} {r[key_pct]:>{width_pct}.1f}% {ess_str}")

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
    # Column means
    def _col_mean(key):
        vals = [r[key] for r in balance_records if r[key] == r[key]]  # exclude nan
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

    # -------------------------------------------------------------------------
    # Table 3: Large-group test frequency across all splits
    # -------------------------------------------------------------------------
    # Table 3: Large-group test frequency (pre-scheduled) + full profile analysis
    all_large_groups = sorted(buckets['large'])
    large_freq = Counter(
        grp for split_grps in large_schedule for grp in split_grps
    )
    lines.append("")
    lines.append("=" * 70)
    lines.append("LARGE-GROUP TEST-FREQUENCY TABLE (pre-scheduled, guaranteed coverage)")
    lines.append("=" * 70)
    lines.append(f"  {'Group':<12} {'Size':>6}  {'Test count':>10}  {'Splits in test'}")
    lines.append("  " + "-" * 65)
    for grp in all_large_groups:
        sz         = group_sizes[grp]
        cnt        = large_freq.get(grp, 0)
        split_list = [str(s + 1) for s, sg in enumerate(large_schedule) if grp in sg]
        lines.append(f"  {grp:<12} {sz:>6}  {cnt:>10}  [{', '.join(split_list)}]")
    lines.append("  " + "-" * 65)
    freq_vals = [large_freq.get(g, 0) for g in all_large_groups]
    lines.append(f"  {'Min':>18}: {min(freq_vals)}")
    lines.append(f"  {'Max':>18}: {max(freq_vals)}")
    lines.append(f"  {'Mean':>18}: {sum(freq_vals)/len(freq_vals):.2f}")

    # Bucket profile analysis (same as corum_merged_random_ess_split.py)
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