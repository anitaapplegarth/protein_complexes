"""
Is the label a property of the complex?

If essentiality belongs to whole complexes rather than to well-connected
proteins, members of the same complex should share a label more often than
expected by chance. Test: the % of complexes whose labelled members are ALL
positive or ALL negative ("pure"), compared with a permutation null that
shuffles labels across proteins while keeping every complex's membership
fixed. Drug target tasks are included as a contrast.

The membership file also records stoichiometry (0 = not recorded), so the
test is repeated for complexes with and without recorded stoichiometry, to
check the result is not driven by better-studied complexes.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =======================================================
# CONFIG
# =======================================================
DATA_DIR = Path("../../data/lookup_tables/cp/")

# One row per complex-protein pair; Stoichiometry = 0 means not recorded
MEMBERSHIP_FILE = "stoich_protein.csv"
COMPLEX_COL     = "ComplexId"
PROTEIN_COL     = "ProteinId"
STOICH_COL      = "Stoichiometry"

TASKS = {
    "Essentiality": dict(
        SPLITS_FILE="ess_protein_splits.csv",
        LABEL_MAP={'Essential': 1, 'Non-essential': 0},
    ),
    "Drug target (HPA)": dict(
        SPLITS_FILE="hpa_protein_splits.csv",
        LABEL_MAP={'Drug_target': 1, 'Non_target': 0},
    ),
    "Drug target (ChEMBL)": dict(
        SPLITS_FILE="chembl_protein_splits.csv",
        LABEL_MAP={'Drug_target': 1, 'Non_target': 0},
    ),
}

# Which complexes to test. A complex is "recorded" if every member has a
# stoichiometry > 0, "not recorded" if none do; mixed complexes appear
# only under "All complexes".
COMPLEX_SUBSETS = {
    "All complexes":          "all",
    "Stoichiometry recorded": "recorded",
    "Not recorded":           "not_recorded",
}

# Complex Portal lists many variants of one complex (e.g. SCF ligases that
# share SKP1-CUL1-RBX1 and differ only in the F-box protein). Variants are
# not independent, so the test is repeated after collapsing them: complexes
# whose member sets overlap by at least this Jaccard index are grouped
# (single linkage) and one representative is kept per group (the one with
# most members). None = keep every complex.
VARIANT_MODES = {
    "All complexes listed": None,
    "Variants collapsed":   0.5,
}

MIN_LABELLED = 2       # complexes need at least this many labelled members
N_PERM       = 1000
RANDOM_STATE = 42

# Bins for the % of labelled members that are positive, per complex
FRAC_BINS   = [-0.001, 0, 1 / 3, 2 / 3, 0.999999, 1]
FRAC_LABELS = ["0%", "1–33%", "34–66%", "67–99%", "100%"]

OBS_COLOUR  = "#C0392B"
NULL_COLOUR = "#95A5A6"

OUTPUT_DIR = Path("./feature_distributions/by_label")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size':        12,
    'axes.titlesize':   14,
    'axes.labelsize':   12,
    'xtick.labelsize':  12,
    'ytick.labelsize':  12,
    'legend.fontsize':  12,
    'figure.titlesize': 16,
})

# =======================================================
# Load
# =======================================================
membership = (pd.read_csv(DATA_DIR / MEMBERSHIP_FILE)
              .rename(columns={PROTEIN_COL: 'ProteinId',
                               COMPLEX_COL: 'ComplexId',
                               STOICH_COL: 'Stoichiometry'})
              [['ComplexId', 'ProteinId', 'Stoichiometry']]
              .drop_duplicates(['ComplexId', 'ProteinId']))

rec = (membership.assign(rec=membership['Stoichiometry'] > 0)
       .groupby('ComplexId')['rec'].agg(['all', 'any']))
complex_status = pd.Series('mixed', index=rec.index)
complex_status[rec['all']] = 'recorded'
complex_status[~rec['any']] = 'not_recorded'
membership['status'] = membership['ComplexId'].map(complex_status)

print(f"Membership: {membership['ComplexId'].nunique()} complexes, "
      f"{membership['ProteinId'].nunique()} proteins")
print("Complexes by stoichiometry status:")
print(complex_status.value_counts().to_string())


def collapse_variants(mem, threshold):
    """Group complexes with Jaccard overlap >= threshold (single linkage)
    and keep the largest complex of each group."""
    sets = mem.groupby('ComplexId')['ProteinId'].apply(frozenset)
    ids = list(sets.index)
    pos = {c: k for k, c in enumerate(ids)}
    parent = list(range(len(ids)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    # Only compare complexes that share at least one protein
    by_prot = mem.groupby('ProteinId')['ComplexId'].apply(list)
    seen = set()
    for cxs in by_prot:
        for a_i in range(len(cxs)):
            for b_i in range(a_i + 1, len(cxs)):
                a, b = pos[cxs[a_i]], pos[cxs[b_i]]
                key = (min(a, b), max(a, b))
                if key in seen:
                    continue
                seen.add(key)
                sa, sb = sets.iloc[a], sets.iloc[b]
                if len(sa & sb) / len(sa | sb) >= threshold:
                    parent[find(a)] = find(b)

    groups = pd.Series([find(k) for k in range(len(ids))], index=ids)
    size = sets.map(len)
    reps = (pd.DataFrame({'group': groups, 'size': size})
            .reset_index(names='ComplexId')
            .sort_values(['size', 'ComplexId'], ascending=[False, True])
            .drop_duplicates('group')['ComplexId'])
    group_sizes = groups.value_counts()
    print(f"\nCollapsing variants (Jaccard >= {threshold}): "
          f"{len(ids)} complexes -> {len(reps)} groups; "
          f"largest group has {group_sizes.max()} complexes")
    # Check for chaining: list the biggest groups by their representative
    rep_of = (pd.DataFrame({'group': groups, 'size': size})
              .loc[list(reps)].reset_index(names='rep')
              .set_index('group')['rep'])
    print("Largest variant groups (representative complex: n complexes):")
    for g, n in group_sizes.head(8).items():
        print(f"  {rep_of[g]}: {n}")
    pd.DataFrame({'ComplexId': ids, 'representative': groups.map(rep_of).values}
                 ).to_csv(OUTPUT_DIR / f'variant_groups_jaccard{threshold}.csv',
                          index=False)
    return mem[mem['ComplexId'].isin(set(reps))]


def load_labels(splits_file, label_map):
    df = pd.read_csv(DATA_DIR / splits_file).rename(
        columns={'UniProt_AC': 'ProteinId'})
    df = df[df['label_mask']].drop_duplicates('ProteinId')
    df['target'] = df['protein_label'].map(label_map)
    return df.dropna(subset=['target'])[['ProteinId', 'target']].astype(
        {'target': int})


rng = np.random.default_rng(RANDOM_STATE)


def complex_fractions(y_by_row, starts, counts):
    return np.add.reduceat(y_by_row, starts) / counts


def pct_pure(frac):
    return 100 * np.mean((frac == 0) | (frac == 1))


def frac_hist(frac):
    cats = pd.Series(pd.cut(frac, FRAC_BINS, labels=FRAC_LABELS))
    return (cats.value_counts(normalize=True)
            .reindex(FRAC_LABELS).fillna(0).values * 100)


def run_test(mem):
    """Permutation test on one set of complex memberships with labels."""
    counts = mem.groupby('ComplexId').size()
    keep = counts[counts >= MIN_LABELLED].index
    mem = mem[mem['ComplexId'].isin(keep)].sort_values('ComplexId')
    if mem.empty:
        return None
    counts = mem.groupby('ComplexId', sort=True).size().values
    starts = np.r_[0, np.cumsum(counts)[:-1]]

    # Permute labels across unique proteins (a protein in several complexes
    # keeps one label everywhere), not across membership rows.
    prots = mem['ProteinId'].unique()
    p_index = pd.Series(np.arange(len(prots)), index=prots)
    row_to_prot = p_index[mem['ProteinId']].values
    y_prot = (mem.drop_duplicates('ProteinId').set_index('ProteinId')
              .loc[prots, 'target'].values)

    frac_obs = complex_fractions(y_prot[row_to_prot], starts, counts)
    obs = pct_pure(frac_obs)

    null = np.empty(N_PERM)
    null_hist = np.zeros(len(FRAC_LABELS))
    for b in range(N_PERM):
        f = complex_fractions(rng.permutation(y_prot)[row_to_prot],
                              starts, counts)
        null[b] = pct_pure(f)
        null_hist += frac_hist(f)
    null_hist /= N_PERM

    return dict(
        n_complexes=len(counts), n_proteins=len(prots),
        pct_positive=100 * y_prot.mean(), obs=obs, null=null,
        obs_hist=frac_hist(frac_obs), null_hist=null_hist,
        p=(np.sum(null >= obs) + 1) / (N_PERM + 1),
    )


# =======================================================
# Test per task and complex subset, for each variant mode
# =======================================================
all_results, all_hist = [], []

for mode_name, threshold in VARIANT_MODES.items():
    mem_mode = (membership if threshold is None
                else collapse_variants(membership, threshold))
    suffix = "" if threshold is None else "_variants_collapsed"

    n_sub, n_task = len(COMPLEX_SUBSETS), len(TASKS)
    fig, axes = plt.subplots(n_sub, n_task,
                             figsize=(5.5 * n_task, 4.2 * n_sub),
                             squeeze=False)

    for j, (task_name, cfg) in enumerate(TASKS.items()):
        labels = load_labels(cfg["SPLITS_FILE"], cfg["LABEL_MAP"])
        mem_all = mem_mode.merge(labels, on='ProteinId', how='inner')

        for i, (sub_name, status) in enumerate(COMPLEX_SUBSETS.items()):
            mem = (mem_all if status == 'all'
                   else mem_all[mem_all['status'] == status])
            r = run_test(mem)
            ax = axes[i, j]
            if r is None:
                ax.set_visible(False)
                continue

            all_results.append({
                'variants':           mode_name,
                'task':               task_name,
                'complexes':          sub_name,
                'n_complexes':        r['n_complexes'],
                'n_proteins':         r['n_proteins'],
                'pct_positive':       round(r['pct_positive'], 1),
                'pct_pure_observed':  round(r['obs'], 1),
                'pct_pure_null_mean': round(r['null'].mean(), 1),
                'pct_pure_null_lo':   round(np.percentile(r['null'], 2.5), 1),
                'pct_pure_null_hi':   round(np.percentile(r['null'], 97.5), 1),
                'p_value':            r['p'],
            })
            for lab, o, n in zip(FRAC_LABELS, r['obs_hist'], r['null_hist']):
                all_hist.append({'variants': mode_name, 'task': task_name,
                                 'complexes': sub_name, 'bin': lab,
                                 'pct_complexes_observed': round(o, 1),
                                 'pct_complexes_chance': round(n, 1)})

            x, w = np.arange(len(FRAC_LABELS)), 0.4
            ax.bar(x - w / 2, r['obs_hist'], w, color=OBS_COLOUR,
                   label="Observed")
            ax.bar(x + w / 2, r['null_hist'], w, color=NULL_COLOUR,
                   label="Expected by chance")
            ax.set_xticks(x)
            ax.set_xticklabels(FRAC_LABELS)
            ax.set_xlabel("Labelled members that are positive"
                          if i == n_sub - 1 else "")
            ax.set_ylabel(f"{sub_name}\n% of complexes" if j == 0 else "")
            title = f"{sub_name} ({r['n_complexes']} complexes)"
            ax.set_title(f"{task_name}\n{title}" if i == 0 else title,
                         fontsize=12)
            ax.grid(axis='y', alpha=0.3)
            ax.set_axisbelow(True)
            if i == 0 and j == 0:
                ax.legend(loc='upper center', frameon=False)

    fig.suptitle(f"Do members of a complex share a label? CP ({mode_name})",
                 y=1.01)
    fig.tight_layout()
    out = OUTPUT_DIR / f"complex_label_homogeneity{suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")

res_df = pd.DataFrame(all_results)
res_df.to_csv(OUTPUT_DIR / 'complex_label_homogeneity.csv', index=False)
pd.DataFrame(all_hist).to_csv(
    OUTPUT_DIR / 'complex_label_homogeneity_bins.csv', index=False)

print("\n% of complexes whose labelled members all share one label:")
with pd.option_context('display.width', 220):
    print(res_df.to_string(index=False))
