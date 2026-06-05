import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats

lu     = pd.read_csv('../../../data/lookup_tables/lu_essentiality_protein.csv')
cp_raw = pd.read_csv('../../../data/lookup_tables/cp_stoich_protein.csv')

# Cancer treated as Non-essential; CORUM excluded (<4% stoich coverage)
BINS   = [0, 1, 2, 3, 4, 5, 100]
LABELS = ['1', '2', '3', '4', '5', '6+']
BLUE = '#4C72B0'
GREY = '#8C8C8C'
BG   = '#F9F9F9'

agg = cp_raw.groupby('ProteinId')['Stoichiometry'].max().reset_index()
agg.columns = ['ProteinId', 'MaxStoich']
merged = agg.merge(lu, left_on='ProteinId', right_on='Protein', how='inner')
merged['is_essential'] = merged['essential_category'].isin(['Core']).astype(int)
known = merged[merged['MaxStoich'] > 0].copy()
known['plot_cat']   = known['essential_category'].replace({'Cancer': 'Non-essential'})
known['stoich_bin'] = pd.cut(known['MaxStoich'], bins=BINS, labels=LABELS, right=True)

grp = known.groupby('stoich_bin', observed=True).agg(
    n=('is_essential', 'count'),
    n_essential=('is_essential', 'sum'),
    rate=('is_essential', 'mean')
).reset_index()
grp['ci_low']   = grp.apply(lambda r: stats.binom.ppf(0.025, r['n'], max(r['rate'], 1e-9)) / r['n'], axis=1)
grp['ci_high']  = grp.apply(lambda r: stats.binom.ppf(0.975, r['n'], max(r['rate'], 1e-9)) / r['n'], axis=1)
grp['err_low']  = grp['rate'] - grp['ci_low']
grp['err_high'] = grp['ci_high'] - grp['rate']
baseline = known['is_essential'].mean()
ess_vals = known.loc[known['is_essential'] == 1, 'MaxStoich']
non_vals = known.loc[known['is_essential'] == 0, 'MaxStoich']
_, mw_p  = stats.mannwhitneyu(ess_vals, non_vals, alternative='two-sided')

# ── Figure 1: Rate bar chart ──────────────────────────────────────────────────
fig1, ax = plt.subplots(figsize=(8, 5), facecolor='white')
fig1.suptitle(
    'Essentiality rate by stoichiometry bin — Complex Portal (CP)\n'
    '(Cancer treated as Non-essential)',
    fontsize=12, fontweight='bold', y=1.02
)
ax.set_facecolor(BG)
x = np.arange(len(grp))

ax.bar(x, grp['rate'] * 100, color=BLUE, alpha=0.85, width=0.6,
       zorder=3, edgecolor='white', linewidth=0.8)
ax.errorbar(x, grp['rate'] * 100,
            yerr=[grp['err_low'] * 100, grp['err_high'] * 100],
            fmt='none', color='#333333', capsize=4, linewidth=1.3, zorder=4)

label_offset = grp['err_high'].max() * 100 + 0.8
for xi, row in grp.iterrows():
    ci_top = row['rate'] * 100 + row['err_high'] * 100
    ax.text(xi, ci_top + 0.8,
            f"{int(row['n_essential'])}/{int(row['n'])}",
            ha='center', va='bottom', fontsize=8, color='#555')

ax.axhline(baseline * 100, color=GREY, ls='--', lw=1.4, zorder=2,
           label=f'Overall: {baseline * 100:.1f}%')
ax.set_xticks(x)
ax.set_xticklabels(LABELS, fontsize=10)
ax.set_xlabel('Max stoichiometry in complex', fontsize=10)
ax.set_ylabel('% proteins labelled Core-essential', fontsize=10)
ax.text(0.97, 0.97, f'MW p = {mw_p:.2e}', transform=ax.transAxes,
        ha='right', va='top', fontsize=8.5, color='#444', style='italic')
ax.legend(fontsize=8.5, framealpha=0.6)
ax.set_ylim(0, min(50, grp['rate'].max() * 100 + grp['err_high'].max() * 100 + 10))
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', alpha=0.35, zorder=1)

plt.tight_layout()
fig1.savefig('stoich_essentiality_rate.png', dpi=150, bbox_inches='tight')

# ── Figure 2: Stacked bar chart ───────────────────────────────────────────────
fig2, ax = plt.subplots(figsize=(7, 5), facecolor='white')
fig2.suptitle(
    'Stoichiometry bin composition by essentiality — Complex Portal (CP)\n'
    '(Cancer treated as Non-essential)',
    fontsize=12, fontweight='bold', y=1.02
)
ax.set_facecolor(BG)

groups   = ['Non-essential', 'Core']
n_groups = [len(known[known['plot_cat'] == g]) for g in groups]
x        = np.arange(len(groups))

comp = (known.groupby(['plot_cat', 'stoich_bin'], observed=True)
             .size().reset_index(name='count'))
totals = comp.groupby('plot_cat')['count'].transform('sum')
comp['pct'] = comp['count'] / totals * 100

bin_colours = ['#cfe2f3', '#93c4e0', '#4a9eca', '#1d6fa4', '#0d4b75', '#052a45']
bottoms = np.zeros(len(groups))
for lbl, bc in zip(LABELS, bin_colours):
    heights = []
    for g in groups:
        row = comp[(comp['plot_cat'] == g) & (comp['stoich_bin'] == lbl)]
        heights.append(row['pct'].values[0] if len(row) else 0)
    heights = np.array(heights)
    ax.bar(x, heights, bottom=bottoms, color=bc, width=0.5,
           label=f'Stoich {lbl}', edgecolor='white', linewidth=0.6)
    for xi, (h, b) in enumerate(zip(heights, bottoms)):
        if h >= 4:
            ax.text(xi, b + h / 2, f'{h:.0f}%',
                    ha='center', va='center', fontsize=8,
                    color='white', fontweight='bold')
    bottoms += heights

ax.set_xticks(x)
ax.set_xticklabels([f'{g}\n(n={n})' for g, n in zip(groups, n_groups)], fontsize=10)
ax.set_ylabel('% of proteins', fontsize=10)
ax.set_ylim(0, 105)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
ax.legend(title='Stoich bin', fontsize=8.5, bbox_to_anchor=(1.01, 1),
          loc='upper left', framealpha=0.7)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', alpha=0.25, zorder=1)

plt.tight_layout()
fig2.savefig('stoich_essentiality_stacked.png', dpi=150, bbox_inches='tight')

print('Done.')