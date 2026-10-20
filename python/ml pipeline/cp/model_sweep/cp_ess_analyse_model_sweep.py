"""
=======================================================================
MODEL-VARIANCE SWEEP  —  analysis and figures
=======================================================================
Reads sweep_results.csv (long format) and produces:

  fig_sweep_panels.png   Panel A  PR-AUC by tier, pooled over models
                         Panel B  paired per-split deltas, pooled
                         Panel C  forest plot, one row per model

  fig_sweep_by_model.png small multiples — Panel A per model, which
                         shows invariance directly rather than hiding
                         it in a pooled distribution

  sweep_summary.csv      per-model means, deltas, sign tests

Panel A is the descriptive figure. It will show heavy overlap, because
pooling models mixes between-model variance (large) with between-split
variance (small) — logistic regression and TabPFN differ far more than
the tiers do. Panels B and C are the ones that carry the claim: model
identity cancels within a split, so the paired difference isolates the
representation effect.
=======================================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binomtest

plt.rcParams.update({
    'font.size':       12,
    'axes.titlesize':  14,
    'axes.labelsize':  12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 16,
})

CONFIG = {
    "RESULTS_CSV": Path("./cp_ess/sweep_results.csv"),
    "OUTPUT_DIR":  Path("./cp_ess"),
    "TASK_LABEL":  "CP — gene essentiality",
}

TIERS  = ['pairwise', 'hypergraph', 'hb_graph']
LABELS = {'pairwise': 'Pairwise', 'hypergraph': 'Hypergraph',
          'hb_graph': 'HB-graph'}
COLOURS = {'pairwise': '#999999', 'hypergraph': '#87CEEB',
           'hb_graph': '#4682B4'}


def load() -> pd.DataFrame:
    df = pd.read_csv(CONFIG["RESULTS_CSV"])

    # Guard against duplicate rows from a resumed run.
    before = len(df)
    df = df.drop_duplicates(subset=['model', 'tier', 'split_index'],
                            keep='last')
    if len(df) < before:
        print(f"   Dropped {before - len(df)} duplicate rows from resumes.")

    n_fail = int(df['pr_auc'].isna().sum())
    if n_fail:
        print(f"   WARNING: {n_fail} failed cells (NaN PR-AUC).")

    print(f"   {df['model'].nunique()} models, "
          f"{df['split_index'].nunique()} splits, {len(df)} rows")
    return df


def to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, split), one column per tier. Paired by construction."""
    wide = df.pivot_table(index=['model', 'family', 'split_index'],
                          columns='tier', values='pr_auc').reset_index()
    wide = wide.dropna(subset=TIERS)
    wide['d_hyper_pair'] = wide['hypergraph'] - wide['pairwise']
    wide['d_hb_hyper']   = wide['hb_graph']   - wide['hypergraph']
    wide['d_hb_pair']    = wide['hb_graph']   - wide['pairwise']
    return wide


def sign_test(d: np.ndarray) -> dict:
    """One-sided sign test, H1: median difference > 0. Ties excluded."""
    d = d[~np.isnan(d)]
    wins, losses = int((d > 0).sum()), int((d < 0).sum())
    n = wins + losses
    p = binomtest(wins, n, 0.5, alternative='greater').pvalue if n else np.nan
    sd = d.std(ddof=1)
    return {'mean': d.mean(), 'std': sd,
            'dz': d.mean() / sd if sd > 0 else np.nan,
            'wins': wins, 'losses': losses, 'ties': int((d == 0).sum()),
            'p': p, 'n': len(d)}


def ci95(d: np.ndarray) -> tuple:
    d = d[~np.isnan(d)]
    se = d.std(ddof=1) / np.sqrt(len(d))
    return d.mean() - 1.96 * se, d.mean() + 1.96 * se


# =======================================================
# FIGURE 1 — three panels
# =======================================================
def figure_panels(wide: pd.DataFrame, out: Path):
    fig = plt.figure(figsize=(16, 5.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], wspace=0.28)

    # ---- Panel A: pooled PR-AUC by tier ----
    ax = fig.add_subplot(gs[0])
    lo = wide[TIERS].to_numpy().min()
    hi = wide[TIERS].to_numpy().max()
    bins = np.linspace(lo, hi, 40)
    for t in TIERS:
        ax.hist(wide[t], bins=bins, alpha=0.55, label=LABELS[t],
                color=COLOURS[t], edgecolor='none')
    for t in TIERS:
        ax.axvline(wide[t].mean(), color=COLOURS[t], lw=2, ls='--')
    ax.set_xlabel('PR-AUC')
    ax.set_ylabel('Count (model x split)')
    ax.set_title('A. PR-AUC by representation\n(pooled over models)')
    ax.legend(frameon=False)

    # ---- Panel B: paired per-split deltas ----
    ax = fig.add_subplot(gs[1])
    deltas = [('d_hyper_pair', 'Hypergraph - Pairwise', '#87CEEB'),
              ('d_hb_hyper',   'HB-graph - Hypergraph', '#4682B4')]
    allv = np.concatenate([wide[c].to_numpy() for c, _, _ in deltas])
    lim = np.nanmax(np.abs(allv)) * 1.05
    bins = np.linspace(-lim, lim, 45)
    for col, lab, colour in deltas:
        s = sign_test(wide[col].to_numpy())
        ax.hist(wide[col], bins=bins, alpha=0.6, color=colour,
                edgecolor='none',
                label=f"{lab}\n{s['wins']}/{s['losses']} W/L, p={s['p']:.1e}")
    ax.axvline(0, color='black', lw=1.5)
    ax.set_xlabel(r'$\Delta$ PR-AUC (paired, within split)')
    ax.set_ylabel('Count (model x split)')
    ax.set_title('B. Paired differences\n(model identity cancels)')
    ax.legend(frameon=False, fontsize=10)

    # ---- Panel C: forest plot, one row per model ----
    ax = fig.add_subplot(gs[2])
    rows = []
    for (model, family), g in wide.groupby(['model', 'family']):
        d = g['d_hb_pair'].to_numpy()
        lo_, hi_ = ci95(d)
        rows.append({'model': model, 'family': family,
                     'mean': d.mean(), 'lo': lo_, 'hi': hi_,
                     'p': sign_test(d)['p']})
    fr = pd.DataFrame(rows).sort_values(['family', 'mean'])

    y = np.arange(len(fr))
    fams = sorted(fr['family'].unique())
    cmap = dict(zip(fams, plt.cm.tab10(np.linspace(0, 1, max(len(fams), 2)))))
    for i, r in enumerate(fr.itertuples()):
        ax.plot([r.lo, r.hi], [i, i], color=cmap[r.family], lw=2.5)
        ax.plot(r.mean, i, 'o', color=cmap[r.family], ms=8)
    ax.axvline(0, color='black', lw=1.5)

    pooled = wide['d_hb_pair'].to_numpy()
    ax.axvline(pooled.mean(), color='grey', ls='--', lw=1.5)
    ax.set_yticks(y)
    ax.set_yticklabels(fr['model'].tolist())
    ax.set_xlabel(r'$\Delta$ PR-AUC (HB-graph - Pairwise)')
    ax.set_title('C. Effect by model\n(mean, 95% CI)')
    ax.margins(y=0.08)
    handles = [plt.Line2D([], [], color=cmap[f], lw=2.5, marker='o',
                          label=f) for f in fams]
    ax.legend(handles=handles, frameon=False, fontsize=10,
              loc='lower right')

    fig.suptitle(CONFIG["TASK_LABEL"], y=1.02)
    fig.savefig(out / 'fig_sweep_panels.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("   Saved: fig_sweep_panels.png")
    return fr


# =======================================================
# FIGURE 2 — small multiples
# =======================================================
def figure_by_model(wide: pd.DataFrame, out: Path):
    models = sorted(wide['model'].unique())
    ncol = 3
    nrow = int(np.ceil(len(models) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.4 * nrow),
                             sharex=True)
    axes = np.atleast_1d(axes).ravel()

    lo = wide[TIERS].to_numpy().min()
    hi = wide[TIERS].to_numpy().max()
    bins = np.linspace(lo, hi, 28)

    for ax, model in zip(axes, models):
        g = wide[wide['model'] == model]
        for t in TIERS:
            ax.hist(g[t], bins=bins, alpha=0.6, color=COLOURS[t],
                    edgecolor='none', label=LABELS[t])
            ax.axvline(g[t].mean(), color=COLOURS[t], lw=1.8, ls='--')
        d = sign_test(g['d_hb_pair'].to_numpy())
        ax.set_title(f"{model}  ({d['wins']}/{d['losses']} W/L)")
        ax.set_ylabel('Count')

    for ax in axes[len(models):]:
        ax.set_visible(False)
    # Label the lowest visible axis in each column, not the last n by index.
    for c in range(ncol):
        idxs = [i for i in range(len(models)) if i % ncol == c]
        if idxs:
            axes[idxs[-1]].set_xlabel('PR-AUC')
    axes[0].legend(frameon=False, fontsize=10)

    fig.suptitle(f'{CONFIG["TASK_LABEL"]} — PR-AUC by representation, '
                 f'per model', y=1.0)
    fig.tight_layout()
    fig.savefig(out / 'fig_sweep_by_model.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("   Saved: fig_sweep_by_model.png")


# =======================================================
# SUMMARY TABLE
# =======================================================
def summary_table(wide: pd.DataFrame, out: Path):
    rows = []
    for (model, family), g in wide.groupby(['model', 'family']):
        row = {'model': model, 'family': family, 'n_splits': len(g)}
        for t in TIERS:
            row[f'{t}_mean'] = g[t].mean()
            row[f'{t}_std']  = g[t].std(ddof=1)
        for col, tag in [('d_hyper_pair', 'H1'), ('d_hb_hyper', 'H2'),
                         ('d_hb_pair', 'total')]:
            s = sign_test(g[col].to_numpy())
            row[f'{tag}_mean_diff'] = s['mean']
            row[f'{tag}_dz']        = s['dz']
            row[f'{tag}_wins']      = s['wins']
            row[f'{tag}_losses']    = s['losses']
            row[f'{tag}_p']         = s['p']
        rows.append(row)

    summ = pd.DataFrame(rows).sort_values(['family', 'model'])
    summ.to_csv(out / 'sweep_summary.csv', index=False)
    print("   Saved: sweep_summary.csv")

    print(f"\n{'Model':<20} {'Family':<15} {'Pair':>7} {'Hyper':>7} "
          f"{'HB':>7} {'H1 W/L':>9} {'H2 W/L':>9} {'H3 W/L':>9}")
    print('-' * 92)
    for r in summ.itertuples():
        print(f"{r.model:<20} {r.family:<15} "
              f"{r.pairwise_mean:>7.4f} {r.hypergraph_mean:>7.4f} "
              f"{r.hb_graph_mean:>7.4f} "
              f"{r.H1_wins:>4}/{r.H1_losses:<4} "
              f"{r.H2_wins:>4}/{r.H2_losses:<4} "
              f"{r.total_wins:>4}/{r.total_losses:<4}")

    n_models = len(summ)
    h1_pos = int((summ['H1_mean_diff'] > 0).sum())
    h2_pos = int((summ['H2_mean_diff'] > 0).sum())
    h3_pos = int((summ['total_mean_diff'] > 0).sum())
    print(f"\n   H1 (hypergraph > pairwise)  positive in "
          f"{h1_pos}/{n_models} models")
    print(f"   H2 (hb-graph > hypergraph)  positive in "
          f"{h2_pos}/{n_models} models")
    print(f"   H3 (hb-graph > pairwise)    positive in "
          f"{h3_pos}/{n_models} models")
    return summ


def main():
    print("=" * 70)
    print("  MODEL-VARIANCE SWEEP — ANALYSIS")
    print("=" * 70)

    out = CONFIG["OUTPUT_DIR"]
    out.mkdir(parents=True, exist_ok=True)

    print("\n1. Loading results")
    df = load()
    wide = to_wide(df)
    print(f"   {len(wide)} complete (model, split) pairs")

    print("\n2. Figures")
    figure_panels(wide, out)
    figure_by_model(wide, out)

    print("\n3. Summary")
    summary_table(wide, out)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
