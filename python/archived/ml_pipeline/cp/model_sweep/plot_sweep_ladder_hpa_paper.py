"""
=======================================================================
MODEL-VARIANCE SWEEP  —  ladder figure
=======================================================================
Renders the summary table visually, as two panels:

  A. Slopegraph. One line per model across the three tiers. This is the
     three mean columns of the table, drawn. A rising line = the ordering
     holds for that model; a line that falls at the last step = it does
     not. Reading the whole panel answers "does the ordering depend on
     the learner?" without any statistics.

  B. Two-step effects with 95% CI. The ladder has TWO steps, and they
     are not equally supported, so they get separate markers:
        H1 = hypergraph - pairwise   (adding set structure)
        H2 = hb-graph  - hypergraph  (adding stoichiometry)
     Panel C of the old figure collapsed both into one number, which is
     why it was hard to read. Whiskers are 95% CI on the paired mean
     difference; W/L counts are printed at the right.

CIs are reconstructed from the summary alone: dz = mean_diff / sd_diff,
so sd_diff = mean_diff / dz, and se = sd_diff / sqrt(n_splits). No need
for the raw per-split file.

Input: a single sweep_summary.csv, written by
cp_hpa_analyse_model_sweep.py for the task named in TASK_LABEL.
=======================================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size':        12,
    'axes.titlesize':   14,
    'axes.labelsize':   13,
    'xtick.labelsize':  12,
    'ytick.labelsize':  12,
    'legend.fontsize':  12,
    'figure.titlesize': 16,
})

CONFIG = {
    "SUMMARY_FILE": Path("./cp_hpa/sweep_summary_paper.csv"),
    "OUTPUT_A":   Path("./cp_hpa/hpa_ladder_PR-AUC_paper.png"),
    "OUTPUT_B":   Path("./cp_hpa/hpa_ladder_deltas_paper.png"),
    "SHOW_TOTAL": True,
    "TASK_LABEL": "Complex Portal \u2014 drug-target status (HPA)",
    # Positive-class prevalence for this task — reported as a text note
    # on Panel A (axis is zoomed to the tier differences, so the actual
    # baseline sits below the visible range and isn't drawn as a line).
    "PREVALENCE": 0.079,
}

TIERS      = ['pairwise', 'hypergraph', 'hb_graph']
TIER_LABEL = ['Pairwise\n(4 features)', 'Hypergraph\n(8)', 'HB-graph\n(14)']


def load() -> pd.DataFrame:
    path = CONFIG["SUMMARY_FILE"]
    if not path.exists():
        raise SystemExit(f"Summary file not found: {path}")
    df = pd.read_csv(path)
    # Read downwards: best total gain at the top.
    return df.sort_values('total_mean_diff', ascending=False).reset_index(drop=True)


def ci95(mean_diff, dz, n):
    """95% CI on a paired mean difference, recovered from Cohen's dz."""
    dz = np.where(np.abs(dz) < 1e-12, np.nan, dz)
    sd = mean_diff / dz
    se = np.abs(sd) / np.sqrt(n)
    return 1.96 * se


def panel_a(ax, df, colours, prevalence=None):
    x = np.arange(3)
    ends = []
    for _, r in df.iterrows():
        y = [r['pairwise_mean'], r['hypergraph_mean'], r['hb_graph_mean']]
        falls = y[2] < y[1]
        ax.plot(x, y, marker='o', ms=7, lw=2.2, color=colours[r['model']],
                alpha=0.95, zorder=3,
                ls='--' if falls else '-')
        ends.append((y[2], r['model']))

    if prevalence is not None:
        ax.text(0.98, 0.02, f'Random baseline ({prevalence:.3f})',
                 transform=ax.transAxes, fontsize=10, color='0.4',
                 va='bottom', ha='right', style='italic')

    # Nudge end labels apart so near-identical finishers stay readable.
    span = (max(e for e, _ in ends) - min(e for e, _ in ends)) or 1.0
    gap  = 0.045 * span
    ends.sort()
    placed = []
    for val, name in ends:
        pos = val if not placed or val - placed[-1] >= gap else placed[-1] + gap
        placed.append(pos)
        ax.annotate(name, (x[2], pos), xytext=(8, 0),
                    textcoords='offset points', va='center',
                    fontsize=12, color=colours[name])
    ax.set_xticks(x)
    ax.set_xticklabels(TIER_LABEL)
    ax.set_xlim(-0.25, 2.95)
    ax.set_ylabel('PR-AUC (mean over 50 splits)')
    ax.grid(axis='y', alpha=0.25)
    ax.spines[['top', 'right']].set_visible(False)

def panel_b(ax, df):
    y = np.arange(len(df))[::-1]          # first row at the top
    # Total is NOT redundant: its point estimate is H1+H2, but its CI and
    # W/L are not derivable from the steps, since the two steps are
    # correlated within a split.
    spec = [
        ('H1',    +0.24, 'o', '#6ec6e8', 'hypergraph \u2212 pairwise'),
        ('H2',     0.00, 'o', '#8ccf7e', 'hb-graph \u2212 hypergraph'),
    ]
    if CONFIG["SHOW_TOTAL"]:
        spec.append(
            ('total', -0.24, 'D', '#5b2c6f', 'hb-graph \u2212 pairwise'))

    for key, dy, mk, col, lab in spec:
        m   = df[f'{key}_mean_diff'].to_numpy()
        err = ci95(m, df[f'{key}_dz'].to_numpy(), df['n_splits'].to_numpy())
        ax.errorbar(m, y + dy, xerr=err, fmt=mk, ms=7, lw=2.2,
                    capsize=3, color=col, label=lab, zorder=3)
        for yi, w, l in zip(y + dy, df[f'{key}_wins'], df[f'{key}_losses']):
            ax.annotate(f'{w}/{l}', (0.985, yi), xycoords=('axes fraction',
                        'data'), ha='right', va='center', fontsize=12,
                        color=col)
    # Faint separator between models, so the marker triples read as groups.
    for yi in y[:-1]:
        ax.axhline(yi - 0.5, color='0.85', lw=0.8, zorder=1)

    ax.axvline(0, color='black', lw=1.4, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(df['model'])
    ax.set_ylim(-0.6, len(df) - 0.4)
    ax.set_xlabel('\u0394 PR-AUC (paired, within split)')
    ax.legend(frameon=False, loc='upper center',
              bbox_to_anchor=(0.5, -0.07), ncol=3,
              columnspacing=1.8, handletextpad=0.4)
    ax.grid(axis='x', alpha=0.25)
    ax.spines[['top', 'right']].set_visible(False)
    # Headroom on the right for the W/L annotations.
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, hi + 0.35 * (hi - lo))


def main():
    df = load()
    cmap = plt.cm.tab10(np.linspace(0, 1, 10))
    colours = {m: cmap[i % 10] for i, m in enumerate(df['model'])}
    task = CONFIG["TASK_LABEL"]

    # --- Panel A, standalone ---
    figA, axA = plt.subplots(figsize=(8.0, 6.4))
    panel_a(axA, df, colours, prevalence=CONFIG["PREVALENCE"])
    axA.set_title(f'{task}\nPR-AUC by representation, per model')
    figA.tight_layout()
    figA.savefig(CONFIG["OUTPUT_A"], dpi=300, bbox_inches='tight')
    print(f"Saved: {CONFIG['OUTPUT_A']}")

    # --- Panel B, standalone. Height scales with the number of models so
    #     each marker triple keeps its vertical breathing room. ---
    figB, axB = plt.subplots(figsize=(9.5, 1.05 * len(df) + 2.4))
    panel_b(axB, df)
    axB.set_title(f'{task}\nPaired effect of each ladder step '
                  f'(mean, 95% CI)')
    figB.tight_layout()
    figB.savefig(CONFIG["OUTPUT_B"], dpi=300, bbox_inches='tight')
    print(f"Saved: {CONFIG['OUTPUT_B']}  ({len(df)} models)")
if __name__ == "__main__":
    main()