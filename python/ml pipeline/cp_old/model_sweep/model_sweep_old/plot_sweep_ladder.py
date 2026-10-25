"""
=======================================================================
MODEL-VARIANCE SWEEP  —  ladder figure
=======================================================================
Renders each task's summary table visually, as two panels:

  A. Slopegraph. One line per model across the three tiers. This is the
     three mean columns of the table, drawn. A rising line = the ordering
     holds for that model; a line that falls at the last step = it does
     not. Reading the whole panel answers "does the ordering depend on
     the learner?" without any statistics.

  B. Two-step effects with 95% CI. The ladder has TWO steps, and they
     are not equally supported, so they get separate markers:
        H1 = hypergraph - pairwise   (adding set structure)
        H2 = hb-graph  - hypergraph  (adding stoichiometry)
     Whiskers are 95% CI on the paired mean difference; W/L counts are
     printed at the right.

CIs are reconstructed from the summary alone: dz = mean_diff / sd_diff,
so sd_diff = mean_diff / dz, and se = sd_diff / sqrt(n_splits). No need
for the raw per-split file.

This replaces plot_sweep_ladder_ess.py / _hpa.py / _hpa_paper.py, which
were identical apart from which summary CSV they read and cosmetic
config. TASKS below holds the one thing that differs per task; run the
whole file to get every task's individual panels plus two combined
landscape figures (one row per task) for the paper.

MODEL_COLOURS is fixed globally, not assigned per-task by sort order,
so the same model keeps the same colour in every figure — the previous
per-task cmap[i % 10] assignment could give a model a different colour
in each task's plot, since panels are sorted by that task's total
effect size.
=======================================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size':        13,
    'axes.titlesize':   15,
    'axes.labelsize':   14,
    'xtick.labelsize':  13,
    'ytick.labelsize':  13,
    'legend.fontsize':  13,
    'figure.titlesize': 17,
})

# -----------------------------------------------------------------------
# One entry per task. Everything that differed between the three old
# scripts lives here; the plotting code below is shared.
# -----------------------------------------------------------------------
TASKS = {
    "essentiality": dict(
        summary_file=Path("./cp_ess/sweep_summary.csv"),
        task_label="Gene essentiality",
        prevalence=0.224,
    ),
    "chembl": dict(
        summary_file=Path("./cp_chembl/sweep_summary.csv"),
        task_label="Drug-target status (ChEMBL)",
        prevalence=0.060,
    ),
    "hpa": dict(
        summary_file=Path("./cp_hpa/sweep_summary.csv"),
        task_label="Drug-target status (HPA)",
        prevalence=0.079,
    ),
}
# Row order for the combined paper figures.
TASK_ORDER = ["essentiality", "chembl", "hpa"]

OUTPUT_DIR = Path("./figs/paper")

# Set False for the full SI sweep (all 7 models); True restricts every
# figure (individual and combined) to the four models kept in the main
# text, matching Table \ref{tab:main_results}.
PAPER_MODELS_ONLY = True
PAPER_MODELS = ["TabPFN", "RandomForest", "XGBoost", "LogisticRegression"]

# Fixed colour per model, shared across every task and every figure.
MODEL_COLOURS = {
    "TabPFN":             "#4C72B0",
    "RandomForest":       "#DD8452",
    "LightGBM":           "#55A868",
    "XGBoost":            "#C44E52",
    "RBF-SVM":            "#8172B2",
    "LogisticRegression": "#937860",
    "MLP":                "#DA8BC3",
}

TIERS      = ['pairwise', 'hypergraph', 'hb_graph']
TIER_LABEL = ['Pairwise\n(4 features)', 'Hypergraph\n(8)', 'HB-graph\n(14)']


# =======================================================================
# Data
# =======================================================================
def load(cfg: dict) -> pd.DataFrame:
    path = cfg["summary_file"]
    if not path.exists():
        raise SystemExit(f"Summary file not found: {path}")
    df = pd.read_csv(path)
    if PAPER_MODELS_ONLY:
        df = df[df['model'].isin(PAPER_MODELS)]
        df['model'] = pd.Categorical(df['model'], categories=PAPER_MODELS,
                                      ordered=True)
        df = df.sort_values('model')
    else:
        # Read downwards: best total gain at the top.
        df = df.sort_values('total_mean_diff', ascending=False)
    return df.reset_index(drop=True)


def ci95(mean_diff, dz, n):
    """95% CI on a paired mean difference, recovered from Cohen's dz."""
    dz = np.where(np.abs(dz) < 1e-12, np.nan, dz)
    sd = mean_diff / dz
    se = np.abs(sd) / np.sqrt(n)
    return 1.96 * se


# =======================================================================
# Panel A — slopegraph
# =======================================================================
def panel_a(ax, df, prevalence=None, label_fontsize=12):
    x = np.arange(3)
    ends = []
    for _, r in df.iterrows():
        y = [r['pairwise_mean'], r['hypergraph_mean'], r['hb_graph_mean']]
        falls = y[2] < y[1]
        ax.plot(x, y, marker='o', ms=7, lw=2.2,
                color=MODEL_COLOURS[r['model']],
                alpha=0.95, zorder=3,
                ls='--' if falls else '-')
        ends.append((y[2], r['model']))

    if prevalence is not None:
        ax.text(0.98, 0.02, f'Random baseline ({prevalence:.3f})',
                 transform=ax.transAxes, fontsize=label_fontsize - 1,
                 color='0.4', va='bottom', ha='right', style='italic')

    # Nudge end labels apart so near-identical finishers stay readable.
    span = (max(e for e, _ in ends) - min(e for e, _ in ends)) or 1.0
    gap  = 0.11 * span
    ends.sort()
    placed = []
    for val, name in ends:
        pos = val if not placed or val - placed[-1] >= gap else placed[-1] + gap
        placed.append(pos)
        ax.annotate(name, (x[2], pos), xytext=(8, 0),
                    textcoords='offset points', va='center',
                    fontsize=label_fontsize, color=MODEL_COLOURS[name])
    ax.set_xticks(x)
    ax.set_xticklabels(TIER_LABEL)
    ax.set_xlim(-0.25, 3.35)
    ax.set_ylabel('PR-AUC (mean over 50 splits)')
    ax.grid(axis='y', alpha=0.25)
    ax.spines[['top', 'right']].set_visible(False)


# =======================================================================
# Panel B — ladder effects with CIs
# =======================================================================
def panel_b(ax, df, show_legend=True, wl_fontsize=12):
    y = np.arange(len(df))[::-1]          # first row at the top
    # Total is NOT redundant: its point estimate is H1+H2, but its CI and
    # W/L are not derivable from the steps, since the two steps are
    # correlated within a split.
    spec = [
        ('H1',    +0.24, 'o', '#6ec6e8', 'hypergraph \u2212 pairwise'),
        ('H2',     0.00, 'o', '#8ccf7e', 'hb-graph \u2212 hypergraph'),
        ('total', -0.24, 'D', '#5b2c6f', 'hb-graph \u2212 pairwise'),
    ]

    for key, dy, mk, col, lab in spec:
        m   = df[f'{key}_mean_diff'].to_numpy()
        err = ci95(m, df[f'{key}_dz'].to_numpy(), df['n_splits'].to_numpy())
        ax.errorbar(m, y + dy, xerr=err, fmt=mk, ms=7, lw=2.2,
                    capsize=3, color=col, label=lab, zorder=3)
        for yi, w, l in zip(y + dy, df[f'{key}_wins'], df[f'{key}_losses']):
            ax.annotate(f'{w}/{l}', (0.985, yi), xycoords=('axes fraction',
                        'data'), ha='right', va='center',
                        fontsize=wl_fontsize, color=col)
    # Faint separator between models, so the marker triples read as groups.
    for yi in y[:-1]:
        ax.axhline(yi - 0.5, color='0.85', lw=0.8, zorder=1)

    ax.axvline(0, color='black', lw=1.4, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(df['model'])
    ax.set_ylim(-0.6, len(df) - 0.4)
    ax.set_xlabel('\u0394 PR-AUC (paired, within split)')
    if show_legend:
        ax.legend(frameon=False, loc='upper center',
                  bbox_to_anchor=(0.5, -0.14), ncol=3,
                  columnspacing=1.8, handletextpad=0.4)
    ax.grid(axis='x', alpha=0.25)
    ax.spines[['top', 'right']].set_visible(False)
    # Headroom on the right for the W/L annotations.
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, hi + 0.35 * (hi - lo))


# =======================================================================
# Per-task standalone figures (unchanged in spirit from the old scripts)
# =======================================================================
def make_individual_figures(name, cfg, df):
    task = cfg["task_label"]

    figA, axA = plt.subplots(figsize=(8.0, 6.4))
    panel_a(axA, df, prevalence=cfg["prevalence"])
    axA.set_title(f'{task}\nPR-AUC by representation, per model')
    figA.tight_layout()
    outA = OUTPUT_DIR / f"fig_{name}_ladder_A_ordering.png"
    figA.savefig(outA, dpi=300, bbox_inches='tight')
    plt.close(figA)
    print(f"Saved: {outA}")

    figB, axB = plt.subplots(figsize=(9.5, 1.05 * len(df) + 2.6))
    panel_b(axB, df)
    axB.set_title(f'{task}\nPaired effect of each ladder step '
                  f'(mean, 95% CI)')
    figB.tight_layout()
    outB = OUTPUT_DIR / f"fig_{name}_ladder_B_effects.png"
    figB.savefig(outB, dpi=300, bbox_inches='tight')
    plt.close(figB)
    print(f"Saved: {outB}  ({len(df)} models)")


# =======================================================================
# Combined 1x3 landscape figures for the paper
# =======================================================================
def make_combined_panel_a(dfs):
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.2), sharey=False)
    labels = ['(a) Gene essentiality', '(b) Drug target (ChEMBL)',
              '(c) Drug target (HPA)']
    for ax, name, lab in zip(axes, TASK_ORDER, labels):
        df = dfs[name]
        panel_a(ax, df, prevalence=TASKS[name]["prevalence"],
                label_fontsize=12)
        ax.set_title(lab, fontsize=15)
    # Only the leftmost panel needs the y-axis label.
    for ax in axes[1:]:
        ax.set_ylabel('')
    fig.tight_layout()
    out = OUTPUT_DIR / "fig_combined_ladder_A_ordering.png"
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


def make_combined_panel_b(dfs):
    fig, axes = plt.subplots(1, 3, figsize=(19, 1.05 * max(len(d) for d in
                              dfs.values()) + 3.0), sharex=False,
                              gridspec_kw={'wspace': 0.55})
    labels = ['(a) Gene essentiality', '(b) Drug target (ChEMBL)',
              '(c) Drug target (HPA)']
    for i, (ax, name, lab) in enumerate(zip(axes, TASK_ORDER, labels)):
        df = dfs[name]
        panel_b(ax, df, show_legend=(i == 1), wl_fontsize=11)
        ax.set_title(lab, fontsize=15)
    fig.tight_layout(rect=[0.02, 0.06, 0.98, 1])
    out = OUTPUT_DIR / "fig_combined_ladder_B_effects.png"
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dfs = {}
    for name, cfg in TASKS.items():
        df = load(cfg)
        dfs[name] = df
        make_individual_figures(name, cfg, df)

    make_combined_panel_a(dfs)
    make_combined_panel_b(dfs)


if __name__ == "__main__":
    main()