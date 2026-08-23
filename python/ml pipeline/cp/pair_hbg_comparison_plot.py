"""
Combined paired-comparison figure for the three prediction tasks.

Reads the `split_results.csv` written by each task's pipeline and draws one
landscape row of paired scatterplots (HB-graph vs pairwise PR-AUC, one point
per split), sized for a full-width (two-column) journal figure.

Run with no arguments; edit CONFIG below.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =======================================================
# CONFIGURATION
# =======================================================
CONFIG = {
    # --- Inputs: one entry per panel, left to right ---------------------
    "TASKS": [
        {"title": "Gene essentiality",
         "path": Path("./randomforest/cp_ess_second_testA/split_results.csv")},
        {"title": "Drug target (ChEMBL)",
         "path": Path("./randomforest/cp_chembl_second_testA/split_results.csv")},
        {"title": "Drug target (HPA)",
         "path": Path("./randomforest/cp_hpa_second_testA/split_results.csv")},
    ],

    # --- Which contrast to plot -----------------------------------------
    "X_COL":   "pairwise_pr_auc",
    "Y_COL":   "hb_graph_pr_auc",
    "X_LABEL": "Pairwise PR-AUC",
    "Y_LABEL": "HB-graph PR-AUC",

    # --- Random-classifier baseline --------------------------------------
    # For PR-AUC the no-skill baseline is the positive prevalence of the test
    # set, which the pipeline stores per split as 'test_ess_pct' (a percentage).
    # Drawn as faint horizontal and vertical lines at the mean prevalence.
    "SHOW_BASELINE":   True,
    "BASELINE_COL":    "test_ess_pct",
    "BASELINE_SCALE":  0.01,          # percentage -> proportion
    "BASELINE_COLOUR": "0.40",

    # --- Output ----------------------------------------------------------
    "OUTPUT_DIR":  Path("./figures"),
    "OUTPUT_STEM": "paired_comparison_three_tasks_shared",
    "FORMATS":     ["pdf", "png"],
    "DPI":         600,

    # --- Layout ----------------------------------------------------------
    # Drawn at the final printed size, so FONT_SIZE is the true point size on
    # the page. Bioinformatics double-column width is 180 mm (7.09 in). Include
    # at natural size:
    #   \includegraphics[width=\textwidth]{paired_comparison_three_tasks.pdf}
    #
    # FIG_HEIGHT_IN = "auto" is recommended. The panels are square (equal
    # aspect), so their size is set by the width available per panel, NOT by the
    # figure height -- adding height only adds white space above and below.
    # "auto" measures the drawn panel and shrinks the canvas to fit it exactly.
    "FIG_WIDTH_IN":  7.09,
    "FIG_HEIGHT_IN": "auto",     # or a number, e.g. 2.85
    "FONT_SIZE":     12,

    # --- Axis limits -----------------------------------------------------
    # "fixed"  : [0, 1] on every panel (matches the per-task figures)
    # "shared" : one square range covering all three tasks
    # "auto"   : per-panel square limits padded around that task's data
    "AXIS_MODE":   "shared",
    "AXIS_PAD":    0.05,
    # Ticks used in "fixed" mode. Five labels do not fit across a ~2 in panel at
    # 12 pt -- that is what produced the run-together 0.000.250.500.751.00 axis.
    "FIXED_TICKS": [0.0, 0.5, 1.0],
    # Share the y-axis so only the leftmost panel carries tick labels, giving
    # every panel more width (and so more height). "fixed"/"shared" modes only.
    "SHARE_Y":     True,

    # --- Cosmetics -------------------------------------------------------
    "POINT_COLOUR":    "steelblue",
    "POINT_SIZE":      40,
    "POINT_ALPHA":     0.7,
    "DIAGONAL_COLOUR": "red",
    "BOX_COLOUR":      "lightgreen",
    "SHOW_WIN_COUNTS": True,
    "SHOW_LEGEND":     True,
    # "below"  : one legend in a row underneath the panels (no data hidden)
    # "panel"  : inside LEGEND_PANEL at LEGEND_LOC
    "LEGEND_MODE":     "below",
    "LEGEND_PANEL":    1,             # "panel" mode only
    "LEGEND_LOC":      "upper left",  # "panel" mode only
    "PANEL_LETTERS":   True,
}


# =======================================================
# HELPERS
# =======================================================
def load_task(task: dict) -> pd.DataFrame:
    """Loads one task's split_results.csv and checks the required columns."""
    path = Path(task["path"])
    if not path.exists():
        raise FileNotFoundError(f"Missing results file for '{task['title']}': {path}")

    df = pd.read_csv(path)
    for col in (CONFIG["X_COL"], CONFIG["Y_COL"]):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in {path}. "
                           f"Available: {list(df.columns)}")
    return df


def square_limits(x: np.ndarray, y: np.ndarray, pad: float) -> tuple:
    """Square axis limits covering both arrays, with a proportional margin."""
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    span = hi - lo
    if span <= 0:
        span = max(abs(hi), 0.01)
    margin = pad * span
    return lo - margin, hi + margin


def add_legend_below(fig, axes, handles, labels):
    """Puts one legend in a row under the panels, growing the canvas to fit it.

    The panels keep their size: the figure is made taller by exactly the height
    of the legend and the axes are shifted up by the same amount.
    """
    fig_w, fig_h = fig.get_size_inches()
    inv = fig.dpi_scale_trans.inverted()

    leg = fig.legend(handles, labels, loc='lower center',
                     bbox_to_anchor=(0.5, 0.0), ncol=len(labels),
                     frameon=False, handlelength=1.8,
                     borderpad=0.2, columnspacing=1.6)

    fig.canvas.draw()
    leg_h = leg.get_window_extent().transformed(inv).height
    pad_in = 0.04

    # Record where the axes sit now, in inches from the bottom.
    boxes = [(ax, ax.get_position()) for ax in axes]

    new_h = fig_h + leg_h + pad_in
    fig.set_size_inches(fig_w, new_h)

    for ax, pos in boxes:
        y0_in = pos.y0 * fig_h + leg_h + pad_in
        h_in  = pos.height * fig_h
        ax.set_position([pos.x0, y0_in / new_h, pos.width, h_in / new_h])

    return new_h


def fit_height_to_panels(fig, axes, layout_kwargs):
    """Shrinks the canvas so the square panels fill it, leaving no dead space.

    With equal-aspect panels the axes size is set by the available width, so any
    extra figure height becomes margin rather than a bigger plot. This measures
    the vertical space taken by titles, labels and ticks, then resizes the
    figure to exactly panel height + that space.
    """
    fig_w = fig.get_size_inches()[0]
    inv = fig.dpi_scale_trans.inverted()
    pad_in = 2 * layout_kwargs.get("pad", 0.4) * plt.rcParams["font.size"] / 72

    def measure(ax):
        """Returns (panel size, vertical space used by title/labels/ticks)."""
        drawn = ax.get_window_extent().transformed(inv)           # the axes box
        tight = ax.get_tightbbox().transformed(inv)               # box + labels
        return drawn.width, drawn.height, tight.height - drawn.height

    # First pass at a deliberately generous height, so the panels are limited by
    # the available WIDTH. That width is the panel size we want to keep.
    fig.canvas.draw()
    panel_w, _, decoration_in = measure(axes[0])

    # Second pass: the canvas is now tight, so re-measure the decorations (a
    # title can wrap differently, ticks can re-label) and settle on that height.
    for _ in range(2):
        target_h = panel_w + decoration_in + pad_in
        fig.set_size_inches(fig_w, target_h)
        fig.tight_layout(**layout_kwargs)
        fig.canvas.draw()
        _, _, decoration_in = measure(axes[0])

    target_h = panel_w + decoration_in + pad_in
    fig.set_size_inches(fig_w, target_h)
    fig.tight_layout(**layout_kwargs)
    return target_h


# =======================================================
# PLOTTING
# =======================================================
def plot_three_tasks():
    plt.rcParams.update({
        'font.size':        CONFIG["FONT_SIZE"],
        'axes.titlesize':   CONFIG["FONT_SIZE"],
        'axes.labelsize':   CONFIG["FONT_SIZE"],
        'xtick.labelsize':  CONFIG["FONT_SIZE"],
        'ytick.labelsize':  CONFIG["FONT_SIZE"],
        'legend.fontsize':  CONFIG["FONT_SIZE"],
        'figure.titlesize': CONFIG["FONT_SIZE"],
        'axes.linewidth':   0.8,
        'pdf.fonttype':     42,
        'ps.fonttype':      42,
    })

    tasks  = CONFIG["TASKS"]
    frames = [load_task(t) for t in tasks]

    xs = [df[CONFIG["X_COL"]].to_numpy() for df in frames]
    ys = [df[CONFIG["Y_COL"]].to_numpy() for df in frames]

    # --- Random baseline per task (mean test-set positive prevalence) ---
    baselines = []
    for df in frames:
        col = CONFIG["BASELINE_COL"]
        if CONFIG["SHOW_BASELINE"] and col in df.columns:
            baselines.append(float(df[col].mean()) * CONFIG["BASELINE_SCALE"])
        else:
            baselines.append(None)

    # --- Axis limits ---
    mode = CONFIG["AXIS_MODE"]
    if mode == "fixed":
        limits = [(0.0, 1.0)] * len(tasks)
    elif mode == "shared":
        shared = square_limits(np.concatenate(xs), np.concatenate(ys),
                               CONFIG["AXIS_PAD"])
        limits = [shared] * len(tasks)
    else:
        limits = [square_limits(x, y, CONFIG["AXIS_PAD"]) for x, y in zip(xs, ys)]

    share_y = CONFIG["SHARE_Y"] and mode in ("fixed", "shared")

    trial_h = 4.0 if CONFIG["FIG_HEIGHT_IN"] == "auto" else CONFIG["FIG_HEIGHT_IN"]
    fig, axes = plt.subplots(
        1, len(tasks),
        figsize=(CONFIG["FIG_WIDTH_IN"], trial_h),
        sharey=share_y,
    )
    axes = np.atleast_1d(axes)

    letters = "abcdefgh"

    for i, (ax, task, x, y, (lo, hi), base) in enumerate(
            zip(axes, tasks, xs, ys, limits, baselines)):

        # Random-classifier baseline, behind everything else
        if base is not None and lo <= base <= hi:
            ax.axhline(base, color=CONFIG["BASELINE_COLOUR"], linestyle=':',
                       linewidth=1.2, zorder=1)
            ax.axvline(base, color=CONFIG["BASELINE_COLOUR"], linestyle=':',
                       linewidth=1.2, zorder=1,
                       label='random baseline' if i == 0 else None)

        # y = x reference line
        ax.plot([lo, hi], [lo, hi], linestyle='--', linewidth=1.6,
                color=CONFIG["DIAGONAL_COLOUR"], zorder=2,
                label='y = x' if i == 0 else None)

        ax.scatter(x, y,
                   s=CONFIG["POINT_SIZE"],
                   alpha=CONFIG["POINT_ALPHA"],
                   color=CONFIG["POINT_COLOUR"],
                   edgecolors='none',
                   zorder=3)

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal', adjustable='box')
        ax.tick_params(direction='out', length=3, width=0.8)

        if mode == "fixed":
            ax.set_xticks(CONFIG["FIXED_TICKS"])
            ax.set_yticks(CONFIG["FIXED_TICKS"])
        else:
            ax.locator_params(axis='both', nbins=4)

        title = task["title"]
        if CONFIG["PANEL_LETTERS"]:
            title = f"({letters[i]}) {title}"
        ax.set_title(title, pad=6)

        ax.set_xlabel(CONFIG["X_LABEL"])
        if i == 0:
            ax.set_ylabel(CONFIG["Y_LABEL"])

        if CONFIG["SHOW_WIN_COUNTS"]:
            above = int(np.sum(y > x))
            below = int(np.sum(y < x))
            ax.text(0.95, 0.05, f"{above}/{below} splits",
                    transform=ax.transAxes, ha='right', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor=CONFIG["BOX_COLOUR"], alpha=0.5,
                              edgecolor='0.6', linewidth=0.6))

    layout_kwargs = dict(pad=0.4, w_pad=0.8)
    fig.tight_layout(**layout_kwargs)
    if CONFIG["FIG_HEIGHT_IN"] == "auto":
        height = fit_height_to_panels(fig, axes, layout_kwargs)
    else:
        height = fig.get_size_inches()[1]

    # Legend last, so the panel geometry above is already settled
    if CONFIG["SHOW_LEGEND"]:
        handles, labels = axes[0].get_legend_handles_labels()
        if CONFIG["LEGEND_MODE"] == "below":
            height = add_legend_below(fig, axes, handles, labels)
        else:
            axes[CONFIG["LEGEND_PANEL"]].legend(
                handles, labels, loc=CONFIG["LEGEND_LOC"],
                frameon=True, framealpha=0.9,
                handlelength=1.6, borderpad=0.3, labelspacing=0.25)

    print(f"   Figure size: {CONFIG['FIG_WIDTH_IN']:.2f} x {height:.2f} in")

    out_dir = Path(CONFIG["OUTPUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in CONFIG["FORMATS"]:
        out_path = out_dir / f"{CONFIG['OUTPUT_STEM']}.{fmt}"
        # No bbox_inches='tight': saved at exactly the size drawn, so
        # \includegraphics[width=\textwidth] applies no rescaling and FONT_SIZE
        # is the true point size on the page.
        fig.savefig(out_path, dpi=CONFIG["DPI"])
        print(f"   Saved: {out_path}")

    # Baseline values, for the caption
    for task, base in zip(tasks, baselines):
        if base is not None:
            print(f"   Random baseline, {task['title']}: {base:.3f}")

    plt.close(fig)


if __name__ == "__main__":
    plot_three_tasks()