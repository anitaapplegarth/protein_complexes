"""
Redraws the noise-injection seed-variability figure from a saved
`noise_injection_records_*.csv`. No models are refitted -- everything the plot
needs is already in the CSV, so this runs in seconds.

One figure is written per task listed in CONFIG["TASKS"].
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# =======================================================
# CONFIGURATION
# =======================================================
CONFIG = {
    # --- Inputs: one figure per entry -----------------------------------
    "TASKS": [
        # {"key": "hpa",
        #  "display": "Drug target prediction (HPA)",
        #  "csv": Path("./randomforest/noise_injection/randomforest_hpa_noise_injection/noise_injection_records.csv")},
        # {"key": "ess",
        #  "display": "Gene essentiality",
        #  "csv": Path("./randomforest/noise_injection/randomforest_ess_noise_injection/noise_injection_records_ess.csv")},
        {"key": "chembl",
         "display": "Drug target prediction (ChEMBL)",
         "csv": Path("./randomforest/noise_injection/randomforest_chembl_noise_injection/noise_injection_records_chembl.csv")},
    ],

    # --- Output ----------------------------------------------------------
    "OUTPUT_DIR":  Path("./figures"),
    "OUTPUT_STEM": "noise_seed_variability",   # task key is appended
    "FORMATS":     ["pdf", "png"],
    "DPI":         600,

    # --- Layout ----------------------------------------------------------
    # "horizontal" puts features on the y-axis: long names read left to right
    # at full size with no rotation, which is the real fix for the unreadable
    # labels. "vertical" reproduces the original rotated-label layout.
    "ORIENTATION": "horizontal",

    # Drawn at final printed size, so FONT_SIZE is the true point size on the
    # page. 7.09 in = 180 mm = Bioinformatics double-column width.
    # Include with \includegraphics[width=\textwidth]{...}.
    "FIG_WIDTH_IN":   7.09,
    "ROW_HEIGHT_IN":  0.26,   # vertical space per feature (horizontal mode)
    "PANEL_HEIGHT_IN": 3.2,   # panel height (vertical mode only)
    "FONT_SIZE":      12,

    # --- Panels ----------------------------------------------------------
    # (column prefix in the CSV, panel title, box colour)
    "PANELS": [
        ("pair",  "(a) Pairwise + noise",  "#4393c3"),
        ("hyper", "(b) Hb-graph + noise",  "#d6604d"),
    ],
    "NOISE_PREFIX":  "rand_",
    "NOISE_COLOUR":  "#cccccc",
    "COLOUR_LABELS": True,    # tint tick labels to match their boxes

    # --- Title -----------------------------------------------------------
    # A figure title duplicates the LaTeX caption; off by default for the
    # manuscript, on for slides or a quick look.
    "SHOW_SUPTITLE": True,

    # --- Feature name display --------------------------------------------
    # Underscores render awkwardly at 12 pt; replace them with thin spaces.
    "TIDY_NAMES": True,
    # Any manual overrides, e.g. {"base_AvgNeighbourDegree": "avg. neighbour degree"}
    "RENAME": {},
}


# =======================================================
# HELPERS
# =======================================================
def tidy(name: str) -> str:
    """Cosmetic tidy-up of a feature name for the axis."""
    if name in CONFIG["RENAME"]:
        return CONFIG["RENAME"][name]
    if CONFIG["TIDY_NAMES"]:
        return name.replace("_", " ")
    return name


def collect_entries(df: pd.DataFrame, prefix: str) -> list:
    """Pulls every importance column for one model, sorted by mean descending.

    Real and noise features are ranked together, so a noise feature that beats a
    real one shows up in its honest position.
    """
    tag = prefix + "__"
    cols = [c for c in df.columns if c.startswith(tag)]
    if not cols:
        raise KeyError(f"No columns starting with '{tag}' in the CSV.")

    entries = []
    for col in cols:
        feature = col[len(tag):]
        vals = df[col].dropna().to_numpy()
        entries.append({
            "feature":  feature,
            "vals":     vals,
            "mean":     float(np.mean(vals)) if vals.size else 0.0,
            "is_noise": feature.startswith(CONFIG["NOISE_PREFIX"]),
        })
    entries.sort(key=lambda e: e["mean"], reverse=True)
    return entries


def draw_panel(ax, entries, colour, title, horizontal: bool):
    data    = [e["vals"] for e in entries]
    labels  = [tidy(e["feature"]) for e in entries]
    colours = [CONFIG["NOISE_COLOUR"] if e["is_noise"] else colour for e in entries]

    positions = np.arange(1, len(entries) + 1)
    bp = ax.boxplot(
        data,
        positions=positions,
        vert=not horizontal,
        patch_artist=True,
        widths=0.65,
        medianprops=dict(color="black", linewidth=1.4),
        flierprops=dict(marker="o", markersize=1.8, markerfacecolor="none",
                        markeredgewidth=0.5, markeredgecolor="0.4"),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
        boxprops=dict(linewidth=0.8),
    )
    for patch, c in zip(bp["boxes"], colours):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)

    importance_label = "Permutation importance (PR-AUC drop)"

    if horizontal:
        ax.set_yticks(positions)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()                      # best feature at the top
        ax.set_xlabel(importance_label)
        ax.axvline(0, color="black", linewidth=0.8, linestyle=":")
        ax.set_ylim(len(entries) + 0.6, 0.4)
        tick_labels = ax.get_yticklabels()
    else:
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel(importance_label)
        ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
        tick_labels = ax.get_xticklabels()

    if CONFIG["COLOUR_LABELS"]:
        for tick_label, c in zip(tick_labels, colours):
            tick_label.set_color("0.45" if c == CONFIG["NOISE_COLOUR"] else c)

    ax.set_title(title, loc="left", pad=6)
    ax.tick_params(direction="out", length=3, width=0.8)

    real_patch  = mpatches.Patch(facecolor=colour, alpha=0.75,
                                 edgecolor="0.3", linewidth=0.8,
                                 label="Real features")
    noise_patch = mpatches.Patch(facecolor=CONFIG["NOISE_COLOUR"], alpha=0.75,
                                 edgecolor="0.3", linewidth=0.8,
                                 label="Noise features")
    ax.legend(handles=[real_patch, noise_patch],
              loc="lower right" if horizontal else "upper right",
              frameon=True, framealpha=0.9, borderpad=0.3, labelspacing=0.25)


# =======================================================
# MAIN
# =======================================================
def replot_task(task: dict):
    csv_path = Path(task["csv"])
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing records file: {csv_path}")

    df = pd.read_csv(csv_path)
    n_splits = df["split_index"].nunique() if "split_index" in df else None
    n_seeds  = df["seed_index"].nunique() if "seed_index" in df else None

    panels = [(prefix, title, colour, collect_entries(df, prefix))
              for prefix, title, colour in CONFIG["PANELS"]]

    horizontal = CONFIG["ORIENTATION"] == "horizontal"

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

    if horizontal:
        # Panels sized in proportion to how many features they hold, so the
        # boxes are the same thickness in both.
        counts  = [len(entries) for *_, entries in panels]
        heights = [c * CONFIG["ROW_HEIGHT_IN"] for c in counts]
        fig_h   = sum(heights) + 1.4          # titles, x-labels, padding
        fig, axes = plt.subplots(
            len(panels), 1,
            figsize=(CONFIG["FIG_WIDTH_IN"], fig_h),
            gridspec_kw={"height_ratios": counts},
        )
    else:
        fig_h = len(panels) * CONFIG["PANEL_HEIGHT_IN"] + 1.8
        fig, axes = plt.subplots(
            len(panels), 1,
            figsize=(CONFIG["FIG_WIDTH_IN"], fig_h),
        )

    axes = np.atleast_1d(axes)

    for ax, (prefix, title, colour, entries) in zip(axes, panels):
        draw_panel(ax, entries, colour, title, horizontal)

    if CONFIG["SHOW_SUPTITLE"]:
        n_noise = sum(e["is_noise"] for e in panels[0][3])
        fig.suptitle(f'Seed-to-seed variability — {task["display"]}\n'
                     f'({n_noise} noise features x {n_seeds} seeds)')

    fig.tight_layout(pad=0.4, h_pad=1.2)

    out_dir = Path(CONFIG["OUTPUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in CONFIG["FORMATS"]:
        out_path = out_dir / f"{CONFIG['OUTPUT_STEM']}_{task['key']}.{fmt}"
        fig.savefig(out_path, dpi=CONFIG["DPI"])
        print(f"   Saved: {out_path}")
    plt.close(fig)

    # --- Summary for the text: does each real feature beat the noise? ---
    print(f"   {task['display']}: {n_splits} splits x {n_seeds} seeds "
          f"= {len(df)} records")
    for prefix, title, _colour, entries in panels:
        noise_means = [e["mean"] for e in entries if e["is_noise"]]
        ceiling = max(noise_means) if noise_means else 0.0
        below = [e["feature"] for e in entries
                 if not e["is_noise"] and e["mean"] <= ceiling]
        print(f"     {prefix}: max noise mean = {ceiling:+.4f}; "
              f"real features at or below it: {below if below else 'none'}")


def main():
    for task in CONFIG["TASKS"]:
        replot_task(task)


if __name__ == "__main__":
    main()