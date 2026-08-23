# """
# Feature-importance heatmaps from the saved *_hb_graph_feature_importance.csv
# files. One figure per database: Complex Portal for the main paper, CORUM for
# the Supplementary Information.

# Cells are coloured by importance relative to the strongest feature in that task
# (row maximum = 1), because absolute permutation importance scales with the
# model's PR-AUC and is therefore not comparable across tasks. The printed number
# in each cell is the raw mean importance.

# Run with no arguments; edit CONFIG below.
# """

# from pathlib import Path

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap

# # =======================================================
# # CONFIGURATION
# # =======================================================
# CONFIG = {
#     # --- Inputs: one figure per database --------------------------------
#     # Paths below are resolved against INPUT_DIR unless they are already
#     # absolute, so point INPUT_DIR at wherever the pipeline wrote its results
#     # and leave the filenames alone. Each task path can also be given in full
#     # (e.g. Path("/Users/anita/.../randomforest/cp_hpa/hb_graph_feature_importance.csv"))
#     # if the six files live in different run directories.
#     "INPUT_DIR": Path("./randomforest"),

#     "DATABASES": [
#         {"key": "cp",
#          "display": "Complex Portal",
#          "tasks": [
#              ("Essentiality",  Path("cp_ess_second_testA/cp_ess_hb_graph_feature_importance.csv")),
#              ("HPA",           Path("cp_hpa_second_testA/cp_hpa_hb_graph_feature_importance.csv")),
#              ("ChEMBL",        Path("cp_chembl_second_testA/cp_chembl_hb_graph_feature_importance.csv")),
#          ]},
#         {"key": "corum",
#          "display": "CORUM",
#          "tasks": [
#              ("Essentiality",  Path("../../corum/randomforest/corum_ess_second_testA/corum_ess_hb_graph_feature_importance.csv")),
#              ("HPA",           Path("../../corum/randomforest/corum_hpa_second_testA/corum_hpa_hb_graph_feature_importance.csv")),
#              ("ChEMBL",        Path("../../corum/randomforest/corum_chembl_second_testA/corum_chembl_hb_graph_feature_importance.csv")),
#          ]},
#     ],


#     # --- What to produce --------------------------------------------------
#     # COMBINED: both databases in one figure, sharing one set of feature
#     #           labels (for a single LaTeX float with one caption).
#     # SEPARATE: one figure per database, as before.
#     "COMBINED": True,
#     "SEPARATE": True,

#     # --- Output ----------------------------------------------------------
#     "OUTPUT_DIR":  Path("./figures"),
#     "OUTPUT_STEM": "feature_importance_heatmap",   # database key is appended
#     "FORMATS":     ["pdf", "png"],
#     "DPI":         600,

#     # --- Layout ----------------------------------------------------------
#     # 7.09 in = 180 mm = Bioinformatics double-column width. Drawn at final
#     # size, so FONT_SIZE is the true point size on the page.
#     "FIG_WIDTH_IN": 7.09,
#     "CELL_HEIGHT_IN": 0.50,
#     "FONT_SIZE":    12,

#     # --- Feature grouping ------------------------------------------------
#     # Fixed order in both figures so CP and CORUM can be read side by side.
#     # Groups are separated by a gap and labelled above the columns.
#     "GROUPS": [
#         ("Membership", [
#             "base_Degree",
#             "base_UniquePartners",
#             "base_TriangleCount",
#             "base_AvgNeighbourDegree",
#             "base_LocalClustCoeff",
#             "protein_MedComplexNodes",
#             "protein_RangeComplexNodes",
#             "protein_RangeUniqueRatio",
#         ]),
#         ("Stoichiometry", [
#             "stoich_MedianRatio",
#             "stoich_MedComplexSize",
#             "stoich_AvgNeighbourDegreeStoich",
#             "stoich_WeightedTriangles",
#             "stoich_RangeComplexSize",
#             "stoich_RangeRatio",
#         ]),
#     ],

#     # --- Colour ----------------------------------------------------------
#     # Normalisation: "row" scales each task by its own strongest feature,
#     # "global" uses one scale across the whole figure.
#     "NORMALISE":  "row",
#     "PALETTE":    ["#f7f7f5", "#a8ccc4", "#3d7f77", "#1b3b3a"],  # muted teal
#     "ANNOTATE":   False,   # colour only; exact means live in the SI table
#     "ANNOT_FMT":  "{:.3f}",
#     "SHOW_CBAR":  True,
#     "CBAR_LABEL": "Relative importance",

#     # --- Names -----------------------------------------------------------
#     # Prefixes are shown by the group headings, so they are stripped here.
#     "STRIP_PREFIXES": True,
#     "RENAME": {
#         "base_Degree":                     "Degree",
#         "base_UniquePartners":             "Unique partners",
#         "base_TriangleCount":              "Triangle count",
#         "base_AvgNeighbourDegree":         "Avg. neighbour degree",
#         "base_LocalClustCoeff":            "Local clust. coeff.",
#         "protein_MedComplexNodes":         "Median complex size",
#         "protein_RangeComplexNodes":       "Range complex size",
#         "protein_RangeUniqueRatio":        "Range unique ratio",
#         "stoich_MedianRatio":              "Median ratio",
#         "stoich_MedComplexSize":           "Median total stoich.",
#         "stoich_AvgNeighbourDegreeStoich": "Avg. neighbour stoich.",
#         "stoich_WeightedTriangles":        "Weighted triangles",
#         "stoich_RangeComplexSize":         "Range total stoich.",
#         "stoich_RangeRatio":               "Range ratio",
#     },
# }


# # =======================================================
# # HELPERS
# # =======================================================
# def resolve(path) -> Path:
#     """Absolute paths are used as given; relative ones sit under INPUT_DIR."""
#     path = Path(path)
#     if path.is_absolute():
#         return path
#     return Path(CONFIG["INPUT_DIR"]) / path


# def label_for(feature: str) -> str:
#     if feature in CONFIG["RENAME"]:
#         return CONFIG["RENAME"][feature]
#     if CONFIG["STRIP_PREFIXES"]:
#         return feature.split("_", 1)[-1].replace("_", " ")
#     return feature


# def build_matrix(db: dict):
#     """Returns (raw means, task labels, ordered features, group spans)."""
#     features, spans, cursor = [], [], 0
#     for group_name, group_features in CONFIG["GROUPS"]:
#         features.extend(group_features)
#         spans.append((group_name, cursor, cursor + len(group_features) - 1))
#         cursor += len(group_features)

#     rows, task_labels = [], []
#     for task_label, path in db["tasks"]:
#         path = resolve(path)
#         if not path.exists():
#             raise FileNotFoundError(f"Missing importance file: {path}")
#         df = pd.read_csv(path).set_index("feature")

#         missing = [f for f in features if f not in df.index]
#         if missing:
#             raise KeyError(f"{path.name} has no rows for: {missing}")
#         extra = [f for f in df.index if f not in features]
#         if extra:
#             print(f"   Note: {path.name} contains features not in CONFIG['GROUPS'] "
#                   f"and they are not plotted: {extra}")

#         rows.append([float(df.loc[f, "mean"]) for f in features])
#         task_labels.append(task_label)

#     return np.array(rows), task_labels, features, spans


# def normalise(raw: np.ndarray) -> np.ndarray:
#     """Scales importances to [0, 1] for colouring; negatives clip to zero."""
#     clipped = np.clip(raw, 0.0, None)
#     if CONFIG["NORMALISE"] == "row":
#         denom = clipped.max(axis=1, keepdims=True)
#     else:
#         denom = np.full((clipped.shape[0], 1), clipped.max())
#     denom[denom == 0] = 1.0
#     return clipped / denom


# # =======================================================
# # PLOTTING
# # =======================================================
# def draw_heatmap(ax, raw, shaded, row_labels, features, spans, cmap,
#                  show_group_headings=True, show_xlabels=True):
#     """Draws one heatmap block onto an existing axes."""
#     n_rows, n_cols = raw.shape

#     mesh = ax.imshow(shaded, cmap=cmap, vmin=0, vmax=1, aspect="auto")

#     ax.set_xticks(np.arange(n_cols))
#     if show_xlabels:
#         ax.set_xticklabels([label_for(f) for f in features],
#                            rotation=45, ha="right", rotation_mode="anchor")
#     else:
#         ax.set_xticklabels([])
#     ax.set_yticks(np.arange(n_rows))
#     ax.set_yticklabels(row_labels)

#     # Thin white gridlines instead of cell borders: cleaner at small sizes
#     ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
#     ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
#     ax.grid(which="minor", color="white", linewidth=1.5)
#     ax.tick_params(which="minor", length=0)
#     ax.tick_params(which="major", length=0)
#     for spine in ax.spines.values():
#         spine.set_visible(False)

#     if CONFIG["ANNOTATE"]:
#         for i in range(n_rows):
#             for j in range(n_cols):
#                 value = raw[i, j]
#                 text = CONFIG["ANNOT_FMT"].format(value)
#                 if value < 0:
#                     text = "\u2212" + text.lstrip("-")   # proper minus sign
#                 ax.text(j, i, text, ha="center", va="center",
#                         fontsize=CONFIG["FONT_SIZE"] - 2,
#                         color="white" if shaded[i, j] > 0.55 else "0.15")

#     # Group separator, and headings above the top block only
#     for group_name, start, end in spans:
#         if show_group_headings:
#             ax.text((start + end) / 2, -0.85, group_name,
#                     ha="center", va="bottom", fontsize=CONFIG["FONT_SIZE"],
#                     fontstyle="italic")
#         if end + 1 < n_cols:
#             ax.axvline(end + 0.5, color="0.25", linewidth=1.2)

#     ax.set_ylim(n_rows - 0.5, -0.6 if show_group_headings else -0.5)
#     return mesh


# def add_colourbar(fig, mesh, ax):
#     cbar = fig.colorbar(mesh, ax=ax, orientation="vertical",
#                         fraction=0.025, pad=0.02, aspect=12)
#     cbar.set_label(CONFIG["CBAR_LABEL"], fontsize=CONFIG["FONT_SIZE"] - 1)
#     cbar.set_ticks([0, 0.5, 1.0])
#     cbar.outline.set_visible(False)
#     cbar.ax.tick_params(length=3, width=0.8,
#                         labelsize=CONFIG["FONT_SIZE"] - 1)


# def report_top_features(db_display, task_labels, raw, features):
#     for label, row in zip(task_labels, raw):
#         order = np.argsort(row)[::-1]
#         top = ", ".join(f"{label_for(features[k])} ({row[k]:+.3f})"
#                         for k in order[:3])
#         print(f"   {db_display} / {label}: {top}")


# def save(fig, suffix):
#     out_dir = Path(CONFIG["OUTPUT_DIR"])
#     out_dir.mkdir(parents=True, exist_ok=True)
#     for fmt in CONFIG["FORMATS"]:
#         out_path = out_dir / f"{CONFIG['OUTPUT_STEM']}_{suffix}.{fmt}"
#         fig.savefig(out_path, dpi=CONFIG["DPI"])
#         print(f"   Saved: {out_path}")
#     plt.close(fig)


# def set_rc():
#     plt.rcParams.update({
#         'font.size':       CONFIG["FONT_SIZE"],
#         'axes.titlesize':  CONFIG["FONT_SIZE"],
#         'axes.labelsize':  CONFIG["FONT_SIZE"],
#         'xtick.labelsize': CONFIG["FONT_SIZE"],
#         'ytick.labelsize': CONFIG["FONT_SIZE"],
#         'pdf.fonttype':    42,
#         'ps.fonttype':     42,
#     })


# def make_cmap():
#     return LinearSegmentedColormap.from_list("importance", CONFIG["PALETTE"])


# def plot_combined():
#     """Both databases in one float: the fourteen column labels appear once.

#     Stacking the two separate figures would repeat the rotated feature labels,
#     which is most of the vertical space. Sharing them puts the CORUM
#     stoichiometry block directly beneath the Complex Portal one, which is the
#     comparison the figure exists to make.
#     """
#     set_rc()
#     cmap = make_cmap()

#     blocks = []
#     for db in CONFIG["DATABASES"]:
#         raw, task_labels, features, spans = build_matrix(db)
#         blocks.append((db, raw, normalise(raw), task_labels, features, spans))

#     n_rows = sum(b[1].shape[0] for b in blocks)

#     # Explicit margins in inches: tight_layout cannot place a colourbar that
#     # spans several axes without stealing width from them.
#     W = CONFIG["FIG_WIDTH_IN"]
#     left_in, right_in, top_in, bottom_in = 1.65, 0.95, 0.45, 1.50
#     gap_in = 0.45
#     fig_h = (n_rows * CONFIG["CELL_HEIGHT_IN"] + top_in + bottom_in
#              + gap_in * (len(blocks) - 1))

#     fig, axes = plt.subplots(
#         len(blocks), 1,
#         figsize=(W, fig_h),
#         gridspec_kw={"height_ratios": [b[1].shape[0] for b in blocks]},
#     )
#     axes = np.atleast_1d(axes)

#     fig.subplots_adjust(
#         left=left_in / W, right=1 - right_in / W,
#         top=1 - top_in / fig_h, bottom=bottom_in / fig_h,
#         hspace=gap_in / (n_rows * CONFIG["CELL_HEIGHT_IN"] / len(blocks)),
#     )

#     mesh = None
#     for i, (ax, (db, raw, shaded, task_labels, features, spans)) in enumerate(
#             zip(axes, blocks)):
#         is_first = (i == 0)
#         is_last = (i == len(blocks) - 1)
#         mesh = draw_heatmap(ax, raw, shaded, task_labels, features, spans, cmap,
#                             show_group_headings=is_first,
#                             show_xlabels=is_last)
#         # Database name to the left of the task labels: a title here would
#         # collide with the group headings above the first block.
#         ax.set_ylabel(db["display"], fontsize=CONFIG["FONT_SIZE"], labelpad=10)

#     if CONFIG["SHOW_CBAR"]:
#         plot_bottom = bottom_in / fig_h
#         plot_top = 1 - top_in / fig_h
#         cax = fig.add_axes([
#             1 - (right_in - 0.25) / W,
#             plot_bottom + 0.25 * (plot_top - plot_bottom),
#             0.16 / W,
#             0.5 * (plot_top - plot_bottom),
#         ])
#         cbar = fig.colorbar(mesh, cax=cax)
#         cbar.set_label(CONFIG["CBAR_LABEL"], fontsize=CONFIG["FONT_SIZE"] - 1)
#         cbar.set_ticks([0, 0.5, 1.0])
#         cbar.outline.set_visible(False)
#         cbar.ax.tick_params(length=3, width=0.8,
#                             labelsize=CONFIG["FONT_SIZE"] - 1)

#     save(fig, "combined")

#     for db, raw, _shaded, task_labels, features, _spans in blocks:
#         report_top_features(db["display"], task_labels, raw, features)


# def plot_database(db: dict):
#     raw, task_labels, features, spans = build_matrix(db)
#     shaded = normalise(raw)
#     n_rows, n_cols = raw.shape

#     set_rc()
#     cmap = make_cmap()

#     fig_h = n_rows * CONFIG["CELL_HEIGHT_IN"] + 1.55
#     fig, ax = plt.subplots(figsize=(CONFIG["FIG_WIDTH_IN"], fig_h))

#     mesh = draw_heatmap(ax, raw, shaded, task_labels, features, spans, cmap)

#     if CONFIG["SHOW_CBAR"]:
#         add_colourbar(fig, mesh, ax)

#     fig.tight_layout(pad=0.4)
#     save(fig, db["key"])
#     report_top_features(db["display"], task_labels, raw, features)


# def main():
#     if CONFIG["COMBINED"]:
#         plot_combined()
#     if CONFIG["SEPARATE"]:
#         for db in CONFIG["DATABASES"]:
#             plot_database(db)


# if __name__ == "__main__":
#     main()

"""
Feature-importance heatmaps from the saved *_hb_graph_feature_importance.csv
files. One figure per database: Complex Portal for the main paper, CORUM for
the Supplementary Information.

Cells are coloured by importance relative to the strongest feature in that task
(row maximum = 1), because absolute permutation importance scales with the
model's PR-AUC and is therefore not comparable across tasks. The printed number
in each cell is the raw mean importance.

Run with no arguments; edit CONFIG below.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# =======================================================
# CONFIGURATION
# =======================================================
CONFIG = {
    # --- Inputs -----------------------------------------------------------
    # Each file path is built as:
    #     INPUT_DIR / <database "dir"> / FILENAME_TEMPLATE
    # with {db}, {task} and {model} substituted. The default template includes
    # the per-run subdirectory, so it resolves to e.g.
    #     ./randomforest/cp_hpa_second_testA/cp_hpa_hb_graph_feature_importance.csv
    # A database "dir" may be relative (as CORUM is here) or absolute, in which
    # case INPUT_DIR is ignored for that database.
    "INPUT_DIR": Path("./randomforest"),
    "FILENAME_TEMPLATE":
        "{db}_{task}_second_testA/{db}_{task}_{model}_feature_importance.csv",

    # Which model's importances to plot: "hb_graph" or "pairwise". Importances
    # are measured within a fitted model, so the two cannot share a figure --
    # run the script once per model.
    "MODEL": "pairwise",

    "DATABASES": [
        {"key": "cp",    "display": "Complex Portal", "dir": Path(".")},
        {"key": "corum", "display": "CORUM",
         "dir": Path("../../corum/randomforest")},
    ],
    # (display label, filename token)
    "TASKS": [
        ("Essentiality", "ess"),
        ("HPA",          "hpa"),
        ("ChEMBL",       "chembl"),
    ],

    # --- What to produce --------------------------------------------------
    # COMBINED: both databases in one figure, sharing one set of feature
    #           labels (for a single LaTeX float with one caption).
    # SEPARATE: one figure per database, as before.
    "COMBINED": True,
    "SEPARATE": True,

    # --- Output ----------------------------------------------------------
    "OUTPUT_DIR":  Path("./figures"),
    "OUTPUT_STEM": "feature_importance_heatmap",   # model and database are appended
    "FORMATS":     ["pdf", "png"],
    "DPI":         600,

    # --- Layout ----------------------------------------------------------
    # 7.09 in = 180 mm = Bioinformatics double-column width. Drawn at final
    # size, so FONT_SIZE is the true point size on the page.
    "FIG_WIDTH_IN": 7.09,
    "CELL_HEIGHT_IN": 0.50,
    "FONT_SIZE":    12,

    # --- Feature grouping ------------------------------------------------
    # Fixed order so CP and CORUM can be read against each other. Groups are
    # separated by a rule and labelled above the columns. One entry per model;
    # the pairwise model has a single group, so its heading is suppressed.
    "GROUPS_BY_MODEL": {
        "hb_graph": [
            ("Membership", [
                "base_Degree",
                "base_UniquePartners",
                "base_TriangleCount",
                "base_AvgNeighbourDegree",
                "base_LocalClustCoeff",
                "protein_MedComplexNodes",
                "protein_RangeComplexNodes",
                "protein_RangeUniqueRatio",
            ]),
            ("Stoichiometry", [
                "stoich_MedianRatio",
                "stoich_MedComplexSize",
                "stoich_AvgNeighbourDegreeStoich",
                "stoich_WeightedTriangles",
                "stoich_RangeComplexSize",
                "stoich_RangeRatio",
            ]),
        ],
        "pairwise": [
            ("", [
                "pair_Degree",
                "pair_TriangleCount",
                "pair_AvgNeighborDegree",
                "pair_LocalClustCoeff",
            ]),
        ],
    },

    # --- Colour ----------------------------------------------------------
    # Normalisation: "row" scales each task by its own strongest feature,
    # "global" uses one scale across the whole figure.
    "NORMALISE":  "row",
    "PALETTE":    ["#f7f7f5", "#a8ccc4", "#3d7f77", "#1b3b3a"],  # muted teal
    "ANNOTATE":   False,   # colour only; exact means live in the SI table
    "ANNOT_FMT":  "{:.3f}",
    "SHOW_CBAR":  True,
    "CBAR_LABEL": "Relative importance",

    # --- Names -----------------------------------------------------------
    # Prefixes are shown by the group headings, so they are stripped here.
    "STRIP_PREFIXES": True,
    "RENAME": {
        "base_Degree":                     "Degree",
        "base_UniquePartners":             "Unique partners",
        "base_TriangleCount":              "Triangle count",
        "base_AvgNeighbourDegree":         "Avg. neighbour degree",
        "base_LocalClustCoeff":            "Local clust. coeff.",
        "protein_MedComplexNodes":         "Median complex size",
        "protein_RangeComplexNodes":       "Range complex size",
        "protein_RangeUniqueRatio":        "Range unique ratio",
        "stoich_MedianRatio":              "Median ratio",
        "stoich_MedComplexSize":           "Median total stoich.",
        "stoich_AvgNeighbourDegreeStoich": "Avg. neighbour stoich.",
        "stoich_WeightedTriangles":        "Weighted triangles",
        "stoich_RangeComplexSize":         "Range total stoich.",
        "stoich_RangeRatio":               "Range ratio",
        "pair_Degree":                     "Degree",
        "pair_TriangleCount":              "Triangle count",
        "pair_AvgNeighborDegree":          "Avg. neighbour degree",
        "pair_LocalClustCoeff":            "Local clust. coeff.",
    },
}


# =======================================================
# HELPERS
# =======================================================
def resolve(path) -> Path:
    """Absolute paths are used as given; relative ones sit under INPUT_DIR."""
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(CONFIG["INPUT_DIR"]) / path


def label_for(feature: str) -> str:
    if feature in CONFIG["RENAME"]:
        return CONFIG["RENAME"][feature]
    if CONFIG["STRIP_PREFIXES"]:
        return feature.split("_", 1)[-1].replace("_", " ")
    return feature


def groups() -> list:
    """Feature groups for the currently selected model."""
    model = CONFIG["MODEL"]
    if model not in CONFIG["GROUPS_BY_MODEL"]:
        raise KeyError(f"No GROUPS_BY_MODEL entry for MODEL='{model}'. "
                       f"Available: {list(CONFIG['GROUPS_BY_MODEL'])}")
    return CONFIG["GROUPS_BY_MODEL"][model]


def path_for(db: dict, task_token: str) -> Path:
    name = CONFIG["FILENAME_TEMPLATE"].format(
        db=db["key"], task=task_token, model=CONFIG["MODEL"])
    return resolve(Path(db.get("dir", ".")) / name)


def build_matrix(db: dict):
    """Returns (raw means, task labels, ordered features, group spans)."""
    features, spans, cursor = [], [], 0
    for group_name, group_features in groups():
        features.extend(group_features)
        spans.append((group_name, cursor, cursor + len(group_features) - 1))
        cursor += len(group_features)

    rows, task_labels = [], []
    for task_label, task_token in CONFIG["TASKS"]:
        path = path_for(db, task_token)
        if not path.exists():
            raise FileNotFoundError(f"Missing importance file: {path}")
        df = pd.read_csv(path).set_index("feature")

        missing = [f for f in features if f not in df.index]
        if missing:
            raise KeyError(f"{path.name} has no rows for: {missing}. "
                           f"It contains: {list(df.index)}")
        extra = [f for f in df.index if f not in features]
        if extra:
            print(f"   Note: {path.name} contains features not listed for this "
                  f"model, which are not plotted: {extra}")

        rows.append([float(df.loc[f, "mean"]) for f in features])
        task_labels.append(task_label)

    return np.array(rows), task_labels, features, spans


def normalise(raw: np.ndarray) -> np.ndarray:
    """Scales importances to [0, 1] for colouring; negatives clip to zero."""
    clipped = np.clip(raw, 0.0, None)
    if CONFIG["NORMALISE"] == "row":
        denom = clipped.max(axis=1, keepdims=True)
    else:
        denom = np.full((clipped.shape[0], 1), clipped.max())
    denom[denom == 0] = 1.0
    return clipped / denom


# =======================================================
# PLOTTING
# =======================================================
def draw_heatmap(ax, raw, shaded, row_labels, features, spans, cmap,
                 show_group_headings=True, show_xlabels=True):
    """Draws one heatmap block onto an existing axes."""
    n_rows, n_cols = raw.shape

    mesh = ax.imshow(shaded, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(np.arange(n_cols))
    if show_xlabels:
        ax.set_xticklabels([label_for(f) for f in features],
                           rotation=45, ha="right", rotation_mode="anchor")
    else:
        ax.set_xticklabels([])
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(row_labels)

    # Thin white gridlines instead of cell borders: cleaner at small sizes
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if CONFIG["ANNOTATE"]:
        for i in range(n_rows):
            for j in range(n_cols):
                value = raw[i, j]
                text = CONFIG["ANNOT_FMT"].format(value)
                if value < 0:
                    text = "\u2212" + text.lstrip("-")   # proper minus sign
                ax.text(j, i, text, ha="center", va="center",
                        fontsize=CONFIG["FONT_SIZE"] - 2,
                        color="white" if shaded[i, j] > 0.55 else "0.15")

    # Group separator, and headings above the top block only
    for group_name, start, end in spans:
        if show_group_headings and group_name:
            ax.text((start + end) / 2, -0.85, group_name,
                    ha="center", va="bottom", fontsize=CONFIG["FONT_SIZE"],
                    fontstyle="italic")
        if end + 1 < n_cols:
            ax.axvline(end + 0.5, color="0.25", linewidth=1.2)

    has_headings = show_group_headings and any(name for name, _, _ in spans)
    ax.set_ylim(n_rows - 0.5, -0.6 if has_headings else -0.5)
    return mesh


def add_colourbar(fig, mesh, ax):
    cbar = fig.colorbar(mesh, ax=ax, orientation="vertical",
                        fraction=0.025, pad=0.02, aspect=12)
    cbar.set_label(CONFIG["CBAR_LABEL"], fontsize=CONFIG["FONT_SIZE"] - 1)
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=3, width=0.8,
                        labelsize=CONFIG["FONT_SIZE"] - 1)


def report_top_features(db_display, task_labels, raw, features):
    for label, row in zip(task_labels, raw):
        order = np.argsort(row)[::-1]
        top = ", ".join(f"{label_for(features[k])} ({row[k]:+.3f})"
                        for k in order[:3])
        print(f"   {db_display} / {label}: {top}")


def save(fig, suffix):
    out_dir = Path(CONFIG["OUTPUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in CONFIG["FORMATS"]:
        out_path = out_dir / f"{CONFIG['OUTPUT_STEM']}_{CONFIG['MODEL']}_{suffix}.{fmt}"
        fig.savefig(out_path, dpi=CONFIG["DPI"])
        print(f"   Saved: {out_path}")
    plt.close(fig)


def set_rc():
    plt.rcParams.update({
        'font.size':       CONFIG["FONT_SIZE"],
        'axes.titlesize':  CONFIG["FONT_SIZE"],
        'axes.labelsize':  CONFIG["FONT_SIZE"],
        'xtick.labelsize': CONFIG["FONT_SIZE"],
        'ytick.labelsize': CONFIG["FONT_SIZE"],
        'pdf.fonttype':    42,
        'ps.fonttype':     42,
    })


def make_cmap():
    return LinearSegmentedColormap.from_list("importance", CONFIG["PALETTE"])


def plot_combined():
    """Both databases in one float: the fourteen column labels appear once.

    Stacking the two separate figures would repeat the rotated feature labels,
    which is most of the vertical space. Sharing them puts the CORUM
    stoichiometry block directly beneath the Complex Portal one, which is the
    comparison the figure exists to make.
    """
    set_rc()
    cmap = make_cmap()

    blocks = []
    for db in CONFIG["DATABASES"]:
        raw, task_labels, features, spans = build_matrix(db)
        blocks.append((db, raw, normalise(raw), task_labels, features, spans))

    n_rows = sum(b[1].shape[0] for b in blocks)

    # Explicit margins in inches: tight_layout cannot place a colourbar that
    # spans several axes without stealing width from them.
    W = CONFIG["FIG_WIDTH_IN"]
    left_in, right_in, top_in, bottom_in = 1.65, 0.95, 0.45, 1.50
    gap_in = 0.45
    fig_h = (n_rows * CONFIG["CELL_HEIGHT_IN"] + top_in + bottom_in
             + gap_in * (len(blocks) - 1))

    fig, axes = plt.subplots(
        len(blocks), 1,
        figsize=(W, fig_h),
        gridspec_kw={"height_ratios": [b[1].shape[0] for b in blocks]},
    )
    axes = np.atleast_1d(axes)

    fig.subplots_adjust(
        left=left_in / W, right=1 - right_in / W,
        top=1 - top_in / fig_h, bottom=bottom_in / fig_h,
        hspace=gap_in / (n_rows * CONFIG["CELL_HEIGHT_IN"] / len(blocks)),
    )

    mesh = None
    for i, (ax, (db, raw, shaded, task_labels, features, spans)) in enumerate(
            zip(axes, blocks)):
        is_first = (i == 0)
        is_last = (i == len(blocks) - 1)
        mesh = draw_heatmap(ax, raw, shaded, task_labels, features, spans, cmap,
                            show_group_headings=is_first,
                            show_xlabels=is_last)
        # Database name to the left of the task labels: a title here would
        # collide with the group headings above the first block.
        ax.set_ylabel(db["display"], fontsize=CONFIG["FONT_SIZE"], labelpad=10)

    if CONFIG["SHOW_CBAR"]:
        plot_bottom = bottom_in / fig_h
        plot_top = 1 - top_in / fig_h
        cax = fig.add_axes([
            1 - (right_in - 0.25) / W,
            plot_bottom + 0.25 * (plot_top - plot_bottom),
            0.16 / W,
            0.5 * (plot_top - plot_bottom),
        ])
        cbar = fig.colorbar(mesh, cax=cax)
        cbar.set_label(CONFIG["CBAR_LABEL"], fontsize=CONFIG["FONT_SIZE"] - 1)
        cbar.set_ticks([0, 0.5, 1.0])
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(length=3, width=0.8,
                            labelsize=CONFIG["FONT_SIZE"] - 1)

    save(fig, "combined")

    for db, raw, _shaded, task_labels, features, _spans in blocks:
        report_top_features(db["display"], task_labels, raw, features)


def plot_database(db: dict):
    raw, task_labels, features, spans = build_matrix(db)
    shaded = normalise(raw)
    n_rows, n_cols = raw.shape

    set_rc()
    cmap = make_cmap()

    fig_h = n_rows * CONFIG["CELL_HEIGHT_IN"] + 1.55
    fig, ax = plt.subplots(figsize=(CONFIG["FIG_WIDTH_IN"], fig_h))

    mesh = draw_heatmap(ax, raw, shaded, task_labels, features, spans, cmap)

    if CONFIG["SHOW_CBAR"]:
        add_colourbar(fig, mesh, ax)

    fig.tight_layout(pad=0.4)
    save(fig, db["key"])
    report_top_features(db["display"], task_labels, raw, features)


def main():
    if CONFIG["COMBINED"]:
        plot_combined()
    if CONFIG["SEPARATE"]:
        for db in CONFIG["DATABASES"]:
            plot_database(db)


if __name__ == "__main__":
    main()