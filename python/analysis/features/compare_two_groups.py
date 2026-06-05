"""
compare_groups.py
-----------------
Compares ALL hypergraph vs pairwise features for two CORUM groups with
matched AP size (51 proteins each) but opposite essentiality profiles:

  grp_0038 : 96.1% Essential  (+78.7pp vs global 17.3%)
  grp_0761 : 0.0%  Essential  (-17.3pp vs global)

Layout per section
------------------
ONE-HOP
  Row 1 : [HG] Degree | [PW] Degree          ← the one paired feature
  Row 2 : [HG] UniquePartners (HG-only)
  Rows 3-4 : remaining 8 HG-only features, 4 per row

TWO-HOP & GLOBAL
  Paired features → mini-column per pair: [HG top / PW bottom],
  columns arranged left-to-right horizontally.
  HG-only features appended in further columns (PW cell = grey placeholder).

Outputs:
  1. Summary CSV  (mean ± SD, MWU p-value, significance)
  2. Three PNGs   (one per section)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy.stats import mannwhitneyu

# ── file paths ────────────────────────────────────────────────────────────────
# INPUTS
SPLITS_PATH  = "../../../data/lookup_tables/corum_ess_merged_splits.csv"
HG_PATH      = "../../../data/lookup_tables/corum_hypergraph_features.csv"
PW_PATH      = "../../../data/lookup_tables/corum_pairwise_features.csv"

# OUTPUTS
OUT_CSV           = "group_feature_summary.csv"
OUT_PLOT_ONEHOP   = "comparison_one_hop.png"
OUT_PLOT_TWOHOP   = "comparison_two_hop.png"
OUT_PLOT_GLOBAL   = "comparison_global.png"


# ── groups ────────────────────────────────────────────────────────────────────
GROUP_A  = "grp_0038"
GROUP_B  = "grp_0761"
LABEL_A  = "grp_0038\n(96% Ess)"
LABEL_B  = "grp_0761\n(0% Ess)"
COLOUR_A = "#c0392b"
COLOUR_B = "#2980b9"
BG_HG    = "#fff8f8"
BG_PW    = "#f8f8ff"
BG_EMPTY = "#f0f0f0"

# ── feature layout ────────────────────────────────────────────────────────────
# (hg_feature, pw_feature) — None where no equivalent exists

ONE_HOP_PAIRS = [
    ("base_Degree",                 "pair_Degree"),   # the only paired one
    ("base_UniquePartners",         None),
    ("stoich_RangeComplexSize",     None),
    ("stoich_MedComplexSize",       None),
    ("stoich_MedianRatio",          None),
    ("stoich_RangeRatio",           None),
    ("protein_MedianUniqueRatio",   None),
    ("protein_RangeUniqueRatio",    None),
    ("protein_MedComplexNodes",     None),
    ("protein_RangeComplexNodes",   None),
]

TWO_HOP_PAIRS = [
    ("base_LocalClustCoeff",            "pair_LocalClustCoeff"),
    ("base_TriangleCount",              "pair_TriangleCount"),
    ("base_AvgNeighbourDegree",         "pair_AvgNeighborDegree"),
    ("stoich_WeightedTriangles",        None),
    ("stoich_AvgNeighbourDegreeStoich", None),
]

GLOBAL_PAIRS = [
    ("base_KatzCentrality",         "pair_KatzCentrality"),
    ("base_EigenvectorCentrality",  "pair_EigenvectorCentrality"),
    ("base_BetweennessCentrality",  "pair_BetweennessCentrality"),
    ("base_ComponentSize",          "pair_ComponentSize"),
    ("base_ComponentEdgeNodeRatio", None),
]

# ── load & merge ──────────────────────────────────────────────────────────────
splits = pd.read_csv(SPLITS_PATH)
hg     = pd.read_csv(HG_PATH)
pw     = pd.read_csv(PW_PATH)

ap = (splits[splits["label_mask"]]
      .drop_duplicates("UniProt_AC")
      [["UniProt_AC", "group_id"]])

feat = hg.merge(pw, on="ProteinId", how="outer")
feat = feat.merge(ap.rename(columns={"UniProt_AC": "ProteinId"}),
                  on="ProteinId", how="inner")

df = feat[feat["group_id"].isin([GROUP_A, GROUP_B])].copy()
print(f"Proteins: {df['group_id'].value_counts().to_dict()}")

# ── helpers ───────────────────────────────────────────────────────────────────
def mwu_stats(feature):
    if not feature or feature not in df.columns:
        return None
    a = df.loc[df["group_id"] == GROUP_A, feature].dropna()
    b = df.loc[df["group_id"] == GROUP_B, feature].dropna()
    if len(a) == 0 or len(b) == 0:
        return None
    _, pval = mannwhitneyu(a, b, alternative="two-sided")
    sig = ("***" if pval < 0.001 else "**" if pval < 0.01
           else "*" if pval < 0.05 else "ns")
    return dict(a=a, b=b, a_mean=a.mean(), b_mean=b.mean(),
                a_sd=a.std(), b_sd=b.std(), pval=pval, sig=sig)

def short(name):
    for pfx in ("base_", "stoich_", "protein_", "pair_"):
        if name.startswith(pfx):
            return name[len(pfx):]
    return name

def draw_panel(ax, stats, title, kind):
    """kind = 'HG' or 'PW'"""
    ax.set_facecolor(BG_HG if kind == "HG" else BG_PW)
    a_v, b_v = stats["a"].values, stats["b"].values

    vp = ax.violinplot([a_v, b_v], positions=[0, 1],
                       showmedians=True, widths=0.55)
    for body, col in zip(vp["bodies"], [COLOUR_A, COLOUR_B]):
        body.set_facecolor(col); body.set_alpha(0.30)
    for part in ("cmedians", "cmins", "cmaxes", "cbars"):
        vp[part].set_color("#333333"); vp[part].set_linewidth(0.8)

    rng = np.random.default_rng(42)
    for xpos, vals, col in [(0, a_v, COLOUR_A), (1, b_v, COLOUR_B)]:
        ax.scatter(xpos + rng.uniform(-0.10, 0.10, len(vals)),
                   vals, color=col, alpha=0.65, s=16, linewidths=0, zorder=3)

    ax.set_xticks([0, 1])
    ax.set_xticklabels([LABEL_A, LABEL_B], fontsize=12)
    sig   = stats["sig"]
    pval  = stats["pval"]
    sig_label = sig if sig != "ns" else f"p={pval:.2f}"
    title_col = "#8B0000" if kind == "HG" else "#00008B"
    ax.set_title(f"[{kind}] {title}  {sig_label}", fontsize=12, pad=4, color=title_col)
    ax.tick_params(axis="y", labelsize=12)
    ax.spines[["top", "right"]].set_visible(False)

def draw_empty(ax, msg="No pairwise\nequivalent"):
    ax.set_facecolor(BG_EMPTY)
    ax.text(0.5, 0.5, msg, ha="center", va="center",
            transform=ax.transAxes, fontsize=12,
            color="#aaaaaa", style="italic")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)

# ── summary CSV ───────────────────────────────────────────────────────────────
SECTIONS_DEF = [
    ("One-hop (neighbours)",               ONE_HOP_PAIRS,  OUT_PLOT_ONEHOP),
    ("Two-hop (neighbours of neighbours)", TWO_HOP_PAIRS,  OUT_PLOT_TWOHOP),
    ("Global",                             GLOBAL_PAIRS,   OUT_PLOT_GLOBAL),
]
rows = []
for sec, pairs, _ in SECTIONS_DEF:
    for hf, pf in pairs:
        for fname, ftype in [(hf, "Hypergraph"), (pf, "Pairwise")]:
            if not fname: continue
            s = mwu_stats(fname)
            if not s:
                print(f"  WARNING: {fname} missing — skipped"); continue
            rows.append({"Section": sec, "Type": ftype, "Feature": fname,
                         "grp_0038 mean": f"{s['a_mean']:.4g}",
                         "grp_0038 SD":   f"{s['a_sd']:.4g}",
                         "grp_0761 mean": f"{s['b_mean']:.4g}",
                         "grp_0761 SD":   f"{s['b_sd']:.4g}",
                         "MWU p-value":   f"{s['pval']:.4g}",
                         "Significant":   s["sig"]})
summary = pd.DataFrame(rows)
summary.to_csv(OUT_CSV, index=False)
print(f"\nSummary saved → {OUT_CSV}")
print(summary.to_string(index=False))

legend_patches = [
    Patch(facecolor=COLOUR_A, alpha=0.7, label="grp_0038 (96% Essential)"),
    Patch(facecolor=COLOUR_B, alpha=0.7, label="grp_0761 (0% Essential)"),
    Patch(facecolor=BG_HG, edgecolor="#8B0000", label="Hypergraph feature"),
    Patch(facecolor=BG_PW, edgecolor="#00008B", label="Pairwise feature"),
]

SUPTITLE = (
    "{section}\n"
    "grp_0038 (96% Essential, red)  vs  grp_0761 (0% Essential, blue)\n"
    "Mann–Whitney U:  * p<0.05   ** p<0.01   *** p<0.001"
)

# ═══════════════════════════════════════════════════════════════════════════════
# ONE-HOP
# Layout:
#   Row 0  : [HG Degree] [PW Degree]  (paired, side-by-side)
#   Row 1  : [HG UniquePartners]       (HG-only, full width = 2 cols merged)
#   Rows 2-3: remaining 8 HG-only, 4 per row
# ═══════════════════════════════════════════════════════════════════════════════
NCOLS_OH = 4
paired_oh    = ONE_HOP_PAIRS[0]           # (base_Degree, pair_Degree)
unique_oh    = ONE_HOP_PAIRS[1]           # (base_UniquePartners, None)
remaining_oh = [p[0] for p in ONE_HOP_PAIRS[2:]]   # 8 HG-only names

n_rem_rows = int(np.ceil(len(remaining_oh) / NCOLS_OH))   # = 2
total_rows_oh = 1 + 1 + n_rem_rows                         # = 4

fig_oh = plt.figure(figsize=(NCOLS_OH * 4.5, total_rows_oh * 4.2))
fig_oh.suptitle(SUPTITLE.format(section="One-hop (neighbours)"),
                fontsize=13, y=1.01)

gs_oh = gridspec.GridSpec(total_rows_oh, NCOLS_OH,
                          figure=fig_oh, hspace=0.55, wspace=0.35)

# Row 0: paired Degree (HG left two cols, PW right two cols)
ax_hg_deg = fig_oh.add_subplot(gs_oh[0, :2])
ax_pw_deg = fig_oh.add_subplot(gs_oh[0, 2:])
s = mwu_stats(paired_oh[0])
draw_panel(ax_hg_deg, s, short(paired_oh[0]), "HG") if s else draw_empty(ax_hg_deg)
s = mwu_stats(paired_oh[1])
draw_panel(ax_pw_deg, s, short(paired_oh[1]), "PW") if s else draw_empty(ax_pw_deg)

# Row 1: UniquePartners spanning all 4 cols — place in first 2 cols, leave right 2 empty
ax_uniq = fig_oh.add_subplot(gs_oh[1, :2])
s = mwu_stats(unique_oh[0])
draw_panel(ax_uniq, s, short(unique_oh[0]), "HG") if s else draw_empty(ax_uniq)
# Right half of row 1: empty/invisible
ax_blank = fig_oh.add_subplot(gs_oh[1, 2:])
ax_blank.set_visible(False)

# Rows 2-3: remaining 8 HG-only features
for i, fname in enumerate(remaining_oh):
    r = 2 + i // NCOLS_OH
    c = i % NCOLS_OH
    ax = fig_oh.add_subplot(gs_oh[r, c])
    s = mwu_stats(fname)
    draw_panel(ax, s, short(fname), "HG") if s else draw_empty(ax, f"{short(fname)}\n(no data)")

fig_oh.legend(handles=legend_patches, loc="lower center", ncol=4,
              fontsize=12, frameon=True, bbox_to_anchor=(0.5, -0.02))
fig_oh.savefig(OUT_PLOT_ONEHOP, dpi=150, bbox_inches="tight")
plt.close(fig_oh)
print(f"Plot saved → {OUT_PLOT_ONEHOP}")


# ═══════════════════════════════════════════════════════════════════════════════
# TWO-HOP  &  GLOBAL  — shared layout logic
# Each (hg, pw) pair → one column: HG top, PW bottom.
# HG-only pairs → one column: HG top, grey placeholder bottom.
# ═══════════════════════════════════════════════════════════════════════════════
def plot_horizontal_section(pairs, section_name, out_path):
    n_cols = len(pairs)
    n_rows = 2   # HG row, PW row

    fig = plt.figure(figsize=(n_cols * 4.5, n_rows * 4.2))
    fig.suptitle(SUPTITLE.format(section=section_name), fontsize=13, y=1.01)
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                           hspace=0.45, wspace=0.35)

    for col, (hf, pf) in enumerate(pairs):
        # HG (top)
        ax_hg = fig.add_subplot(gs[0, col])
        s = mwu_stats(hf)
        draw_panel(ax_hg, s, short(hf), "HG") if s else draw_empty(ax_hg, f"{short(hf)}\n(no data)")

        # PW (bottom)
        ax_pw = fig.add_subplot(gs[1, col])
        if pf:
            s = mwu_stats(pf)
            draw_panel(ax_pw, s, short(pf), "PW") if s else draw_empty(ax_pw, f"{short(pf)}\n(no data)")
        else:
            draw_empty(ax_pw, "No pairwise\nequivalent")

    fig.legend(handles=legend_patches, loc="lower center", ncol=4,
               fontsize=12, frameon=True, bbox_to_anchor=(0.5, -0.03))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {out_path}")

plot_horizontal_section(TWO_HOP_PAIRS,
                        "Two-hop (neighbours of neighbours)",
                        OUT_PLOT_TWOHOP)

plot_horizontal_section(GLOBAL_PAIRS,
                        "Global",
                        OUT_PLOT_GLOBAL)