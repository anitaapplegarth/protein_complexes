"""
Figure: structure -> representation -> matrix, for two GABA-A receptor complexes.

Row 1  top-down PyMOL cartoon renders of CPX-2167 and CPX-8701
Row 2  pairwise (merged projection) / hypergraph / hb-graph
Row 3  adjacency matrix / incidence matrix / multiplicity-valued incidence matrix

Proteins
    P1  P28472  GABRB3  beta-3   shared, x2 in CPX-2167 and x4 in CPX-8701
    P2  P14867  GABRA1  alpha-1  x2 in CPX-2167
    P3  P18507  GABRG2  gamma-2  x1 in CPX-2167
    P4  P31644  GABRA5  alpha-5  x1 in CPX-8701
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from matplotlib.lines import Line2D
import matplotlib.image as mpimg
import numpy as np
from PIL import Image

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "font.family": "DejaVu Sans",
})

# PyMOL colours, so the panels match the renders exactly
C_P1 = (0.70, 0.30, 0.40)   # raspberry, beta-3, shared
C_P2 = (0.50, 0.80, 1.00)   # skyblue,  alpha-1
C_P3 = (0.65, 0.90, 0.65)   # palegreen, gamma-2
C_P4 = (1.00, 0.50, 0.00)   # orange, alpha-5

EDGE = "0.25"
DASH = "0.45"

IMG1 = "/Users/anitaapplegarth/Documents/dphil/paper/cpx2167_6i53_topdown.png"
IMG2 = "/Users/anitaapplegarth/Documents/dphil/paper/cpx8701_6a96_topdown.png"

R = 0.058          # node radius in axes coords
RS = 0.042         # smaller radius for the hb-graph copies


def autocrop(path, pad=8):
    """Trim the white border PyMOL leaves around the molecule."""
    im = Image.open(path).convert("RGBA")
    a = np.array(im)
    opaque = a[:, :, 3] > 0
    ink = (a[:, :, :3].sum(axis=2) < 720) & opaque
    if not ink.any():
        return np.array(im.convert("RGB"))
    ys, xs = np.where(ink)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, a.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, a.shape[1])
    out = a[y0:y1, x0:x1]
    rgb = out[:, :, :3].astype(float)
    al = (out[:, :, 3:4].astype(float)) / 255.0
    return (rgb * al + 255.0 * (1 - al)).astype("uint8")


def node(ax, x, y, colour, label=None, r=R):
    ax.add_patch(Circle((x, y), r, facecolor=colour, edgecolor=EDGE,
                        linewidth=0.8, zorder=3))
    if label:
        ax.text(x, y, label, ha="center", va="center", fontsize=12,
                color="0.1", zorder=4)


def blank(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")


def matrix(ax, rows, rowlabels, collabels, title):
    """Draw a bracketed matrix with row and column labels."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ncol = len(rows[0])
    nrow = len(rows)

    x0, x1 = 0.34, 0.34 + 0.13 * ncol
    ytop, ybot = 0.74, 0.74 - 0.135 * nrow
    colx = [x0 + 0.13 * (j + 0.5) for j in range(ncol)]
    rowy = [ytop - 0.135 * (i + 0.5) for i in range(nrow)]

    ax.text((x0 + x1) / 2, 0.95, title, ha="center", va="center", fontsize=12)

    for j, cl in enumerate(collabels):
        ax.text(colx[j], ytop + 0.055, cl, ha="center", va="center",
                fontsize=12, color="0.35")
    for i, rl in enumerate(rowlabels):
        ax.text(x0 - 0.075, rowy[i], rl, ha="right", va="center",
                fontsize=12, color="0.35")

    b = 0.035
    for xb, sgn in ((x0 - 0.035, 1), (x1 + 0.035, -1)):
        ax.add_line(Line2D([xb, xb], [ybot, ytop], color="0.2", lw=1.1))
        ax.add_line(Line2D([xb, xb + sgn * b], [ytop, ytop], color="0.2", lw=1.1))
        ax.add_line(Line2D([xb, xb + sgn * b], [ybot, ybot], color="0.2", lw=1.1))

    for i in range(nrow):
        for j in range(ncol):
            v = rows[i][j]
            ax.text(colx[j], rowy[i], str(v), ha="center", va="center",
                    fontsize=12,
                    color=(0.70, 0.15, 0.15) if (i == 0 and v not in (0, 1)) else "0.1")


fig = plt.figure(figsize=(7.2, 6.5))
gs = fig.add_gridspec(
    4, 6,
    height_ratios=[1.12, 1.12, 0.14, 0.75],
    hspace=0.20, wspace=0.10,
    left=0.02, right=0.98, top=0.985, bottom=0.03,
)

# ---------------------------------------------------------------- row 1
axA = fig.add_subplot(gs[0, 0:3])
axB = fig.add_subplot(gs[0, 3:6])
for ax, path, name in ((axA, IMG1, "CPX-2167"), (axB, IMG2, "CPX-8701")):
    ax.imshow(autocrop(path))
    ax.axis("off")
axA.set_title("GABA-A receptor:  α1-β3-γ2", fontsize=13, pad=4)
axB.set_title("GABA-A receptor:  α5-β3", fontsize=13, pad=4)

# ---------------------------------------------------------------- row 2
# shared node geometry, identical across the pairwise and hypergraph panels
PX = {"P2": (0.26, 0.72), "P3": (0.26, 0.30), "P1": (0.52, 0.51),
      "P4": (0.80, 0.51)}

ax1 = fig.add_subplot(gs[1, 0:2])
blank(ax1)
ax1.set_title("Pairwise", fontsize=13, pad=4)
for a, b in (("P2", "P3"), ("P2", "P1"), ("P3", "P1"), ("P1", "P4")):
    ax1.add_line(Line2D([PX[a][0], PX[b][0]], [PX[a][1], PX[b][1]],
                        color=EDGE, lw=1.0, zorder=1))
for k, c in (("P1", C_P1), ("P2", C_P2), ("P3", C_P3), ("P4", C_P4)):
    node(ax1, *PX[k], c, k)

ax2 = fig.add_subplot(gs[1, 2:4])
blank(ax2)
ax2.set_title("Hypergraph", fontsize=13, pad=4)
ax2.add_patch(Ellipse((0.38, 0.51), 0.62, 0.80, facecolor="none",
                      edgecolor=DASH, ls=(0, (4, 3)), lw=1.0, clip_on=False))
ax2.add_patch(Ellipse((0.68, 0.51), 0.50, 0.46, facecolor="none",
                      edgecolor=DASH, ls=(0, (4, 3)), lw=1.0, clip_on=False))
for k, c in (("P1", C_P1), ("P2", C_P2), ("P3", C_P3), ("P4", C_P4)):
    node(ax2, *PX[k], c, k)
ax2.text(0.10, 0.82, "$e_1$", fontsize=12, color=DASH, ha="center", va="center")
ax2.text(0.90, 0.68, "$e_2$", fontsize=12, color=DASH, ha="center", va="center")

ax3 = fig.add_subplot(gs[1, 4:6])
blank(ax3)
ax3.set_title("hb-graph", fontsize=13, pad=4)
ax3.add_patch(Ellipse((0.30, 0.51), 0.60, 0.88, facecolor="none",
                      edgecolor=DASH, ls=(0, (4, 3)), lw=1.0, clip_on=False))
ax3.add_patch(Ellipse((0.66, 0.51), 0.66, 0.84, facecolor="none",
                      edgecolor=DASH, ls=(0, (4, 3)), lw=1.0, clip_on=False))
ax3.text(0.04, 0.86, "$e_1$", fontsize=12, color=DASH, ha="center", va="center")
ax3.text(0.94, 0.84, "$e_2$", fontsize=12, color=DASH, ha="center", va="center")

# e1 only: P2 x2, P3 x1
for x, y in ((0.17, 0.70), (0.17, 0.48)):
    node(ax3, x, y, C_P2, "P2")
node(ax3, 0.17, 0.26, C_P3, "P3")
# intersection: the two P1 copies shared by both hyperedges
for x, y in ((0.46, 0.64), (0.46, 0.42)):
    node(ax3, x, y, C_P1, "P1")
# e2 only: the other two P1 copies, plus P4
for x, y in ((0.76, 0.64), (0.76, 0.42)):
    node(ax3, x, y, C_P1, "P1")
node(ax3, 0.76, 0.22, C_P4, "P4")

# ---------------------------------------------------------------- row 3
rl = ["P1", "P2", "P3", "P4"]
matrix(fig.add_subplot(gs[3, 0:2]),
       [[0, 1, 1, 1], [1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]],
       rl, rl, "Adjacency matrix")
matrix(fig.add_subplot(gs[3, 2:4]),
       [[1, 1], [1, 0], [1, 0], [0, 1]],
       rl, ["$e_1$", "$e_2$"], "Incidence matrix")
matrix(fig.add_subplot(gs[3, 4:6]),
       [[2, 4], [2, 0], [1, 0], [0, 1]],
       rl, ["$e_1$", "$e_2$"], "Incidence matrix")

# ---------------------------------------------------------------- key
handles = [
    Line2D([], [], marker="o", ls="none", markersize=9, markeredgecolor=EDGE,
           markeredgewidth=0.8, markerfacecolor=C_P1, label="P1  P28472 (shared)"),
    Line2D([], [], marker="o", ls="none", markersize=9, markeredgecolor=EDGE,
           markeredgewidth=0.8, markerfacecolor=C_P2, label="P2  P14867"),
    Line2D([], [], marker="o", ls="none", markersize=9, markeredgecolor=EDGE,
           markeredgewidth=0.8, markerfacecolor=C_P3, label="P3  P18507"),
    Line2D([], [], marker="o", ls="none", markersize=9, markeredgecolor=EDGE,
           markeredgewidth=0.8, markerfacecolor=C_P4, label="P4  P31644"),
]
axkey = fig.add_subplot(gs[2, 0:6])
axkey.axis("off")
axkey.legend(handles=handles, loc="center", ncol=4, frameon=False,
             fontsize=12, handletextpad=0.4, columnspacing=1.4)

fig.savefig("/Users/anitaapplegarth/github/dphil/protein_complexes/python/analysis/representation_figure.pdf",
            bbox_inches="tight", dpi=300)
fig.savefig("/Users/anitaapplegarth/github/dphil/protein_complexes/python/analysis/representation_figure.png",
            bbox_inches="tight", dpi=200)
print("written")