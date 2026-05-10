"""
Gaussian + Square Mask Visualizer
==================================
Overlays one or both masks on scatter data loaded from an external file.
Equivalent to the gnuplot command:

    plot 'file.cor' u 3:2 w p pt 7

Mask modes
----------
MASK_MODE = "gaussian"   → Gaussian blob only
MASK_MODE = "square"     → Flat square/rectangle only
MASK_MODE = "both"       → Gaussian blob + square region together

Parameters
----------
FWHM         : Full Width at Half Maximum of the Gaussian (data units).
CENTER       : (x, y) peak / square-center in data units.
               Set to None to auto-place at the data centroid.
XRANGE       : [x_min, x_max] — like gnuplot  set xrange [0:60]
YRANGE       : [y_min, y_max] — like gnuplot  set yrange [0:5]
               Set either to None to auto-fit.
SQUARE_WIDTH : Width  of the square mask in data units.
SQUARE_HEIGHT: Height of the square mask (= SQUARE_WIDTH for a true square).
SQUARE_COLOR : Fill color of the square, e.g. "royalblue", "#2ca02c", "red".
SQUARE_ALPHA : Opacity of the square fill  (0 = invisible, 1 = opaque).
RESOLUTION   : Pixels per data unit for the Gaussian (higher = sharper).
DATA_FILE    : Path to the whitespace-delimited scatter data file.
COL_X        : 1-based column for X  (gnuplot 'u 3:2' → COL_X=3).
COL_Y        : 1-based column for Y  (gnuplot 'u 3:2' → COL_Y=2).
COMMENT_CHAR : Lines starting with this character are skipped.
OUTPUT_FILE  : Output image path  (png / pdf / svg …).
"""

import matplotlib
matplotlib.use("Agg")   # non-interactive — no display needed

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# ─────────────────────────────────────────────────────────────
#  USER PARAMETERS  ← change these freely
# ─────────────────────────────────────────────────────────────

MASK_MODE     = "both"    # "gaussian" | "square" | "both"

# ── Gaussian ─────────────────────────────────────────────────
FWHM          = 0.6796       # Full Width at Half Maximum (data units)

# ── Shared center (Gaussian peak AND square center) ──────────
CENTER        = [0,1]      # e.g. (5.0, 2.3)  or  None → auto centroid
SQUARE_CENTER = (0,1)
# ── Square mask ──────────────────────────────────────────────
SQUARE_CENTER = None      # (x, y) center of the square — None → follows CENTER
SQUARE_WIDTH  = 2.0       # width  in data units
SQUARE_HEIGHT = 1.0       # height in data units (set = SQUARE_WIDTH for square)
SQUARE_COLOR  = "royalblue"
SQUARE_ALPHA  = 0.20      # opacity: 0 (invisible) → 1 (solid)

# ── Axis ranges — like gnuplot set xrange / set yrange ───────
XRANGE        = [-1,3]      # e.g. [0, 60]  or  None → auto
YRANGE        = [0,2]      # e.g. [0, 5]   or  None → auto

# ── Data / output ─────────────────────────────────────────────
RESOLUTION    = 10
DATA_FILE     = "e2627_v1j2_1M_vfjf.cor"
COL_X         = 3
COL_Y         = 2
COMMENT_CHAR  = "#"
OUTPUT_FILE   = "gaussian_mask_output.png"

# ─────────────────────────────────────────────────────────────


def fwhm_to_sigma(fwhm):
    return fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def load_scatter_file(filepath, col_x, col_y, comment_char="#"):
    data = np.loadtxt(filepath, comments=comment_char)
    return data[:, col_x - 1], data[:, col_y - 1]


def make_gaussian_mask(extent, center, fwhm, resolution=10):
    """2-D Gaussian array built in real data coordinates."""
    sigma = fwhm_to_sigma(fwhm)
    x_min, x_max, y_min, y_max = extent
    cx, cy = center
    nx = max(2, int((x_max - x_min) * resolution))
    ny = max(2, int((y_max - y_min) * resolution))
    X, Y = np.meshgrid(np.linspace(x_min, x_max, nx),
                       np.linspace(y_min, y_max, ny))
    return np.exp(-((X - cx)**2 + (Y - cy)**2) / (2.0 * sigma**2))


def make_red_transparent_cmap():
    stops = [
        (1.0, 1.0,  1.0,  0.00),
        (1.0, 0.85, 0.85, 0.40),
        (0.9, 0.3,  0.2,  0.75),
        (0.6, 0.0,  0.0,  1.00),
    ]
    return LinearSegmentedColormap.from_list("red_transparent", stops)


def plot_gaussian_mask(
    mask_mode=MASK_MODE,
    fwhm=FWHM, center=CENTER,
    square_center=SQUARE_CENTER,
    square_width=SQUARE_WIDTH, square_height=SQUARE_HEIGHT,
    square_color=SQUARE_COLOR, square_alpha=SQUARE_ALPHA,
    xrange=XRANGE, yrange=YRANGE,
    resolution=RESOLUTION, data_file=DATA_FILE,
    col_x=COL_X, col_y=COL_Y, comment_char=COMMENT_CHAR,
    output_file=OUTPUT_FILE,
):
    # 1. Load scatter data
    print(f"Loading '{data_file}'  (col {col_x} → X, col {col_y} → Y) …")
    sx, sy = load_scatter_file(data_file, col_x, col_y, comment_char)
    print(f"  {len(sx):,} points  |  X [{sx.min():.3f}, {sx.max():.3f}]"
          f"  |  Y [{sy.min():.3f}, {sy.max():.3f}]")

    # 2. Axis ranges — like gnuplot set xrange / set yrange
    margin = fwhm
    x_min, x_max = xrange if xrange is not None else (sx.min() - margin, sx.max() + margin)
    y_min, y_max = yrange if yrange is not None else (sy.min() - margin, sy.max() + margin)
    extent = [x_min, x_max, y_min, y_max]
    print(f"  xrange [{x_min:.3f}, {x_max:.3f}]  yrange [{y_min:.3f}, {y_max:.3f}]"
          f"{'  (auto)' if xrange is None else ''}")

    # 3. Center
    if center is None:
        center = (float(np.mean(sx)), float(np.mean(sy)))
        print(f"  Auto-center → ({center[0]:.3f}, {center[1]:.3f})")
    else:
        print(f"  Center = ({center[0]:.3f}, {center[1]:.3f})")

    cx, cy = center

    # Square center — independent from Gaussian center
    if square_center is None:
        scx, scy = cx, cy   # default: follow Gaussian center
    else:
        scx, scy = square_center
    print(f"  Square center = ({scx:.3f}, {scy:.3f})"
          f"{'  (follows Gaussian)' if square_center is None else ''}")

    # 4. Plot
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f7f7f7")

    legend_handles = []

    # ── Gaussian mask ──────────────────────────────────────────
    if mask_mode in ("gaussian", "both"):
        Z    = make_gaussian_mask(extent, center, fwhm, resolution)
        cmap = make_red_transparent_cmap()
        im   = ax.imshow(
            Z, extent=extent, origin="lower",
            cmap=cmap, vmin=0, vmax=1,
            aspect="auto", interpolation="bilinear", zorder=1,
        )
        # FWHM circle
        circle = mpatches.Circle(
            center, fwhm / 2.0,
            color="black", fill=False, linestyle="--",
            linewidth=1.2, zorder=4,
        )
        ax.add_patch(circle)
        legend_handles.append(
            mpatches.Patch(facecolor="#990000", label=f"Gaussian binning function")
        )

    # ── Square / rectangle mask ────────────────────────────────
    if mask_mode in ("square", "both"):
        sq_x = scx - square_width  / 2.0   # bottom-left corner x
        sq_y = scy - square_height / 2.0   # bottom-left corner y
        rect = mpatches.FancyBboxPatch(
            (sq_x, sq_y), square_width, square_height,
            boxstyle="square,pad=0",
            linewidth=1.4, edgecolor=square_color,
            facecolor=square_color, alpha=square_alpha,
            zorder=2,
        )
        ax.add_patch(rect)
        legend_handles.append(
            mpatches.Patch(
                facecolor=square_color, alpha=square_alpha,
                edgecolor=square_color,
                label=f"Standard histogram binning function",
            )
        )

    # ── Crosshairs through center ──────────────────────────────
    ax.axhline(cy, color="black", lw=0.6, ls="-", alpha=0.35, zorder=3)
    ax.axvline(cx, color="black", lw=0.6, ls="-", alpha=0.35, zorder=3)

    # ── Scatter ────────────────────────────────────────────────
    ax.scatter(sx, sy, s=3, c="black", alpha=0.35, zorder=5,
               linewidths=0)
    legend_handles.append(
        plt.Line2D([0], [0], marker='o', color='w',
                    markerfacecolor='black', markeredgecolor='black', markersize=5, alpha=0.5,
                    label=f"Trajectories"))

    # ── Colorbar (Gaussian only) ───────────────────────────────
    if mask_mode in ("gaussian", "both"):
        cbar = fig.colorbar(im, ax=ax, orientation="horizontal",
                            fraction=0.04, pad=0.18, aspect=40)
        cbar.set_label("Gaussian Density", fontsize=9)
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels(["0 (min)", "0.25", "0.50", "0.75", "1 (max)"])

    # ── Axes ───────────────────────────────────────────────────
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(
        f"Mask: {mask_mode}  |  FWHM={fwhm}  |  "
        f"Centre=({cx:.3f}, {cy:.3f})  |  {data_file}",
        fontsize=10, pad=10,
    )
    ax.set_xlabel(f"$j_f$")
    ax.set_ylabel(f"$v_f$")
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.35, zorder=0)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {output_file}")


if __name__ == "__main__":
    plot_gaussian_mask()
