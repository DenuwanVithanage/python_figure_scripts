"""
Gaussian + Square Mask Visualizer  (multi-panel edition)
=========================================================
Overlays one or both masks on scatter data loaded from an external file.
Equivalent to the gnuplot command:

    plot 'file.cor' u 3:2 w p pt 7

Layout modes
------------
LAYOUT_MODE = "single"     → original single panel (mask overlay only)
LAYOUT_MODE = "stacked"    → two panels stacked vertically, shared x-axis
                              top: scatter  |  bottom: DCS log-y
LAYOUT_MODE = "twinx"      → one panel, scatter on left y-axis,
                              DCS on right y-axis (log scale), shared x-axis

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
DATA_FILE    : Path to the whitespace-delimited scatter data file (.cor).
DCS_FILE     : Path to the classical DCS data file (.dat).
COL_X        : 1-based column for X  (gnuplot 'u 3:2' → COL_X=3).
COL_Y        : 1-based column for Y  (gnuplot 'u 3:2' → COL_Y=2).
VLINE_X      : x-coordinate of the vertical dashed reference line (None = off).
COMMENT_CHAR : Lines starting with this character are skipped.
OUTPUT_FILE  : Output image path  (png / pdf / svg …).
"""

import matplotlib
matplotlib.use("Agg")   # non-interactive — no display needed

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
import os

# ─────────────────────────────────────────────────────────────
#  USER PARAMETERS  ← change these freely
# ─────────────────────────────────────────────────────────────

LAYOUT_MODE   = "stacked"  # "single" | "stacked" | "twinx"

MASK_MODE     = "both"    # "gaussian" | "square" | "both"

# ── Gaussian ─────────────────────────────────────────────────
FWHM          = 0.6796

# ── Shared center (Gaussian peak AND square center) ──────────
CENTER        = [0, 1]     # e.g. (5.0, 2.3)  or  None → auto centroid

# ── Square mask ──────────────────────────────────────────────
SQUARE_CENTER = None       # None → follows CENTER
SQUARE_WIDTH  = 2.0
SQUARE_HEIGHT = 1.0
SQUARE_COLOR  = "royalblue"
SQUARE_ALPHA  = 0.20

# ── Axis ranges ───────────────────────────────────────────────
XRANGE        = [-1, 3]
YRANGE        = [0, 2]         # scatter y-range  (panel 1 / left axis)
DCS_YRANGE    = [0.05, 20000]  # DCS y-range      (panel 2 / right axis)

# ── Vertical reference line ───────────────────────────────────
VLINE_X       = -0.5           # dashed vertical line; set to None to disable

# ── Data / output ─────────────────────────────────────────────
RESOLUTION    = 10
DATA_FILE     = "e2627_v1j2_1M_vfjf.cor"
DCS_FILE      = "e2627_v1j2.jf_profile_v1p0_win1p0.dat"
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


def load_dcs_file(filepath, comment_char="#"):
    """Load DCS file: returns (x, y) from first two columns."""
    data = np.loadtxt(filepath, comments=comment_char)
    return data[:, 0], data[:, 1]


def make_gaussian_mask(extent, center, fwhm, resolution=10):
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


# ─────────────────────────────────────────────────────────────
#  PANEL DRAWING HELPERS
# ─────────────────────────────────────────────────────────────

def draw_scatter_panel(ax, sx, sy, extent, center,
                       mask_mode, fwhm, square_center,
                       square_width, square_height,
                       square_color, square_alpha,
                       resolution, vline_x,
                       show_colorbar=True):
    """Draw the scatter + mask onto *ax*. Returns list of legend handles."""
    x_min, x_max, y_min, y_max = extent
    cx, cy = center
    scx, scy = square_center

    ax.set_facecolor("#f7f7f7")
    legend_handles = []

    # Gaussian mask
    if mask_mode in ("gaussian", "both"):
        Z    = make_gaussian_mask(extent, center, fwhm, resolution)
        cmap = make_red_transparent_cmap()
        im   = ax.imshow(
            Z, extent=extent, origin="lower",
            cmap=cmap, vmin=0, vmax=1,
            aspect="auto", interpolation="bilinear", zorder=1,
        )
        circle = mpatches.Circle(
            center, fwhm / 2.0,
            color="black", fill=False, linestyle="--",
            linewidth=1.2, zorder=4,
        )
        ax.add_patch(circle)
        legend_handles.append(
            mpatches.Patch(facecolor="#990000",
                           label="Gaussian binning function")
        )
        if show_colorbar:
            # Inset colorbar: bottom-right inside the axes, [left, bottom, w, h]
            cax = ax.inset_axes([0.58, 0.12, 0.40, 0.06])
            cbar = plt.colorbar(im, cax=cax, orientation="horizontal")
            cbar.set_label("Gaussian Weight", fontsize=8, labelpad=2)
            cbar.set_ticks([0, 0.5, 1.0])
            cbar.set_ticklabels(["0", "0.5", "1"], fontsize=7)
            cax.tick_params(labelsize=7)
            # Slight white backdrop so it reads over the scatter
            cax.patch.set_facecolor((1, 1, 1, 0.55))

    # Square mask
    if mask_mode in ("square", "both"):
        sq_x = scx - square_width  / 2.0
        sq_y = scy - square_height / 2.0
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
                label="Standard histogram binning function",
            )
        )

    # Crosshairs
    ax.axhline(cy, color="black", lw=0.6, ls="-", alpha=0.35, zorder=3)
    ax.axvline(cx, color="black", lw=0.6, ls="-", alpha=0.35, zorder=3)

    # Vertical reference line
    if vline_x is not None:
        ax.axvline(vline_x, color="black", lw=1.5, ls="--", alpha=0.7, zorder=3)

    # Scatter
    ax.scatter(sx, sy, s=3, c="#888888", alpha=0.45, zorder=5, linewidths=0)
    legend_handles.append(
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='#888888', markeredgecolor='#888888',
                   markersize=5, alpha=0.7,
                   label="Trajectories")
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel(r"$v_f$", fontsize=12)
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.35, zorder=0)
    return legend_handles


def draw_dcs_panel(ax, dx, dy, xrange, yrange, vline_x,
                   show_xlabel=True):
    """Draw the DCS line+points onto *ax*."""
    x_min, x_max = xrange
    y_min, y_max = yrange

    ax.set_facecolor("#f7f7f7")

    ax.semilogy(dx, dy, 'o-', color="magenta",
                markersize=4, linewidth=1.2, zorder=3,
                label="Classical DCS")

    if vline_x is not None:
        ax.axvline(vline_x, color="black", lw=1.5, ls="--", alpha=0.7, zorder=3)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel("Classical DCS", fontsize=12)
    if show_xlabel:
        ax.set_xlabel(r"$j_f$", fontsize=12)

    # Nicer log ticks
    ax.yaxis.set_major_formatter(
        ticker.LogFormatterMathtext(base=10)
    )
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.35, zorder=0, which="both")


# ─────────────────────────────────────────────────────────────
#  MAIN PLOTTING FUNCTION
# ─────────────────────────────────────────────────────────────

def plot_gaussian_mask(
    layout_mode=LAYOUT_MODE,
    mask_mode=MASK_MODE,
    fwhm=FWHM, center=CENTER,
    square_center=SQUARE_CENTER,
    square_width=SQUARE_WIDTH, square_height=SQUARE_HEIGHT,
    square_color=SQUARE_COLOR, square_alpha=SQUARE_ALPHA,
    xrange=XRANGE, yrange=YRANGE, dcs_yrange=DCS_YRANGE,
    vline_x=VLINE_X,
    resolution=RESOLUTION,
    data_file=DATA_FILE, dcs_file=DCS_FILE,
    col_x=COL_X, col_y=COL_Y, comment_char=COMMENT_CHAR,
    output_file=OUTPUT_FILE,
):
    # ── 1. Load scatter data ──────────────────────────────────
    print(f"Loading '{data_file}' …")
    sx, sy = load_scatter_file(data_file, col_x, col_y, comment_char)
    print(f"  {len(sx):,} points  |  X [{sx.min():.3f}, {sx.max():.3f}]"
          f"  |  Y [{sy.min():.3f}, {sy.max():.3f}]")

    # ── 2. Load DCS data (optional) ───────────────────────────
    dcs_available = layout_mode != "single" and os.path.isfile(dcs_file)
    if layout_mode != "single":
        if dcs_available:
            print(f"Loading '{dcs_file}' …")
            dx, dy = load_dcs_file(dcs_file, comment_char)
            print(f"  {len(dx):,} points  |  X [{dx.min():.3f}, {dx.max():.3f}]"
                  f"  |  Y [{dy.min():.3g}, {dy.max():.3g}]")
        else:
            print(f"  WARNING: '{dcs_file}' not found — "
                  f"DCS panel will show a placeholder curve.")
            # Synthetic stand-in so the layout still renders
            dx = np.linspace(xrange[0], xrange[1], 200)
            dy = 500 * np.exp(-0.8 * (dx + 0.5)**2) + 1.0

    # ── 3. Resolve axis limits ────────────────────────────────
    margin = fwhm
    x_min, x_max = (xrange if xrange is not None
                    else (sx.min() - margin, sx.max() + margin))
    y_min, y_max = (yrange if yrange is not None
                    else (sy.min() - margin, sy.max() + margin))
    extent = [x_min, x_max, y_min, y_max]

    # ── 4. Resolve centers ────────────────────────────────────
    if center is None:
        center = (float(np.mean(sx)), float(np.mean(sy)))
    cx, cy = center
    scx, scy = (cx, cy) if square_center is None else square_center
    print(f"  Center=({cx:.3f},{cy:.3f})  Square center=({scx:.3f},{scy:.3f})")

    # ══════════════════════════════════════════════════════════
    #  LAYOUT A — single (original)
    # ══════════════════════════════════════════════════════════
    if layout_mode == "single":
        fig, ax = plt.subplots(figsize=(12, 4))
        fig.patch.set_facecolor("white")
        draw_scatter_panel(
            ax, sx, sy, extent, (cx, cy),
            mask_mode, fwhm, (scx, scy),
            square_width, square_height, square_color, square_alpha,
            resolution, vline_x,
            show_colorbar=True,
        )
        ax.set_xlabel(r"$j_f$", fontsize=12)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved → {output_file}")
        return

    # ══════════════════════════════════════════════════════════
    #  LAYOUT B — stacked two-panel, shared x-axis
    # ══════════════════════════════════════════════════════════
    if layout_mode == "stacked":
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(10, 9),
            sharex=True,
            gridspec_kw={"hspace": 0.06, "height_ratios": [1, 1]},
        )
        fig.patch.set_facecolor("white")

        # Top panel — scatter + mask (colorbar inset, no x-tick labels)
        draw_scatter_panel(
            ax_top, sx, sy, extent, (cx, cy),
            mask_mode, fwhm, (scx, scy),
            square_width, square_height, square_color, square_alpha,
            resolution, vline_x,
            show_colorbar=True,
        )
        ax_top.tick_params(labelbottom=False)

        # Bottom panel — DCS
        draw_dcs_panel(
            ax_bot, dx, dy,
            (x_min, x_max), dcs_yrange, vline_x,
            show_xlabel=True,
        )
        if not dcs_available:
            ax_bot.set_title("(placeholder — supply DCS_FILE)", fontsize=9,
                             color="gray")

        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved → {output_file}")
        return

    # ══════════════════════════════════════════════════════════
    #  LAYOUT C — twinx: scatter (left y) + DCS (right y)
    # ══════════════════════════════════════════════════════════
    if layout_mode == "twinx":
        fig, ax_left = plt.subplots(figsize=(11, 5))
        fig.patch.set_facecolor("white")
        ax_right = ax_left.twinx()

        # ── Right axis first so scatter sits on top ───────────
        ax_right.semilogy(dx, dy, 'o-', color="royalblue",
                          markersize=4, linewidth=1.4, zorder=2,
                          label="Classical DCS (right axis)",
                          alpha=0.75)
        ax_right.set_ylim(*dcs_yrange)
        ax_right.set_ylabel("Classical DCS  (log scale)", fontsize=12,
                             color="royalblue")
        ax_right.tick_params(axis="y", colors="royalblue")
        ax_right.yaxis.set_major_formatter(
            ticker.LogFormatterMathtext(base=10)
        )
        ax_right.grid(False)   # avoid double-grid

        # ── Left axis — scatter + mask ────────────────────────
        draw_scatter_panel(
            ax_left, sx, sy, extent, (cx, cy),
            mask_mode, fwhm, (scx, scy),
            square_width, square_height, square_color, square_alpha,
            resolution, vline_x,
            show_colorbar=True,
        )
        ax_left.set_xlabel(r"$j_f$", fontsize=12)
        ax_left.set_title(
            f"{data_file}  |  Mask: {mask_mode}  FWHM={fwhm}  |  "
            f"twinx layout",
            fontsize=10, pad=8,
        )

        # ── Combined legend ───────────────────────────────────
        h1, l1 = ax_left.get_legend_handles_labels()
        h2, l2 = ax_right.get_legend_handles_labels()
        ax_left.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)

        if not dcs_available:
            ax_right.set_title("right axis: placeholder curve", fontsize=8,
                               color="gray")

        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved → {output_file}")
        return

    raise ValueError(f"Unknown LAYOUT_MODE '{layout_mode}'. "
                     "Choose 'single', 'stacked', or 'twinx'.")


if __name__ == "__main__":
    # ── produce all three layouts at once ─────────────────────
    for mode, fname in [
        ("single",  "output_single.png"),
        ("stacked", "output_stacked.png"),
        ("twinx",   "output_twinx.png"),
    ]:
        print(f"\n{'='*60}")
        print(f"  Layout: {mode}")
        print(f"{'='*60}")
        plot_gaussian_mask(layout_mode=mode, output_file=fname)
