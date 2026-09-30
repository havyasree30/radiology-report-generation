"""Shared, restrained figure style for manuscript-ready EDA figures.

Colours: the reference data-viz palette. Categorical hues are assigned in a
fixed slot order (adjacent slots are the CVD-validated pairs); a neutral grey
encodes "missing/unmentioned" rather than a hue. Sequential = one hue
(blue, light -> dark); diverging = blue <-> red around a grey midpoint.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from src.utils.config import EDA_FIGURES  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
MISSING_GREY = "#d9d8d2"

# Label states, stacked in this order: consecutive palette slots + neutral grey.
STATE_COLORS = {"positive": SERIES[0], "uncertain": SERIES[1], "negative": SERIES[2], "missing": MISSING_GREY}
SPLIT_COLORS = {"train": SERIES[0], "val": SERIES[1], "test": SERIES[2], "official_valid": MUTED}

SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#b02a2a"])


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.labelsize": 10, "axes.labelcolor": INK_2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
        "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.frameon": False, "legend.fontsize": 9,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })


def save_figure(fig: plt.Figure, name: str, out_dir: Path = EDA_FIGURES, pdf: bool = True) -> list[Path]:
    """Save PNG (300 dpi) and, by default, a vector PDF for manuscripts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.png"]
    fig.savefig(paths[0])
    if pdf:
        paths.append(out_dir / f"{name}.pdf")
        fig.savefig(paths[1])
    plt.close(fig)
    return paths


def plain_log_ticks(ax: plt.Axes, axis: str = "x") -> None:
    """Readable tick labels (0.1, 1, 10, 1,000) instead of 10^n on log axes."""
    fmt = plt.FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}")
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(fmt)


def subtitle(ax: plt.Axes, text: str) -> None:
    """Secondary line under the (left-aligned) title."""
    ax.text(0, 1.01, text, transform=ax.transAxes, fontsize=9, color=INK_2, va="bottom")


def heatmap(ax: plt.Axes, M: np.ndarray, labels: list[str], cmap, vmin=None, vmax=None,
            fmt: str = "{:.2f}", annotate: bool = True, cbar_label: str = "") -> None:
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks(range(len(labels)), labels, rotation=55, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    if annotate:
        lo = np.nanmin(M) if vmin is None else vmin
        hi = np.nanmax(M) if vmax is None else vmax
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if np.isnan(v):
                    continue
                frac = (v - lo) / (hi - lo) if hi > lo else 0
                dark_bg = frac > 0.6 if cmap is SEQUENTIAL else abs(frac - 0.5) > 0.35
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=6.5,
                        color="white" if dark_bg else INK)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8)
    if cbar_label:
        cb.set_label(cbar_label, fontsize=9, color=INK_2)
