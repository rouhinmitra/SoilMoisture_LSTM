"""
Shared figure style for RZSM paper figures.

Centralises the conventions already used by the published plotting scripts so new
work matches them without copying constants around.  Read off:
  - plot_spatial_transferability_scatter.py:132-136  (model palette)
  - plot_temporal_transferability_scatter.py:279-283 (identical palette)
  - plot_spatial_transferability_timeseries.py:178-184 (time-series palette)
  - plot_transfer_r2_vs_depth.py:40-42 (larger typography for depth figures)

Note: run_cv.py's internal per-fold scatter still uses the older matplotlib default
triplet (#1f77b4 / #ff7f0e / #2ca02c).  That is the pre-paper style; do not copy it.
The existing scripts are deliberately left untouched so published figures stay
reproducible byte-for-byte.
"""
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

# --- Palettes ---------------------------------------------------------------

MODEL_COLORS: Dict[str, str] = {
    "LSTM": "#5B7FD6",         # blue
    "Exponential": "#E8813B",  # orange
    "RF": "#3BB58F",           # teal
}

TIMESERIES_COLORS: Dict[str, str] = {
    "SSM": "silver",
    "RZSM_obs": "black",
    "LSTM": "blue",
    "RF": "green",
    "Exp": "orange",
}

MODEL_ORDER = ["Exponential", "RF", "LSTM"]

STATIONS = ["Ne1", "Ne2", "Ne3"]


def station_label(station: str) -> str:
    """'Ne1' -> 'US-Ne1' (station titles in the paper figures)."""
    return station if station.startswith("US-") else f"US-{station}"


# --- Typography and layout --------------------------------------------------

AXIS_LABEL_FONTSIZE = 15
TITLE_FONTSIZE = 15
LEGEND_FONTSIZE = 12

# Depth figures use a larger scale (plot_transfer_r2_vs_depth.py:40-42)
DEPTH_AXIS_LABEL_FONTSIZE = 20
DEPTH_TICK_LABEL_FONTSIZE = 18
DEPTH_LEGEND_FONTSIZE = 18

FIGSIZE_1x3 = (12, 4)
FIGSIZE_3x3 = (12, 9)
FIGSIZE_3x1 = (8, 9)

GRID_ALPHA = 0.3
SCATTER_ALPHA = 0.45
DPI = 300

XLABEL_MEASURED = "Measured RZSM 25cm (m³/m³)"
YLABEL_PREDICTED = "Predicted RZSM 25cm (m³/m³)"


def style_axes(ax, xlabel: Optional[str] = None, ylabel: Optional[str] = None,
               title: Optional[str] = None) -> None:
    """Apply the standard grid / label / title treatment to one axis."""
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONTSIZE)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    if title:
        ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax.grid(True, alpha=GRID_ALPHA)


def add_one_to_one(ax, lo: float, hi: float, label: str = "1:1") -> None:
    """Dashed 1:1 reference line spanning [lo, hi]."""
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.5, label=label)


def save_figure(fig, out_path) -> Path:
    """Save at the paper's standard resolution and cropping."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def scatter_panels(station_data: Dict[str, Dict[str, np.ndarray]],
                   title_suffix: str = "") -> "plt.Figure":
    """
    1x3 measured-vs-predicted panels, one per station, in the paper's style.

    station_data: {station: {"y_true": arr, "y_pred": arr, "r2": float}}
    """
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_1x3, sharey=True)

    lo, hi = np.inf, -np.inf
    for d in station_data.values():
        if len(d["y_true"]):
            lo = min(lo, float(np.min(d["y_true"])), float(np.min(d["y_pred"])))
            hi = max(hi, float(np.max(d["y_true"])), float(np.max(d["y_pred"])))
    if not np.isfinite(lo):
        lo, hi = 0.0, 1.0
    pad = 0.02 * (hi - lo)
    lo, hi = lo - pad, hi + pad

    for ax, station in zip(axes, STATIONS):
        d = station_data.get(station)
        if d is None or not len(d["y_true"]):
            ax.set_visible(False)
            continue
        ax.scatter(d["y_true"], d["y_pred"], alpha=SCATTER_ALPHA, s=10,
                   color=MODEL_COLORS["LSTM"],
                   label=f"LSTM ($R^2$={d['r2']:.2f})")
        add_one_to_one(ax, lo, hi)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        title = station_label(station)
        if title_suffix:
            title = f"{title} {title_suffix}"
        style_axes(ax, xlabel=XLABEL_MEASURED, title=title)
        ax.legend(fontsize=LEGEND_FONTSIZE, loc="upper left",
                  framealpha=0.9, markerscale=1.8)

    axes[0].set_ylabel(YLABEL_PREDICTED, fontsize=AXIS_LABEL_FONTSIZE)
    fig.tight_layout()
    return fig
