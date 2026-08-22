#!/usr/bin/env python
"""
Time-series plots of monthly spatial transferability metrics.

Uses spatial_transfer_monthly_metrics.csv (from
compute_spatial_transfer_monthly_metrics.py) to plot, for each station:
 - Average monthly R²
 - Average monthly RMSE
 - Average monthly Bias
averaged over 2017–2023 for each model (LSTM, RF, Exp).
"""

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent
METRICS_CSV = BASE / "outputs" / "spatial_transferability" / "spatial_transfer_monthly_metrics.csv"
OUTPUT_DIR = BASE / "outputs" / "spatial_transferability"

STATIONS = ["Ne1", "Ne2", "Ne3"]
MODELS = ["Exp", "RF", "LSTM"]
MODEL_LABELS: Dict[str, str] = {"Exp": "Exponential", "RF": "RF tuned", "LSTM": "LSTM"}
MODEL_COLORS: Dict[str, str] = {"Exp": "#E8813B", "RF": "#3BB58F", "LSTM": "#5B7FD6"}

MONTHS = [5, 6, 7, 8, 9, 10]
MONTH_LABELS = ["May", "Jun", "Jul", "Aug", "Sep", "Oct"]


def load_monthly_means() -> pd.DataFrame:
    """
    Load CSV and compute mean and 95% CI across years for each station/model/month.

    Returns columns:
      - station, model, month
      - rmse_mean, rmse_lo, rmse_hi
      - bias_mean, bias_lo, bias_hi
      - corr_mean, corr_lo, corr_hi
    """
    df = pd.read_csv(METRICS_CSV)

    # Only per-station rows (exclude ALL)
    df = df[df["station"].isin(STATIONS)].copy()

    grouped = df.groupby(["station", "model", "month"], sort=True)
    rows: List[Dict[str, object]] = []

    for (station, model, month), g in grouped:
        n_years = g["year"].nunique()
        if n_years == 0:
            continue

        def _agg(col: str):
            mean = g[col].mean()
            if n_years > 1:
                std = g[col].std(ddof=1)
                se = std / np.sqrt(n_years)
                ci = 1.96 * se
            else:
                ci = 0.0
            return mean, mean - ci, mean + ci

        rmse_mean, rmse_lo, rmse_hi = _agg("rmse")
        bias_mean, bias_lo, bias_hi = _agg("bias")
        corr_mean, corr_lo, corr_hi = _agg("corr")

        # Clip correlation CIs to valid range [-1, 1]
        corr_lo = max(-1.0, corr_lo)
        corr_hi = min(1.0, corr_hi)

        rows.append(
            {
                "station": station,
                "model": model,
                "month": int(month),
                "rmse_mean": rmse_mean,
                "rmse_lo": rmse_lo,
                "rmse_hi": rmse_hi,
                "bias_mean": bias_mean,
                "bias_lo": bias_lo,
                "bias_hi": bias_hi,
                "corr_mean": corr_mean,
                "corr_lo": corr_lo,
                "corr_hi": corr_hi,
            }
        )

    return pd.DataFrame(rows)


def make_plots() -> None:
    df = load_monthly_means()

    # Order: RMSE, Bias, Correlation
    metrics = [
        ("rmse_mean", "rmse_lo", "rmse_hi", "RMSE (m³/m³)"),
        ("bias_mean", "bias_lo", "bias_hi", "Bias (m³/m³)"),
        ("corr_mean", "corr_lo", "corr_hi", "Correlation"),
    ]

    fig, axes = plt.subplots(
        nrows=len(STATIONS),
        ncols=len(metrics),
        figsize=(12, 8),
        sharex=True,
    )

    for row_idx, station in enumerate(STATIONS):
        station_df = df[df["station"] == station]

        for col_idx, (mean_col, lo_col, hi_col, title) in enumerate(metrics):
            ax = axes[row_idx, col_idx]

            for model in MODELS:
                sub = station_df[station_df["model"] == model]
                if sub.empty:
                    continue

                y_mean, y_lo, y_hi = [], [], []
                for m in MONTHS:
                    row = sub[sub["month"] == m]
                    if row.empty:
                        y_mean.append(np.nan)
                        y_lo.append(np.nan)
                        y_hi.append(np.nan)
                    else:
                        r = row.iloc[0]
                        y_mean.append(r[mean_col])
                        y_lo.append(r[lo_col])
                        y_hi.append(r[hi_col])

                x = MONTHS
                color = MODEL_COLORS[model]

                # Confidence interval band (mean ± 1.96 * SE across years)
                ax.fill_between(
                    x,
                    y_lo,
                    y_hi,
                    color=color,
                    alpha=0.15,
                    linewidth=0,
                )

                # Mean line
                ax.plot(
                    x,
                    y_mean,
                    marker="o",
                    linewidth=1.5,
                    color=color,
                    label=MODEL_LABELS[model],
                )

            ax.set_xticks(MONTHS)
            ax.set_xticklabels(MONTH_LABELS,fontsize=15)
            ax.grid(True, alpha=0.3)

            # Clamp y-limits for RMSE and Bias
            if mean_col.startswith("rmse"):
                ax.set_ylim(0.01, 0.06)
            elif mean_col.startswith("bias"):
                ax.set_ylim(-0.06, 0.02)

            if row_idx == 0:
                ax.set_title(title, fontsize=12, fontweight="bold")

            if col_idx == 0:
                ax.set_ylabel(f"US-{station}", fontsize=15)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(MODELS),
        fontsize=15,
        framealpha=0.9,
    )

    fig.tight_layout(rect=[0.0, 0.05, 1.0, 0.95])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "spatial_transfer_monthly_metrics_timeseries.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    make_plots()

