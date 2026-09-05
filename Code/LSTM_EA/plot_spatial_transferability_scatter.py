#!/usr/bin/env python
"""
Spatial transferability comparison scatter plot for journal figure.

Plots measured vs predicted RZSM (25 cm) for three models
(LSTM, Exponential Filter, Random Forest) across US-Ne1, US-Ne2, US-Ne3
using leave-one-station-out cross-validation results (2017-2023).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent

LSTM_DIR = BASE / "outputs" / "sensitivity" / "seq20" / "baseline_no_doy"
RF_DIR = BASE / "outputs" / "rf_cv_tuned"
EXP_CSV = BASE / "outputs" / "exp_filter_spatial_cv" / "cv_predictions.csv"

OUTPUT_DIR = BASE / "outputs" / "spatial_transferability"

STATIONS = ["Ne1", "Ne2", "Ne3"]

RF_FOLD_MAP = {
    "Ne1": "fold3_test_Ne1",
    "Ne2": "fold2_test_Ne2",
    "Ne3": "fold1_test_Ne3",
}

YEAR_RANGE = (2017, 2023)
MONTH_RANGE = (5, 10)

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _filter_growing_season(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    df[date_col] = pd.to_datetime(df[date_col])
    mask = (
        (df[date_col].dt.year >= YEAR_RANGE[0])
        & (df[date_col].dt.year <= YEAR_RANGE[1])
        & (df[date_col].dt.month >= MONTH_RANGE[0])
        & (df[date_col].dt.month <= MONTH_RANGE[1])
    )
    return df.loc[mask].copy()


def load_lstm(station: str) -> pd.DataFrame:
    path = LSTM_DIR / f"predictions_{station}.csv"
    df = pd.read_csv(path)
    df = df.rename(columns={"Date": "date", "RZSM_obs": "measured", "RZSM_pred": "predicted"})
    df = _filter_growing_season(df)
    df["measured"] = df["measured"] / 100.0
    df["predicted"] = df["predicted"] / 100.0
    return df[["date", "measured", "predicted"]]


def load_rf(station: str) -> pd.DataFrame:
    fold = RF_FOLD_MAP[station]
    path = RF_DIR / fold / "predictions.csv"
    df = pd.read_csv(path)
    df = df.rename(columns={"actual": "measured", "predicted": "predicted"})
    df = _filter_growing_season(df)
    df["measured"] = df["measured"] / 100.0
    df["predicted"] = df["predicted"] / 100.0
    return df[["date", "measured", "predicted"]]


def load_exp(station: str, exp_df: pd.DataFrame) -> pd.DataFrame:
    df = exp_df.loc[exp_df["test_station"] == station].copy()
    df = df.rename(columns={"Date": "date", "RZSM_true": "measured", "RZSM_pred": "predicted"})
    df = _filter_growing_season(df)
    df["measured"] = df["measured"] / 100.0
    df["predicted"] = df["predicted"] / 100.0
    return df[["date", "measured", "predicted"]]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

MODEL_ORDER = ["Exponential", "RF", "LSTM"]


def compute_metrics(measured: np.ndarray, predicted: np.ndarray) -> dict:
    residuals = predicted - measured
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((measured - np.mean(measured)) ** 2)
    return {
        "R2": 1.0 - ss_res / ss_tot,
        "Bias": np.mean(residuals),
        "RMSE": np.sqrt(np.mean(residuals ** 2)),
        "Corr": np.corrcoef(measured, predicted)[0, 1],
    }


def make_metrics_table(station_data: dict) -> pd.DataFrame:
    rows = []
    for station in STATIONS:
        row = {"Station": f"US-{station}"}
        for model in MODEL_ORDER:
            df = station_data[station][model]
            m = compute_metrics(df["measured"].values, df["predicted"].values)
            prefix = {"Exponential": "Exp", "RF": "RF", "LSTM": "LSTM"}[model]
            for key, val in m.items():
                row[f"{prefix}_{key}"] = val
        rows.append(row)

    avg_row = {"Station": "Average"}
    metric_cols = [c for c in rows[0] if c != "Station"]
    for col in metric_cols:
        avg_row[col] = np.mean([r[col] for r in rows])
    rows.append(avg_row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_figure():
    exp_all = pd.read_csv(EXP_CSV)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

    colors = {
        "LSTM": "#5B7FD6",
        "Exponential": "#E8813B",
        "RF": "#3BB58F",
    }

    global_min, global_max = np.inf, -np.inf

    station_data = {}
    for station in STATIONS:
        lstm_df = load_lstm(station)
        rf_df = load_rf(station)
        exp_df = load_exp(station, exp_all)
        station_data[station] = {"LSTM": lstm_df, "Exponential": exp_df, "RF": rf_df}

        for model_df in (lstm_df, rf_df, exp_df):
            vals = np.concatenate([model_df["measured"].values, model_df["predicted"].values])
            global_min = min(global_min, vals.min())
            global_max = max(global_max, vals.max())

    pad = (global_max - global_min) * 0.03
    axis_min = global_min - pad
    axis_max = global_max + pad

    for idx, station in enumerate(STATIONS):
        ax = axes[idx]
        data = station_data[station]

        for model_name in ["Exponential", "RF", "LSTM"]:
            df = data[model_name]
            ax.scatter(
                df["measured"],
                df["predicted"],
                s=12,
                alpha=0.45,
                color=colors[model_name],
                label=model_name,
                edgecolors="none",
                rasterized=True,
            )
            ax.grid(True, alpha=0.3)

        ax.plot(
            [axis_min, axis_max],
            [axis_min, axis_max],
            "k--",
            linewidth=0.8,
            zorder=0,
        )

        ax.set_xlim(axis_min, axis_max)
        ax.set_ylim(axis_min, axis_max)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"US-{station}", fontsize=15, fontweight="bold")
        ax.set_xlabel("Measured RZSM 25cm (m\u00b3/m\u00b3)", fontsize=15)
        ax.tick_params(labelsize=15)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Predicted RZSM 25cm (m\u00b3/m\u00b3)", fontsize=15)
    axes[0].legend(fontsize=12, loc="upper left", framealpha=0.9, markerscale=1.8)

    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "spatial_transferability_scatter.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")

    metrics_df = make_metrics_table(station_data)
    metrics_path = OUTPUT_DIR / "spatial_transferability_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, float_format="%.4f")
    print(f"Saved -> {metrics_path}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    make_figure()
