#!/usr/bin/env python
"""
Temporal transferability comparison scatter plot.

Plots measured vs predicted RZSM (25 cm) for three models
(LSTM, Exponential Filter, Random Forest) using out-of-year
single-station predictions aggregated across all stations/years.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE = Path(__file__).resolve().parent

LSTM_CSV = BASE / "outputs" / "lstm_out_of_year_single_station" / "all_stations_lstm_ooy_predictions.csv"
RF_CSV = BASE / "outputs" / "rf_out_of_year_single_station" / "all_stations_rf_ooy_predictions.csv"
EXP_CSV = BASE / "outputs" / "exp_filter_out_of_year_single_station" / "all_stations_exp_filter_ooy_predictions.csv"

OUTPUT_DIR = BASE / "outputs" / "temporal_transferability"

YEAR_RANGE = (2016, 2023)
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


def _postprocess_common(df: pd.DataFrame) -> pd.DataFrame:
    df = _filter_growing_season(df, "date")
    df["measured"] = df["measured"] / 100.0
    df["predicted"] = df["predicted"] / 100.0
    return df[["date", "station", "test_year", "measured", "predicted"]]


def load_lstm_temporal() -> pd.DataFrame:
    df = pd.read_csv(LSTM_CSV)
    df = df.rename(
        columns={
            "Date": "date",
            "RZSM_true": "measured",
            "RZSM_pred": "predicted",
        }
    )
    # Ensure expected identifier columns exist
    if "station" not in df.columns:
        raise ValueError("Expected 'station' column in LSTM temporal CSV.")
    if "test_year" not in df.columns:
        if "Year" in df.columns:
            df = df.rename(columns={"Year": "test_year"})
        else:
            raise ValueError("Expected 'test_year' or 'Year' column in LSTM temporal CSV.")
    return _postprocess_common(df)


def load_rf_temporal() -> pd.DataFrame:
    df = pd.read_csv(RF_CSV)
    df = df.rename(
        columns={
            "Date": "date",
            "RZSM_true": "measured",
            "RZSM_pred": "predicted",
        }
    )
    if "station" not in df.columns:
        raise ValueError("Expected 'station' column in RF temporal CSV.")
    if "test_year" not in df.columns:
        if "Year" in df.columns:
            df = df.rename(columns={"Year": "test_year"})
        else:
            raise ValueError("Expected 'test_year' or 'Year' column in RF temporal CSV.")
    return _postprocess_common(df)


def load_exp_temporal() -> pd.DataFrame:
    df = pd.read_csv(EXP_CSV)
    df = df.rename(
        columns={
            "Date": "date",
            "RZSM_true": "measured",
            "RZSM_pred": "predicted",
        }
    )
    if "station" not in df.columns:
        raise ValueError("Expected 'station' column in Exponential temporal CSV.")
    if "test_year" not in df.columns:
        if "Year" in df.columns:
            df = df.rename(columns={"Year": "test_year"})
        else:
            raise ValueError("Expected 'test_year' or 'Year' column in Exponential temporal CSV.")
    return _postprocess_common(df)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

MODEL_ORDER = ["Exponential", "RF", "LSTM"]
STATIONS = ["Ne1", "Ne2", "Ne3"]

def compute_metrics(measured: np.ndarray, predicted: np.ndarray) -> dict:
    residuals = predicted - measured
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((measured - np.mean(measured)) ** 2)
    return {
        "R2": 1.0 - ss_res / ss_tot,
        "Bias": np.mean(residuals),
        "RMSE": np.sqrt(np.mean(residuals**2)),
        "Corr": np.corrcoef(measured, predicted)[0, 1],
    }


def make_metrics_table(station_data: dict) -> pd.DataFrame:
    rows = []

    for station in STATIONS:
        row = {"Station": f"US-{station}"}
        for model_name in ["Exponential", "RF", "LSTM"]:
            df = station_data[station][model_name]
            m = compute_metrics(df["measured"].values, df["predicted"].values)
            prefix = {"Exponential": "Exp", "RF": "RF", "LSTM": "LSTM"}[model_name]
            for key, val in m.items():
                row[f"{prefix}_{key}"] = val
        rows.append(row)

    # Optional: add an Average row across stations
    avg_row = {"Station": "Average"}
    metric_cols = [c for c in rows[0].keys() if c != "Station"]
    for col in metric_cols:
        avg_row[col] = np.mean([r[col] for r in rows])
    rows.append(avg_row)

    return pd.DataFrame(rows)


def compute_per_year_metrics(station_data: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    rows: list[dict] = []
    for station in STATIONS:
        for model_name in ["Exponential", "RF", "LSTM"]:
            df = station_data[station][model_name]
            if df.empty:
                continue
            for year, g in df.groupby("test_year"):
                m = compute_metrics(g["measured"].values, g["predicted"].values)
                rows.append(
                    {
                        "Station": f"US-{station}",
                        "Year": int(year),
                        "Model": model_name,
                        **m,
                    }
                )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["Station", "Year", "Model"]).reset_index(drop=True)
    return out


def plot_per_year_bars(
    per_year_df: pd.DataFrame,
    metric: str,
    *,
    out_name: str,
    y_label: str,
    colors: dict[str, str],
) -> Path:
    years = list(range(YEAR_RANGE[0], YEAR_RANGE[1] + 1))
    x = np.arange(len(years))
    width = 0.25
    model_order = ["Exponential", "RF", "LSTM"]
    offsets = {
        "Exponential": -width,
        "RF": 0.0,
        "LSTM": width,
    }

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True, sharey=True)

    for idx, station in enumerate(STATIONS):
        ax = axes[idx]
        station_label = f"US-{station}"
        sdf = per_year_df.loc[per_year_df["Station"] == station_label]

        for model_name in model_order:
            vals = []
            for y in years:
                v = sdf.loc[(sdf["Year"] == y) & (sdf["Model"] == model_name), metric]
                vals.append(float(v.iloc[0]) if len(v) else np.nan)

            ax.bar(
                x + offsets[model_name],
                vals,
                width=width,
                label=model_name,
                color=colors[model_name],
                alpha=0.85,
                edgecolor="none",
            )

        ax.set_title(station_label, fontsize=15)
        ax.set_ylabel(y_label, fontsize=13)
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(labelsize=12)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([str(y) for y in years], fontsize=12)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, framealpha=0.9, fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / out_name
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_figure():
    lstm_df = load_lstm_temporal()
    rf_df = load_rf_temporal()
    exp_df = load_exp_temporal()

    model_data = {
        "LSTM": lstm_df,
        "RF": rf_df,
        "Exponential": exp_df,
    }

    # Split by station for plotting
    station_data: dict[str, dict[str, pd.DataFrame]] = {}
    for station in STATIONS:
        station_data[station] = {
            "LSTM": lstm_df[lstm_df["station"] == station],
            "RF": rf_df[rf_df["station"] == station],
            "Exponential": exp_df[exp_df["station"] == station],
        }

    # Determine global axis limits across all stations and models
    global_min, global_max = np.inf, -np.inf
    for station in STATIONS:
        for model_name in ["Exponential", "RF", "LSTM"]:
            df = station_data[station][model_name]
            if df.empty:
                continue
            vals = np.concatenate([df["measured"].values, df["predicted"].values])
            if vals.size == 0:
                continue
            global_min = min(global_min, vals.min())
            global_max = max(global_max, vals.max())

    pad = (global_max - global_min) * 0.03
    axis_min = global_min - pad
    axis_max = global_max + pad

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

    colors = {
        "LSTM": "#5B7FD6",
        "Exponential": "#E8813B",
        "RF": "#3BB58F",
    }

    for idx, station in enumerate(STATIONS):
        ax = axes[idx]
        data = station_data[station]

        for model_name in ["Exponential", "RF", "LSTM"]:
            df = data[model_name]
            if df.empty:
                continue
            ax.scatter(
                df["measured"],
                df["predicted"],
                s=8,
                alpha=0.4,
                color=colors[model_name],
                label=model_name,
                edgecolors="none",
                rasterized=True,
            )

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
        ax.set_title(f"US-{station}", fontsize=15)
        ax.set_xlabel("Measured RZSM 25cm (m³/m³)", fontsize=15)
        ax.tick_params(labelsize=15)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Predicted RZSM 25cm (m³/m³)", fontsize=15)
    axes[0].legend(fontsize=12, loc="upper left", framealpha=0.9, markerscale=1.8)

    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "temporal_transferability_scatter.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")

    metrics_df = make_metrics_table(station_data)
    metrics_path = OUTPUT_DIR / "temporal_transferability_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, float_format="%.4f")
    print(f"Saved -> {metrics_path}")
    print(metrics_df.to_string(index=False))

    per_year_df = compute_per_year_metrics(station_data)
    per_year_path = OUTPUT_DIR / "temporal_transferability_per_year_metrics.csv"
    per_year_df.to_csv(per_year_path, index=False, float_format="%.4f")
    print(f"Saved -> {per_year_path}")

    if not per_year_df.empty:
        r2_path = plot_per_year_bars(
            per_year_df,
            "R2",
            out_name="temporal_transferability_r2_bars.png",
            y_label="R²",
            colors=colors,
        )
        print(f"Saved -> {r2_path}")

        bias_path = plot_per_year_bars(
            per_year_df,
            "Bias",
            out_name="temporal_transferability_bias_bars.png",
            y_label="Bias (m³/m³)",
            colors=colors,
        )
        print(f"Saved -> {bias_path}")

        rmse_path = plot_per_year_bars(
            per_year_df,
            "RMSE",
            out_name="temporal_transferability_rmse_bars.png",
            y_label="RMSE (m³/m³)",
            colors=colors,
        )
        print(f"Saved -> {rmse_path}")


if __name__ == "__main__":
    make_figure()

