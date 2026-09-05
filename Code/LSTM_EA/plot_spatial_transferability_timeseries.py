#!/usr/bin/env python
"""
Time-series panels for spatial transferability figure.

Creates a 3x3 grid:
 - Rows: US-Ne1, US-Ne2, US-Ne3
 - Columns: Poor, Moderate, Good fit years
Year selection is based on LSTM R² over May–October for each station.
Lines per panel:
 - SSM_avg observed
 - RZSM_25 observed
 - RZSM predicted by Exponential Filter
 - RZSM predicted by RF (tuned)
 - RZSM predicted by LSTM
"""

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

YEAR_RANGE: Tuple[int, int] = (2017, 2023)
MONTH_RANGE: Tuple[int, int] = (5, 10)


def _filter_growing_season(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    df[date_col] = pd.to_datetime(df[date_col])
    mask = (
        (df[date_col].dt.year >= YEAR_RANGE[0])
        & (df[date_col].dt.year <= YEAR_RANGE[1])
        & (df[date_col].dt.month >= MONTH_RANGE[0])
        & (df[date_col].dt.month <= MONTH_RANGE[1])
    )
    return df.loc[mask].copy()


def _compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    residuals = y_pred - y_true
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def load_lstm_raw(station: str) -> pd.DataFrame:
    """Load LSTM predictions with SSM and RZSM in percentage units."""
    path = LSTM_DIR / f"predictions_{station}.csv"
    df = pd.read_csv(path)
    df = df.rename(
        columns={
            "Date": "date",
            "SSM_avg_obs": "ssm",
            "RZSM_obs": "rzsm_obs",
            "RZSM_pred": "rzsm_lstm",
        }
    )
    df = _filter_growing_season(df)
    return df


def load_rf_raw(station: str) -> pd.DataFrame:
    """Load RF tuned predictions in percentage units."""
    fold = RF_FOLD_MAP[station]
    path = RF_DIR / fold / "predictions.csv"
    df = pd.read_csv(path)
    df = df.rename(columns={"date": "date", "actual": "rzsm_obs", "predicted": "rzsm_rf"})
    df = _filter_growing_season(df)
    return df


def load_exp_raw(exp_all: pd.DataFrame, station: str) -> pd.DataFrame:
    """Load exponential filter predictions in percentage units."""
    df = exp_all.loc[exp_all["test_station"] == station].copy()
    df = df.rename(columns={"Date": "date", "RZSM_true": "rzsm_obs", "RZSM_pred": "rzsm_exp"})
    df = _filter_growing_season(df)
    return df


def select_years_by_r2(lstm_df: pd.DataFrame) -> Dict[str, int]:
    """Return mapping {'poor': year_10pct, 'moderate': year_50pct, 'good': year_90pct}."""
    lstm_df = lstm_df.copy()
    lstm_df["date"] = pd.to_datetime(lstm_df["date"])
    lstm_df["year"] = lstm_df["date"].dt.year

    r2_by_year: Dict[int, float] = {}
    for year in range(YEAR_RANGE[0], YEAR_RANGE[1] + 1):
        df_y = lstm_df[lstm_df["year"] == year]
        if len(df_y) < 10:
            continue
        r2 = _compute_r2(df_y["rzsm_obs"].values, df_y["rzsm_lstm"].values)
        if not np.isnan(r2):
            r2_by_year[year] = r2

    if not r2_by_year:
        raise ValueError("No valid years found for LSTM R² computation.")

    years = np.array(sorted(r2_by_year.keys()))
    r2_values = np.array([r2_by_year[y] for y in years])

    def _pick_year(p: float) -> int:
        target = np.percentile(r2_values, p)
        idx = int(np.argmin(np.abs(r2_values - target)))
        return int(years[idx])

    return {
        "poor": _pick_year(10.0),
        "moderate": _pick_year(50.0),
        "good": _pick_year(90.0),
    }


def build_panel_data(
    station: str,
    year: int,
    lstm_df: pd.DataFrame,
    rf_df: pd.DataFrame,
    exp_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return merged DataFrame with aligned series for a station and year."""
    mask_lstm = pd.to_datetime(lstm_df["date"]).dt.year == year
    mask_rf = pd.to_datetime(rf_df["date"]).dt.year == year
    mask_exp = pd.to_datetime(exp_df["date"]).dt.year == year

    lstm_y = lstm_df.loc[mask_lstm, ["date", "ssm", "rzsm_obs", "rzsm_lstm"]].copy()
    rf_y = rf_df.loc[mask_rf, ["date", "rzsm_rf"]].copy()
    exp_y = exp_df.loc[mask_exp, ["date", "rzsm_exp"]].copy()

    merged = lstm_y.merge(rf_y, on="date", how="left").merge(exp_y, on="date", how="left")

    for col in ["ssm", "rzsm_obs", "rzsm_lstm", "rzsm_rf", "rzsm_exp"]:
        merged[col] = merged[col] / 100.0

    merged = merged.sort_values("date")
    return merged


def make_timeseries_figure() -> None:
    exp_all = pd.read_csv(EXP_CSV)

    lstm_by_station: Dict[str, pd.DataFrame] = {}
    rf_by_station: Dict[str, pd.DataFrame] = {}
    exp_by_station: Dict[str, pd.DataFrame] = {}
    year_choices: Dict[str, Dict[str, int]] = {}

    for station in STATIONS:
        lstm_df = load_lstm_raw(station)
        rf_df = load_rf_raw(station)
        exp_df = load_exp_raw(exp_all, station)

        lstm_by_station[station] = lstm_df
        rf_by_station[station] = rf_df
        exp_by_station[station] = exp_df
        year_choices[station] = select_years_by_r2(lstm_df)

    REFERENCE_YEAR = 2000

    fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True, sharey="row")

    colors = {
        "SSM": "silver",
        "RZSM_obs": "black",
        "LSTM": "blue",
        "RF": "green",
        "Exp": "orange",
    }

    fit_labels = ["Poor fit", "Moderate fit", "Good fit"]
    fit_keys = ["poor", "moderate", "good"]

    for row_idx, station in enumerate(STATIONS):
        for col_idx, (fit_key, fit_label) in enumerate(zip(fit_keys, fit_labels)):
            ax = axes[row_idx, col_idx]
            year = year_choices[station][fit_key]

            panel_df = build_panel_data(
                station,
                year,
                lstm_by_station[station],
                rf_by_station[station],
                exp_by_station[station],
            )

            real_dates = pd.to_datetime(panel_df["date"])
            plot_dates = real_dates.map(
                lambda d: d.replace(year=REFERENCE_YEAR)
            )

            r2 = _compute_r2(
                (panel_df["rzsm_obs"] * 100).values,
                (panel_df["rzsm_lstm"] * 100).values,
            )

            ax.plot(plot_dates, panel_df["ssm"], color=colors["SSM"], linewidth=1.0,linestyle="-.", label="SSM observed")
            ax.plot(plot_dates, panel_df["rzsm_obs"], color=colors["RZSM_obs"], linewidth=1.3, label="RZSM 25cm observed")
            ax.plot(plot_dates, panel_df["rzsm_lstm"], color=colors["LSTM"], linewidth=1.2, label="RZSM 25cm LSTM")
            ax.plot(plot_dates, panel_df["rzsm_rf"], color=colors["RF"], linewidth=1.0, label="RZSM 25cm RF tuned")
            ax.plot(plot_dates, panel_df["rzsm_exp"], color=colors["Exp"], linewidth=1.0, label="RZSM 25cm Exponential")

            ax.grid(True, alpha=0.3)

            title_station = f"US-{station}"
            ax.set_title(f"{title_station} ({fit_label})", fontsize=20)

            text_str = f"R\u00b2 = {r2:.3f}\n{year}"
            ax.text(
                0.95,
                0.27,
                text_str,
                transform=ax.transAxes,
                fontsize=10,
                va="top",
                ha="right",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2),
            )

        axes[row_idx, 0].set_ylabel("RZSM 25cm (m\u00b3/m\u00b3)", fontsize=15)

    ref_start = pd.Timestamp(f"{REFERENCE_YEAR}-05-01")
    ref_end = pd.Timestamp(f"{REFERENCE_YEAR}-10-31")
    for ax in axes.flat:
        ax.set_xlim(ref_start, ref_end)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.tick_params(axis="x", labelsize=15)

    for ax in axes[-1, :]:
        ax.set_xlabel("", fontsize=9)

    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.055),  # x=0.5 (center), y<1 lowers it

        ncol=5,
        fontsize=12,
        framealpha=0.9,
    )

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "spatial_transferability_timeseries.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")

    rows: List[Dict[str, object]] = []
    for station in STATIONS:
        for fit_key in fit_keys:
            year = year_choices[station][fit_key]
            lstm_df = lstm_by_station[station]
            mask_year = pd.to_datetime(lstm_df["date"]).dt.year == year
            df_y = lstm_df.loc[mask_year]
            r2 = _compute_r2(df_y["rzsm_obs"].values, df_y["rzsm_lstm"].values)
            rows.append(
                {
                    "station": station,
                    "fit_category": fit_key,
                    "year": year,
                    "lstm_r2_may_oct": r2,
                }
            )

    years_df = pd.DataFrame(rows)
    years_path = OUTPUT_DIR / "spatial_transferability_selected_years.csv"
    years_df.to_csv(years_path, index=False, float_format="%.4f")
    print(f"Saved -> {years_path}")


if __name__ == "__main__":
    make_timeseries_figure()

