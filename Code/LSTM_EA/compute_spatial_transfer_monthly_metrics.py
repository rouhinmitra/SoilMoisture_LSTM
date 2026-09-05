#!/usr/bin/env python
"""
Compute monthly spatial transferability metrics for LSTM, RF, and
Exponential Filter models.

Outputs a CSV with R2, Bias, RMSE, Corr for each:
  - station (Ne1, Ne2, Ne3, and ALL),
  - model (LSTM, RF, Exp),
  - year (2017–2023),
  - month (May–October).
"""

from pathlib import Path
from typing import Dict, List

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

YEAR_RANGE = (2017, 2023)
MONTH_RANGE = (5, 10)


def _filter_growing_season(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    df[date_col] = pd.to_datetime(df[date_col])
    mask = (
        (df[date_col].dt.year >= YEAR_RANGE[0])
        & (df[date_col].dt.year <= YEAR_RANGE[1])
        & (df[date_col].dt.month >= MONTH_RANGE[0])
        & (df[date_col].dt.month <= MONTH_RANGE[1])
    )
    return df.loc[mask].copy()


def compute_metrics(measured: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    residuals = predicted - measured
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((measured - np.mean(measured)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    bias = float(np.mean(residuals))
    rmse = float(np.sqrt(np.mean(residuals**2)))
    corr = float(np.corrcoef(measured, predicted)[0, 1]) if len(measured) > 1 else np.nan
    return {"r2": r2, "bias": bias, "rmse": rmse, "corr": corr}


def load_lstm(station: str) -> pd.DataFrame:
    path = LSTM_DIR / f"predictions_{station}.csv"
    df = pd.read_csv(path)
    df = df.rename(columns={"Date": "date", "RZSM_obs": "measured", "RZSM_pred": "predicted"})
    df = _filter_growing_season(df)
    df["model"] = "LSTM"
    df["station"] = station
    df["measured"] = df["measured"] / 100.0
    df["predicted"] = df["predicted"] / 100.0
    return df[["station", "model", "date", "measured", "predicted"]]


def load_rf(station: str) -> pd.DataFrame:
    fold = RF_FOLD_MAP[station]
    path = RF_DIR / fold / "predictions.csv"
    df = pd.read_csv(path)
    df = df.rename(columns={"date": "date", "actual": "measured", "predicted": "predicted"})
    df = _filter_growing_season(df)
    df["model"] = "RF"
    df["station"] = station
    df["measured"] = df["measured"] / 100.0
    df["predicted"] = df["predicted"] / 100.0
    return df[["station", "model", "date", "measured", "predicted"]]


def load_exp(exp_all: pd.DataFrame, station: str) -> pd.DataFrame:
    df = exp_all.loc[exp_all["test_station"] == station].copy()
    df = df.rename(columns={"Date": "date", "RZSM_true": "measured", "RZSM_pred": "predicted"})
    df = _filter_growing_season(df)
    df["model"] = "Exp"
    df["station"] = station
    df["measured"] = df["measured"] / 100.0
    df["predicted"] = df["predicted"] / 100.0
    return df[["station", "model", "date", "measured", "predicted"]]


def build_long_dataframe() -> pd.DataFrame:
    exp_all = pd.read_csv(EXP_CSV)

    frames: List[pd.DataFrame] = []
    for station in STATIONS:
        frames.append(load_lstm(station))
        frames.append(load_rf(station))
        frames.append(load_exp(exp_all, station))

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    return df


def compute_monthly_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    grouped = df.groupby(["station", "model", "year", "month"], sort=True)
    for (station, model, year, month), g in grouped:
        n = len(g)
        if n < 10:
            continue
        metrics = compute_metrics(g["measured"].values, g["predicted"].values)
        rows.append(
            {
                "station": station,
                "model": model,
                "year": int(year),
                "month": int(month),
                "n_samples": int(n),
                **metrics,
            }
        )

    station_df = pd.DataFrame(rows)

    if station_df.empty:
        return station_df

    # Average across stations for each model/year/month
    avg_rows: List[Dict[str, object]] = []
    grouped_all = station_df.groupby(["model", "year", "month"], sort=True)
    for (model, year, month), g in grouped_all:
        avg_rows.append(
            {
                "station": "ALL",
                "model": model,
                "year": int(year),
                "month": int(month),
                "n_samples": int(g["n_samples"].sum()),
                "r2": float(g["r2"].mean()),
                "bias": float(g["bias"].mean()),
                "rmse": float(g["rmse"].mean()),
                "corr": float(g["corr"].mean()),
            }
        )

    avg_df = pd.DataFrame(avg_rows)
    full_df = pd.concat([station_df, avg_df], ignore_index=True)
    full_df = full_df.sort_values(["station", "model", "year", "month"]).reset_index(drop=True)
    return full_df


def main() -> None:
    df = build_long_dataframe()
    metrics_df = compute_monthly_metrics(df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "spatial_transfer_monthly_metrics.csv"
    metrics_df.to_csv(out_path, index=False, float_format="%.4f")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()

