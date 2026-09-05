#!/usr/bin/env python
"""
Out-of-year (year-wise held-out) cross-validation for the Exponential Filter
at a single station, looping over all stations.

The exponential filter estimates root zone soil moisture (RZSM) from surface
soil moisture (SSM) via a recursive Soil Water Index (SWI) formulation
governed by a single characteristic time parameter T.

For each station (Ne1, Ne2, Ne3):
- Load daily data and filter to growing season / year range
- Split by year: each fold holds out one year as test, trains on all others
- Optimise T (1-50 days) on training years by maximising NSE
- Apply the filter with optimal T to the held-out year using training
  normalisation parameters (no data leakage)

Outputs:
- Per-station, per-fold metrics and prediction CSVs
- Per-station aggregated metrics (including OVERALL row)
- All-stations metrics and predictions CSVs
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from utils.logging_utils import setup_logging


STATIONS: Dict[str, str] = {
    "Ne1": "ne1_1_maize.csv",
    "Ne2": "ne2_1_maize.csv",
    "Ne3": "ne3_1_maize.csv",
}


@dataclass
class ExpFilterOOYConfig:
    """Configuration for exponential filter out-of-year temporal transfer."""

    data_dir: str = "/Users/rouhinmitra/SM_work/Data/Base/"
    output_dir: str = "outputs/exp_filter_out_of_year_single_station"
    target_col: str = "RZSM_25_avg"
    ssm_col: str = "SSM_avg"
    # Mirror original script behaviour: use years > 2015 and exclude 2023
    # i.e., effectively 2016–2022 inclusive.
    year_range: Tuple[int, int] = (2016, 2023)
    month_range: Tuple[int, int] = (5, 10)
    T_range: Tuple[int, int] = (1, 50)
    T_step: int = 1
    min_train_years: int = 2


# ---------------------------------------------------------------------------
# Exponential Filter
# ---------------------------------------------------------------------------

class ExponentialFilter:
    """
    Recursive exponential filter (Soil Water Index) for estimating RZSM
    from SSM.  Governed by a single parameter T (characteristic time in days).
    """

    def __init__(self, T: float = 10):
        self.T = T

    @staticmethod
    def _normalize(data: np.ndarray):
        mn, mx = np.min(data), np.max(data)
        return (data - mn) / (mx - mn), mn, mx

    @staticmethod
    def _denormalize(norm: np.ndarray, mn: float, mx: float):
        return norm * (mx - mn) + mn

    def _run_filter(self, ssm_norm: np.ndarray, dates: np.ndarray) -> np.ndarray:
        n = len(ssm_norm)
        swi = np.zeros(n)
        k = np.zeros(n)
        swi[0] = ssm_norm[0]
        k[0] = 1.0
        for i in range(1, n):
            dt = (dates[i] - dates[i - 1]).days
            k[i] = k[i - 1] / (k[i - 1] + np.exp(-dt / self.T)) if k[i - 1] != 0 else 1.0
            swi[i] = swi[i - 1] + k[i] * (ssm_norm[i] - swi[i - 1])
        return swi

    def fit(
        self,
        ssm: np.ndarray,
        rzsm: np.ndarray,
        dates: np.ndarray,
    ) -> Tuple[np.ndarray, float, float, float, float]:
        """Apply filter on training data; return predictions + norm params."""
        dates = pd.to_datetime(dates)
        ssm_norm, ssm_min, ssm_max = self._normalize(ssm)
        _, rzsm_min, rzsm_max = self._normalize(rzsm)
        swi_norm = self._run_filter(ssm_norm, dates)
        swi_denorm = self._denormalize(swi_norm, rzsm_min, rzsm_max)
        return swi_denorm, ssm_min, ssm_max, rzsm_min, rzsm_max

    def predict(
        self,
        ssm: np.ndarray,
        dates: np.ndarray,
        ssm_min: float,
        ssm_max: float,
        rzsm_min: float,
        rzsm_max: float,
    ) -> np.ndarray:
        """Apply filter on new data using training normalisation params."""
        dates = pd.to_datetime(dates)
        ssm_norm = (ssm - ssm_min) / (ssm_max - ssm_min)
        swi_norm = self._run_filter(ssm_norm, dates)
        return self._denormalize(swi_norm, rzsm_min, rzsm_max)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _nse(obs: np.ndarray, pred: np.ndarray) -> float:
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1.0 - ss_res / ss_tot


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mean_true = np.mean(y_true)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))
    denom = mean_true if mean_true != 0 and not np.isnan(mean_true) else np.nan
    nrmse_pct = (rmse / denom) * 100 if denom and not np.isnan(denom) else np.nan
    bias_pct = (bias / denom) * 100 if denom and not np.isnan(denom) else np.nan
    ubrmse = np.sqrt(max(rmse ** 2 - bias ** 2, 0.0))
    ubrmse_pct = (ubrmse / denom) * 100 if denom and not np.isnan(denom) else np.nan
    try:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    except Exception:
        corr = np.nan

    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(rmse),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "bias": bias,
        "correlation": corr,
        "nse": float(_nse(y_true, y_pred)),
        "nrmse_pct": float(nrmse_pct) if not np.isnan(nrmse_pct) else np.nan,
        "ubrmse_pct": float(ubrmse_pct) if not np.isnan(ubrmse_pct) else np.nan,
        "bias_pct": float(bias_pct) if not np.isnan(bias_pct) else np.nan,
        "n_samples": len(y_true),
    }


# ---------------------------------------------------------------------------
# Data loading (lightweight, no sliding windows)
# ---------------------------------------------------------------------------

def load_station_data(
    data_file: str,
    config,
    apply_month_filter: bool = False,
) -> pd.DataFrame:
    """Load a station CSV and apply year/month filters."""
    path = Path(config.data_dir) / data_file
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month

    # Dynamic target depth selection (mirror old script):
    # Prefer 50 cm if available and non-NaN, else 25 cm, else fallback.
    target_col = config.target_col
    if "RZSM_50_avg" in df.columns and df["RZSM_50_avg"].notna().any():
        target_col = "RZSM_50_avg"
    elif "RZSM_25_avg" in df.columns and df["RZSM_25_avg"].notna().any():
        target_col = "RZSM_25_avg"
    config.target_col = target_col

    # Drop rows with missing key variables now that target_col is known
    df = df.dropna(subset=["Date", config.ssm_col, config.target_col])

    # Year filtering: mirror original behaviour via year_range (2016–2022)
    start_yr, end_yr = config.year_range
    df = df[(df["Year"] >= start_yr) & (df["Year"] <= end_yr)]

    # Optional month filter (typically used to restrict to growing season
    # before cross-validation, matching the original script).
    if apply_month_filter:
        start_m, end_m = config.month_range
        df = df[(df["Month"] >= start_m) & (df["Month"] <= end_m)]

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Out-of-year cross-validation
# ---------------------------------------------------------------------------

def _growing_season_mask(df: pd.DataFrame, config) -> np.ndarray:
    """Boolean mask for rows within the configured growing-season months."""
    start_m, end_m = config.month_range
    return ((df["Month"] >= start_m) & (df["Month"] <= end_m)).values


def _optimise_T(
    train_df: pd.DataFrame,
    config: ExpFilterOOYConfig,
) -> Tuple[int, float]:
    """Grid-search T on training data, return (best_T, best_nse).
    """
    rzsm = train_df[config.target_col].values

    best_T, best_nse = 1, -np.inf
    for T in range(config.T_range[0], config.T_range[1] + 1, config.T_step):
        ef = ExponentialFilter(T=T)
        pred, *_ = ef.fit(
            train_df[config.ssm_col].values,
            rzsm,
            train_df["Date"].values,
        )
        # NSE over all (already month-filtered) training rows, matching
        # the original optimisation objective.
        nse = _nse(rzsm, pred)
        if nse > best_nse:
            best_nse, best_T = nse, T
    return best_T, best_nse


def run_ooy_for_station(
    station_name: str,
    data_file: str,
    config: ExpFilterOOYConfig,
    log: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run out-of-year exponential filter CV for one station."""
    log.info("\n" + "=" * 70)
    log.info("EXP-FILTER OUT-OF-YEAR CV: Station %s", station_name)
    log.info("=" * 70)

    station_dir = Path(config.output_dir) / station_name
    station_dir.mkdir(parents=True, exist_ok=True)

    # Load month-filtered data so both training and testing operate on the
    # same growing-season subset, mirroring the original script.
    df = load_station_data(data_file, config, apply_month_filter=True)
    log.info(
        "Loaded %d growing-season rows for %s (years %d-%d, months %d-%d)",
        len(df),
        station_name,
        config.year_range[0],
        config.year_range[1],
        config.month_range[0],
        config.month_range[1],
    )

    unique_years = sorted(df["Year"].unique())
    log.info("Available years: %s", unique_years)

    if len(unique_years) < config.min_train_years + 1:
        log.warning("Not enough years for OOY CV at station %s; skipping.", station_name)
        return pd.DataFrame(), pd.DataFrame()

    fold_metric_rows: List[Dict] = []
    fold_pred_dfs: List[pd.DataFrame] = []
    all_y_true: List[np.ndarray] = []
    all_y_pred: List[np.ndarray] = []

    for fold_idx, test_year in enumerate(unique_years, start=1):
        train_df = df[df["Year"] != test_year].sort_values("Date").reset_index(drop=True)
        test_df = df[df["Year"] == test_year].sort_values("Date").reset_index(drop=True)

        train_years = sorted(train_df["Year"].unique())
        if len(train_years) < config.min_train_years or len(test_df) == 0:
            log.info(
                "Skipping year %d: %d training years (min %d), %d test rows",
                test_year,
                len(train_years),
                config.min_train_years,
                len(test_df),
            )
            continue

        fold_name = f"fold_{fold_idx}_test_{test_year}"

        log.info(
            "\n%s  Fold %d (%s): test_year=%d, n_train=%d, n_test=%d, train_years=%s",
            "-" * 60,
            fold_idx,
            fold_name,
            test_year,
            len(train_df),
            len(test_df),
            train_years,
        )

        # Optimise T on full-year training data, NSE evaluated on growing season
        best_T, best_train_nse = _optimise_T(train_df, config)
        log.info("  Best T=%d  (train NSE=%.4f)", best_T, best_train_nse)

        # Normalisation params from training data (already month-filtered)
        ef = ExponentialFilter(T=best_T)
        _, ssm_min, ssm_max = ef._normalize(train_df[config.ssm_col].values)
        _, rzsm_min, rzsm_max = ef._normalize(train_df[config.target_col].values)

        # Run filter on test year (already month-filtered)
        y_pred = ef.predict(
            test_df[config.ssm_col].values,
            test_df["Date"].values,
            ssm_min,
            ssm_max,
            rzsm_min,
            rzsm_max,
        )
        y_true = test_df[config.target_col].values
        dates_fold = test_df["Date"].values

        metrics = evaluate_metrics(y_true, y_pred)
        log.info(
            "  Test: R2=%.4f, RMSE=%.4f, MAE=%.4f, NSE=%.4f, Corr=%.4f",
            metrics["r2"],
            metrics["rmse"],
            metrics["mae"],
            metrics["nse"],
            metrics["correlation"],
        )

        metric_row = {
            "station": station_name,
            "fold": fold_name,
            "test_year": int(test_year),
            "best_T": best_T,
            **metrics,
        }
        fold_metric_rows.append(metric_row)

        dates_fold = pd.to_datetime(dates_fold)
        pred_df = pd.DataFrame(
            {
                "station": station_name,
                "fold": fold_name,
                "test_year": int(test_year),
                "Date": dates_fold,
                "Year": dates_fold.year,
                "RZSM_true": y_true,
                "RZSM_pred": y_pred,
            }
        )
        fold_pred_dfs.append(pred_df)
        all_y_true.append(y_true)
        all_y_pred.append(y_pred)

        # Save per-fold predictions
        fold_dir = station_dir / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(fold_dir / "predictions.csv", index=False)

    if not fold_metric_rows:
        log.warning("No successful folds for station %s.", station_name)
        return pd.DataFrame(), pd.DataFrame()

    # OVERALL pooled metrics
    all_true = np.concatenate(all_y_true)
    all_pred = np.concatenate(all_y_pred)
    overall = evaluate_metrics(all_true, all_pred)
    overall_row = {
        "station": station_name,
        "fold": "OVERALL",
        "test_year": np.nan,
        "best_T": np.nan,
        **overall,
    }

    station_metrics_df = pd.concat(
        [pd.DataFrame(fold_metric_rows), pd.DataFrame([overall_row])],
        ignore_index=True,
    )
    station_preds_df = pd.concat(fold_pred_dfs, ignore_index=True)

    metrics_path = station_dir / f"{station_name}_exp_filter_ooy_metrics.csv"
    preds_path = station_dir / f"{station_name}_exp_filter_ooy_predictions.csv"
    station_metrics_df.to_csv(metrics_path, index=False)
    station_preds_df.to_csv(preds_path, index=False)
    log.info("Saved station metrics  -> %s", metrics_path)
    log.info("Saved station predictions -> %s", preds_path)

    return station_metrics_df, station_preds_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    config = ExpFilterOOYConfig()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(str(output_dir), log_level="INFO")
    log = logging.getLogger(__name__)

    log.info("=" * 80)
    log.info("EXPONENTIAL FILTER OUT-OF-YEAR (TEMPORAL TRANSFER) - SINGLE STATION")
    log.info("=" * 80)
    log.info("Stations: %s", list(STATIONS.keys()))
    log.info("Year range: %d-%d", config.year_range[0], config.year_range[1])
    log.info("Filter runs on full-year data; evaluation on months %d-%d", config.month_range[0], config.month_range[1])
    log.info("T search range: %d-%d (step %d)", config.T_range[0], config.T_range[1], config.T_step)
    log.info("Target: %s | Input: %s", config.target_col, config.ssm_col)

    all_station_metrics: List[pd.DataFrame] = []
    all_station_preds: List[pd.DataFrame] = []

    for station_name, data_file in STATIONS.items():
        data_path = Path(config.data_dir) / data_file
        if not data_path.exists():
            log.warning("Data file for %s not found: %s", station_name, data_path)
            continue
        try:
            m_df, p_df = run_ooy_for_station(station_name, data_file, config, log)
            if not m_df.empty:
                all_station_metrics.append(m_df)
            if not p_df.empty:
                all_station_preds.append(p_df)
        except Exception as e:
            log.exception("Error processing station %s: %s", station_name, e)

    if all_station_metrics:
        combined_metrics = pd.concat(all_station_metrics, ignore_index=True)
        summary_path = output_dir / "all_stations_exp_filter_ooy_metrics.csv"
        combined_metrics.to_csv(summary_path, index=False)
        log.info("\nAll-stations metrics saved -> %s", summary_path)

        # Print summary table
        fold_only = combined_metrics[combined_metrics["fold"] != "OVERALL"]
        log.info("\n" + "=" * 80)
        log.info("ALL-STATIONS EXPONENTIAL FILTER OUT-OF-YEAR SUMMARY")
        log.info("=" * 80)
        log.info(
            "\n%s",
            fold_only[
                ["station", "fold", "test_year", "best_T", "r2", "rmse", "mae", "nse", "n_samples"]
            ].to_string(index=False),
        )

        overall_only = combined_metrics[combined_metrics["fold"] == "OVERALL"]
        log.info("\nOVERALL (pooled across folds):")
        log.info(
            "\n%s",
            overall_only[
                ["station", "r2", "rmse", "mae", "nse", "correlation", "n_samples"]
            ].to_string(index=False),
        )

    if all_station_preds:
        combined_preds = pd.concat(all_station_preds, ignore_index=True)
        preds_path = output_dir / "all_stations_exp_filter_ooy_predictions.csv"
        combined_preds.to_csv(preds_path, index=False)
        log.info("All-stations predictions saved -> %s", preds_path)

    log.info("\n" + "=" * 80)
    log.info("EXPONENTIAL FILTER OUT-OF-YEAR PROCESSING COMPLETED")
    log.info("=" * 80)


if __name__ == "__main__":
    main()
