#!/usr/bin/env python
"""
Leave-One-Station-Out Cross-Validation for the Exponential Filter.

Trains on 2 stations, tests on the 3rd, for all 3 combinations:
- Fold 1: Train Ne1+Ne2, Test Ne3
- Fold 2: Train Ne1+Ne3, Test Ne2
- Fold 3: Train Ne2+Ne3, Test Ne1

Training approach: concatenate the two training sites' daily time series
end-to-end (each sorted by date).  The large dt gap between sites causes
exp(-dt/T) -> 0, which resets the filter gain K to ~1 automatically -- no
special boundary handling needed.

Normalisation at test time uses pooled min/max from both training sites:
    ssm_min = min(site_A_ssm_min, site_B_ssm_min)
    ssm_max = max(site_A_ssm_max, site_B_ssm_max)
    (same for RZSM)

This mirrors how the LSTM (run_cv.py) and RF spatial CV scripts pool
training data from both sites.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from run_exp_filter_out_of_year_single_station import (
    ExponentialFilter,
    _growing_season_mask,
    _nse,
    evaluate_metrics,
    load_station_data,
)
from utils.logging_utils import setup_logging


STATIONS: Dict[str, str] = {
    "Ne1": "ne1_1_maize.csv",
    "Ne2": "ne2_1_maize.csv",
    "Ne3": "ne3_1_maize.csv",
}

CV_FOLDS = [
    {"train": ["Ne1", "Ne2"], "test": "Ne3", "name": "fold1_test_Ne3"},
    {"train": ["Ne1", "Ne3"], "test": "Ne2", "name": "fold2_test_Ne2"},
    {"train": ["Ne2", "Ne3"], "test": "Ne1", "name": "fold3_test_Ne1"},
]


@dataclass
class ExpFilterSpatialConfig:
    """Configuration for exponential filter leave-one-station-out CV."""

    data_dir: str = "/Users/rouhinmitra/SM_work/Data/Base/"
    output_dir: str = "outputs/exp_filter_spatial_cv"
    target_col: str = "RZSM_25_avg"
    ssm_col: str = "SSM_avg"
    year_range: Tuple[int, int] = (2017, 2023)
    month_range: Tuple[int, int] = (5, 10)
    T_range: Tuple[int, int] = (1, 50)
    T_step: int = 1


# ---------------------------------------------------------------------------
# Spatial fold
# ---------------------------------------------------------------------------

def _optimise_T_spatial(
    train_dfs: List[pd.DataFrame],
    config: ExpFilterSpatialConfig,
) -> Tuple[int, float]:
    """
    Grid-search T by running the filter independently on each training site
    (full-year data for continuity) and selecting the T that maximises the
    mean growing-season NSE across sites.
    """
    gs_masks = [_growing_season_mask(df, config) for df in train_dfs]
    best_T, best_mean_nse = 1, -np.inf

    for T in range(config.T_range[0], config.T_range[1] + 1, config.T_step):
        nse_values = []
        for df, gs_mask in zip(train_dfs, gs_masks):
            ef = ExponentialFilter(T=T)
            pred, *_ = ef.fit(
                df[config.ssm_col].values,
                df[config.target_col].values,
                df["Date"].values,
            )
            rzsm_gs = df[config.target_col].values[gs_mask]
            pred_gs = pred[gs_mask]
            nse_values.append(_nse(rzsm_gs, pred_gs))
        mean_nse = float(np.mean(nse_values))
        if mean_nse > best_mean_nse:
            best_mean_nse, best_T = mean_nse, T

    return best_T, best_mean_nse


def run_spatial_fold(
    fold: Dict,
    config: ExpFilterSpatialConfig,
    log: logging.Logger,
) -> Dict:
    """
    Run a single leave-one-station-out fold.

    Returns a dict with metrics, predictions array, dates, and fold info.
    """
    fold_name = fold["name"]
    train_stations = fold["train"]
    test_station = fold["test"]

    log.info("\n" + "=" * 70)
    log.info("FOLD: %s", fold_name)
    log.info("Training on: %s", train_stations)
    log.info("Testing on:  %s", test_station)
    log.info("=" * 70)

    # Load each training station (full year, no month filter)
    train_dfs: List[pd.DataFrame] = []
    n_train_total = 0
    for stn in train_stations:
        df = load_station_data(STATIONS[stn], config, apply_month_filter=False)
        df = df.sort_values("Date").reset_index(drop=True)
        log.info("  Train station %s: %d full-year rows", stn, len(df))
        train_dfs.append(df)
        n_train_total += len(df)
    log.info("  Combined training rows: %d", n_train_total)

    # Load test station (full year for filter continuity)
    test_df = load_station_data(STATIONS[test_station], config, apply_month_filter=False)
    test_df = test_df.sort_values("Date").reset_index(drop=True)
    log.info("  Test station %s: %d full-year rows", test_station, len(test_df))

    # Optimise T independently on each training site (full year),
    # NSE evaluated on growing season only
    best_T, best_train_nse = _optimise_T_spatial(train_dfs, config)
    log.info("  Best T=%d  (mean train growing-season NSE=%.4f)", best_T, best_train_nse)

    # Pool normalisation params from both training sites
    ssm_min = min(df[config.ssm_col].min() for df in train_dfs)
    ssm_max = max(df[config.ssm_col].max() for df in train_dfs)
    rzsm_min = min(df[config.target_col].min() for df in train_dfs)
    rzsm_max = max(df[config.target_col].max() for df in train_dfs)
    log.info(
        "  Pooled norms: SSM [%.2f, %.2f], RZSM [%.2f, %.2f]",
        ssm_min, ssm_max, rzsm_min, rzsm_max,
    )

    # Run filter on full test year, then extract growing-season rows
    ef = ExponentialFilter(T=best_T)
    y_pred_full = ef.predict(
        test_df[config.ssm_col].values,
        test_df["Date"].values,
        ssm_min,
        ssm_max,
        rzsm_min,
        rzsm_max,
    )
    gs_mask = _growing_season_mask(test_df, config)
    y_pred = y_pred_full[gs_mask]
    y_true = test_df[config.target_col].values[gs_mask]
    dates_gs = test_df.loc[gs_mask, "Date"].values

    metrics = evaluate_metrics(y_true, y_pred)
    log.info(
        "  Test (growing season): R2=%.4f, RMSE=%.4f, MAE=%.4f, NSE=%.4f, Corr=%.4f",
        metrics["r2"],
        metrics["rmse"],
        metrics["mae"],
        metrics["nse"],
        metrics["correlation"],
    )

    # Build predictions DataFrame (growing season only)
    dates_test = pd.to_datetime(dates_gs)
    pred_df = pd.DataFrame(
        {
            "test_station": test_station,
            "train_stations": "+".join(train_stations),
            "Date": dates_test,
            "Year": dates_test.year,
            "RZSM_true": y_true,
            "RZSM_pred": y_pred,
        }
    )

    return {
        "fold_name": fold_name,
        "test_station": test_station,
        "train_stations": train_stations,
        "best_T": best_T,
        "metrics": metrics,
        "y_pred": y_pred,
        "y_true": y_true,
        "dates_test": dates_test,
        "pred_df": pred_df,
        "n_train": n_train_total,
        "n_test": len(y_true),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    config = ExpFilterSpatialConfig()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(str(output_dir), log_level="INFO")
    log = logging.getLogger(__name__)

    log.info("=" * 80)
    log.info("EXPONENTIAL FILTER LEAVE-ONE-STATION-OUT (SPATIAL TRANSFER)")
    log.info("=" * 80)
    log.info("Stations: %s", list(STATIONS.keys()))
    log.info("Folds: %d", len(CV_FOLDS))
    log.info("Year range: %d-%d", config.year_range[0], config.year_range[1])
    log.info("Filter runs on full-year data; evaluation on months %d-%d", config.month_range[0], config.month_range[1])
    log.info("T search range: %d-%d (step %d)", config.T_range[0], config.T_range[1], config.T_step)
    log.info("Target: %s | Input: %s", config.target_col, config.ssm_col)

    results: List[Dict] = []
    for fold in CV_FOLDS:
        try:
            result = run_spatial_fold(fold, config, log)
            results.append(result)
        except Exception as e:
            log.exception("Error in fold %s: %s", fold["name"], e)

    if not results:
        log.warning("No successful folds. Exiting.")
        return

    # ---- Save per-fold results CSV ----------------------------------------
    rows = []
    for r in results:
        rows.append(
            {
                "test_station": r["test_station"],
                "train_stations": "+".join(r["train_stations"]),
                "best_T": r["best_T"],
                **r["metrics"],
                "n_train": r["n_train"],
                "n_test": r["n_test"],
            }
        )
    results_df = pd.DataFrame(rows)
    results_path = output_dir / "cv_results.csv"
    results_df.to_csv(results_path, index=False)
    log.info("Results saved -> %s", results_path)

    # ---- Save combined predictions CSV ------------------------------------
    pred_dfs = [r["pred_df"] for r in results]
    combined_preds = pd.concat(pred_dfs, ignore_index=True)
    preds_path = output_dir / "cv_predictions.csv"
    combined_preds.to_csv(preds_path, index=False)
    log.info("Predictions saved -> %s", preds_path)

    # ---- Print summary ----------------------------------------------------
    log.info("\n" + "=" * 100)
    log.info("LEAVE-ONE-STATION-OUT EXPONENTIAL FILTER RESULTS")
    log.info("=" * 100)

    header = (
        f"{'Test':<8} {'Train':<12} {'T':>4} "
        f"{'R2':>8} {'RMSE':>8} {'MAE':>8} {'NSE':>8} {'Corr':>8} "
        f"{'nRMSE%':>8} {'ubRMSE%':>9} {'Bias%':>8} "
        f"{'N_train':>8} {'N_test':>8}"
    )
    log.info(header)
    log.info("-" * len(header))

    r2_scores, rmse_scores = [], []
    for r in results:
        m = r["metrics"]
        line = (
            f"{r['test_station']:<8} {'+'.join(r['train_stations']):<12} {r['best_T']:>4} "
            f"{m['r2']:>8.4f} {m['rmse']:>8.4f} {m['mae']:>8.4f} "
            f"{m['nse']:>8.4f} {m['correlation']:>8.4f} "
            f"{m['nrmse_pct']:>8.2f} {m['ubrmse_pct']:>9.2f} {m['bias_pct']:>8.2f} "
            f"{r['n_train']:>8} {r['n_test']:>8}"
        )
        log.info(line)
        r2_scores.append(m["r2"])
        rmse_scores.append(m["rmse"])

    log.info("-" * len(header))
    log.info(
        "MEAN  R2=%.4f  RMSE=%.4f  |  STD  R2=%.4f  RMSE=%.4f",
        np.mean(r2_scores),
        np.mean(rmse_scores),
        np.std(r2_scores),
        np.std(rmse_scores),
    )
    log.info("=" * 100)
    log.info("Exponential filter spatial CV complete.")


if __name__ == "__main__":
    main()
