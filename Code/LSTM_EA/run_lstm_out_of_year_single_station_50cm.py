#!/usr/bin/env python
"""
Out-of-year (year-wise held-out) cross-validation for LSTM / EA-LSTM
at a single station — 50cm root zone soil moisture.

Mirrors `run_lstm_out_of_year_single_station.py` but:
- predicts `RZSM_50_avg` instead of `RZSM_25_avg`
- runs only the year range (inclusive) 2016–2023
- does not do any additional anchored multi-window sweep
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from run_lstm_out_of_year_single_station import OOYCVConfig, STATIONS, run_ooy_for_station
from utils.logging_utils import setup_logging


def main() -> None:
    base_config = OOYCVConfig()
    base_config = replace(
        base_config,
        target_col="RZSM_50_avg",
        output_dir="outputs/lstm_out_of_year_single_station_50cm",
        year_range=(2016, 2023),
    )

    base_output_dir = Path(base_config.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    _ = setup_logging(str(base_output_dir), log_level="INFO")
    log = logging.getLogger(__name__)

    log.info("=" * 80)
    log.info("LSTM OUT-OF-YEAR (YEAR-WISE HELD-OUT) CV - SINGLE STATION (50cm)")
    log.info("=" * 80)
    log.info("Model type: %s", base_config.model_type)
    log.info("Stations: %s", list(STATIONS.keys()))
    log.info("Year range: %d-%d (single run)", base_config.year_range[0], base_config.year_range[1])
    log.info("Month range: May–October (run_ooy_for_station uses month_range=(4,10))")
    log.info("Target col: %s", base_config.target_col)

    windows: List[Tuple[int, int]] = [(base_config.year_range[0], base_config.year_range[1])]

    all_window_metrics: List[pd.DataFrame] = []
    per_station_window_means: List[pd.DataFrame] = []

    for window_start, window_end in windows:
        window_tag = f"years_{window_start}_{window_end}"
        window_output_dir = base_output_dir / window_tag
        window_output_dir.mkdir(parents=True, exist_ok=True)

        log.info("\n" + "=" * 80)
        log.info("WINDOW %s: %d-%d", window_tag, window_start, window_end)
        log.info("=" * 80)

        window_config = replace(
            base_config,
            year_range=(window_start, window_end),
            output_dir=str(window_output_dir),
        )

        window_station_metrics: List[pd.DataFrame] = []
        window_station_preds: List[pd.DataFrame] = []

        for station_name, data_file in STATIONS.items():
            data_path = Path(window_config.data_dir) / data_file
            if not data_path.exists():
                log.warning(
                    "Data file for station %s not found for window %s: %s",
                    station_name,
                    window_tag,
                    data_path,
                )
                continue

            try:
                station_metrics_df, station_preds_df = run_ooy_for_station(
                    station_name=station_name,
                    data_file=data_file,
                    config=window_config,
                    logger=log,
                )

                if not station_metrics_df.empty:
                    station_metrics_df = station_metrics_df.copy()
                    station_metrics_df["window_start"] = window_start
                    station_metrics_df["window_end"] = window_end
                    station_metrics_df["window_years"] = window_end - window_start + 1
                    window_station_metrics.append(station_metrics_df)
                    all_window_metrics.append(station_metrics_df)

                if not station_preds_df.empty:
                    station_preds_df = station_preds_df.copy()
                    station_preds_df["window_start"] = window_start
                    station_preds_df["window_end"] = window_end
                    station_preds_df["window_years"] = window_end - window_start + 1
                    window_station_preds.append(station_preds_df)

            except Exception as e:
                log.exception(
                    "Error processing station %s for window %s: %s",
                    station_name,
                    window_tag,
                    e,
                )
                continue

        # Per-window aggregation across stations
        if window_station_metrics:
            window_metrics_df = pd.concat(window_station_metrics, ignore_index=True)
            summary_path = window_output_dir / "all_stations_lstm_ooy_metrics.csv"
            window_metrics_df.to_csv(summary_path, index=False)

            fold_only = window_metrics_df[window_metrics_df["fold"] != "OVERALL"].copy()
            if not fold_only.empty:
                mean_by_station = (
                    fold_only.groupby("station", as_index=False)
                    .agg(
                        mean_r2=("r2", "mean"),
                        mean_rmse=("rmse", "mean"),
                        mean_mae=("mae", "mean"),
                        n_folds=("fold", "count"),
                    )
                    .sort_values("station")
                )
                mean_by_station["window_start"] = window_start
                mean_by_station["window_end"] = window_end
                mean_by_station["window_years"] = window_end - window_start + 1
                per_station_window_means.append(mean_by_station)

            log.info("Saved window metrics to %s", summary_path)

        if window_station_preds:
            combined_preds_df = pd.concat(window_station_preds, ignore_index=True)
            combined_preds_path = window_output_dir / "all_stations_lstm_ooy_predictions.csv"
            combined_preds_df.to_csv(combined_preds_path, index=False)
            log.info("Saved window predictions to %s", combined_preds_path)

    # Aggregate across all windows and stations (single window, but keep compatibility)
    if all_window_metrics:
        all_windows_df = pd.concat(all_window_metrics, ignore_index=True)
        all_windows_summary_path = base_output_dir / "all_windows_lstm_ooy_metrics.csv"
        all_windows_df.to_csv(all_windows_summary_path, index=False)
        log.info("Saved all-windows summary metrics to %s", all_windows_summary_path)

    if per_station_window_means:
        station_means_df = pd.concat(per_station_window_means, ignore_index=True)
        station_means_path = base_output_dir / "anchored_windows_station_mean_metrics.csv"
        station_means_df.to_csv(station_means_path, index=False)
        log.info("Saved anchored-window station mean metrics to %s", station_means_path)

    log.info("\n" + "=" * 80)
    log.info("OUT-OF-YEAR PROCESSING (50cm) COMPLETED")
    log.info("=" * 80)


if __name__ == "__main__":
    main()

