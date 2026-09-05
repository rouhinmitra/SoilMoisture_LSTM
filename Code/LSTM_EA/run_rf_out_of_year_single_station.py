#!/usr/bin/env python
"""
Out-of-year (year-wise held-out) cross-validation for Random Forest
using the LSTM_EA data pipeline (Base data, sliding windows, Presto/static
features), evaluated per station.

For each station (Ne1, Ne2, Ne3):
- Build sliding-window sequences with DataProcessor
- Convert sequences to last-timestep tabular features (dynamic + static)
- Perform out-of-year splits: each calendar year is held out in turn
- Within each fold, optionally tune RF hyperparameters via random search
  on an inner validation year, then train on all training years and
  evaluate on the held-out test year.

Outputs:
- Per-station, per-fold prediction CSVs and metrics
- Per-station aggregated metrics (including optional OVERALL row)
- All-stations metrics and predictions CSVs
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.config import DataConfig, FeatureConfig
from src.data_loader import DataProcessor
from utils.logging_utils import setup_logging
from random_forest_tuned import (
    RFTunedConfig,
    _compute_metrics,
    _split_train_val_by_year,
    tune_random_forest,
)


# Same station mapping as LSTM and RF spatial scripts
STATIONS: Dict[str, str] = {
    "Ne1": "ne1_1_maize.csv",
    "Ne2": "ne2_1_maize.csv",
    "Ne3": "ne3_1_maize.csv",
}


@dataclass
class RFOOYConfig(RFTunedConfig):
    """
    Random Forest out-of-year (temporal transfer) configuration.

    Inherits data/feature/hyperparameter defaults from RFTunedConfig but:
    - Uses a different output_dir
    - Restricts year_range to exclude 2024
    - Allows enabling temporal features for parity with LSTM temporal runs.
    """

    output_dir: str = "outputs/rf_out_of_year_single_station"
    # Exclude 2024 from temporal transfer
    year_range: Tuple[int, int] = (2016, 2023)
    # Optional temporal features (day-of-year encoding)
    add_temporal: bool = True


def build_out_of_year_splits(
    dates: np.ndarray,
    min_train_years: int = 2,
    verbose: bool = True,
) -> List[Dict]:
    """
    Build out-of-year splits at the sequence level.

    Parameters
    ----------
    dates : np.ndarray
        Array of per-sequence target dates (1D, datetime64)
    min_train_years : int
        Minimum number of distinct training years required to keep a fold
    verbose : bool
        If True, print basic information about the splits
    """
    if dates.size == 0:
        return []

    years = pd.to_datetime(dates).year.values
    unique_years = np.sort(np.unique(years))

    if verbose:
        print("\n=== OUT-OF-YEAR SPLITS (RF, SINGLE STATION) ===")
        print(f"Available years in sequences: {list(unique_years)}")

    if unique_years.size <= 1:
        print("Not enough distinct years for out-of-year CV.")
        return []

    splits: List[Dict] = []
    all_indices = np.arange(len(years))

    for fold_idx, test_year in enumerate(unique_years, start=1):
        test_mask = years == test_year
        train_mask = ~test_mask

        test_idx = all_indices[test_mask]
        train_idx = all_indices[train_mask]

        train_years = np.unique(years[train_idx])

        if train_idx.size == 0 or train_years.size < min_train_years:
            print(
                f"Skipping year {test_year}: only {train_years.size} "
                f"training years available (min_train_years={min_train_years})"
            )
            continue

        split = {
            "name": f"fold_{fold_idx}_test_{int(test_year)}",
            "test_year": int(test_year),
            "train_idx": train_idx,
            "test_idx": test_idx,
        }
        splits.append(split)

        if verbose:
            print(
                f"Fold {fold_idx}: test_year={test_year}, "
                f"train_years={list(train_years)}, "
                f"n_train={len(train_idx)}, n_test={len(test_idx)}"
            )

    return splits


def run_ooy_for_station(
    station_name: str,
    data_file: str,
    config: RFOOYConfig,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run out-of-year RF temporal transfer for a single station.

    Returns
    -------
    (station_metrics_df, station_predictions_df)
    """
    logger.info("\n" + "=" * 70)
    logger.info("RF OUT-OF-YEAR CV: Station %s", station_name)
    logger.info("=" * 70)

    station_output_dir = Path(config.output_dir) / station_name
    station_output_dir.mkdir(parents=True, exist_ok=True)

    # Configure data loading: single station as both "train" and "test"
    data_config = DataConfig(
        train_files=[data_file],
        test_files=[data_file],
        data_dir=config.data_dir,
        seq_length=config.seq_length,
        nan_threshold=config.nan_threshold,
        same_year_constraint=True,
        month_range=(4, 10),  # April–October, consistent with LSTM temporal
        year_range=config.year_range,
    )

    # Resolve Presto embeddings path
    presto_path = config.presto_embeddings_path
    if config.use_presto_static:
        p = Path(presto_path)
        if not p.is_absolute():
            presto_path = str(Path(__file__).resolve().parent / p)
        if not Path(presto_path).exists():
            raise FileNotFoundError(
                f"Presto embeddings file not found: {presto_path}. "
                "Set use_presto_static=False or provide a valid presto_embeddings_path."
            )

    feature_config = FeatureConfig(
        dynamic_cols=config.dynamic_cols,
        static_cols=config.static_cols,
        target_col=config.target_col,
        add_temporal=config.add_temporal,
        temporal_features=["doy_sin", "doy_cos"],
        use_presto_static=config.use_presto_static,
        presto_embeddings_path=presto_path if config.use_presto_static else None,
    )

    logger.info("Preparing data for station %s ...", station_name)
    processor = DataProcessor(data_config, feature_config)
    # Use val_years for scaler fitting (as in tuned RF / LSTM scripts)
    val_years_for_scaler = (
        config.val_years if getattr(config, "val_years", None) is not None else None
    )
    data = processor.prepare_data(
        data_config.train_files,
        data_config.test_files,
        val_years=val_years_for_scaler,
    )

    X_d = data["X_d_train"]
    X_s = data["X_s_train"]
    y = data["y_train"]
    dates_train = data["dates_train"]

    logger.info("Sequences for %s: %d", station_name, len(y))
    if len(y) == 0:
        logger.warning("No sequences available for station %s; skipping.", station_name)
        return pd.DataFrame(), pd.DataFrame()

    # Build last-timestep tabular features and target in original units
    X_full = np.concatenate([X_d[:, -1, :], X_s[:, -1, :]], axis=1)
    y_full_scaled = y[:, -1]
    y_full = processor.scaler_y.inverse_transform(
        y_full_scaled.reshape(-1, 1)
    ).ravel()

    # Out-of-year splits for this station
    splits = build_out_of_year_splits(
        dates_train,
        min_train_years=2,
        verbose=True,
    )
    if not splits:
        logger.warning("No valid out-of-year splits for station %s; skipping.", station_name)
        return pd.DataFrame(), pd.DataFrame()

    years_all = pd.to_datetime(dates_train).year.values

    fold_metric_rows: List[Dict] = []
    fold_pred_dfs: List[pd.DataFrame] = []
    all_y_true: List[np.ndarray] = []
    all_y_pred: List[np.ndarray] = []

    for fold_idx, split in enumerate(splits, start=1):
        fold_name = split["name"]
        test_year = split["test_year"]
        train_idx = split["train_idx"]
        test_idx = split["test_idx"]

        logger.info("\n%s", "-" * 70)
        logger.info(
            "Fold %d (%s): test_year=%d, n_train=%d, n_test=%d",
            fold_idx,
            fold_name,
            test_year,
            len(train_idx),
            len(test_idx),
        )

        X_train_fold = X_full[train_idx]
        y_train_fold = y_full[train_idx]
        X_test_fold = X_full[test_idx]
        y_test_fold = y_full[test_idx]

        # Inner train/validation split within training years
        train_inner_idx, val_idx = _split_train_val_by_year(
            dates_train[train_idx], config.val_years
        )
        X_inner_train = X_train_fold[train_inner_idx]
        y_inner_train = y_train_fold[train_inner_idx]
        X_val = X_train_fold[val_idx]
        y_val = y_train_fold[val_idx]

        logger.info(
            "Inner (year-based) split sizes for fold %s: train=%d, val=%d (outer train=%d)",
            fold_name,
            len(y_inner_train),
            len(y_val),
            len(y_train_fold),
        )

        # Tune RF on inner train/val split (random search style)
        best_params = tune_random_forest(
            X_inner_train,
            y_inner_train,
            X_val,
            y_val,
            config,
            logger,
        )

        # Refit final RF on all outer training windows (train + val)
        n_estimators = int(best_params["best_n_estimators"])
        max_depth = best_params["best_max_depth"]
        min_samples_split = int(best_params["best_min_samples_split"])
        min_samples_leaf = int(best_params["best_min_samples_leaf"])
        max_features = best_params["best_max_features"]
        bootstrap = bool(best_params["best_bootstrap"])
        max_samples = best_params["best_max_samples"]

        final_model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            max_samples=max_samples,
            random_state=config.seed,
            n_jobs=-1,
        )
        final_model.fit(X_train_fold, y_train_fold)

        # Evaluate on held-out test year
        y_pred_fold = final_model.predict(X_test_fold)
        metrics = _compute_metrics(y_pred_fold, y_test_fold)

        # May–Oct R² on this fold
        dates_fold = pd.to_datetime(dates_train[test_idx])
        months_fold = dates_fold.month
        mask_mo = (months_fold >= 5) & (months_fold <= 10)
        n_may_oct = int(np.sum(mask_mo))
        r2_may_oct = np.nan
        if n_may_oct > 1:
            r2_may_oct = float(
                1
                - np.sum((y_test_fold[mask_mo] - y_pred_fold[mask_mo]) ** 2)
                / np.sum(
                    (y_test_fold[mask_mo] - np.mean(y_test_fold[mask_mo])) ** 2
                )
            )

        logger.info(
            "Fold %s metrics: R2=%.4f, RMSE=%.4f, MAE=%.4f",
            fold_name,
            metrics["r2"],
            metrics["rmse"],
            metrics["mae"],
        )

        fold_dir = station_output_dir / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Save per-fold predictions
        pred_df = pd.DataFrame(
            {
                "station": station_name,
                "fold": fold_name,
                "test_year": int(test_year),
                "Date": dates_fold,
                "Year": dates_fold.year.values,
                "RZSM_true": y_test_fold,
                "RZSM_pred": y_pred_fold,
            }
        )
        pred_path = fold_dir / "predictions.csv"
        pred_df.to_csv(pred_path, index=False)
        logger.info("Fold predictions saved to %s", pred_path)

        # Collect fold-level metrics
        metric_row = {
            "station": station_name,
            "fold": fold_name,
            "test_year": int(test_year),
            "r2": metrics["r2"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "bias": metrics["bias"],
            "correlation": metrics["correlation"],
            "n_samples": metrics["n_samples"],
            "n_train_windows": int(len(y_train_fold)),
            "n_val_windows": int(len(y_val)),
            "n_test_windows": int(len(y_test_fold)),
            "r2_may_oct": r2_may_oct,
            "n_may_oct": n_may_oct,
            "best_n_estimators": best_params.get("best_n_estimators"),
            "best_max_depth": best_params.get("best_max_depth"),
            "best_min_samples_split": best_params.get("best_min_samples_split"),
            "best_min_samples_leaf": best_params.get("best_min_samples_leaf"),
            "best_max_features": best_params.get("best_max_features"),
            "best_bootstrap": best_params.get("best_bootstrap"),
            "best_max_samples": best_params.get("best_max_samples"),
            "best_val_r2": best_params.get("best_val_r2"),
            "best_val_rmse": best_params.get("best_val_rmse"),
            "best_val_mae": best_params.get("best_val_mae"),
        }
        fold_metric_rows.append(metric_row)

        fold_pred_dfs.append(pred_df)
        all_y_true.append(y_test_fold)
        all_y_pred.append(y_pred_fold)

    if not fold_metric_rows:
        logger.warning("No successful folds for station %s.", station_name)
        return pd.DataFrame(), pd.DataFrame()

    station_metrics_df = pd.DataFrame(fold_metric_rows)
    station_preds_df = pd.concat(fold_pred_dfs, ignore_index=True)

    # Optional overall pooled metrics for this station
    all_y_true_arr = np.concatenate(all_y_true)
    all_y_pred_arr = np.concatenate(all_y_pred)
    overall_metrics = _compute_metrics(all_y_pred_arr, all_y_true_arr)
    overall_row = {
        "station": station_name,
        "fold": "OVERALL",
        "test_year": np.nan,
        "r2": overall_metrics["r2"],
        "rmse": overall_metrics["rmse"],
        "mae": overall_metrics["mae"],
        "bias": overall_metrics["bias"],
        "correlation": overall_metrics["correlation"],
        "n_samples": overall_metrics["n_samples"],
        "n_train_windows": np.nan,
        "n_val_windows": np.nan,
        "n_test_windows": np.nan,
        "r2_may_oct": np.nan,
        "n_may_oct": np.nan,
        "best_n_estimators": np.nan,
        "best_max_depth": np.nan,
        "best_min_samples_split": np.nan,
        "best_min_samples_leaf": np.nan,
        "best_max_features": np.nan,
        "best_bootstrap": np.nan,
        "best_max_samples": np.nan,
        "best_val_r2": np.nan,
        "best_val_rmse": np.nan,
        "best_val_mae": np.nan,
    }
    station_metrics_df = pd.concat(
        [station_metrics_df, pd.DataFrame([overall_row])],
        ignore_index=True,
    )

    # Save per-station outputs
    metrics_path = station_output_dir / f"{station_name}_rf_ooy_metrics.csv"
    preds_path = station_output_dir / f"{station_name}_rf_ooy_predictions.csv"
    station_metrics_df.to_csv(metrics_path, index=False)
    station_preds_df.to_csv(preds_path, index=False)
    logger.info("Saved station metrics to %s", metrics_path)
    logger.info("Saved station predictions to %s", preds_path)

    return station_metrics_df, station_preds_df


def main() -> None:
    """Run RF out-of-year temporal transfer for all stations."""
    config = RFOOYConfig()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(str(output_dir), log_level="INFO")
    log = logging.getLogger(__name__)

    log.info("=" * 80)
    log.info("RANDOM FOREST OUT-OF-YEAR (TEMPORAL TRANSFER) - SINGLE STATION")
    log.info("=" * 80)
    log.info("Stations: %s", list(STATIONS.keys()))
    log.info("Year range: %d-%d", config.year_range[0], config.year_range[1])
    log.info("Month range: April–October (4-10)")
    log.info("Sequence length: %d", config.seq_length)
    log.info("Validation years for tuning: %s", str(config.val_years))
    log.info("Output directory: %s", str(output_dir))

    if config.use_presto_static:
        presto_path = Path(config.presto_embeddings_path)
        if not presto_path.is_absolute():
            presto_path = Path(__file__).resolve().parent / presto_path
        log.info("Presto embeddings: ENABLED")
        log.info("  Path: %s", str(presto_path))
        if not presto_path.exists():
            raise FileNotFoundError(
                f"Presto embeddings file not found: {presto_path}. "
                "Set use_presto_static=False or provide a valid presto_embeddings_path."
            )
    else:
        log.info("Presto embeddings: disabled")

    np.random.seed(config.seed)

    all_station_metrics: List[pd.DataFrame] = []
    all_station_preds: List[pd.DataFrame] = []

    for station_name, data_file in STATIONS.items():
        data_path = Path(config.data_dir) / data_file
        if not data_path.exists():
            log.warning("Data file for station %s not found: %s", station_name, data_path)
            continue

        try:
            station_metrics_df, station_preds_df = run_ooy_for_station(
                station_name, data_file, config, log
            )
            if not station_metrics_df.empty:
                all_station_metrics.append(station_metrics_df)
            if not station_preds_df.empty:
                all_station_preds.append(station_preds_df)
        except Exception as e:
            log.exception("Error processing station %s: %s", station_name, e)
            continue

    # Aggregate across stations
    if all_station_metrics:
        all_metrics_df = pd.concat(all_station_metrics, ignore_index=True)
        summary_path = output_dir / "all_stations_rf_ooy_metrics.csv"
        all_metrics_df.to_csv(summary_path, index=False)

        log.info("\n" + "=" * 80)
        log.info("ALL-STATIONS RF OUT-OF-YEAR SUMMARY")
        log.info("=" * 80)
        log.info("Per-fold metrics:")
        log.info(
            all_metrics_df[
                [
                    "station",
                    "fold",
                    "test_year",
                    "r2",
                    "rmse",
                    "mae",
                    "n_samples",
                ]
            ].to_string(index=False)
        )
        log.info("Summary metrics saved to %s", summary_path)

    if all_station_preds:
        combined_preds_df = pd.concat(all_station_preds, ignore_index=True)
        combined_preds_path = output_dir / "all_stations_rf_ooy_predictions.csv"
        combined_preds_df.to_csv(combined_preds_path, index=False)
        log.info("Combined predictions saved to %s", combined_preds_path)

    log.info("\n" + "=" * 80)
    log.info("RF OUT-OF-YEAR PROCESSING COMPLETED")
    log.info("=" * 80)


if __name__ == "__main__":
    main()

