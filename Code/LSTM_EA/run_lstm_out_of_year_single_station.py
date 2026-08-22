#!/usr/bin/env python
"""
Out-of-year (year-wise held-out) cross-validation for LSTM / EA-LSTM
at a single station, looping over all stations.

For each station (Ne1, Ne2, Ne3):
- Build sliding-window sequences with the existing DataProcessor
- Split sequences by year: each fold holds out one year as test,
  trains on all other years
- Within the training years, create a validation split (by val_years
  if provided, otherwise by random fraction)
- Train an LSTM / EA-LSTM per fold and evaluate on the held-out year

Outputs:
- Per-station, per-fold prediction CSVs and metrics
- Per-station aggregated metrics (including optional overall row)
- All-stations metrics summary CSV
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from dataclasses import dataclass, replace
from typing import List, Dict, Tuple, Optional, Union, Sequence
import logging

from src.config import DataConfig, FeatureConfig, ModelConfig, TrainingConfig
from src.data_loader import DataProcessor, RZSMDataset
from src.models import get_model
from src.trainer import Trainer
from src.evaluator import Evaluator
from utils.logging_utils import setup_logging


# Station mapping (same CSVs as other experiments)
STATIONS: Dict[str, str] = {
    "Ne1": "ne1_1_maize.csv",
    "Ne2": "ne2_1_maize.csv",
    "Ne3": "ne3_1_maize.csv",
}


def generate_year_windows(
    start_year: int,
    end_year: int,
    min_years: int = 3,
) -> List[Tuple[int, int]]:
    """
    Generate contiguous (start_year, end_year) windows within a base range.

    Windows are ordered from largest to smallest in terms of duration, and for
    each duration all possible contiguous windows are included.
    """
    if end_year < start_year:
        raise ValueError(f"end_year ({end_year}) must be >= start_year ({start_year})")

    total_years = end_year - start_year + 1
    if min_years < 1 or min_years > total_years:
        raise ValueError(
            f"min_years must be in [1, {total_years}], got {min_years}"
        )

    windows: List[Tuple[int, int]] = []
    for window_size in range(total_years, min_years - 1, -1):
        for offset in range(0, total_years - window_size + 1):
            ws = start_year + offset
            we = ws + window_size - 1
            windows.append((ws, we))

    return windows


def generate_anchored_year_windows(
    anchor_start_year: int,
    end_year: int,
    min_years: int = 3,
) -> List[Tuple[int, int]]:
    """
    Generate a single window per length, always anchored at anchor_start_year:
    (anchor_start_year, end_year), (anchor_start_year, end_year-1), ...,
    down to the minimum length.
    """
    if end_year < anchor_start_year:
        raise ValueError(
            f"end_year ({end_year}) must be >= anchor_start_year ({anchor_start_year})"
        )

    total_years = end_year - anchor_start_year + 1
    if min_years < 1 or min_years > total_years:
        raise ValueError(
            f"min_years must be in [1, {total_years}], got {min_years}"
        )

    windows: List[Tuple[int, int]] = []
    for window_end in range(end_year, anchor_start_year + min_years - 2, -1):
        windows.append((anchor_start_year, window_end))
    return windows


def generate_end_anchored_year_windows(
    start_year: int,
    anchor_end_year: int,
    min_years: int = 3,
) -> List[Tuple[int, int]]:
    """
    Generate windows anchored at a fixed end year:
    (start_year, anchor_end_year), (start_year+1, anchor_end_year), ...,
    down to the minimum length.
    """
    if anchor_end_year < start_year:
        raise ValueError(
            f"anchor_end_year ({anchor_end_year}) must be >= start_year ({start_year})"
        )

    total_years = anchor_end_year - start_year + 1
    if min_years < 1 or min_years > total_years:
        raise ValueError(
            f"min_years must be in [1, {total_years}], got {min_years}"
        )

    windows: List[Tuple[int, int]] = []
    # Last valid start keeps at least min_years in the window (inclusive range)
    last_start = anchor_end_year - min_years + 1
    for window_start in range(start_year, last_start + 1):
        windows.append((window_start, anchor_end_year))
    return windows


@dataclass
class OOYCVConfig:
    """
    Out-of-year CV configuration (single-station, year-wise held-out).
    Mirrors CVConfig from run_cv.py with minor adaptations.
    """

    data_dir: str = "/Users/rouhinmitra/SM_work/Data/Base/"
    output_dir: str = "outputs/lstm_out_of_year_single_station"
    seq_length: int = 20
    nan_threshold: float = 0.1

    # LSTM / EA-LSTM hyperparameters (copied from run_cv CVConfig)
    hidden_dim: int = 16
    dropout: float = 0.20
    batch_size: int = 32
    epochs: int = 150
    learning_rate: float = 0.0005
    early_stopping_patience: int = 20
    weight_decay: float = 0.001
    add_temporal: bool = False
    seed: int = 42
    model_type: str = "LSTM"  # "EALSTM" or "LSTM"
    num_layers: int = 1  # LSTM layers (only used when model_type="LSTM")

    # Validation strategy
    val_fraction: float = 0.15  # used if val_years is None
    val_years: Optional[Union[int, Sequence[int]]] = None

    # Restrict experiment data to this year range (inclusive). Excludes 2023 and 2024.
    year_range: Tuple[int, int] = (2016, 2023)

    # Minimum distinct training years required per fold
    min_train_years: int = 2

    # Presto embeddings as static features (added to Alpha Earth when True)
    use_presto_static: bool = False
    presto_embeddings_path: str = (
        "/Users/rouhinmitra/SM_work/Code/Data/s2_pixels/"
        "presto_embeddings_fused_interpolated.csv"
    )
    # When True, do not append Excel irrigation as an extra static feature
    exclude_irrigation_static: bool = False

    # Features
    dynamic_cols: List[str] = None
    static_cols: List[str] = None
    target_col: str = "RZSM_25_avg"

    def __post_init__(self):
        # Adjust default output_dir when using LSTM to be explicit
        if self.model_type == "LSTM" and self.output_dir == "outputs/cv_results_tuned":
            self.output_dir = "outputs/lstm_out_of_year_single_station"

        if self.dynamic_cols is None:
            self.dynamic_cols = [
                # Original features
                "SSM",
                "SSM_avg",
                "SWC_PI_F_2_1_1",
                "SWC_PI_F_3_1_1",
                "P_PI_F_1_1_1",
                "P_PI_F_2_2_1",
                "I",
                "TA_1_1_1",
                "RH_1_1_1",
                "LE_1_1_1",
                "NETRAD_1_1_1",
                # Sentinel-2 features
                # "ndvi",
                # "b11",
                # "b12",
                # "b2",
                # "b3",
                # "b4",
                # "b5",
                # "b6",
                # "b7",
                # "b8",
                # "b8a",
                # irrigation is intentionally not in dynamic_cols; it is added
                # as a static feature in the DataProcessor to control the input
                # gate (similar to Alpha Earth embeddings).
            ]

        if self.static_cols is None:
            # When use_presto_static: daily Presto (emb_*) merged in loader
            # + Alpha Earth + precip_jan_apr
            if self.use_presto_static:
                self.static_cols = [
                    *(f"emb_{k}" for k in range(128)),
                    *(f"A{i:02d}" for i in range(64)),
                    "precip_jan_apr",
                ]
            else:
                self.static_cols = [
                    # *(f"A{i:02d}" for i in range(64)),
                    "precip_jan_apr",
                    # "precip_may_oct",
                ]


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
        print("\n=== OUT-OF-YEAR SPLITS (SINGLE STATION) ===")
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
    config: OOYCVConfig,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run out-of-year CV for a single station.

    Returns
    -------
    (station_metrics_df, station_predictions_df)
    """
    logger.info("\n" + "=" * 70)
    logger.info(f"OUT-OF-YEAR CV: Station {station_name}")
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
        month_range=(4, 10),  # April–October (growing season, matches spatial CV)
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
        exclude_irrigation_static=config.exclude_irrigation_static,
    )

    model_config = ModelConfig(
        model_type=config.model_type,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        num_layers=config.num_layers,
    )

    training_config = TrainingConfig(
        batch_size=config.batch_size,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        early_stopping_patience=config.early_stopping_patience,
        weight_decay=config.weight_decay,
    )

    # Prepare data once for this station
    logger.info("Preparing data for station %s ...", station_name)
    processor = DataProcessor(data_config, feature_config)
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

    # Build out-of-year splits
    splits = build_out_of_year_splits(
        dates_train,
        min_train_years=config.min_train_years,
        verbose=True,
    )
    if not splits:
        logger.warning("No valid out-of-year splits for station %s; skipping.", station_name)
        return pd.DataFrame(), pd.DataFrame()

    # Full dataset (scaled sequences)
    full_dataset = RZSMDataset(X_d, X_s, y)
    years_all = pd.to_datetime(dates_train).year.values
    dyn_dim = X_d.shape[2]
    stat_dim = X_s.shape[2]

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

        # Validation split within training indices
        val_indices: List[int] = []
        train_indices_final: List[int] = []

        if getattr(config, "val_years", None) is not None:
            val_years_raw = config.val_years
            val_years_seq = (
                (val_years_raw,) if isinstance(val_years_raw, int) else tuple(val_years_raw)
            )
            val_years_set = set(val_years_seq)

            for idx in train_idx:
                if years_all[idx] in val_years_set:
                    val_indices.append(int(idx))
                else:
                    train_indices_final.append(int(idx))

            # Fallback to random split if no val indices or no train indices
            if len(val_indices) == 0 or len(train_indices_final) == 0:
                logger.info(
                    "Validation years %s yielded empty or degenerate split; "
                    "falling back to random val_fraction=%.2f",
                    sorted(val_years_set),
                    config.val_fraction,
                )
                train_indices_final = []
                val_indices = []

        if not train_indices_final and not val_indices:
            # Random split over train_idx
            train_idx_arr = np.array(train_idx, dtype=int)
            n_total = train_idx_arr.size
            val_size = int(max(1, round(n_total * config.val_fraction))) if n_total > 1 else 0

            rng = np.random.RandomState(config.seed + fold_idx)
            rng.shuffle(train_idx_arr)
            if val_size > 0:
                val_indices = train_idx_arr[:val_size].tolist()
                train_indices_final = train_idx_arr[val_size:].tolist()
            else:
                train_indices_final = train_idx_arr.tolist()
                val_indices = []

        logger.info(
            "Fold %s: n_train=%d, n_val=%d, n_test=%d",
            fold_name,
            len(train_indices_final),
            len(val_indices),
            len(test_idx),
        )

        train_subset = Subset(full_dataset, train_indices_final)
        val_subset = Subset(full_dataset, val_indices) if val_indices else None
        test_subset = Subset(full_dataset, test_idx.tolist())

        train_loader = DataLoader(
            train_subset,
            batch_size=config.batch_size,
            shuffle=True,
        )
        val_loader = (
            DataLoader(
                val_subset,
                batch_size=config.batch_size,
                shuffle=False,
            )
            if val_subset is not None
            else None
        )
        test_loader = DataLoader(
            test_subset,
            batch_size=config.batch_size,
            shuffle=False,
        )

        # Create model and trainer for this fold
        logger.info("Creating model for fold %s ...", fold_name)
        model = get_model(model_config, dyn_dim, stat_dim)

        fold_dir = station_output_dir / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trainer = Trainer(model, training_config, str(fold_dir), device=str(device))

        logger.info("Training fold %s ...", fold_name)
        history = trainer.train(train_loader, val_loader)

        # Evaluate on held-out year
        logger.info("Evaluating fold %s on held-out year %d ...", fold_name, test_year)
        evaluator = Evaluator(trainer.model, processor.scaler_y, str(fold_dir), device=str(device))
        y_pred, y_true = evaluator.predict(test_loader)
        metrics = evaluator.compute_metrics(y_pred, y_true)

        # Save training history plot
        evaluator.plot_training_history(history, f"{station_name}_{fold_name}")

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
            "n_train": len(train_indices_final),
            "n_val": len(val_indices),
            "n_test": len(test_idx),
        }
        fold_metric_rows.append(metric_row)

        # Align predictions with dates
        dates_fold = pd.to_datetime(dates_train[test_idx])
        years_fold = dates_fold.year.values

        fold_df = pd.DataFrame(
            {
                "station": station_name,
                "fold": fold_name,
                "test_year": int(test_year),
                "Date": dates_fold,
                "Year": years_fold,
                "RZSM_true": y_true,
                "RZSM_pred": y_pred,
            }
        )
        fold_pred_dfs.append(fold_df)

        all_y_true.append(y_true)
        all_y_pred.append(y_pred)

        logger.info(
            "Fold %s metrics: R2=%.4f, RMSE=%.4f, MAE=%.4f",
            fold_name,
            metrics["r2"],
            metrics["rmse"],
            metrics["mae"],
        )

    if not fold_metric_rows:
        logger.warning("No successful folds for station %s.", station_name)
        return pd.DataFrame(), pd.DataFrame()

    station_metrics_df = pd.DataFrame(fold_metric_rows)
    station_preds_df = pd.concat(fold_pred_dfs, ignore_index=True)

    # Optional overall pooled metrics for this station
    all_y_true_arr = np.concatenate(all_y_true)
    all_y_pred_arr = np.concatenate(all_y_pred)
    overall_evaluator = Evaluator(
        trainer.model, processor.scaler_y, str(station_output_dir), device=str(device)
    )
    overall_metrics = overall_evaluator.compute_metrics(all_y_pred_arr, all_y_true_arr)
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
        "n_train": np.nan,
        "n_val": np.nan,
        "n_test": np.nan,
    }
    station_metrics_df = pd.concat(
        [station_metrics_df, pd.DataFrame([overall_row])],
        ignore_index=True,
    )

    # Save per-station outputs
    metrics_path = station_output_dir / f"{station_name}_lstm_ooy_metrics.csv"
    preds_path = station_output_dir / f"{station_name}_lstm_ooy_predictions.csv"
    station_metrics_df.to_csv(metrics_path, index=False)
    station_preds_df.to_csv(preds_path, index=False)
    logger.info("Saved station metrics to %s", metrics_path)
    logger.info("Saved station predictions to %s", preds_path)

    return station_metrics_df, station_preds_df


def main() -> None:
    """Run out-of-year cross-validation for all stations."""
    config = OOYCVConfig()

    base_output_dir = Path(config.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(str(base_output_dir), log_level="INFO")
    log = logging.getLogger(__name__)

    log.info("=" * 80)
    log.info("LSTM OUT-OF-YEAR (YEAR-WISE HELD-OUT) CV - SINGLE STATION")
    log.info("=" * 80)
    log.info("Model type: %s", config.model_type)
    log.info("Stations: %s", list(STATIONS.keys()))
    log.info("Base year range: %d-%d", config.year_range[0], config.year_range[1])
    log.info("Month range: May–October (5-10)")
    if config.use_presto_static:
        presto_path = Path(config.presto_embeddings_path)
        if not presto_path.is_absolute():
            presto_path = Path(__file__).resolve().parent / presto_path
        log.info(
            "Presto embeddings: ENABLED at daily scale "
            "(merged on Date+Site; static = Presto 128-d + AE + precip)"
        )
        log.info("  Path: %s", presto_path)
    else:
        log.info("Presto embeddings: disabled (static = Alpha Earth A00-A63 + precip only)")

    # Seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    start_year, end_year = config.year_range
    # End-anchored windowing: (2014–2023), (2015–2023), ..., down to 3 years.
    min_years = 3
    windows = generate_end_anchored_year_windows(start_year, end_year, min_years=min_years)
    log.info(
        "Running end-anchored OOY experiments (%d windows): %d-%d down to %d years.",
        len(windows),
        start_year,
        end_year,
        min_years,
    )

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
            config,
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
                    station_name,
                    data_file,
                    window_config,
                    log,
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

            # Unweighted mean across held-out years (folds) for each station in this window
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

            log.info("\n" + "=" * 80)
            log.info("ALL-STATIONS OUT-OF-YEAR SUMMARY FOR WINDOW %s", window_tag)
            log.info("=" * 80)
            log.info("Per-fold metrics:")
            log.info(
                window_metrics_df[
                    ["station", "fold", "test_year", "r2", "rmse", "mae", "n_samples"]
                ].to_string(index=False)
            )
            log.info("Summary metrics for window %s saved to %s", window_tag, summary_path)

        if window_station_preds:
            combined_preds_df = pd.concat(window_station_preds, ignore_index=True)
            combined_preds_path = window_output_dir / "all_stations_lstm_ooy_predictions.csv"
            combined_preds_df.to_csv(combined_preds_path, index=False)
            log.info(
                "Combined predictions for window %s saved to %s",
                window_tag,
                combined_preds_path,
            )

    # Aggregate across all windows and stations
    if all_window_metrics:
        all_windows_df = pd.concat(all_window_metrics, ignore_index=True)
        all_windows_summary_path = base_output_dir / "all_windows_lstm_ooy_metrics.csv"
        all_windows_df.to_csv(all_windows_summary_path, index=False)
        log.info("\n" + "=" * 80)
        log.info("ALL-WINDOWS OUT-OF-YEAR SUMMARY")
        log.info("=" * 80)
        log.info(
            all_windows_df[
                [
                    "station",
                    "fold",
                    "test_year",
                    "r2",
                    "rmse",
                    "mae",
                    "n_samples",
                    "window_start",
                    "window_end",
                    "window_years",
                ]
            ].to_string(index=False)
        )
        log.info("All-windows summary metrics saved to %s", all_windows_summary_path)

    if per_station_window_means:
        station_means_df = pd.concat(per_station_window_means, ignore_index=True)
        station_means_path = base_output_dir / "anchored_windows_station_mean_metrics.csv"
        station_means_df.to_csv(station_means_path, index=False)
        log.info("Anchored-window per-station mean metrics saved to %s", station_means_path)

    log.info("\n" + "=" * 80)
    log.info("OUT-OF-YEAR MULTI-WINDOW PROCESSING COMPLETED")
    log.info("=" * 80)


if __name__ == "__main__":
    main()

