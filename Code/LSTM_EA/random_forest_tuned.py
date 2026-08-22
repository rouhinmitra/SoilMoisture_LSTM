#!/usr/bin/env python
"""
Leave-One-Station-Out Cross-Validation for Random Forest (RZSM regression) with tuning.

This script mirrors `random_forest.py` but performs hyperparameter tuning using
an inner validation year drawn from the training stations for each fold.

Workflow per fold:
- Train stations: 2 sites (e.g., Ne1+Ne2), Test station: 1 site (e.g., Ne3)
- Windows: same 20-day sliding windows as the LSTM pipeline
- Features: last-timestep dynamic + static features (same feature set as baseline LSTM)
- Target: RZSM_25_avg at last timestep, inverse-transformed to original units
- Tuning: grid search on a validation year (default 2022) within the training stations
"""

import logging
import random
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.config import DataConfig, FeatureConfig
from src.data_loader import DataProcessor
from utils.logging_utils import setup_logging
from random_forest import STATIONS, CV_FOLDS  # reuse definitions


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² (1 - SS_res/SS_tot). Returns np.nan if SS_tot is 0 or len < 2."""
    if len(y_true) < 2:
        return np.nan
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return np.nan
    return float(1 - (ss_res / ss_tot))


def _compute_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    """Compute R², RMSE, MAE, bias, correlation, NSE, n_samples."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    bias = np.mean(y_pred - y_true)
    corr = np.corrcoef(y_pred, y_true)[0, 1] if len(y_true) > 1 else np.nan
    nse = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
    return {
        "r2": float(r2),
        "rmse": float(rmse),
        "mae": float(mae),
        "bias": float(bias),
        "correlation": float(corr) if not np.isnan(corr) else np.nan,
        "nse": float(nse) if not np.isnan(nse) else np.nan,
        "n_samples": len(y_true),
    }


@dataclass
class RFTunedConfig:
    """
    Tuned Random Forest CV configuration.

    Matches the LSTM baseline in terms of features and windowing, but adds
    inner-loop hyperparameter tuning using a validation year.
    """

    # Data / windowing
    data_dir: str = "/Users/rouhinmitra/SM_work/Data/Base/"
    output_dir: str = "outputs/rf_cv_tuned"
    seq_length: int = 20
    nan_threshold: float = 0.1
    year_range: Tuple[int, int] = (2017, 2024)
    seed: int = 42

    # Validation year(s) within the training stations (used for tuning)
    val_years: Optional[Union[int, Sequence[int]]] = (2022,)

    # Static feature configuration (Presto + Alpha Earth + precip)
    use_presto_static: bool = True
    presto_embeddings_path: str = (
        "/Users/rouhinmitra/SM_work/Code/Data/s2_pixels/"
        "presto_embeddings_fused_interpolated.csv"
    )

    # Feature definitions (default to baseline LSTM features)
    dynamic_cols: Optional[List[str]] = None
    static_cols: Optional[List[str]] = None
    target_col: str = "RZSM_25_avg"

    # Hyperparameter ranges / choices (RandomizedSearch-style)
    # These mirror the older sklearn script's distributions.
    n_estimators_range: Tuple[int, int] = (100, 1000)  # randint(100, 1000)
    max_depth_choices: Sequence[Optional[int]] = (
        None,
        5,
        10,
        15,
        20,
        25,
        30,
    )
    min_samples_split_range: Tuple[int, int] = (2, 20)  # randint(2, 20)
    min_samples_leaf_range: Tuple[int, int] = (1, 10)  # randint(1, 10)
    max_features_choices: Sequence[Optional[Union[str, float]]] = (
        "sqrt",
        "log2",
        0.5,
        0.7,
        0.9,
    )
    bootstrap_choices: Sequence[bool] = (True, False)
    max_samples_choices: Sequence[Optional[float]] = (None, 0.7, 0.8, 0.9)

    # Random search configuration
    n_iter_search: int = 200
    random_search_seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.random_search_seed is None:
            self.random_search_seed = self.seed
        if self.dynamic_cols is None:
            self.dynamic_cols = [
                # Original dynamic features
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
                "ndvi",
                "b11",
                "b12",
                "b2",
                "b3",
                "b4",
                "b5",
                "b6",
                "b7",
                "b8",
                "b8a",
            ]
        if self.static_cols is None:
            if self.use_presto_static:
                self.static_cols = [
                    *(f"emb_{k}" for k in range(128)),
                    *(f"A{i:02d}" for i in range(64)),
                    "precip_jan_apr",
                ]
            else:
                self.static_cols = [
                    *(f"A{i:02d}" for i in range(64)),
                    "precip_jan_apr",
                    # "precip_may_oct",  # can be added if desired
                ]


def _split_train_val_by_year(
    dates_train: np.ndarray,
    val_years: Optional[Union[int, Sequence[int]]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split training indices into inner-train and validation subsets based on year.

    Returns (train_idx, val_idx) as index arrays.
    """
    if val_years is None:
        n = len(dates_train)
        all_idx = np.arange(n)
        return all_idx, np.array([], dtype=int)

    years = pd.to_datetime(dates_train).year
    if isinstance(val_years, int):
        val_years_seq = (val_years,)
    else:
        val_years_seq = tuple(val_years)
    val_set = set(val_years_seq)

    val_mask = np.isin(years, list(val_set))
    val_idx = np.where(val_mask)[0]
    train_idx = np.where(~val_mask)[0]
    return train_idx, val_idx


def tune_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: RFTunedConfig,
    logger: logging.Logger,
) -> Dict[str, Union[float, int, Optional[int]]]:
    """
    Random search over RF hyperparameters using validation R².

    Returns a dict with best hyperparameters and their validation metrics.
    """
    if X_val.size == 0 or y_val.size == 0 or X_train.size == 0 or y_train.size == 0:
        logger.warning(
            "No validation or training samples available for tuning; "
            "falling back to default hyperparameters."
        )
        return {
            "best_n_estimators": int(
                (config.n_estimators_range[0] + config.n_estimators_range[1]) / 2
            ),
            "best_max_depth": None,
            "best_min_samples_split": int(
                (config.min_samples_split_range[0] + config.min_samples_split_range[1])
                / 2
            ),
            "best_min_samples_leaf": int(
                (config.min_samples_leaf_range[0] + config.min_samples_leaf_range[1])
                / 2
            ),
            "best_max_features": "sqrt",
            "best_bootstrap": True,
            "best_max_samples": None,
            "best_val_r2": np.nan,
            "best_val_rmse": np.nan,
            "best_val_mae": np.nan,
        }

    rng = np.random.default_rng(config.random_search_seed or config.seed)
    best_params: Dict[str, Union[float, int, Optional[int], bool, float, str]] = {}
    best_r2 = -np.inf
    best_rmse = np.inf

    logger.info(
        "Tuning Random Forest hyperparameters on validation subset "
        f"(n_train={len(y_train)}, n_val={len(y_val)}; n_iter={config.n_iter_search})..."
    )

    for i in range(config.n_iter_search):
        # Sample hyperparameters from the specified ranges / choices
        n_estimators = int(
            rng.integers(
                config.n_estimators_range[0], config.n_estimators_range[1]
            )
        )
        # Sample max_depth from discrete choices (keep original Python types)
        max_depth_idx = int(
            rng.integers(0, len(config.max_depth_choices))
        )
        max_depth = config.max_depth_choices[max_depth_idx]
        min_samples_split = int(
            rng.integers(
                config.min_samples_split_range[0],
                config.min_samples_split_range[1],
            )
        )
        min_samples_leaf = int(
            rng.integers(
                config.min_samples_leaf_range[0],
                config.min_samples_leaf_range[1],
            )
        )
        # Sample categorical / mixed-type params by index to preserve types
        max_feat_idx = int(
            rng.integers(0, len(config.max_features_choices))
        )
        max_features = config.max_features_choices[max_feat_idx]

        bootstrap_idx = int(
            rng.integers(0, len(config.bootstrap_choices))
        )
        bootstrap = bool(config.bootstrap_choices[bootstrap_idx])

        max_samples_idx = int(
            rng.integers(0, len(config.max_samples_choices))
        )
        max_samples = config.max_samples_choices[max_samples_idx]

        # Ensure max_samples is only set when bootstrap=True (sklearn constraint)
        if not bootstrap:
            max_samples = None

        model = RandomForestRegressor(
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
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        metrics = _compute_metrics(y_val_pred, y_val)
        r2 = metrics["r2"]
        rmse = metrics["rmse"]

        logger.info(
            "  Sample %3d: n_estimators=%s, max_depth=%s, "
            "min_samples_split=%s, min_samples_leaf=%s, max_features=%s, "
            "bootstrap=%s, max_samples=%s -> val R²=%.4f, RMSE=%.4f",
            i + 1,
            n_estimators,
            str(max_depth),
            min_samples_split,
            min_samples_leaf,
            str(max_features),
            bootstrap,
            str(max_samples),
            r2,
            rmse,
        )

        # Select by higher R², break ties with lower RMSE
        if (r2 > best_r2) or (np.isclose(r2, best_r2) and rmse < best_rmse):
            best_r2 = r2
            best_rmse = rmse
            best_params = {
                "best_n_estimators": int(n_estimators),
                "best_max_depth": max_depth,
                "best_min_samples_split": int(min_samples_split),
                "best_min_samples_leaf": int(min_samples_leaf),
                "best_max_features": max_features,
                "best_bootstrap": bootstrap,
                "best_max_samples": max_samples,
                "best_val_r2": float(r2),
                "best_val_rmse": float(rmse),
                "best_val_mae": float(metrics["mae"]),
            }

    logger.info(
        "Best RF params: n_estimators=%s, max_depth=%s, min_samples_split=%s, "
        "min_samples_leaf=%s, max_features=%s, bootstrap=%s, max_samples=%s "
        "(val R²=%.4f, RMSE=%.4f)",
        best_params["best_n_estimators"],
        str(best_params["best_max_depth"]),
        best_params["best_min_samples_split"],
        best_params["best_min_samples_leaf"],
        str(best_params["best_max_features"]),
        str(best_params["best_bootstrap"]),
        str(best_params["best_max_samples"]),
        best_params["best_val_r2"],
        best_params["best_val_rmse"],
    )

    return best_params


def run_single_fold_tuned(
    fold: Dict,
    config: RFTunedConfig,
    logger: logging.Logger,
) -> Dict:
    """
    Run a single CV fold with inner-loop RF tuning on a validation year.

    Returns dict with predictions, actuals, metrics, tuning info, and station info.
    """
    fold_name = fold["name"]
    train_stations = fold["train"]
    test_station = fold["test"]

    logger.info("\n" + "=" * 70)
    logger.info("FOLD (tuned RF): %s", fold_name)
    logger.info("Training on: %s", train_stations)
    logger.info("Testing on: %s", test_station)
    logger.info("=" * 70)

    data_config = DataConfig(
        train_files=[STATIONS[s] for s in train_stations],
        test_files=[STATIONS[test_station]],
        data_dir=config.data_dir,
        seq_length=config.seq_length,
        nan_threshold=config.nan_threshold,
        same_year_constraint=True,
        month_range=(4, 10),
        year_range=config.year_range,
    )

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
        use_presto_static=config.use_presto_static,
        presto_embeddings_path=presto_path if config.use_presto_static else None,
    )

    logger.info("Preparing data (with val_years=%s)...", str(config.val_years))
    processor = DataProcessor(data_config, feature_config)
    data = processor.prepare_data(
        data_config.train_files,
        data_config.test_files,
        val_years=config.val_years,
    )

    X_d_train = data["X_d_train"]  # (n_train, seq_len, dyn_dim)
    X_s_train = data["X_s_train"]  # (n_train, seq_len, stat_dim)
    y_train = data["y_train"]  # (n_train, seq_len)
    X_d_test = data["X_d_test"]
    X_s_test = data["X_s_test"]
    y_test = data["y_test"]
    dates_train = data["dates_train"]
    dates_test = data["dates_test"]

    # Build last-timestep features and target (original units) for training windows
    X_full = np.concatenate([X_d_train[:, -1, :], X_s_train[:, -1, :]], axis=1)
    y_full_scaled = y_train[:, -1]
    y_full = processor.scaler_y.inverse_transform(
        y_full_scaled.reshape(-1, 1)
    ).ravel()

    train_idx, val_idx = _split_train_val_by_year(dates_train, config.val_years)
    X_inner_train = X_full[train_idx]
    y_inner_train = y_full[train_idx]
    X_val = X_full[val_idx]
    y_val = y_full[val_idx]

    logger.info(
        "Inner split sizes: train=%d, val=%d (total=%d)",
        len(y_inner_train),
        len(y_val),
        len(y_full),
    )

    # Tune RF on inner train/val split
    best_params = tune_random_forest(
        X_inner_train,
        y_inner_train,
        X_val,
        y_val,
        config,
        logger,
    )

    # Refit final model on all training windows (train + val)
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
    final_model.fit(X_full, y_full)

    # Prepare test features / targets (last timestep, original units)
    X_test = np.concatenate([X_d_test[:, -1, :], X_s_test[:, -1, :]], axis=1)
    y_test_scaled = y_test[:, -1]
    y_test_last = processor.scaler_y.inverse_transform(
        y_test_scaled.reshape(-1, 1)
    ).ravel()

    y_pred = final_model.predict(X_test)
    metrics = _compute_metrics(y_pred, y_test_last)

    # May–Oct R² on test station
    r2_may_oct = np.nan
    n_may_oct = 0
    if dates_test is not None and len(dates_test) == len(y_test_last):
        months = pd.to_datetime(dates_test).month
        mask = (months >= 5) & (months <= 10)
        n_may_oct = int(np.sum(mask))
        if n_may_oct > 1:
            r2_may_oct = _r2(y_test_last[mask], y_pred[mask])

    logger.info("\nFold %s (tuned) Results:", fold_name)
    logger.info("  R²:          %.4f", metrics["r2"])
    logger.info(
        "  R² (May–Oct): %s",
        f"{r2_may_oct:.4f}" if not np.isnan(r2_may_oct) else "n/a",
    )
    logger.info("  RMSE:        %.4f", metrics["rmse"])
    logger.info("  MAE:         %.4f", metrics["mae"])

    fold_output_dir = Path(config.output_dir) / fold_name
    fold_output_dir.mkdir(parents=True, exist_ok=True)

    # Save test predictions
    test_df = pd.DataFrame(
        {
            "date": pd.to_datetime(dates_test),
            "actual": y_test_last,
            "predicted": y_pred,
            "residual": y_test_last - y_pred,
        }
    )
    test_df.to_csv(fold_output_dir / "predictions.csv", index=False)
    logger.info("Test predictions saved to %s", fold_output_dir / "predictions.csv")

    # Optionally save validation predictions if any
    if len(y_val) > 0:
        # Recreate validation predictions using the tuned model trained on X_full
        y_val_pred = final_model.predict(X_val)
        val_df = pd.DataFrame(
            {
                "date": pd.to_datetime(dates_train[val_idx]),
                "actual": y_val,
                "predicted": y_val_pred,
                "residual": y_val - y_val_pred,
            }
        )
        val_df.to_csv(fold_output_dir / "val_predictions.csv", index=False)
        logger.info(
            "Validation predictions saved to %s",
            fold_output_dir / "val_predictions.csv",
        )

    result = {
        "fold_name": fold_name,
        "test_station": test_station,
        "train_stations": train_stations,
        "y_pred": y_pred,
        "y_true": y_test_last,
        "metrics": metrics,
        "dates_test": dates_test,
        "n_train_windows": int(len(y_full)),
        "n_val_windows": int(len(y_val)),
        "n_test_windows": int(len(y_test_last)),
        "r2_may_oct": r2_may_oct,
        "n_may_oct": n_may_oct,
    }
    result.update(best_params)
    return result


def create_combined_scatter_plot(results: List[Dict], output_dir: Path) -> None:
    """Create a combined 3-panel scatter plot with all folds (tuned RF)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for result, ax, color in zip(results, axes, colors):
        y_pred = result["y_pred"]
        y_true = result["y_true"]
        metrics = result["metrics"]
        test_station = result["test_station"]

        ax.scatter(y_true, y_pred, alpha=0.3, s=10, c=color)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            linewidth=2,
            label="1:1 Line",
        )
        z = np.polyfit(y_true, y_pred, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min_val, max_val, 100)
        ax.plot(
            x_line,
            p(x_line),
            "k-",
            linewidth=1.5,
            alpha=0.7,
            label="Best Fit",
        )

        ax.set_xlabel("Observed RZSM (%)", fontsize=11)
        ax.set_ylabel("Predicted RZSM (%)", fontsize=11)
        ax.set_title(
            f"Test: {test_station}\nR² = {metrics['r2']:.3f}, "
            f"RMSE = {metrics['rmse']:.2f}",
            fontsize=12,
        )
        ax.set_aspect("equal", "box")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)

    plt.suptitle(
        "Leave-One-Station-Out CV: Random Forest (tuned)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    save_path = output_dir / "cv_scatter_plots_tuned.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Combined tuned RF scatter plot saved to: {save_path}")


def create_summary_plot(results: List[Dict], output_dir: Path) -> None:
    """Create R² summary bar plot with mean line for tuned RF."""
    fig, ax = plt.subplots(figsize=(8, 5))
    stations = [r["test_station"] for r in results]
    r2_scores = [r["metrics"]["r2"] for r in results]

    x = np.arange(len(stations))
    width = 0.35
    bars1 = ax.bar(x - width / 2, r2_scores, width, label="R²", color="steelblue")
    for bar, val in zip(bars1, r2_scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    mean_r2 = float(np.mean(r2_scores))
    ax.axhline(
        y=mean_r2,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean R² = {mean_r2:.3f}",
    )
    ax.set_ylabel("R² Score", fontsize=12)
    ax.set_xlabel("Test Station", fontsize=12)
    ax.set_title(
        "Leave-One-Station-Out CV: R² by Test Station (Random Forest tuned)",
        fontsize=14,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"Test: {s}" for s in stations])
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    save_path = output_dir / "cv_r2_summary_tuned.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Tuned RF R² summary plot saved to: {save_path}")


def main() -> List[Dict]:
    """Run leave-one-station-out cross-validation with tuned Random Forest."""
    config = RFTunedConfig()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(str(output_dir), log_level="INFO")
    log = logging.getLogger(__name__)

    log.info("=" * 70)
    log.info("LEAVE-ONE-STATION-OUT CROSS-VALIDATION (RANDOM FOREST, TUNED)")
    log.info("=" * 70)
    log.info("Stations: %s", list(STATIONS.keys()))
    log.info("Number of folds: %d", len(CV_FOLDS))
    log.info(
        "Year range: %d–%d (inclusive)",
        config.year_range[0],
        config.year_range[1],
    )
    log.info("Sequence length: %d", config.seq_length)
    log.info("Validation years for tuning: %s", str(config.val_years))
    log.info(
        "RF random search space: "
        "n_estimators in [%d, %d), max_depth in %s, "
        "min_samples_split in [%d, %d), min_samples_leaf in [%d, %d), "
        "max_features in %s, bootstrap in %s, max_samples in %s; "
        "n_iter_search=%d",
        config.n_estimators_range[0],
        config.n_estimators_range[1],
        list(config.max_depth_choices),
        config.min_samples_split_range[0],
        config.min_samples_split_range[1],
        config.min_samples_leaf_range[0],
        config.min_samples_leaf_range[1],
        list(config.max_features_choices),
        list(config.bootstrap_choices),
        list(config.max_samples_choices),
        config.n_iter_search,
    )
    if config.use_presto_static:
        presto_path = Path(config.presto_embeddings_path)
        if not presto_path.is_absolute():
            presto_path = Path(__file__).resolve().parent / presto_path
        log.info("Presto embeddings: ENABLED")
        log.info("  Path: %s", str(presto_path))
    else:
        log.info("Presto embeddings: disabled")

    if config.use_presto_static:
        p = Path(config.presto_embeddings_path)
        if not p.is_absolute():
            p = Path(__file__).resolve().parent / p
        if not p.exists():
            raise FileNotFoundError(
                f"Presto embeddings file not found: {p}. "
                "Set use_presto_static=False or provide a valid presto_embeddings_path."
            )

    np.random.seed(config.seed)
    random.seed(config.seed)

    results: List[Dict] = []
    for fold in CV_FOLDS:
        result = run_single_fold_tuned(fold, config, log)
        results.append(result)

    log.info("\n" + "=" * 70)
    log.info("CREATING VISUALIZATIONS (tuned RF)")
    log.info("=" * 70)
    create_combined_scatter_plot(results, output_dir)
    create_summary_plot(results, output_dir)

    log.info("\n" + "=" * 70)
    log.info("CROSS-VALIDATION SUMMARY (tuned RF)")
    log.info("=" * 70)

    print("\n" + "=" * 100)
    print("RANDOM FOREST (TUNED) CV RESULTS")
    print("=" * 100)
    print(
        f"{'Test Station':<15} {'Train Stations':<20} {'R²':>8} "
        f"{'R²_MayOct':>10} {'RMSE':>8} {'MAE':>8} "
        f"{'N_train':>10} {'N_val':>10} {'N_test':>10}"
    )
    print("-" * 100)

    r2_scores: List[float] = []
    r2_may_oct_scores: List[float] = []
    rmse_scores: List[float] = []

    for result in results:
        test = result["test_station"]
        train = "+".join(result["train_stations"])
        r2 = result["metrics"]["r2"]
        r2_mo = result.get("r2_may_oct", np.nan)
        rmse = result["metrics"]["rmse"]
        mae = result["metrics"]["mae"]
        n_train = result.get("n_train_windows", "")
        n_val = result.get("n_val_windows", "")
        n_test = result.get("n_test_windows", "")

        r2_scores.append(r2)
        if not np.isnan(r2_mo):
            r2_may_oct_scores.append(r2_mo)
        rmse_scores.append(rmse)

        r2_mo_str = f"{r2_mo:.4f}" if not np.isnan(r2_mo) else "n/a"
        print(
            f"{test:<15} {train:<20} {r2:>8.4f} {r2_mo_str:>10} "
            f"{rmse:>8.4f} {mae:>8.4f} {n_train:>10} {n_val:>10} {n_test:>10}"
        )

    # Overall R² for growing season (May–Oct), pooled across all folds
    all_y_true = np.concatenate([r["y_true"] for r in results])
    all_y_pred = np.concatenate([r["y_pred"] for r in results])
    all_dates = np.concatenate([r["dates_test"] for r in results])
    months = pd.to_datetime(all_dates).month
    mask_mo = (months >= 5) & (months <= 10)
    r2_overall_may_oct = np.nan
    if np.sum(mask_mo) > 1:
        r2_overall_may_oct = _r2(all_y_true[mask_mo], all_y_pred[mask_mo])

    print("-" * 100)
    mean_r2_mo = np.mean(r2_may_oct_scores) if r2_may_oct_scores else np.nan
    mean_r2_mo_str = f"{mean_r2_mo:.4f}" if not np.isnan(mean_r2_mo) else "n/a"
    print(
        f"{'MEAN':<15} {'':<20} {np.mean(r2_scores):>8.4f} {mean_r2_mo_str:>10} "
        f"{np.mean(rmse_scores):>8.4f} {'':>8} {'':>10} {'':>10}"
    )
    print(
        f"{'STD':<15} {'':<20} {np.std(r2_scores):>8.4f} "
        f"{'':>10} {np.std(rmse_scores):>8.4f}"
    )
    print("-" * 100)
    r2_overall_mo_str = (
        f"{r2_overall_may_oct:.4f}" if not np.isnan(r2_overall_may_oct) else "n/a"
    )
    print(f"Overall R² (May–Oct, pooled): {r2_overall_mo_str}")
    print("=" * 100)

    # Save results to CSV with tuned hyperparameters
    results_df = pd.DataFrame(
        [
            {
                "test_station": r["test_station"],
                "train_stations": "+".join(r["train_stations"]),
                "r2": r["metrics"]["r2"],
                "r2_may_oct": r.get("r2_may_oct"),
                "rmse": r["metrics"]["rmse"],
                "mae": r["metrics"]["mae"],
                "bias": r["metrics"]["bias"],
                "correlation": r["metrics"]["correlation"],
                "n_samples": r["metrics"]["n_samples"],
                "n_may_oct": r.get("n_may_oct"),
                "n_train_windows": r.get("n_train_windows"),
                "n_val_windows": r.get("n_val_windows"),
                "n_test_windows": r.get("n_test_windows"),
                "best_n_estimators": r.get("best_n_estimators"),
                "best_max_depth": r.get("best_max_depth"),
                "best_min_samples_split": r.get("best_min_samples_split"),
                "best_min_samples_leaf": r.get("best_min_samples_leaf"),
                "best_max_features": r.get("best_max_features"),
                "best_bootstrap": r.get("best_bootstrap"),
                "best_max_samples": r.get("best_max_samples"),
                "best_val_r2": r.get("best_val_r2"),
                "best_val_rmse": r.get("best_val_rmse"),
                "best_val_mae": r.get("best_val_mae"),
            }
            for r in results
        ]
    )
    csv_path = output_dir / "cv_results_tuned.csv"
    results_df.to_csv(csv_path, index=False)
    log.info("\nTuned RF results saved to: %s", str(csv_path))

    # Append overall May–Oct summary row
    if not np.isnan(r2_overall_may_oct):
        log.info(
            "Overall R² (May–Oct, pooled, tuned RF): %.4f", r2_overall_may_oct
        )
        summary_row = pd.DataFrame(
            [
                {
                    "test_station": "OVERALL_MayOct",
                    "train_stations": "",
                    "r2": np.nan,
                    "r2_may_oct": r2_overall_may_oct,
                    "rmse": np.nan,
                    "mae": np.nan,
                    "bias": np.nan,
                    "correlation": np.nan,
                    "n_samples": int(np.sum(mask_mo)),
                    "n_may_oct": int(np.sum(mask_mo)),
                    "n_train_windows": np.nan,
                    "n_val_windows": np.nan,
                    "n_test_windows": np.nan,
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
            ]
        )
        results_df = pd.concat([results_df, summary_row], ignore_index=True)
        results_df.to_csv(csv_path, index=False)

    log.info("\nOutputs saved to: %s", str(output_dir))
    log.info("Tuned RF CV complete!")
    return results


if __name__ == "__main__":
    main()

