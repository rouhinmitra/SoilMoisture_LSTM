#!/usr/bin/env python
"""
Leave-One-Station-Out Cross-Validation for EA-LSTM/LSTM — 100cm RZSM.

Trains on 2 stations, tests on the 3rd, for all 3 combinations:
- Fold 1: Train Ne1+Ne2, Test Ne3
- Fold 2: Train Ne1+Ne3, Test Ne2
- Fold 3: Train Ne2+Ne3, Test Ne1

Target: RZSM_100_avg (100cm root zone soil moisture).

This script optionally augments the input sequence with:
- LSTM-predicted 25cm RZSM as dynamic feature `RZSM25_pred`
- LSTM-predicted 50cm RZSM as dynamic feature `RZSM50_pred`

Both prediction sources are expected as per-station CSV files containing columns:
`Date` and `RZSM_pred`, named `predictions_<Station>.csv` (e.g., predictions_Ne1.csv).
"""

import logging
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset, random_split

from src.config import DataConfig, FeatureConfig, ModelConfig, TrainingConfig
from src.data_loader import DataProcessor, RZSMDataset
from src.evaluator import Evaluator
from src.models import get_model
from src.trainer import Trainer
from utils.logging_utils import setup_logging

# --- Plotting utilities ---
def _plot_yearly_timeseries(
    *,
    fold_output_dir: Path,
    test_csv_path: Path,
    dates_test: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_label: str,
):
    """
    Plot one time series per year for the held-out station with:
    - SSM_avg (from the test CSV)
    - RZSM_obs (y_true)
    - RZSM_pred (y_pred)
    """
    if dates_test is None or len(dates_test) != len(y_true):
        return

    df_pred = pd.DataFrame(
        {
            "Date": pd.to_datetime(dates_test),
            "RZSM_obs": y_true,
            "RZSM_pred": y_pred,
        }
    )

    df_test = pd.read_csv(test_csv_path)
    if "Date" not in df_test.columns or "SSM_avg" not in df_test.columns:
        raise ValueError(f"Test file {test_csv_path} must contain Date and SSM_avg columns")
    df_test = df_test.copy()
    df_test["Date"] = pd.to_datetime(df_test["Date"], errors="coerce")
    df_test = df_test.dropna(subset=["Date"])
    df_test = df_test.sort_values("Date").drop_duplicates(subset="Date", keep="first")

    df = df_pred.merge(df_test[["Date", "SSM_avg"]], on="Date", how="inner").sort_values("Date")
    if df.empty:
        return

    out_dir = fold_output_dir / "yearly_timeseries"
    out_dir.mkdir(parents=True, exist_ok=True)

    df["Year"] = df["Date"].dt.year
    for year, g in df.groupby("Year"):
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(g["Date"], g["SSM_avg"], label="SSM_avg", linewidth=1.5, alpha=0.9)
        ax.plot(g["Date"], g["RZSM_obs"], label=f"{target_label} obs", linewidth=1.8, alpha=0.9)
        ax.plot(g["Date"], g["RZSM_pred"], label=f"{target_label} pred", linewidth=1.8, alpha=0.9, linestyle="--")
        ax.set_title(f"{target_label} vs SSM_avg — {year}", fontsize=12)
        ax.set_xlabel("Date")
        ax.set_ylabel("Soil moisture")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / f"timeseries_{int(year)}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

# Configuration
STATIONS = {
    "Ne1": "ne1_1_maize.csv",
    "Ne2": "ne2_1_maize.csv",
    "Ne3": "ne3_1_maize.csv",
}

CV_FOLDS = [
    {"train": ["Ne1", "Ne2"], "test": "Ne3", "name": "fold1_test_Ne3"},
    {"train": ["Ne1", "Ne3"], "test": "Ne2", "name": "fold2_test_Ne2"},
    {"train": ["Ne2", "Ne3"], "test": "Ne1", "name": "fold3_test_Ne1"},
]

TARGET_COL = "RZSM_100_avg"  # 100cm root zone soil moisture


def _load_predictions_for_station(
    station: str,
    pred_dir: Union[str, Path],
    out_col: str,
) -> pd.DataFrame:
    """
    Load LSTM-predicted RZSM for a given station from a predictions directory.

    Expects a file named `predictions_<Station>.csv` containing columns:
    - Date
    - RZSM_pred
    """
    if station not in STATIONS:
        raise ValueError(f"Unknown station: {station}. Expected one of {sorted(STATIONS.keys())}")

    pred_dir = Path(pred_dir)
    pred_path = pred_dir / f"predictions_{station}.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions file not found for {station}: {pred_path}")

    df = pd.read_csv(pred_path)
    if "Date" not in df.columns or "RZSM_pred" not in df.columns:
        raise ValueError(f"Predictions file {pred_path} must contain 'Date' and 'RZSM_pred' columns")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[["Date", "RZSM_pred"]].rename(columns={"RZSM_pred": out_col})
    return df


@dataclass
class CVConfig:
    """Cross-validation configuration — mirrors run_cv_50cm.py, but 100cm target."""

    data_dir: str = "/Users/rouhinmitra/SM_work/Data/Base/"
    output_dir: str = "outputs/cv_results_tuned_100cm"
    seq_length: int = 30
    nan_threshold: float = 0.1
    # Tuned hyperparameters (kept consistent with 25cm/50cm script defaults)
    hidden_dim: int = 32
    dropout: float = 0.5
    batch_size: int = 32
    epochs: int = 150
    learning_rate: float = 0.0005
    early_stopping_patience: int = 20
    weight_decay: float = 0.001
    add_temporal: bool = True
    seed: int = 42
    model_type: str = "LSTM"  # "EALSTM" or "LSTM"
    num_layers: int = 1  # only used when model_type="LSTM"
    val_fraction: float = 0.15  # fraction of training data held out for validation (used when val_years is None)
    val_years: Optional[Union[int, Sequence[int]]] = 2022  # if set, use these years from each training station as validation
    year_range: Tuple[int, int] = (2017, 2024)
    month_range: Tuple[int, int] = (4, 10)

    # Presto embeddings as static features (added to Alpha Earth when True)
    use_presto_static: bool = True
    presto_embeddings_path: str = "/Users/rouhinmitra/SM_work/Code/Data/s2_pixels/presto_embeddings_fused_interpolated.csv"

    # Auxiliary prediction inputs
    pred_dir_25: str = "/Users/rouhinmitra/SM_work/Code/LSTM_EA/outputs/sensitivity/seq20/baseline"
    pred_dir_50: str = "/Users/rouhinmitra/SM_work/Code/LSTM_EA/outputs/cv_results_lstm_50cm"  # set this to the directory containing predictions_Ne1.csv/Ne2/Ne3.csv for 50cm

    # Optional auxiliary predictions merged into base CSV on Date
    use_aux_predictions: bool = False

    # Features
    dynamic_cols: List[str] = None
    static_cols: List[str] = None
    target_col: str = TARGET_COL

    def __post_init__(self):
        if self.model_type == "LSTM" and self.output_dir == "outputs/cv_results_tuned_100cm":
            self.output_dir = "outputs/cv_results_lstm_100cm"

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
                # "RZSM_25_avg",
                # "RZSM_50_avg",
                # Note: irrigation is NOT in dynamic_cols (it will be appended as static feature)
            ]
            if self.use_aux_predictions:
                self.dynamic_cols = self.dynamic_cols + ["RZSM25_pred", "RZSM50_pred"]

        if self.static_cols is None:
            if self.use_presto_static:
                self.static_cols = [f"emb_{k}" for k in range(128)] + [f"A{i:02d}" for i in range(64)] + [
                    "precip_jan_apr",
                    # "precip_may_oct",
                ]
            else:
                self.static_cols = [f"A{i:02d}" for i in range(64)] + [
                    "precip_jan_apr",
                    # "precip_may_oct",
                ]


def _create_merged_station_files(
    config: CVConfig,
    train_stations: List[str],
    test_station: str,
    fold_output_dir: Path,
) -> Dict[str, Union[str, List[str]]]:
    """
    For each station, merge base CSV with predicted 25cm and 50cm RZSM on Date and
    write merged CSVs into a fold-local directory. Returns updated data_dir and
    train/test file lists suitable for DataConfig.
    """
    merged_dir = fold_output_dir / "merged_with_predictions"
    merged_dir.mkdir(parents=True, exist_ok=True)

    if not config.pred_dir_50:
        raise ValueError(
            "pred_dir_50 is empty. Set CVConfig.pred_dir_50 to the directory containing "
            "predictions_Ne1.csv/predictions_Ne2.csv/predictions_Ne3.csv for 50cm."
        )

    def _merge_single_station(station: str) -> str:
        base_filename = STATIONS[station]
        base_path = Path(config.data_dir) / base_filename
        if not base_path.exists():
            raise FileNotFoundError(f"Base data file not found for station {station}: {base_path}")

        base_df = pd.read_csv(base_path)
        if "Date" not in base_df.columns:
            raise ValueError(f"Base data file {base_path} must contain a 'Date' column")
        if config.target_col not in base_df.columns:
            raise ValueError(f"Base data file {base_path} must contain target column '{config.target_col}'")

        base_df = base_df.copy()
        base_df["Date"] = pd.to_datetime(base_df["Date"])

        pred25 = _load_predictions_for_station(station, config.pred_dir_25, out_col="RZSM25_pred")
        pred50 = _load_predictions_for_station(station, config.pred_dir_50, out_col="RZSM50_pred")

        merged = base_df.merge(pred25, on="Date", how="inner").merge(pred50, on="Date", how="inner")

        out_name = base_filename.replace(".csv", "_with_pred.csv")
        out_path = merged_dir / out_name
        merged.to_csv(out_path, index=False)
        return out_name

    train_files_merged = [_merge_single_station(s) for s in train_stations]
    test_file_merged = _merge_single_station(test_station)

    return {
        "data_dir": str(merged_dir),
        "train_files": train_files_merged,
        "test_files": [test_file_merged],
    }


def run_single_fold(
    fold: Dict,
    config: CVConfig,
    logger: logging.Logger,
) -> Dict:
    """
    Run a single CV fold.

    Returns dict with predictions, actuals, metrics, and station info.
    """
    fold_name = fold["name"]
    train_stations = fold["train"]
    test_station = fold["test"]

    logger.info(f"\n{'=' * 70}")
    logger.info(f"FOLD: {fold_name}")
    logger.info(f"Training on: {train_stations}")
    logger.info(f"Testing on: {test_station}")
    logger.info(f"{'=' * 70}")

    fold_output_dir = Path(config.output_dir) / fold_name
    fold_output_dir.mkdir(parents=True, exist_ok=True)

    if config.use_aux_predictions:
        logger.info("Aux predictions: ENABLED (merging 25cm+50cm predictions into base CSVs on Date)")
        merged_info = _create_merged_station_files(config, train_stations, test_station, fold_output_dir)
        train_files = merged_info["train_files"]
        test_files = merged_info["test_files"]
        data_dir = merged_info["data_dir"]
    else:
        logger.info("Aux predictions: disabled (using base station CSVs only)")
        train_files = [STATIONS[s] for s in train_stations]
        test_files = [STATIONS[test_station]]
        data_dir = config.data_dir

    data_config = DataConfig(
        train_files=train_files,
        test_files=test_files,
        data_dir=data_dir,
        seq_length=config.seq_length,
        nan_threshold=config.nan_threshold,
        same_year_constraint=True,
        month_range=config.month_range,
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
        add_temporal=config.add_temporal,
        temporal_features=["doy_sin", "doy_cos"],
        use_presto_static=config.use_presto_static,
        presto_embeddings_path=presto_path if config.use_presto_static else None,
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

    logger.info("Preparing data...")
    processor = DataProcessor(data_config, feature_config)
    val_years_for_scaler = config.val_years if getattr(config, "val_years", None) is not None else None
    data = processor.prepare_data(
        data_config.train_files,
        data_config.test_files,
        val_years=val_years_for_scaler,
    )

    full_train_dataset = RZSMDataset(data["X_d_train"], data["X_s_train"], data["y_train"])
    test_dataset = RZSMDataset(data["X_d_test"], data["X_s_test"], data["y_test"])

    dates_train = data["dates_train"]
    years = pd.to_datetime(dates_train).year
    if getattr(config, "val_years", None) is not None:
        val_years_raw = config.val_years
        val_years_seq = (val_years_raw,) if isinstance(val_years_raw, int) else val_years_raw
        val_years_set = set(val_years_seq)
        val_indices = [i for i, y in enumerate(years) if y in val_years_set]
        train_indices = [i for i, y in enumerate(years) if y not in val_years_set]
        train_subset = Subset(full_train_dataset, train_indices)
        val_subset = Subset(full_train_dataset, val_indices)
        train_size, val_size = len(train_indices), len(val_indices)
        logger.info(f"Validation set: years {sorted(val_years_set)} from training stations ({val_size} samples)")
    else:
        val_size = int(len(full_train_dataset) * config.val_fraction)
        train_size = len(full_train_dataset) - val_size
        train_subset, val_subset = random_split(
            full_train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(config.seed),
        )

    train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    logger.info(f"Train/Val/Test split: {train_size} / {val_size} / {len(test_dataset)} samples")

    dyn_dim = data["X_d_train"].shape[2]
    stat_dim = data["X_s_train"].shape[2]

    logger.info("Creating model...")
    model = get_model(model_config, dyn_dim, stat_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(model, training_config, str(fold_output_dir), device=str(device))

    logger.info("Training (val_loader from training stations, test station held out)...")
    history = trainer.train(train_loader, val_loader)

    logger.info("Evaluating...")
    evaluator = Evaluator(trainer.model, processor.scaler_y, str(fold_output_dir), device=str(device))
    y_pred, y_true = evaluator.predict(test_loader)
    metrics = evaluator.compute_metrics(y_pred, y_true)

    # Plot train/val loss curves for this fold (same as run_cv.py)
    evaluator.plot_training_history(history, "loss")

    logger.info(f"\nFold {fold_name} Results:")
    logger.info(f"  R²:   {metrics['r2']:.4f}")
    logger.info(f"  RMSE: {metrics['rmse']:.4f}")
    logger.info(f"  MAE:  {metrics['mae']:.4f}")

    # Plot per-year time series for test station: SSM_avg, obs, pred
    dates_test = data.get("dates_test")
    try:
        test_csv_path = Path(data_config.data_dir) / data_config.test_files[0]
        _plot_yearly_timeseries(
            fold_output_dir=fold_output_dir,
            test_csv_path=test_csv_path,
            dates_test=dates_test,
            y_true=y_true,
            y_pred=y_pred,
            target_label="RZSM_100",
        )
    except Exception as e:
        logger.warning(f"Failed to create yearly time series plots for {test_station}: {e}")

    return {
        "fold_name": fold_name,
        "test_station": test_station,
        "train_stations": train_stations,
        "y_pred": y_pred,
        "y_true": y_true,
        "metrics": metrics,
        "history": history,
        "dates_test": dates_test,
    }


def create_combined_scatter_plot(results: List[Dict], output_dir: Path, model_type: str = "EALSTM"):
    """Create a combined scatter plot with all folds."""
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
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="1:1 Line")

        z = np.polyfit(y_true, y_pred, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min_val, max_val, 100)
        ax.plot(x_line, p(x_line), "k-", linewidth=1.5, alpha=0.7, label="Best Fit")

        ax.set_xlabel("Observed RZSM (%)", fontsize=11)
        ax.set_ylabel("Predicted RZSM (%)", fontsize=11)
        ax.set_title(f"Test: {test_station}\nR² = {metrics['r2']:.3f}, RMSE = {metrics['rmse']:.2f}", fontsize=12)
        ax.set_aspect("equal", "box")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)

    plt.suptitle(f"Leave-One-Station-Out Cross-Validation: {model_type} (100cm RZSM)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    save_path = output_dir / "cv_scatter_plots.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Combined scatter plot saved to: {save_path}")


def create_summary_plot(results: List[Dict], output_dir: Path):
    """Create a summary bar plot of R² scores."""
    fig, ax = plt.subplots(figsize=(8, 5))

    stations = [r["test_station"] for r in results]
    r2_scores = [r["metrics"]["r2"] for r in results]

    x = np.arange(len(stations))
    width = 0.35

    bars1 = ax.bar(x - width / 2, r2_scores, width, label="R²", color="steelblue")

    for bar, val in zip(bars1, r2_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    mean_r2 = np.mean(r2_scores)
    ax.axhline(y=mean_r2, color="red", linestyle="--", linewidth=2, label=f"Mean R² = {mean_r2:.3f}")

    ax.set_ylabel("R² Score", fontsize=12)
    ax.set_xlabel("Test Station", fontsize=12)
    ax.set_title("Leave-One-Station-Out CV: R² by Test Station (100cm RZSM)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Test: {s}" for s in stations])
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    save_path = output_dir / "cv_r2_summary.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"R² summary plot saved to: {save_path}")


def main():
    """Run leave-one-station-out cross-validation for 100cm RZSM."""
    parser = argparse.ArgumentParser(description="Leave-one-station-out CV for 100cm RZSM.")
    parser.add_argument("--use-aux-predictions", action="store_true", help="Merge 25cm+50cm predictions into base CSVs on Date.")
    parser.add_argument("--pred-dir-25", type=str, default=None, help="Directory containing predictions_<Station>.csv for 25cm (Date,RZSM_pred).")
    parser.add_argument("--pred-dir-50", type=str, default=None, help="Directory containing predictions_<Station>.csv for 50cm (Date,RZSM_pred).")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output_dir (default from CVConfig).")
    args = parser.parse_args()

    config = CVConfig(use_aux_predictions=bool(args.use_aux_predictions))
    if args.pred_dir_25:
        config.pred_dir_25 = args.pred_dir_25
    if args.pred_dir_50:
        config.pred_dir_50 = args.pred_dir_50
    if args.output_dir:
        config.output_dir = args.output_dir

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _ = setup_logging(str(output_dir), log_level="INFO")
    log = logging.getLogger(__name__)

    log.info("=" * 70)
    log.info("LEAVE-ONE-STATION-OUT CROSS-VALIDATION — 100cm RZSM")
    log.info("=" * 70)
    log.info(f"Target: {TARGET_COL}")
    log.info(f"Model: {config.model_type}")
    log.info(f"Year range: {config.year_range[0]}–{config.year_range[1]} (inclusive)")
    log.info(f"Stations: {list(STATIONS.keys())}")
    log.info(f"Number of folds: {len(CV_FOLDS)}")
    log.info(f"Data: month_range={config.month_range}")
    log.info(f"Aux predictions: {'ENABLED' if config.use_aux_predictions else 'disabled'}")

    if config.use_presto_static:
        presto_path = Path(config.presto_embeddings_path)
        if not presto_path.is_absolute():
            presto_path = Path(__file__).resolve().parent / presto_path
        log.info("Presto embeddings: ENABLED at daily scale (merged on Date+Site; static = Presto 128-d + AE + precip)")
        log.info(f"  Path: {presto_path}")
    else:
        log.info("Presto embeddings: disabled (static = Alpha Earth A00-A63 + precip only)")

    log.info("Irrigation feature: Used as STATIC feature (input gate control)")

    if config.use_presto_static:
        p = Path(config.presto_embeddings_path)
        if not p.is_absolute():
            p = Path(__file__).resolve().parent / p
        if not p.exists():
            raise FileNotFoundError(
                f"Presto embeddings file not found: {p}. "
                "Set use_presto_static=False or provide a valid presto_embeddings_path."
            )

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    results = []
    for fold in CV_FOLDS:
        result = run_single_fold(fold, config, log)
        results.append(result)

    log.info("\n" + "=" * 70)
    log.info("CREATING VISUALIZATIONS")
    log.info("=" * 70)

    create_combined_scatter_plot(results, output_dir, model_type=config.model_type)
    create_summary_plot(results, output_dir)

    log.info("\n" + "=" * 70)
    log.info("CROSS-VALIDATION SUMMARY")
    log.info("=" * 70)

    print("\n" + "=" * 70)
    print("CROSS-VALIDATION RESULTS (100cm RZSM)")
    print("=" * 70)
    print(f"{'Test Station':<15} {'Train Stations':<20} {'R²':>10} {'RMSE':>10} {'MAE':>10}")
    print("-" * 70)

    r2_scores = []
    rmse_scores = []

    for result in results:
        test = result["test_station"]
        train = "+".join(result["train_stations"])
        r2 = result["metrics"]["r2"]
        rmse = result["metrics"]["rmse"]
        mae = result["metrics"]["mae"]

        r2_scores.append(r2)
        rmse_scores.append(rmse)

        print(f"{test:<15} {train:<20} {r2:>10.4f} {rmse:>10.4f} {mae:>10.4f}")

    print("-" * 70)
    print(f"{'MEAN':<15} {'':<20} {np.mean(r2_scores):>10.4f} {np.mean(rmse_scores):>10.4f}")
    print(f"{'STD':<15} {'':<20} {np.std(r2_scores):>10.4f} {np.std(rmse_scores):>10.4f}")
    print("=" * 70)

    results_df = pd.DataFrame(
        [
            {
                "test_station": r["test_station"],
                "train_stations": "+".join(r["train_stations"]),
                "r2": r["metrics"]["r2"],
                "rmse": r["metrics"]["rmse"],
                "mae": r["metrics"]["mae"],
                "bias": r["metrics"]["bias"],
                "correlation": r["metrics"]["correlation"],
                "n_samples": r["metrics"]["n_samples"],
            }
            for r in results
        ]
    )

    results_df.to_csv(output_dir / "cv_results.csv", index=False)
    log.info(f"\nResults saved to: {output_dir / 'cv_results.csv'}")

    log.info(f"\nOutputs saved to: {output_dir}")
    log.info("CV complete!")

    return results


if __name__ == "__main__":
    results = main()

