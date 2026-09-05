#!/usr/bin/env python
"""
Spatial Transfer with Varying Training Window Length.

Runs leave-one-station-out (spatial transfer) LSTM experiments while
progressively reducing the number of training years.  The training window
is anchored at 2017 and shrinks from the end:

  End 2023 -> train {2017-2021, 2023}  (6 years)
  End 2021 -> train {2017-2021}         (5 years)
  End 2020 -> train {2017-2020}         (4 years)
  End 2019 -> train {2017-2019}         (3 years)
  End 2018 -> train {2017-2018}         (2 years)

Year 2022 is always reserved as the validation set (from training stations).
The test set (held-out station) always spans the full 2017-2023 range for a
fair comparison across windows.

Produces:
  - Per-window, per-fold model artifacts
  - all_windows_spatial_metrics.csv
  - R² vs. training window length line plots per test station
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from src.config import DataConfig, FeatureConfig, ModelConfig, TrainingConfig
from src.data_loader import DataProcessor, RZSMDataset
from src.evaluator import Evaluator
from src.models import get_model
from src.trainer import Trainer
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

VAL_YEAR = 2022
ANCHOR_START = 2017
TRAINING_END_YEARS = [2024, 2023, 2021, 2020, 2019, 2018]

TITLE_FONTSIZE = 26
AXIS_LABEL_FONTSIZE = 20
TICK_LABEL_FONTSIZE = 18
LEGEND_FONTSIZE = 16
LINE_COLORS = ["red", "blue", "green"]


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return np.nan
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return np.nan
    return float(1 - ss_res / ss_tot)


@dataclass
class SpatialWindowConfig:
    """Configuration for spatial-transfer training-window experiment."""

    data_dir: str = "/Users/rouhinmitra/SM_work/Data/Base/"
    output_dir: str = "outputs/spatial_transfer_training_windows"
    seq_length: int = 20
    nan_threshold: float = 0.1

    hidden_dim: int = 32
    dropout: float = 0.50
    batch_size: int = 32
    epochs: int = 150
    learning_rate: float = 0.0005
    early_stopping_patience: int = 20
    weight_decay: float = 0.001
    add_temporal: bool = True
    seed: int = 42
    model_type: str = "LSTM"
    num_layers: int = 1
    val_fraction: float = 0.15

    # Always load the full range; training-year filtering is done post-hoc
    year_range: Tuple[int, int] = (2017, 2024)

    use_presto_static: bool = True
    presto_embeddings_path: str = (
        "/Users/rouhinmitra/SM_work/Code/Data/s2_pixels/"
        "presto_embeddings_fused_interpolated.csv"
    )

    dynamic_cols: List[str] = None
    static_cols: List[str] = None
    target_col: str = "RZSM_25_avg"

    def __post_init__(self):
        if self.dynamic_cols is None:
            self.dynamic_cols = [
                "SSM", "SSM_avg", "SWC_PI_F_2_1_1", "SWC_PI_F_3_1_1",
                "P_PI_F_1_1_1", "P_PI_F_2_2_1", "I", "TA_1_1_1",
                "RH_1_1_1", "LE_1_1_1", "NETRAD_1_1_1",
                "ndvi", "b11", "b12", "b2", "b3", "b4", "b5", "b6",
                "b7", "b8", "b8a",
            ]
        if self.static_cols is None:
            if self.use_presto_static:
                self.static_cols = (
                    [f"emb_{k}" for k in range(128)]
                    + [f"A{i:02d}" for i in range(64)]
                    + ["precip_jan_apr"]
                )
            else:
                self.static_cols = (
                    [f"A{i:02d}" for i in range(64)]
                    + ["precip_jan_apr", "precip_may_oct"]
                )


# ---------------------------------------------------------------------------
# Modified fold runner
# ---------------------------------------------------------------------------

def run_fold_with_train_filter(
    fold: Dict,
    config: SpatialWindowConfig,
    allowed_train_years: Set[int],
    fold_output_dir: Path,
    logger: logging.Logger,
) -> Dict:
    """
    Run a single leave-one-station-out fold, restricting the training
    sequences to *allowed_train_years* while always validating on VAL_YEAR.
    """
    fold_name = fold["name"]
    train_stations = fold["train"]
    test_station = fold["test"]

    logger.info("  FOLD: %s | Train: %s | Test: %s", fold_name, train_stations, test_station)

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
                "Set use_presto_static=False or provide a valid path."
            )

    feature_config = FeatureConfig(
        dynamic_cols=config.dynamic_cols,
        static_cols=config.static_cols,
        target_col=config.target_col,
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

    # Prepare data (full year range for both train and test stations)
    processor = DataProcessor(data_config, feature_config)
    data = processor.prepare_data(
        data_config.train_files,
        data_config.test_files,
        val_years=VAL_YEAR,
    )

    full_train_dataset = RZSMDataset(
        data["X_d_train"], data["X_s_train"], data["y_train"]
    )
    test_dataset = RZSMDataset(
        data["X_d_test"], data["X_s_test"], data["y_test"]
    )

    dates_train = data["dates_train"]
    years = pd.to_datetime(dates_train).year

    # Validation: year == VAL_YEAR
    val_indices = [i for i, y in enumerate(years) if y == VAL_YEAR]

    # Training: year in allowed_train_years (which already excludes VAL_YEAR)
    train_indices = [i for i, y in enumerate(years) if y in allowed_train_years]

    train_size = len(train_indices)
    val_size = len(val_indices)

    if train_size == 0:
        logger.warning("  No training samples for fold %s; skipping.", fold_name)
        return None

    # Fall back to random val split if no val samples from 2022
    if val_size == 0:
        logger.warning(
            "  No validation samples from year %d; falling back to %.0f%% random split.",
            VAL_YEAR, config.val_fraction * 100,
        )
        rng = np.random.RandomState(config.seed)
        arr = np.array(train_indices)
        rng.shuffle(arr)
        n_val = max(1, int(len(arr) * config.val_fraction))
        val_indices = arr[:n_val].tolist()
        train_indices = arr[n_val:].tolist()
        train_size = len(train_indices)
        val_size = len(val_indices)

    train_subset = Subset(full_train_dataset, train_indices)
    val_subset = Subset(full_train_dataset, val_indices)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False)

    logger.info(
        "  Train/Val/Test: %d / %d / %d samples",
        train_size, val_size, len(test_dataset),
    )

    dyn_dim = data["X_d_train"].shape[2]
    stat_dim = data["X_s_train"].shape[2]

    model = get_model(model_config, dyn_dim, stat_dim)
    fold_output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(model, training_config, str(fold_output_dir), device=str(device))

    history = trainer.train(train_loader, val_loader)

    evaluator = Evaluator(
        trainer.model, processor.scaler_y, str(fold_output_dir), device=str(device)
    )
    y_pred, y_true = evaluator.predict(test_loader)
    metrics = evaluator.compute_metrics(y_pred, y_true)

    evaluator.plot_training_history(history, "loss")

    dates_test = data.get("dates_test")
    r2_may_oct = np.nan
    n_may_oct = 0
    if dates_test is not None and len(dates_test) == len(y_true):
        months = pd.to_datetime(dates_test).month
        mask = (months >= 5) & (months <= 10)
        n_may_oct = int(np.sum(mask))
        if n_may_oct > 1:
            r2_may_oct = _r2(y_true[mask], y_pred[mask])

    logger.info(
        "  Results: R²=%.4f  R²(May-Oct)=%.4f  RMSE=%.4f  MAE=%.4f",
        metrics["r2"],
        r2_may_oct if not np.isnan(r2_may_oct) else -999,
        metrics["rmse"],
        metrics["mae"],
    )

    return {
        "fold_name": fold_name,
        "test_station": test_station,
        "train_stations": train_stations,
        "y_pred": y_pred,
        "y_true": y_true,
        "metrics": metrics,
        "history": history,
        "dates_test": dates_test,
        "n_train": train_size,
        "n_val": val_size,
        "n_test": len(test_dataset),
        "r2_may_oct": r2_may_oct,
        "n_may_oct": n_may_oct,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_r2_vs_training_years(
    metrics_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Line plot of R² vs. number of training years, one line per test station."""

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, station in enumerate(sorted(metrics_df["test_station"].unique())):
        sub = (
            metrics_df[metrics_df["test_station"] == station]
            .groupby("train_years", as_index=False)["r2"]
            .mean()
            .sort_values("train_years")
        )
        ax.plot(
            sub["train_years"],
            sub["r2"],
            marker="o",
            markersize=6,
            linewidth=2,
            color=LINE_COLORS[i % len(LINE_COLORS)],
            label=station,
        )

    ax.set_xlabel("Training window length (years)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("R\u00b2", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(
        "Spatial Transfer LSTM: R\u00b2 vs. training window length",
        fontsize=TITLE_FONTSIZE,
    )
    ax.legend(fontsize=LEGEND_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    fig.tight_layout()

    out_path = output_dir / "r2_vs_training_years_spatial.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")

    # Mean across stations
    grp = metrics_df.groupby("train_years")["r2"]
    mean_df = pd.DataFrame({
        "train_years": grp.mean().index,
        "mean": grp.mean().values,
        "std": grp.std().values,
    })

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(
        mean_df["train_years"],
        mean_df["mean"],
        marker="s",
        markersize=8,
        color="blue",
        linewidth=2,
        label="Mean across stations",
    )
    if (mean_df["std"] > 0).any():
        ax2.fill_between(
            mean_df["train_years"],
            mean_df["mean"] - mean_df["std"],
            mean_df["mean"] + mean_df["std"],
            alpha=0.2,
            color="gray",
        )
    ax2.set_xlabel("Training window length (years)", fontsize=AXIS_LABEL_FONTSIZE)
    ax2.set_ylabel("Mean R\u00b2 across stations", fontsize=AXIS_LABEL_FONTSIZE)
    ax2.set_title(
        "Spatial Transfer LSTM: mean R\u00b2 vs. training window length",
        fontsize=TITLE_FONTSIZE,
    )
    ax2.legend(fontsize=LEGEND_FONTSIZE)
    ax2.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)
    fig2.tight_layout()

    out_path2 = output_dir / "r2_vs_training_years_spatial_mean.png"
    fig2.savefig(out_path2, dpi=150)
    plt.close(fig2)
    print(f"Saved {out_path2}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    config = SpatialWindowConfig()

    base_output_dir = Path(config.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(str(base_output_dir), log_level="INFO")
    log = logging.getLogger(__name__)

    log.info("=" * 80)
    log.info("SPATIAL TRANSFER: R² vs. TRAINING WINDOW LENGTH")
    log.info("=" * 80)
    log.info("Model: %s", config.model_type)
    log.info("Stations: %s", list(STATIONS.keys()))
    log.info("Data loaded with year_range=%s (test always full range)", config.year_range)
    log.info("Validation year: %d (always)", VAL_YEAR)
    log.info("Training end years: %s", TRAINING_END_YEARS)
    log.info("Target: %s", config.target_col)

    if config.use_presto_static:
        p = Path(config.presto_embeddings_path)
        if not p.is_absolute():
            p = Path(__file__).resolve().parent / p
        if not p.exists():
            raise FileNotFoundError(
                f"Presto embeddings file not found: {p}. "
                "Set use_presto_static=False or provide a valid path."
            )
        log.info("Presto embeddings: ENABLED (%s)", p)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    all_rows: List[Dict] = []

    for train_end in TRAINING_END_YEARS:
        allowed_train_years = set(range(ANCHOR_START, train_end + 1)) - {VAL_YEAR}
        n_train_years = len(allowed_train_years)
        window_tag = f"end{train_end}_train{n_train_years}yr"

        log.info("\n" + "=" * 80)
        log.info(
            "WINDOW end=%d | training years=%s (%d yr) | val=%d",
            train_end,
            sorted(allowed_train_years),
            n_train_years,
            VAL_YEAR,
        )
        log.info("=" * 80)

        window_dir = base_output_dir / window_tag
        window_dir.mkdir(parents=True, exist_ok=True)

        for fold in CV_FOLDS:
            fold_dir = window_dir / fold["name"]
            try:
                result = run_fold_with_train_filter(
                    fold, config, allowed_train_years, fold_dir, log,
                )
            except Exception as e:
                log.exception("Error in fold %s window %s: %s", fold["name"], window_tag, e)
                continue

            if result is None:
                continue

            all_rows.append({
                "test_station": result["test_station"],
                "train_stations": "+".join(result["train_stations"]),
                "train_end_year": train_end,
                "train_years": n_train_years,
                "allowed_years": str(sorted(allowed_train_years)),
                "r2": result["metrics"]["r2"],
                "r2_may_oct": result["r2_may_oct"],
                "rmse": result["metrics"]["rmse"],
                "mae": result["metrics"]["mae"],
                "bias": result["metrics"]["bias"],
                "correlation": result["metrics"]["correlation"],
                "n_train": result["n_train"],
                "n_val": result["n_val"],
                "n_test": result["n_test"],
            })

    if not all_rows:
        log.warning("No successful runs. Exiting.")
        return

    metrics_df = pd.DataFrame(all_rows)
    csv_path = base_output_dir / "all_windows_spatial_metrics.csv"
    metrics_df.to_csv(csv_path, index=False)
    log.info("Metrics saved to %s", csv_path)

    # Print summary table
    log.info("\n" + "=" * 110)
    log.info("SUMMARY")
    log.info("=" * 110)
    header = (
        f"{'Test':<6} {'Train':<10} {'EndYr':>6} {'#Yr':>4} "
        f"{'R2':>8} {'R2_MO':>8} {'RMSE':>8} {'MAE':>8} "
        f"{'N_tr':>7} {'N_val':>7} {'N_te':>7}"
    )
    log.info(header)
    log.info("-" * 110)
    for _, row in metrics_df.iterrows():
        r2mo = f"{row['r2_may_oct']:.4f}" if not np.isnan(row["r2_may_oct"]) else "n/a"
        log.info(
            f"{row['test_station']:<6} {row['train_stations']:<10} "
            f"{row['train_end_year']:>6} {row['train_years']:>4} "
            f"{row['r2']:>8.4f} {r2mo:>8} {row['rmse']:>8.4f} {row['mae']:>8.4f} "
            f"{row['n_train']:>7} {row['n_val']:>7} {row['n_test']:>7}"
        )

    # Per-window mean R²
    log.info("-" * 110)
    for ty in sorted(metrics_df["train_years"].unique()):
        sub = metrics_df[metrics_df["train_years"] == ty]
        log.info(
            "  %d training years: mean R²=%.4f  std=%.4f",
            ty, sub["r2"].mean(), sub["r2"].std(),
        )
    log.info("=" * 110)

    # Plots
    plot_r2_vs_training_years(metrics_df, base_output_dir)

    log.info("Spatial transfer training-window experiment complete.")


if __name__ == "__main__":
    main()
