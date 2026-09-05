#!/usr/bin/env python
"""
Hyperparameter tuning for LSTM RZSM_25_avg using Optuna.

- Uses the same pipeline as run_cv.py.
- Optimizes mean validation R² across LOSO folds.
"""

import argparse
import logging
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import optuna
import pandas as pd
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from torch.utils.data import DataLoader, Subset

from src.config import DataConfig, FeatureConfig, ModelConfig, TrainingConfig
from src.data_loader import DataProcessor, RZSMDataset
from src.evaluator import Evaluator
from src.models import get_model
from src.trainer import Trainer

# --------------------------------------------------------------------------
# Basic setup (mirrors run_cv.py)
# --------------------------------------------------------------------------

STATIONS: Dict[str, str] = {
    "Ne1": "ne1_1_maize.csv",
    "Ne2": "ne2_1_maize.csv",
    "Ne3": "ne3_1_maize.csv",
}

CV_FOLDS: List[Dict] = [
    {"train": ["Ne1", "Ne2"], "test": "Ne3", "name": "fold1_test_Ne3"},
    {"train": ["Ne1", "Ne3"], "test": "Ne2", "name": "fold2_test_Ne2"},
    {"train": ["Ne2", "Ne3"], "test": "Ne1", "name": "fold3_test_Ne1"},
]

DATA_DIR = "/Users/rouhinmitra/SM_work/Data/Base/"
PRESTO_PATH = "/Users/rouhinmitra/SM_work/Code/Data/s2_pixels/presto_embeddings_fused_interpolated.csv"
OUTPUT_DIR = "outputs/hparam_search_rzsm25"

TARGET_COL = "RZSM_25_avg"
YEAR_RANGE = (2017, 2024)
MONTH_RANGE = (4, 10)
VAL_YEAR = 2022  # same as CVConfig.val_years in run_cv.py

# Dynamic / static features copied from CVConfig in run_cv.py
DYNAMIC_COLS = [
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
    # irrigation is handled as static in DataProcessor
]

STATIC_COLS = [f"emb_{k}" for k in range(128)] + [f"A{i:02d}" for i in range(64)] + [
    "precip_jan_apr",
]


# --------------------------------------------------------------------------
# Config builders
# --------------------------------------------------------------------------


def build_configs(trial: optuna.Trial, fold: Dict):
    """Sample hyperparameters and build configs for one fold."""

    # Search space (you can widen/narrow as needed)
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.6, step=0.1)
    lr = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    seq_length = trial.suggest_categorical("seq_length", [15, 20, 30])
    nan_threshold = 0.1  # keep same as run_cv.py

    data_cfg = DataConfig(
        train_files=[STATIONS[s] for s in fold["train"]],
        test_files=[STATIONS[fold["test"]]],
        data_dir=DATA_DIR,
        seq_length=seq_length,
        nan_threshold=nan_threshold,
        same_year_constraint=True,
        month_range=MONTH_RANGE,
        year_range=YEAR_RANGE,
    )

    feat_cfg = FeatureConfig(
        dynamic_cols=DYNAMIC_COLS,
        static_cols=STATIC_COLS,
        target_col=TARGET_COL,
        add_temporal=True,
        temporal_features=["doy_sin", "doy_cos"],
        use_presto_static=True,
        presto_embeddings_path=PRESTO_PATH,
    )

    model_cfg = ModelConfig(
        model_type="LSTM",
        hidden_dim=hidden_dim,
        dropout=dropout,
        num_layers=num_layers,
    )

    train_cfg = TrainingConfig(
        batch_size=batch_size,
        epochs=100,  # early stopping will usually stop sooner
        learning_rate=lr,
        early_stopping_patience=20,
        weight_decay=weight_decay,
    )

    return data_cfg, feat_cfg, model_cfg, train_cfg


# --------------------------------------------------------------------------
# One-fold run (returns validation R²)
# --------------------------------------------------------------------------


def run_fold(trial: optuna.Trial, fold: Dict, *, fold_index: int) -> float:
    """Train and evaluate a single LOSO fold; returns validation R²."""
    data_cfg, feat_cfg, model_cfg, train_cfg = build_configs(trial, fold)

    processor = DataProcessor(data_cfg, feat_cfg)
    # Use VAL_YEAR only for scaler fitting (matches run_cv behavior)
    data = processor.prepare_data(
        data_cfg.train_files,
        data_cfg.test_files,
        val_years=VAL_YEAR,
    )

    full_train_dataset = RZSMDataset(data["X_d_train"], data["X_s_train"], data["y_train"])

    # Split train/val by year (same logic as run_cv.py)
    dates_train = pd.to_datetime(data["dates_train"])
    # pd.to_datetime() can yield a DatetimeIndex (no .dt accessor); use .year which works for both
    years = np.asarray(dates_train.year)
    val_indices = [i for i, y in enumerate(years) if y == VAL_YEAR]
    train_indices = [i for i, y in enumerate(years) if y != VAL_YEAR]

    if len(val_indices) < 10:
        # No / too little validation data → prune this trial
        raise optuna.TrialPruned(f"Not enough validation samples for year {VAL_YEAR}")

    train_subset = Subset(full_train_dataset, train_indices)
    val_subset = Subset(full_train_dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=train_cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=train_cfg.batch_size, shuffle=False)

    dyn_dim = data["X_d_train"].shape[2]
    stat_dim = data["X_s_train"].shape[2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_cfg, dyn_dim, stat_dim)
    # Use a per-trial subdirectory to avoid collisions
    fold_out_dir = Path(OUTPUT_DIR) / f"trial_{trial.number}_{fold['name']}"
    trainer = Trainer(model, train_cfg, str(fold_out_dir), device=str(device))

    # Full training with early stopping (no per-epoch pruning, Trainer API doesn’t expose that)
    history = trainer.train(train_loader, val_loader)

    evaluator = Evaluator(trainer.model, processor.scaler_y, str(fold_out_dir), device=str(device))
    # IMPORTANT: evaluate on validation loader, not test
    y_pred, y_true = evaluator.predict(val_loader)
    metrics = evaluator.compute_metrics(y_pred, y_true)
    r2_val = float(metrics["r2"])

    # Report fold-level score so Optuna can prune between folds
    trial.report(r2_val, step=fold_index)
    if trial.should_prune():
        # best-effort cleanup before pruning
        if fold_out_dir.exists():
            shutil.rmtree(fold_out_dir, ignore_errors=True)
        raise optuna.TrialPruned()

    # Best-effort cleanup to avoid disk explosion (checkpoints/logs per trial×fold)
    if fold_out_dir.exists():
        shutil.rmtree(fold_out_dir, ignore_errors=True)

    return r2_val


# --------------------------------------------------------------------------
# Optuna objective: mean val R² over folds
# --------------------------------------------------------------------------


def objective(trial: optuna.Trial) -> float:
    r2_scores: List[float] = []
    for i, fold in enumerate(CV_FOLDS):
        r2 = run_fold(trial, fold, fold_index=i)
        r2_scores.append(r2)
    return float(np.mean(r2_scores))


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------


def main():
    global OUTPUT_DIR, VAL_YEAR
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning for RZSM_25_avg (LOSO folds).")
    parser.add_argument("--n-trials", type=int, default=60, help="Number of Optuna trials.")
    parser.add_argument("--seed", type=int, default=42, help="Sampler seed.")
    parser.add_argument("--val-year", type=int, default=VAL_YEAR, help="Validation year used within training stations.")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Output directory for optuna.db and results.")
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir
    VAL_YEAR = int(args.val_year)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=int(args.seed)),
        pruner=MedianPruner(n_startup_trials=5),
        study_name="lstm_rzsm25_tuning",
        storage=f"sqlite:///{OUTPUT_DIR}/optuna.db",
        load_if_exists=True,
    )

    print(f"Starting hyperparameter search. Results in: {OUTPUT_DIR}")
    # If any trial fails unexpectedly, don't abort the whole study.
    study.optimize(objective, n_trials=int(args.n_trials), show_progress_bar=True, catch=(Exception,))

    print("\n" + "=" * 60)
    print("BEST TRIAL")
    print("=" * 60)
    best = study.best_trial
    print(f"  Mean val R²: {best.value:.4f}")
    print("\n  Best hyperparameters (paste into CVConfig or your script):")
    for k, v in best.params.items():
        print(f"    {k} = {v!r}")

    # Save all trials
    df = study.trials_dataframe()
    df.to_csv(Path(OUTPUT_DIR) / "all_trials.csv", index=False)
    print(f"\nAll trials saved to {OUTPUT_DIR}/all_trials.csv")

    try:
        import optuna.visualization as vis

        fig = vis.plot_param_importances(study)
        fig.write_html(str(Path(OUTPUT_DIR) / "param_importances.html"))
        print(f"Param importances written to {OUTPUT_DIR}/param_importances.html")
    except Exception:
        print("Skipped importance plot (install plotly + optuna[viz] to enable).")


if __name__ == "__main__":
    main()