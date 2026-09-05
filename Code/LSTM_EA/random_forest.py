#!/usr/bin/env python
"""
Leave-One-Station-Out Cross-Validation for Random Forest (RZSM regression).

Uses the same folds and data pipeline as run_cv.py:
- Fold 1: Train Ne1+Ne2, Test Ne3
- Fold 2: Train Ne1+Ne3, Test Ne2
- Fold 3: Train Ne2+Ne3, Test Ne1

Features: last timestep of each sliding window (dynamic + static concatenated).
Target: RZSM_25_avg at last timestep (inverse-transformed to original units).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Union, Sequence, Tuple
import logging
import random

from sklearn.ensemble import RandomForestRegressor

from src.config import DataConfig, FeatureConfig
from src.data_loader import DataProcessor
from utils.logging_utils import setup_logging

# Same station and fold definitions as run_cv.py
STATIONS = {
    'Ne1': 'ne1_1_maize.csv',
    'Ne2': 'ne2_1_maize.csv',
    'Ne3': 'ne3_1_maize.csv',
}

CV_FOLDS = [
    {'train': ['Ne1', 'Ne2'], 'test': 'Ne3', 'name': 'fold1_test_Ne3'},
    {'train': ['Ne1', 'Ne3'], 'test': 'Ne2', 'name': 'fold2_test_Ne2'},
    {'train': ['Ne2', 'Ne3'], 'test': 'Ne1', 'name': 'fold3_test_Ne1'},
]


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
        'r2': float(r2),
        'rmse': float(rmse),
        'mae': float(mae),
        'bias': float(bias),
        'correlation': float(corr) if not np.isnan(corr) else np.nan,
        'nse': float(nse) if not np.isnan(nse) else np.nan,
        'n_samples': len(y_true),
    }


@dataclass
class RFConfig:
    """Random Forest CV configuration (data/features match run_cv.py; RF-specific params)."""
    data_dir: str = "/Users/rouhinmitra/SM_work/Data/Base/"
    output_dir: str = "outputs/rf_cv_results"
    seq_length: int = 20
    nan_threshold: float = 0.1
    seed: int = 42
    year_range: Tuple[int, int] = (2017, 2024)

    use_presto_static: bool = True
    presto_embeddings_path: str = "/Users/rouhinmitra/SM_work/Code/Data/s2_pixels/presto_embeddings_fused_interpolated.csv"

    dynamic_cols: List[str] = None
    static_cols: List[str] = None
    target_col: str = "RZSM_25_avg"

    # Random Forest hyperparameters
    n_estimators: int = 200
    max_depth: Optional[int] = 20
    min_samples_leaf: int = 5
    random_state: Optional[int] = None  # set from seed in __post_init__ if None

    def __post_init__(self):
        if self.random_state is None:
            self.random_state = self.seed
        if self.dynamic_cols is None:
            self.dynamic_cols = [
                'SSM', 'SSM_avg', 'SWC_PI_F_2_1_1', 'SWC_PI_F_3_1_1',
                'P_PI_F_1_1_1', 'P_PI_F_2_2_1', 'I', 'TA_1_1_1', 'RH_1_1_1', 'LE_1_1_1', 'NETRAD_1_1_1',
                'ndvi', 'b11', 'b12', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b8a',
            ]
        if self.static_cols is None:
            if self.use_presto_static:
                self.static_cols = [f'emb_{k}' for k in range(128)] + [f'A{i:02d}' for i in range(64)] + [
                    'precip_jan_apr',
                ]
            else:
                self.static_cols = [f'A{i:02d}' for i in range(64)] + [
                    'precip_jan_apr',
                    # 'precip_may_oct',
                ]


def run_single_fold(
    fold: Dict,
    config: RFConfig,
    logger: logging.Logger,
) -> Dict:
    """
    Run a single CV fold: prepare data, build last-timestep features, fit RF, evaluate.

    Returns dict with predictions, actuals, metrics, and station info.
    """
    fold_name = fold['name']
    train_stations = fold['train']
    test_station = fold['test']

    logger.info(f"\n{'='*70}")
    logger.info(f"FOLD: {fold_name}")
    logger.info(f"Training on: {train_stations}")
    logger.info(f"Testing on: {test_station}")
    logger.info(f"{'='*70}")

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

    logger.info("Preparing data...")
    processor = DataProcessor(data_config, feature_config)
    data = processor.prepare_data(
        data_config.train_files,
        data_config.test_files,
        val_years=None,
    )

    X_d_train = data['X_d_train']   # (n_train, seq_len, dyn_dim)
    X_s_train = data['X_s_train']   # (n_train, seq_len, stat_dim)
    y_train = data['y_train']       # (n_train, seq_len)
    X_d_test = data['X_d_test']
    X_s_test = data['X_s_test']
    y_test = data['y_test']
    dates_test = data['dates_test']

    # Last-timestep features: concatenate dynamic and static
    X_train = np.concatenate([X_d_train[:, -1, :], X_s_train[:, -1, :]], axis=1)
    X_test = np.concatenate([X_d_test[:, -1, :], X_s_test[:, -1, :]], axis=1)

    y_train_last_scaled = y_train[:, -1]
    y_test_last_scaled = y_test[:, -1]

    y_train_last = processor.scaler_y.inverse_transform(y_train_last_scaled.reshape(-1, 1)).ravel()
    y_test_last = processor.scaler_y.inverse_transform(y_test_last_scaled.reshape(-1, 1)).ravel()

    train_size = len(y_train_last)
    logger.info(f"Train/Test samples: {train_size} / {len(y_test_last)}")

    model = RandomForestRegressor(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        random_state=config.random_state,
    )
    model.fit(X_train, y_train_last)

    y_pred = model.predict(X_test)

    metrics = _compute_metrics(y_pred, y_test_last)

    r2_may_oct = np.nan
    n_may_oct = 0
    if dates_test is not None and len(dates_test) == len(y_test_last):
        months = pd.to_datetime(dates_test).month
        mask = (months >= 5) & (months <= 10)
        n_may_oct = int(np.sum(mask))
        if n_may_oct > 1:
            r2_may_oct = _r2(y_test_last[mask], y_pred[mask])

    logger.info(f"\nFold {fold_name} Results:")
    logger.info(f"  R²:   {metrics['r2']:.4f}")
    logger.info(f"  R² (May–Oct): {r2_may_oct:.4f}" if not np.isnan(r2_may_oct) else "  R² (May–Oct): n/a")
    logger.info(f"  RMSE: {metrics['rmse']:.4f}")
    logger.info(f"  MAE:  {metrics['mae']:.4f}")

    fold_output_dir = Path(config.output_dir) / fold_name
    fold_output_dir.mkdir(parents=True, exist_ok=True)

    # Save per-fold predictions
    pred_df = pd.DataFrame({
        'actual': y_test_last,
        'predicted': y_pred,
        'residual': y_test_last - y_pred,
    })
    pred_df.to_csv(fold_output_dir / 'predictions.csv', index=False)
    logger.info(f"Predictions saved to {fold_output_dir / 'predictions.csv'}")

    return {
        'fold_name': fold_name,
        'test_station': test_station,
        'train_stations': train_stations,
        'y_pred': y_pred,
        'y_true': y_test_last,
        'metrics': metrics,
        'dates_test': dates_test,
        'n_train': train_size,
        'n_test': len(y_test_last),
        'r2_may_oct': r2_may_oct,
        'n_may_oct': n_may_oct,
    }


def create_combined_scatter_plot(results: List[Dict], output_dir: Path):
    """Create a combined 3-panel scatter plot with all folds (Random Forest)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for result, ax, color in zip(results, axes, colors):
        y_pred = result['y_pred']
        y_true = result['y_true']
        metrics = result['metrics']
        test_station = result['test_station']

        ax.scatter(y_true, y_pred, alpha=0.3, s=10, c=color)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 Line')
        z = np.polyfit(y_true, y_pred, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min_val, max_val, 100)
        ax.plot(x_line, p(x_line), 'k-', linewidth=1.5, alpha=0.7, label='Best Fit')

        ax.set_xlabel('Observed RZSM (%)', fontsize=11)
        ax.set_ylabel('Predicted RZSM (%)', fontsize=11)
        ax.set_title(f'Test: {test_station}\nR² = {metrics["r2"]:.3f}, RMSE = {metrics["rmse"]:.2f}', fontsize=12)
        ax.set_aspect('equal', 'box')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=9)

    plt.suptitle('Leave-One-Station-Out Cross-Validation: Random Forest', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = output_dir / 'cv_scatter_plots.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Combined scatter plot saved to: {save_path}")


def create_summary_plot(results: List[Dict], output_dir: Path):
    """Create R² summary bar plot with mean line."""
    fig, ax = plt.subplots(figsize=(8, 5))
    stations = [r['test_station'] for r in results]
    r2_scores = [r['metrics']['r2'] for r in results]

    x = np.arange(len(stations))
    width = 0.35
    bars1 = ax.bar(x - width / 2, r2_scores, width, label='R²', color='steelblue')
    for bar, val in zip(bars1, r2_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    mean_r2 = np.mean(r2_scores)
    ax.axhline(y=mean_r2, color='red', linestyle='--', linewidth=2, label=f'Mean R² = {mean_r2:.3f}')
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_xlabel('Test Station', fontsize=12)
    ax.set_title('Leave-One-Station-Out CV: R² by Test Station (Random Forest)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Test: {s}' for s in stations])
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    save_path = output_dir / 'cv_r2_summary.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"R² summary plot saved to: {save_path}")


def main():
    """Run leave-one-station-out cross-validation with Random Forest."""
    config = RFConfig()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(str(output_dir), log_level="INFO")
    log = logging.getLogger(__name__)

    log.info("=" * 70)
    log.info("LEAVE-ONE-STATION-OUT CROSS-VALIDATION (RANDOM FOREST)")
    log.info("=" * 70)
    log.info(f"Stations: {list(STATIONS.keys())}")
    log.info(f"Number of folds: {len(CV_FOLDS)}")
    log.info(f"Year range: {config.year_range[0]}–{config.year_range[1]} (inclusive)")
    log.info(f"RF: n_estimators={config.n_estimators}, max_depth={config.max_depth}, min_samples_leaf={config.min_samples_leaf}")
    if config.use_presto_static:
        presto_path = Path(config.presto_embeddings_path)
        if not presto_path.is_absolute():
            presto_path = Path(__file__).resolve().parent / presto_path
        log.info("Presto embeddings: ENABLED")
        log.info(f"  Path: {presto_path}")
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

    results = []
    for fold in CV_FOLDS:
        result = run_single_fold(fold, config, log)
        results.append(result)

    log.info("\n" + "=" * 70)
    log.info("CREATING VISUALIZATIONS")
    log.info("=" * 70)
    create_combined_scatter_plot(results, output_dir)
    create_summary_plot(results, output_dir)

    log.info("\n" + "=" * 70)
    log.info("CROSS-VALIDATION SUMMARY")
    log.info("=" * 70)

    print("\n" + "=" * 100)
    print("RANDOM FOREST CV RESULTS")
    print("=" * 100)
    print(f"{'Test Station':<15} {'Train Stations':<20} {'R²':>8} {'R²_MayOct':>10} {'RMSE':>8} {'MAE':>8} {'N_train':>10} {'N_test':>10}")
    print("-" * 100)

    r2_scores = []
    r2_may_oct_scores = []
    rmse_scores = []

    for result in results:
        test = result['test_station']
        train = '+'.join(result['train_stations'])
        r2 = result['metrics']['r2']
        r2_mo = result.get('r2_may_oct', np.nan)
        rmse = result['metrics']['rmse']
        mae = result['metrics']['mae']
        n_train = result.get('n_train', '')
        n_test = result.get('n_test', '')

        r2_scores.append(r2)
        if not np.isnan(r2_mo):
            r2_may_oct_scores.append(r2_mo)
        rmse_scores.append(rmse)

        r2_mo_str = f"{r2_mo:.4f}" if not np.isnan(r2_mo) else "n/a"
        print(f"{test:<15} {train:<20} {r2:>8.4f} {r2_mo_str:>10} {rmse:>8.4f} {mae:>8.4f} {n_train:>10} {n_test:>10}")

    all_y_true = np.concatenate([r['y_true'] for r in results])
    all_y_pred = np.concatenate([r['y_pred'] for r in results])
    all_dates = np.concatenate([r['dates_test'] for r in results])
    months = pd.to_datetime(all_dates).month
    mask_mo = (months >= 5) & (months <= 10)
    r2_overall_may_oct = np.nan
    if np.sum(mask_mo) > 1:
        r2_overall_may_oct = _r2(all_y_true[mask_mo], all_y_pred[mask_mo])

    print("-" * 100)
    mean_r2_mo = np.mean(r2_may_oct_scores) if r2_may_oct_scores else np.nan
    mean_r2_mo_str = f"{mean_r2_mo:.4f}" if not np.isnan(mean_r2_mo) else "n/a"
    print(f"{'MEAN':<15} {'':<20} {np.mean(r2_scores):>8.4f} {mean_r2_mo_str:>10} {np.mean(rmse_scores):>8.4f} {'':>8} {'':>10} {'':>10}")
    print(f"{'STD':<15} {'':<20} {np.std(r2_scores):>8.4f} {'':>10} {np.std(rmse_scores):>8.4f}")
    print("-" * 100)
    r2_overall_mo_str = f"{r2_overall_may_oct:.4f}" if not np.isnan(r2_overall_may_oct) else "n/a"
    print(f"Overall R² (May–Oct, pooled): {r2_overall_mo_str}")
    print("=" * 100)

    results_df = pd.DataFrame([
        {
            'test_station': r['test_station'],
            'train_stations': '+'.join(r['train_stations']),
            'r2': r['metrics']['r2'],
            'r2_may_oct': r.get('r2_may_oct'),
            'rmse': r['metrics']['rmse'],
            'mae': r['metrics']['mae'],
            'bias': r['metrics']['bias'],
            'correlation': r['metrics']['correlation'],
            'n_samples': r['metrics']['n_samples'],
            'n_may_oct': r.get('n_may_oct'),
            'n_train': r.get('n_train'),
            'n_test': r.get('n_test'),
        }
        for r in results
    ])
    results_df.to_csv(output_dir / 'cv_results.csv', index=False)
    log.info(f"\nResults saved to: {output_dir / 'cv_results.csv'}")

    if not np.isnan(r2_overall_may_oct):
        log.info(f"Overall R² (May–Oct, pooled): {r2_overall_may_oct:.4f}")
        summary_row = pd.DataFrame([{
            'test_station': 'OVERALL_MayOct',
            'train_stations': '',
            'r2': np.nan,
            'r2_may_oct': r2_overall_may_oct,
            'rmse': np.nan,
            'mae': np.nan,
            'bias': np.nan,
            'correlation': np.nan,
            'n_samples': int(np.sum(mask_mo)),
            'n_may_oct': int(np.sum(mask_mo)),
            'n_train': np.nan,
            'n_test': np.nan,
        }])
        results_df = pd.concat([results_df, summary_row], ignore_index=True)
        results_df.to_csv(output_dir / 'cv_results.csv', index=False)

    log.info(f"\nOutputs saved to: {output_dir}")
    log.info("RF CV complete!")
    return results


if __name__ == "__main__":
    results = main()
