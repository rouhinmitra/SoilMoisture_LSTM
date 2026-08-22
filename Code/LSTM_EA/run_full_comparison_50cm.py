#!/usr/bin/env python
"""
Full Model Comparison for 50cm Root Zone Soil Moisture

Multi-trial evaluation comparing:
- EALSTM: Entity-aware LSTM (static features control input gate)
- LSTM: Standard LSTM (static features as input)
- Random Forest: Classical ML baseline

Target: RZSM_50_avg (50cm depth)
Tests multiple sequence lengths to find optimal lookback window.
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import logging
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from src.config import DataConfig, FeatureConfig, ModelConfig, TrainingConfig
from src.data_loader import DataProcessor, RZSMDataset
from src.models import get_model
from src.trainer import Trainer
from src.evaluator import Evaluator
from utils.logging_utils import setup_logging

# Configuration
STATIONS = {
    'Ne1': 'ne1_1_maize.csv',
    'Ne2': 'ne2_1_maize.csv', 
    'Ne3': 'ne3_1_maize.csv'
}

CV_FOLDS = [
    {'train': ['Ne1', 'Ne2'], 'test': 'Ne3', 'name': 'fold1_test_Ne3'},
    {'train': ['Ne1', 'Ne3'], 'test': 'Ne2', 'name': 'fold2_test_Ne2'},
    {'train': ['Ne2', 'Ne3'], 'test': 'Ne1', 'name': 'fold3_test_Ne1'},
]

# Experiment settings
N_TRIALS = 5
SEEDS = [42, 123, 456, 789, 1024]
SEQ_LENGTHS = [7, 14, 21, 30]  # Test multiple sequence lengths

# Models to compare
MODELS = ['EALSTM', 'LSTM', 'RandomForest']

# Target depth
TARGET_DEPTH = "50cm"
TARGET_COL = "RZSM_50_avg"


def set_all_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class CVConfig:
    """Cross-validation configuration"""
    data_dir: str = "/Users/rouhinmitra/SM_work/Data/Base/"
    output_dir: str = "outputs/full_comparison_50cm"
    seq_length: int = 14  # Will be updated per experiment
    nan_threshold: float = 0.1
    model_type: str = "EALSTM"
    # Neural network hyperparameters
    hidden_dim: int = 128
    dropout: float = 0.5
    batch_size: int = 32
    epochs: int = 150
    learning_rate: float = 0.0005
    early_stopping_patience: int = 15
    weight_decay: float = 0.001
    add_temporal: bool = True
    # Random Forest hyperparameters
    rf_n_estimators: int = 200
    rf_max_depth: int = 20
    rf_min_samples_split: int = 5
    rf_n_jobs: int = -1
    
    # Features
    dynamic_cols: List[str] = None
    static_cols: List[str] = None
    target_col: str = TARGET_COL  # 50cm depth
    
    def __post_init__(self):
        if self.dynamic_cols is None:
            self.dynamic_cols = [
                'SSM', 'SSM_avg', 'SWC_PI_F_2_1_1', 'SWC_PI_F_3_1_1',
                'P_PI_F_1_1_1', 'P_PI_F_2_2_1', 'I', 'TA_1_1_1', 'RH_1_1_1',
                'ndvi', 'b11', 'b12', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b8a', 'irrigation'
            ]
        if self.static_cols is None:
            self.static_cols = [f'A{i:02d}' for i in range(64)] + [
                'precip_jan_apr',   # accumulated precip Jan 1–Apr 30 (static / input-gate)
                'precip_may_oct',   # accumulated precip May–Oct summer (static / input-gate)
            ]


def save_timeseries_plots_by_year(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: np.ndarray,
    save_dir: Path,
    model_type: str,
    test_station: str,
    trial_num: int,
) -> None:
    """Save observed vs predicted time series plots, one per year (out-of-sample)."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    dates = pd.to_datetime(dates)
    years = dates.year
    for year in np.unique(years):
        mask = years == year
        if mask.sum() == 0:
            continue
        d = np.asarray(dates)[mask]
        yt = y_true[mask]
        yp = y_pred[mask]
        order = np.argsort(d)
        d = d[order]
        yt = yt[order]
        yp = yp[order]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(d, yt, 'o-', label='Observed', color='#1f77b4', markersize=3, alpha=0.8)
        ax.plot(d, yp, 's-', label='Predicted', color='#ff7f0e', markersize=3, alpha=0.8)
        ax.set_xlabel('Date')
        ax.set_ylabel('RZSM')
        ax.set_title(f'{model_type} — {test_station} — {year} (trial {trial_num})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = save_dir / f"timeseries_{model_type}_{test_station}_year{year}_trial{trial_num}.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()


def prepare_data_for_fold(fold: Dict, config: CVConfig):
    """Prepare data for a single fold."""
    train_stations = fold['train']
    test_station = fold['test']
    
    data_config = DataConfig(
        train_files=[STATIONS[s] for s in train_stations],
        test_files=[STATIONS[test_station]],
        data_dir=config.data_dir,
        seq_length=config.seq_length,
        nan_threshold=config.nan_threshold,
        same_year_constraint=True,
        month_range=(5, 10),  # May–October only
    )
    
    feature_config = FeatureConfig(
        dynamic_cols=config.dynamic_cols,
        static_cols=config.static_cols,
        target_col=config.target_col,
        add_temporal=config.add_temporal,
        temporal_features=['doy_sin', 'doy_cos']
    )
    
    processor = DataProcessor(data_config, feature_config)
    data = processor.prepare_data(data_config.train_files, data_config.test_files)
    
    return data, processor


def run_lstm_fold(
    fold: Dict,
    config: CVConfig,
    model_type: str,
    trial_num: int = 1,
    timeseries_dir: Optional[Path] = None,
) -> Dict:
    """Run LSTM/EALSTM for a single fold."""
    data, processor = prepare_data_for_fold(fold, config)
    
    train_dataset = RZSMDataset(data['X_d_train'], data['X_s_train'], data['y_train'])
    test_dataset = RZSMDataset(data['X_d_test'], data['X_s_test'], data['y_test'])
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    dyn_dim = data['X_d_train'].shape[2]
    stat_dim = data['X_s_train'].shape[2]
    
    model_config = ModelConfig(
        model_type=model_type,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout
    )
    
    training_config = TrainingConfig(
        batch_size=config.batch_size,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        early_stopping_patience=config.early_stopping_patience,
        weight_decay=config.weight_decay
    )
    
    model = get_model(model_config, dyn_dim, stat_dim)
    
    fold_output_dir = Path(config.output_dir) / "temp"
    fold_output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = Trainer(model, training_config, str(fold_output_dir), device=str(device))
    trainer.train(train_loader, test_loader)
    
    evaluator = Evaluator(trainer.model, processor.scaler_y, str(fold_output_dir), device=str(device))
    y_pred, y_true = evaluator.predict(test_loader)
    metrics = evaluator.compute_metrics(y_pred, y_true)
    
    if timeseries_dir is not None and 'dates_test' in data:
        save_timeseries_plots_by_year(
            y_true, y_pred, data['dates_test'],
            timeseries_dir, model_type, fold['test'], trial_num,
        )
    
    return {'test_station': fold['test'], 'metrics': metrics}


def run_rf_fold(
    fold: Dict,
    config: CVConfig,
    seed: int,
    trial_num: int = 1,
    timeseries_dir: Optional[Path] = None,
) -> Dict:
    """Run Random Forest for a single fold."""
    data, processor = prepare_data_for_fold(fold, config)
    
    X_d_train = data['X_d_train']
    X_s_train = data['X_s_train']
    y_train = data['y_train']
    
    X_d_test = data['X_d_test']
    X_s_test = data['X_s_test']
    y_test = data['y_test']
    
    def create_rf_features(X_d, X_s):
        X_d_last = X_d[:, -1, :]
        X_d_mean = X_d.mean(axis=1)
        X_d_std = X_d.std(axis=1)
        X_d_min = X_d.min(axis=1)
        X_d_max = X_d.max(axis=1)
        X_s_first = X_s[:, 0, :]
        
        X = np.concatenate([
            X_d_last,
            X_d_mean,
            X_d_std,
            X_d_min,
            X_d_max,
            X_s_first
        ], axis=1)
        
        return X
    
    X_train = create_rf_features(X_d_train, X_s_train)
    X_test = create_rf_features(X_d_test, X_s_test)
    
    y_train_flat = y_train[:, -1]
    y_test_flat = y_test[:, -1]
    
    rf = RandomForestRegressor(
        n_estimators=config.rf_n_estimators,
        max_depth=config.rf_max_depth,
        min_samples_split=config.rf_min_samples_split,
        n_jobs=config.rf_n_jobs,
        random_state=seed
    )
    
    rf.fit(X_train, y_train_flat)
    
    y_pred_scaled = rf.predict(X_test)
    
    y_pred = processor.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = processor.scaler_y.inverse_transform(y_test_flat.reshape(-1, 1)).flatten()
    
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    metrics = {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'n_samples': len(y_true)
    }
    
    if timeseries_dir is not None and 'dates_test' in data:
        save_timeseries_plots_by_year(
            y_true, y_pred, data['dates_test'],
            timeseries_dir, 'RandomForest', fold['test'], trial_num,
        )
    
    return {'test_station': fold['test'], 'metrics': metrics}


def run_single_trial(
    model_type: str,
    seed: int,
    trial_num: int,
    config: CVConfig,
    logger: logging.Logger,
    timeseries_dir: Optional[Path] = None,
) -> Dict:
    """Run a single trial for a model."""
    set_all_seeds(seed)
    
    results = []
    for fold in CV_FOLDS:
        if model_type == 'RandomForest':
            result = run_rf_fold(fold, config, seed, trial_num=trial_num, timeseries_dir=timeseries_dir)
        else:
            result = run_lstm_fold(fold, config, model_type, trial_num=trial_num, timeseries_dir=timeseries_dir)
        results.append(result)
    
    r2_scores = [r['metrics']['r2'] for r in results]
    
    summary = {
        'trial': trial_num,
        'seed': seed,
        'model_type': model_type,
        'seq_length': config.seq_length,
        'mean_r2': np.mean(r2_scores),
        'Ne3_r2': results[0]['metrics']['r2'],
        'Ne2_r2': results[1]['metrics']['r2'],
        'Ne1_r2': results[2]['metrics']['r2'],
    }
    
    logger.info(f"    Trial {trial_num} (seed={seed}): Mean R²={summary['mean_r2']:.4f} "
                f"(Ne1={summary['Ne1_r2']:.4f}, Ne2={summary['Ne2_r2']:.4f}, Ne3={summary['Ne3_r2']:.4f})")
    
    return summary


def run_model_trials(
    model_type: str,
    seq_length: int,
    seeds: List[int],
    config: CVConfig,
    logger: logging.Logger,
    timeseries_dir: Optional[Path] = None,
) -> Dict:
    """Run multiple trials for a model at a specific sequence length."""
    
    # Update config with current sequence length
    config.seq_length = seq_length
    
    logger.info(f"\n{'='*70}")
    logger.info(f"MODEL: {model_type} | SEQ_LENGTH: {seq_length}")
    logger.info(f"Running {len(seeds)} trials...")
    logger.info(f"{'='*70}")
    
    trial_results = []
    for i, seed in enumerate(seeds):
        result = run_single_trial(model_type, seed, i + 1, config, logger, timeseries_dir=timeseries_dir)
        trial_results.append(result)
    
    mean_r2_all = [r['mean_r2'] for r in trial_results]
    ne1_r2_all = [r['Ne1_r2'] for r in trial_results]
    ne2_r2_all = [r['Ne2_r2'] for r in trial_results]
    ne3_r2_all = [r['Ne3_r2'] for r in trial_results]
    
    summary = {
        'model_type': model_type,
        'seq_length': seq_length,
        'n_trials': len(seeds),
        'mean_r2_mean': np.mean(mean_r2_all),
        'mean_r2_std': np.std(mean_r2_all),
        'ne1_r2_mean': np.mean(ne1_r2_all),
        'ne1_r2_std': np.std(ne1_r2_all),
        'ne2_r2_mean': np.mean(ne2_r2_all),
        'ne2_r2_std': np.std(ne2_r2_all),
        'ne3_r2_mean': np.mean(ne3_r2_all),
        'ne3_r2_std': np.std(ne3_r2_all),
        'trial_results': trial_results
    }
    
    logger.info(f"\n  Summary: Mean R² = {summary['mean_r2_mean']:.4f} ± {summary['mean_r2_std']:.4f}")
    
    return summary


def create_comparison_plot(all_results: List[Dict], output_dir: Path):
    """Create comparison plots for model and sequence length analysis."""
    
    # Organize results by model and seq_length
    models = MODELS
    seq_lengths = SEQ_LENGTHS
    
    # Create a matrix of results
    results_matrix = {}
    for r in all_results:
        key = (r['model_type'], r['seq_length'])
        results_matrix[key] = r
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = {'EALSTM': '#1f77b4', 'LSTM': '#ff7f0e', 'RandomForest': '#2ca02c'}
    
    # Plot 1: Mean R² vs Sequence Length for each model
    ax1 = axes[0, 0]
    for model in models:
        r2_means = [results_matrix[(model, sl)]['mean_r2_mean'] for sl in seq_lengths]
        r2_stds = [results_matrix[(model, sl)]['mean_r2_std'] for sl in seq_lengths]
        ax1.errorbar(seq_lengths, r2_means, yerr=r2_stds, marker='o', capsize=5, 
                     label=model, color=colors[model], linewidth=2, markersize=8)
    
    ax1.set_xlabel('Sequence Length (days)', fontsize=12)
    ax1.set_ylabel('Mean CV R²', fontsize=12)
    ax1.set_title(f'RZSM {TARGET_DEPTH}: Mean R² vs Sequence Length', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(seq_lengths)
    ax1.set_ylim(0.4, 0.85)
    
    # Plot 2: Best sequence length comparison (bar chart)
    ax2 = axes[0, 1]
    # Find best seq_length for each model
    best_results = []
    for model in models:
        model_results = [results_matrix[(model, sl)] for sl in seq_lengths]
        best = max(model_results, key=lambda x: x['mean_r2_mean'])
        best_results.append(best)
    
    x = np.arange(len(models))
    mean_r2 = [r['mean_r2_mean'] for r in best_results]
    mean_r2_std = [r['mean_r2_std'] for r in best_results]
    best_seq = [r['seq_length'] for r in best_results]
    
    bars = ax2.bar(x, mean_r2, yerr=mean_r2_std, capsize=5, 
                   color=[colors[m] for m in models], alpha=0.8)
    
    ax2.set_ylabel('Best Mean R²', fontsize=12)
    ax2.set_title(f'RZSM {TARGET_DEPTH}: Best Performance per Model', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{m}\n(seq={s})' for m, s in zip(models, best_seq)])
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0.4, 0.85)
    
    for bar, mean, std, seq in zip(bars, mean_r2, mean_r2_std, best_seq):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Highlight best overall
    best_idx = np.argmax(mean_r2)
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(3)
    
    # Plot 3: Per-station R² for best configuration per model
    ax3 = axes[1, 0]
    width = 0.25
    
    ne1_mean = [r['ne1_r2_mean'] for r in best_results]
    ne2_mean = [r['ne2_r2_mean'] for r in best_results]
    ne3_mean = [r['ne3_r2_mean'] for r in best_results]
    ne1_std = [r['ne1_r2_std'] for r in best_results]
    ne2_std = [r['ne2_r2_std'] for r in best_results]
    ne3_std = [r['ne3_r2_std'] for r in best_results]
    
    ax3.bar(x - width, ne1_mean, width, yerr=ne1_std, label='Ne1 (hard)', capsize=3, color='#d62728')
    ax3.bar(x, ne2_mean, width, yerr=ne2_std, label='Ne2', capsize=3, color='#9467bd')
    ax3.bar(x + width, ne3_mean, width, yerr=ne3_std, label='Ne3', capsize=3, color='#8c564b')
    
    ax3.set_ylabel('R²', fontsize=12)
    ax3.set_title(f'RZSM {TARGET_DEPTH}: Per-Station R² (Best Config)', fontsize=14)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{m}\n(seq={s})' for m, s in zip(models, best_seq)])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(0.2, 1.0)
    
    # Plot 4: Heatmap of R² for all model/seq_length combinations
    ax4 = axes[1, 1]
    
    heatmap_data = np.zeros((len(models), len(seq_lengths)))
    for i, model in enumerate(models):
        for j, sl in enumerate(seq_lengths):
            heatmap_data[i, j] = results_matrix[(model, sl)]['mean_r2_mean']
    
    im = ax4.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=0.8)
    
    ax4.set_xticks(np.arange(len(seq_lengths)))
    ax4.set_yticks(np.arange(len(models)))
    ax4.set_xticklabels(seq_lengths)
    ax4.set_yticklabels(models)
    ax4.set_xlabel('Sequence Length (days)', fontsize=12)
    ax4.set_title(f'RZSM {TARGET_DEPTH}: Mean R² Heatmap', fontsize=14)
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(seq_lengths)):
            val = heatmap_data[i, j]
            text_color = 'white' if val < 0.6 or val > 0.75 else 'black'
            ax4.text(j, i, f'{val:.3f}', ha='center', va='center', 
                     color=text_color, fontweight='bold', fontsize=11)
    
    plt.colorbar(im, ax=ax4, label='Mean R²')
    
    plt.suptitle(f'Model Comparison for Root Zone Soil Moisture at {TARGET_DEPTH}\n'
                 f'({N_TRIALS} trials per configuration)', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = output_dir / 'full_comparison_50cm.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlot saved to: {save_path}")
    
    return best_results


def main():
    """Run full model comparison for 50cm depth with multiple sequence lengths."""
    
    output_dir = Path("outputs/full_comparison_50cm")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(str(output_dir), log_level="INFO")
    log = logging.getLogger(__name__)
    
    log.info("="*70)
    log.info(f"FULL MODEL COMPARISON - RZSM {TARGET_DEPTH}")
    log.info("="*70)
    log.info(f"Target: {TARGET_COL}")
    log.info(f"Models: {MODELS}")
    log.info(f"Sequence Lengths: {SEQ_LENGTHS}")
    log.info(f"Trials per configuration: {N_TRIALS}")
    log.info("Data: May–October only (month_range=(5, 10))")
    log.info("")
    log.info("EALSTM: Static features control input gate (entity-aware)")
    log.info("LSTM: Static features concatenated as input")
    log.info("RandomForest: Classical ML with aggregated sequence features")
    log.info("="*70)
    
    config = CVConfig()
    timeseries_dir = output_dir / "timeseries"
    timeseries_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    total_configs = len(MODELS) * len(SEQ_LENGTHS)
    current = 0
    
    for model_type in MODELS:
        for seq_length in SEQ_LENGTHS:
            current += 1
            log.info(f"\n[{current}/{total_configs}] Running {model_type} with seq_length={seq_length}")
            summary = run_model_trials(model_type, seq_length, SEEDS, config, log, timeseries_dir=timeseries_dir)
            all_results.append(summary)
    
    best_results = create_comparison_plot(all_results, output_dir)
    
    # Print full results table
    print("\n" + "="*110)
    print(f"FULL MODEL COMPARISON RESULTS - RZSM {TARGET_DEPTH}")
    print("="*110)
    print(f"{'Model':<15} {'SeqLen':>8} {'Mean R²':>18} {'Ne1 R²':>18} {'Ne2 R²':>18} {'Ne3 R²':>18}")
    print("-"*110)
    
    for r in all_results:
        print(f"{r['model_type']:<15} {r['seq_length']:>8} "
              f"{r['mean_r2_mean']:>7.4f}±{r['mean_r2_std']:<8.4f} "
              f"{r['ne1_r2_mean']:>7.4f}±{r['ne1_r2_std']:<8.4f} "
              f"{r['ne2_r2_mean']:>7.4f}±{r['ne2_r2_std']:<8.4f} "
              f"{r['ne3_r2_mean']:>7.4f}±{r['ne3_r2_std']:<8.4f}")
    
    print("-"*110)
    
    # Find best per model
    print(f"\n📊 BEST CONFIGURATION PER MODEL:")
    for model in MODELS:
        model_results = [r for r in all_results if r['model_type'] == model]
        best = max(model_results, key=lambda x: x['mean_r2_mean'])
        print(f"  {model}: seq_length={best['seq_length']}, "
              f"Mean R² = {best['mean_r2_mean']:.4f} ± {best['mean_r2_std']:.4f}")
    
    # Find overall best
    best_overall = max(all_results, key=lambda x: x['mean_r2_mean'])
    print(f"\n🏆 BEST OVERALL for {TARGET_DEPTH}: {best_overall['model_type']} (seq={best_overall['seq_length']})")
    print(f"   Mean R² = {best_overall['mean_r2_mean']:.4f} ± {best_overall['mean_r2_std']:.4f}")
    
    # Compare EALSTM best vs others' best
    ealstm_results = [r for r in all_results if r['model_type'] == 'EALSTM']
    ealstm_best = max(ealstm_results, key=lambda x: x['mean_r2_mean'])
    
    print(f"\n📊 EALSTM (best) vs Others (best):")
    for model in MODELS[1:]:  # Skip EALSTM
        model_results = [r for r in all_results if r['model_type'] == model]
        model_best = max(model_results, key=lambda x: x['mean_r2_mean'])
        diff = ealstm_best['mean_r2_mean'] - model_best['mean_r2_mean']
        symbol = "✅" if diff > 0 else "❌"
        print(f"   {symbol} vs {model}: {diff:+.4f} ({diff*100:+.1f}%)")
    
    # Save to CSV
    results_df = pd.DataFrame([
        {
            'target': TARGET_COL,
            'depth': TARGET_DEPTH,
            'model_type': r['model_type'],
            'seq_length': r['seq_length'],
            'n_trials': r['n_trials'],
            'mean_r2_mean': r['mean_r2_mean'],
            'mean_r2_std': r['mean_r2_std'],
            'ne1_r2_mean': r['ne1_r2_mean'],
            'ne1_r2_std': r['ne1_r2_std'],
            'ne2_r2_mean': r['ne2_r2_mean'],
            'ne2_r2_std': r['ne2_r2_std'],
            'ne3_r2_mean': r['ne3_r2_mean'],
            'ne3_r2_std': r['ne3_r2_std'],
        }
        for r in all_results
    ])
    
    results_df.to_csv(output_dir / 'full_comparison_50cm_results.csv', index=False)
    log.info(f"\nResults saved to: {output_dir / 'full_comparison_50cm_results.csv'}")
    
    return all_results


if __name__ == "__main__":
    results = main()
