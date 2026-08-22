#!/usr/bin/env python
"""
Multi-Trial Evaluation for EA-LSTM.

Runs the same experiment multiple times with different seeds
to get robust mean ± std estimates.
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import List, Dict
import logging
import random

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
N_TRIALS = 5  # Number of trials per configuration
SEEDS = [42, 123, 456, 789, 1024]  # Different seeds for each trial

# Configurations to test (best from previous experiments)
CONFIGS = [
    {'model_type': 'EALSTM', 'seq_length': 14},  # Best EALSTM
    {'model_type': 'EALSTM', 'seq_length': 21},  # Second best
    {'model_type': 'EALSTM', 'seq_length': 30},  # Original
]


def set_all_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Make PyTorch deterministic (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class CVConfig:
    """Cross-validation configuration"""
    data_dir: str = "/Users/rouhinmitra/SM_work/Data/Base/"
    output_dir: str = "outputs/multi_trial"
    seq_length: int = 14
    nan_threshold: float = 0.1
    model_type: str = "EALSTM"
    # Tuned hyperparameters
    hidden_dim: int = 128
    dropout: float = 0.5
    batch_size: int = 32
    epochs: int = 150
    learning_rate: float = 0.0005
    early_stopping_patience: int = 15
    weight_decay: float = 0.001
    add_temporal: bool = True
    
    # Features (with Sentinel-2)
    dynamic_cols: List[str] = None
    static_cols: List[str] = None
    target_col: str = "RZSM_25_avg"
    
    def __post_init__(self):
        if self.dynamic_cols is None:
            self.dynamic_cols = [
                # Original features
                'SSM', 'SSM_avg', 'SWC_PI_F_2_1_1', 'SWC_PI_F_3_1_1',
                'P_PI_F_1_1_1', 'P_PI_F_2_2_1', 'I', 'TA_1_1_1', 'RH_1_1_1',
                # Sentinel-2 features
                'ndvi', 'b11', 'b12', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b8a'
            ]
        if self.static_cols is None:
            self.static_cols = [f'A{i:02d}' for i in range(64)] + [
                'precip_jan_apr',   # accumulated precip Jan 1–Apr 30 (static / input-gate)
                'precip_may_oct',   # accumulated precip May–Oct summer (static / input-gate)
            ]


def run_single_fold(fold: Dict, config: CVConfig) -> Dict:
    """Run a single CV fold."""
    fold_name = fold['name']
    train_stations = fold['train']
    test_station = fold['test']
    
    # Create configs
    data_config = DataConfig(
        train_files=[STATIONS[s] for s in train_stations],
        test_files=[STATIONS[test_station]],
        data_dir=config.data_dir,
        seq_length=config.seq_length,
        nan_threshold=config.nan_threshold,
        same_year_constraint=True
    )
    
    feature_config = FeatureConfig(
        dynamic_cols=config.dynamic_cols,
        static_cols=config.static_cols,
        target_col=config.target_col,
        add_temporal=config.add_temporal,
        temporal_features=['doy_sin', 'doy_cos']
    )
    
    model_config = ModelConfig(
        model_type=config.model_type,
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
    
    # Prepare data
    processor = DataProcessor(data_config, feature_config)
    data = processor.prepare_data(data_config.train_files, data_config.test_files)
    
    # Create datasets
    train_dataset = RZSMDataset(data['X_d_train'], data['X_s_train'], data['y_train'])
    test_dataset = RZSMDataset(data['X_d_test'], data['X_s_test'], data['y_test'])
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Get dimensions
    dyn_dim = data['X_d_train'].shape[2]
    stat_dim = data['X_s_train'].shape[2]
    
    # Create model
    model = get_model(model_config, dyn_dim, stat_dim)
    
    # Train
    fold_output_dir = Path(config.output_dir) / "temp"
    fold_output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = Trainer(model, training_config, str(fold_output_dir), device=str(device))
    
    history = trainer.train(train_loader, test_loader)
    
    # Evaluate
    evaluator = Evaluator(trainer.model, processor.scaler_y, str(fold_output_dir), device=str(device))
    y_pred, y_true = evaluator.predict(test_loader)
    metrics = evaluator.compute_metrics(y_pred, y_true)
    
    return {
        'fold_name': fold_name,
        'test_station': test_station,
        'metrics': metrics,
    }


def run_single_trial(config_dict: Dict, seed: int, trial_num: int, logger: logging.Logger) -> Dict:
    """Run a single trial (full CV) with a specific seed."""
    
    # Set seed
    set_all_seeds(seed)
    
    # Create config
    config = CVConfig()
    config.model_type = config_dict['model_type']
    config.seq_length = config_dict['seq_length']
    
    # Run all folds
    results = []
    for fold in CV_FOLDS:
        result = run_single_fold(fold, config)
        results.append(result)
    
    # Calculate summary
    r2_scores = [r['metrics']['r2'] for r in results]
    
    summary = {
        'trial': trial_num,
        'seed': seed,
        'model_type': config_dict['model_type'],
        'seq_length': config_dict['seq_length'],
        'mean_r2': np.mean(r2_scores),
        'Ne3_r2': results[0]['metrics']['r2'],
        'Ne2_r2': results[1]['metrics']['r2'],
        'Ne1_r2': results[2]['metrics']['r2'],
    }
    
    logger.info(f"    Trial {trial_num} (seed={seed}): Mean R²={summary['mean_r2']:.4f} "
                f"(Ne1={summary['Ne1_r2']:.4f}, Ne2={summary['Ne2_r2']:.4f}, Ne3={summary['Ne3_r2']:.4f})")
    
    return summary


def run_multi_trial_experiment(config_dict: Dict, seeds: List[int], logger: logging.Logger) -> Dict:
    """Run multiple trials for a configuration."""
    
    logger.info(f"\n{'='*70}")
    logger.info(f"CONFIG: {config_dict['model_type']} with seq_length={config_dict['seq_length']}")
    logger.info(f"Running {len(seeds)} trials...")
    logger.info(f"{'='*70}")
    
    trial_results = []
    for i, seed in enumerate(seeds):
        result = run_single_trial(config_dict, seed, i+1, logger)
        trial_results.append(result)
    
    # Aggregate results
    mean_r2_all = [r['mean_r2'] for r in trial_results]
    ne1_r2_all = [r['Ne1_r2'] for r in trial_results]
    ne2_r2_all = [r['Ne2_r2'] for r in trial_results]
    ne3_r2_all = [r['Ne3_r2'] for r in trial_results]
    
    summary = {
        'model_type': config_dict['model_type'],
        'seq_length': config_dict['seq_length'],
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
    logger.info(f"           Ne1 R² = {summary['ne1_r2_mean']:.4f} ± {summary['ne1_r2_std']:.4f}")
    logger.info(f"           Ne2 R² = {summary['ne2_r2_mean']:.4f} ± {summary['ne2_r2_std']:.4f}")
    logger.info(f"           Ne3 R² = {summary['ne3_r2_mean']:.4f} ± {summary['ne3_r2_std']:.4f}")
    
    return summary


def create_summary_plot(all_results: List[Dict], output_dir: Path):
    """Create summary plot with error bars."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Extract data
    labels = [f"{r['model_type']}\nseq={r['seq_length']}" for r in all_results]
    mean_r2 = [r['mean_r2_mean'] for r in all_results]
    mean_r2_std = [r['mean_r2_std'] for r in all_results]
    
    # Plot 1: Mean R² with error bars
    ax1 = axes[0]
    x = np.arange(len(labels))
    bars = ax1.bar(x, mean_r2, yerr=mean_r2_std, capsize=5, color='steelblue', alpha=0.8)
    
    ax1.set_ylabel('Mean R²', fontsize=12)
    ax1.set_title(f'Mean CV R² ({all_results[0]["n_trials"]} trials per config)', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0.5, 0.8)
    
    # Add value labels
    for bar, mean, std in zip(bars, mean_r2, mean_r2_std):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Highlight best
    best_idx = np.argmax(mean_r2)
    bars[best_idx].set_color('green')
    
    # Plot 2: Per-station breakdown
    ax2 = axes[1]
    
    width = 0.25
    ne1_mean = [r['ne1_r2_mean'] for r in all_results]
    ne2_mean = [r['ne2_r2_mean'] for r in all_results]
    ne3_mean = [r['ne3_r2_mean'] for r in all_results]
    ne1_std = [r['ne1_r2_std'] for r in all_results]
    ne2_std = [r['ne2_r2_std'] for r in all_results]
    ne3_std = [r['ne3_r2_std'] for r in all_results]
    
    ax2.bar(x - width, ne1_mean, width, yerr=ne1_std, label='Ne1', capsize=3, color='#ff7f0e')
    ax2.bar(x, ne2_mean, width, yerr=ne2_std, label='Ne2', capsize=3, color='#2ca02c')
    ax2.bar(x + width, ne3_mean, width, yerr=ne3_std, label='Ne3', capsize=3, color='#1f77b4')
    
    ax2.set_ylabel('R²', fontsize=12)
    ax2.set_title('Per-Station R² with Error Bars', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0.4, 0.9)
    
    plt.tight_layout()
    
    save_path = output_dir / 'multi_trial_results.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlot saved to: {save_path}")


def main():
    """Run multi-trial evaluation."""
    
    # Setup
    output_dir = Path("outputs/multi_trial")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(str(output_dir), log_level="INFO")
    log = logging.getLogger(__name__)
    
    log.info("="*70)
    log.info("MULTI-TRIAL EVALUATION")
    log.info("="*70)
    log.info(f"Number of trials per config: {N_TRIALS}")
    log.info(f"Seeds: {SEEDS}")
    log.info(f"Configurations to test: {len(CONFIGS)}")
    log.info("="*70)
    
    # Run all experiments
    all_results = []
    
    for config_dict in CONFIGS:
        summary = run_multi_trial_experiment(config_dict, SEEDS, log)
        all_results.append(summary)
    
    # Create summary plot
    create_summary_plot(all_results, output_dir)
    
    # Print final summary
    print("\n" + "="*90)
    print("MULTI-TRIAL RESULTS SUMMARY")
    print("="*90)
    print(f"{'Config':<20} {'Mean R²':>15} {'Ne1 R²':>15} {'Ne2 R²':>15} {'Ne3 R²':>15}")
    print("-"*90)
    
    for r in all_results:
        config_name = f"{r['model_type']}-{r['seq_length']}"
        print(f"{config_name:<20} "
              f"{r['mean_r2_mean']:>6.4f}±{r['mean_r2_std']:<6.4f} "
              f"{r['ne1_r2_mean']:>6.4f}±{r['ne1_r2_std']:<6.4f} "
              f"{r['ne2_r2_mean']:>6.4f}±{r['ne2_r2_std']:<6.4f} "
              f"{r['ne3_r2_mean']:>6.4f}±{r['ne3_r2_std']:<6.4f}")
    
    print("-"*90)
    
    # Find best
    best = max(all_results, key=lambda x: x['mean_r2_mean'])
    print(f"\n🏆 BEST CONFIG: {best['model_type']} with seq_length={best['seq_length']}")
    print(f"   Mean R² = {best['mean_r2_mean']:.4f} ± {best['mean_r2_std']:.4f}")
    
    # Save to CSV
    results_df = pd.DataFrame([
        {
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
    
    results_df.to_csv(output_dir / 'multi_trial_results.csv', index=False)
    log.info(f"\nResults saved to: {output_dir / 'multi_trial_results.csv'}")
    
    return all_results


if __name__ == "__main__":
    results = main()

