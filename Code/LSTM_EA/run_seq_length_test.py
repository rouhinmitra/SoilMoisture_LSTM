#!/usr/bin/env python
"""
Test different sequence lengths for EA-LSTM.

Tests seq_length = [7, 14, 21, 30] to find optimal window size.
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from typing import List, Dict
import logging

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

# Sequence lengths to test
SEQ_LENGTHS = [45, 60, 90]  # Testing longer sequences


@dataclass
class CVConfig:
    """Cross-validation configuration"""
    data_dir: str = "/Users/rouhinmitra/SM_work/Data/Base/"
    output_dir: str = "outputs/seq_length_test_long"
    seq_length: int = 30  # Will be overridden
    nan_threshold: float = 0.1
    # Tuned hyperparameters
    hidden_dim: int = 128
    dropout: float = 0.5
    batch_size: int = 32
    epochs: int = 150
    learning_rate: float = 0.0005
    early_stopping_patience: int = 15
    weight_decay: float = 0.001
    add_temporal: bool = True
    seed: int = 42
    
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


def run_single_fold(fold: Dict, config: CVConfig, logger: logging.Logger) -> Dict:
    """Run a single CV fold."""
    fold_name = fold['name']
    train_stations = fold['train']
    test_station = fold['test']
    
    logger.info(f"  Fold: {fold_name} (Train: {train_stations}, Test: {test_station})")
    
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
        model_type="EALSTM",
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
    fold_output_dir = Path(config.output_dir) / f"seq_{config.seq_length}" / fold_name
    fold_output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = Trainer(model, training_config, str(fold_output_dir), device=str(device))
    
    history = trainer.train(train_loader, test_loader)
    
    # Evaluate
    evaluator = Evaluator(trainer.model, processor.scaler_y, str(fold_output_dir), device=str(device))
    y_pred, y_true = evaluator.predict(test_loader)
    metrics = evaluator.compute_metrics(y_pred, y_true)
    
    logger.info(f"    R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}")
    
    return {
        'fold_name': fold_name,
        'test_station': test_station,
        'train_stations': train_stations,
        'metrics': metrics,
        'n_train': len(train_dataset),
        'n_test': len(test_dataset)
    }


def run_cv_for_seq_length(seq_length: int, base_config: CVConfig, logger: logging.Logger) -> Dict:
    """Run full CV for a specific sequence length."""
    
    logger.info(f"\n{'='*70}")
    logger.info(f"TESTING SEQUENCE LENGTH: {seq_length} days")
    logger.info(f"{'='*70}")
    
    # Update config
    config = CVConfig()
    config.seq_length = seq_length
    config.output_dir = f"outputs/seq_length_test_long/seq_{seq_length}"
    
    # Run all folds
    results = []
    for fold in CV_FOLDS:
        result = run_single_fold(fold, config, logger)
        results.append(result)
    
    # Calculate summary
    r2_scores = [r['metrics']['r2'] for r in results]
    rmse_scores = [r['metrics']['rmse'] for r in results]
    
    summary = {
        'seq_length': seq_length,
        'mean_r2': np.mean(r2_scores),
        'std_r2': np.std(r2_scores),
        'mean_rmse': np.mean(rmse_scores),
        'std_rmse': np.std(rmse_scores),
        'Ne3_r2': results[0]['metrics']['r2'],
        'Ne2_r2': results[1]['metrics']['r2'],
        'Ne1_r2': results[2]['metrics']['r2'],
        'n_train_samples': results[0]['n_train'],
        'n_test_samples': results[0]['n_test'],
        'results': results
    }
    
    logger.info(f"\nSeq Length {seq_length} Summary:")
    logger.info(f"  Mean R²: {summary['mean_r2']:.4f} ± {summary['std_r2']:.4f}")
    logger.info(f"  Mean RMSE: {summary['mean_rmse']:.4f}")
    logger.info(f"  Train samples: {summary['n_train_samples']}")
    
    return summary


def create_comparison_plot(all_results: List[Dict], output_dir: Path):
    """Create comparison plot across sequence lengths."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    seq_lengths = [r['seq_length'] for r in all_results]
    mean_r2 = [r['mean_r2'] for r in all_results]
    std_r2 = [r['std_r2'] for r in all_results]
    
    ne1_r2 = [r['Ne1_r2'] for r in all_results]
    ne2_r2 = [r['Ne2_r2'] for r in all_results]
    ne3_r2 = [r['Ne3_r2'] for r in all_results]
    
    # Plot 1: Mean R² with error bars
    ax1 = axes[0]
    ax1.errorbar(seq_lengths, mean_r2, yerr=std_r2, marker='o', markersize=10, 
                 capsize=5, linewidth=2, label='Mean R²')
    ax1.set_xlabel('Sequence Length (days)', fontsize=12)
    ax1.set_ylabel('R² Score', fontsize=12)
    ax1.set_title('Mean CV R² vs Sequence Length', fontsize=14)
    ax1.set_xticks(seq_lengths)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.4, 0.9)
    
    # Highlight best
    best_idx = np.argmax(mean_r2)
    ax1.scatter([seq_lengths[best_idx]], [mean_r2[best_idx]], 
                color='red', s=200, zorder=5, marker='*', label=f'Best: {seq_lengths[best_idx]} days')
    ax1.legend()
    
    # Plot 2: Per-station R²
    ax2 = axes[1]
    x = np.arange(len(seq_lengths))
    width = 0.25
    
    ax2.bar(x - width, ne1_r2, width, label='Ne1 (hardest)', color='#ff7f0e')
    ax2.bar(x, ne2_r2, width, label='Ne2', color='#2ca02c')
    ax2.bar(x + width, ne3_r2, width, label='Ne3', color='#1f77b4')
    
    ax2.set_xlabel('Sequence Length (days)', fontsize=12)
    ax2.set_ylabel('R² Score', fontsize=12)
    ax2.set_title('Per-Station R² vs Sequence Length', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(seq_lengths)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 1.0)
    
    plt.tight_layout()
    
    save_path = output_dir / 'seq_length_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nComparison plot saved to: {save_path}")


def main():
    """Test multiple sequence lengths."""
    
    # Setup
    output_dir = Path("outputs/seq_length_test_long")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(str(output_dir), log_level="INFO")
    log = logging.getLogger(__name__)
    
    log.info("="*70)
    log.info("SEQUENCE LENGTH EXPERIMENT")
    log.info("="*70)
    log.info(f"Testing sequence lengths: {SEQ_LENGTHS}")
    
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run experiments
    all_results = []
    base_config = CVConfig()
    
    for seq_len in SEQ_LENGTHS:
        summary = run_cv_for_seq_length(seq_len, base_config, log)
        all_results.append(summary)
    
    # Create comparison plot
    create_comparison_plot(all_results, output_dir)
    
    # Print final summary
    print("\n" + "="*70)
    print("SEQUENCE LENGTH COMPARISON")
    print("="*70)
    print(f"{'Seq Len':<10} {'Mean R²':>10} {'Std R²':>10} {'Ne1 R²':>10} {'Ne2 R²':>10} {'Ne3 R²':>10} {'Samples':>10}")
    print("-"*70)
    
    for r in all_results:
        print(f"{r['seq_length']:<10} {r['mean_r2']:>10.4f} {r['std_r2']:>10.4f} "
              f"{r['Ne1_r2']:>10.4f} {r['Ne2_r2']:>10.4f} {r['Ne3_r2']:>10.4f} "
              f"{r['n_train_samples']:>10}")
    
    print("-"*70)
    
    # Find best
    best = max(all_results, key=lambda x: x['mean_r2'])
    print(f"\n🏆 BEST SEQUENCE LENGTH: {best['seq_length']} days (Mean R² = {best['mean_r2']:.4f})")
    
    # Save results to CSV
    results_df = pd.DataFrame([
        {
            'seq_length': r['seq_length'],
            'mean_r2': r['mean_r2'],
            'std_r2': r['std_r2'],
            'mean_rmse': r['mean_rmse'],
            'Ne1_r2': r['Ne1_r2'],
            'Ne2_r2': r['Ne2_r2'],
            'Ne3_r2': r['Ne3_r2'],
            'n_train_samples': r['n_train_samples']
        }
        for r in all_results
    ])
    
    results_df.to_csv(output_dir / 'seq_length_results.csv', index=False)
    log.info(f"\nResults saved to: {output_dir / 'seq_length_results.csv'}")
    
    return all_results


if __name__ == "__main__":
    results = main()

