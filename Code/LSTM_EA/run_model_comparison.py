#!/usr/bin/env python
"""
Compare EA-LSTM vs Regular LSTM across different sequence lengths.

EA-LSTM: Static features (AE embeddings) control the input gate
Regular LSTM: Static features concatenated as regular input features
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
SEQ_LENGTHS = [7, 14, 21, 30]

# Model types to compare
MODEL_TYPES = ['EALSTM', 'LSTM']


@dataclass
class CVConfig:
    """Cross-validation configuration"""
    data_dir: str = "/Users/rouhinmitra/SM_work/Data/Base/"
    output_dir: str = "outputs/model_comparison"
    seq_length: int = 30
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
    fold_output_dir = Path(config.output_dir) / f"{config.model_type}_seq_{config.seq_length}" / fold_name
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
        'train_stations': train_stations,
        'metrics': metrics,
        'n_train': len(train_dataset),
        'n_test': len(test_dataset)
    }


def run_experiment(model_type: str, seq_length: int, logger: logging.Logger) -> Dict:
    """Run full CV for a specific model type and sequence length."""
    
    logger.info(f"  {model_type} with seq_length={seq_length}...")
    
    # Create config
    config = CVConfig()
    config.model_type = model_type
    config.seq_length = seq_length
    config.output_dir = "outputs/model_comparison"
    
    # Run all folds
    results = []
    for fold in CV_FOLDS:
        result = run_single_fold(fold, config, logger)
        results.append(result)
    
    # Calculate summary
    r2_scores = [r['metrics']['r2'] for r in results]
    rmse_scores = [r['metrics']['rmse'] for r in results]
    
    summary = {
        'model_type': model_type,
        'seq_length': seq_length,
        'mean_r2': np.mean(r2_scores),
        'std_r2': np.std(r2_scores),
        'mean_rmse': np.mean(rmse_scores),
        'Ne3_r2': results[0]['metrics']['r2'],
        'Ne2_r2': results[1]['metrics']['r2'],
        'Ne1_r2': results[2]['metrics']['r2'],
        'n_train_samples': results[0]['n_train'],
    }
    
    logger.info(f"    Mean R²: {summary['mean_r2']:.4f} (Ne1: {summary['Ne1_r2']:.4f}, Ne2: {summary['Ne2_r2']:.4f}, Ne3: {summary['Ne3_r2']:.4f})")
    
    return summary


def create_comparison_plots(all_results: List[Dict], output_dir: Path):
    """Create comparison plots."""
    
    # Separate results by model type
    ealstm_results = [r for r in all_results if r['model_type'] == 'EALSTM']
    lstm_results = [r for r in all_results if r['model_type'] == 'LSTM']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    seq_lengths = [r['seq_length'] for r in ealstm_results]
    
    # Plot 1: Mean R² comparison
    ax1 = axes[0, 0]
    ealstm_r2 = [r['mean_r2'] for r in ealstm_results]
    lstm_r2 = [r['mean_r2'] for r in lstm_results]
    
    x = np.arange(len(seq_lengths))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, ealstm_r2, width, label='EA-LSTM', color='#1f77b4')
    bars2 = ax1.bar(x + width/2, lstm_r2, width, label='Standard LSTM', color='#ff7f0e')
    
    ax1.set_xlabel('Sequence Length (days)', fontsize=12)
    ax1.set_ylabel('Mean R²', fontsize=12)
    ax1.set_title('Mean CV R²: EA-LSTM vs Standard LSTM', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(seq_lengths)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0.5, 0.8)
    
    # Add value labels
    for bar, val in zip(bars1, ealstm_r2):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}', 
                ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, lstm_r2):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}', 
                ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Difference (EA-LSTM - LSTM)
    ax2 = axes[0, 1]
    diff = [e - l for e, l in zip(ealstm_r2, lstm_r2)]
    colors = ['green' if d > 0 else 'red' for d in diff]
    bars = ax2.bar(seq_lengths, diff, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Sequence Length (days)', fontsize=12)
    ax2.set_ylabel('R² Difference (EA-LSTM - LSTM)', fontsize=12)
    ax2.set_title('EA-LSTM Advantage by Sequence Length', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, diff):
        ypos = bar.get_height() + 0.005 if val > 0 else bar.get_height() - 0.015
        ax2.text(bar.get_x() + bar.get_width()/2, ypos, f'{val:+.3f}', 
                ha='center', va='bottom' if val > 0 else 'top', fontsize=10, fontweight='bold')
    
    # Plot 3: Ne1 (hardest station) comparison
    ax3 = axes[1, 0]
    ealstm_ne1 = [r['Ne1_r2'] for r in ealstm_results]
    lstm_ne1 = [r['Ne1_r2'] for r in lstm_results]
    
    ax3.plot(seq_lengths, ealstm_ne1, 'o-', linewidth=2, markersize=8, label='EA-LSTM', color='#1f77b4')
    ax3.plot(seq_lengths, lstm_ne1, 's--', linewidth=2, markersize=8, label='Standard LSTM', color='#ff7f0e')
    
    ax3.set_xlabel('Sequence Length (days)', fontsize=12)
    ax3.set_ylabel('R²', fontsize=12)
    ax3.set_title('Ne1 (Hardest Station) Performance', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(seq_lengths)
    
    # Plot 4: All stations for best seq_length
    ax4 = axes[1, 1]
    best_seq = seq_lengths[np.argmax(ealstm_r2)]
    best_ealstm = [r for r in ealstm_results if r['seq_length'] == best_seq][0]
    best_lstm = [r for r in lstm_results if r['seq_length'] == best_seq][0]
    
    stations = ['Ne1', 'Ne2', 'Ne3']
    ealstm_vals = [best_ealstm['Ne1_r2'], best_ealstm['Ne2_r2'], best_ealstm['Ne3_r2']]
    lstm_vals = [best_lstm['Ne1_r2'], best_lstm['Ne2_r2'], best_lstm['Ne3_r2']]
    
    x = np.arange(len(stations))
    bars1 = ax4.bar(x - width/2, ealstm_vals, width, label='EA-LSTM', color='#1f77b4')
    bars2 = ax4.bar(x + width/2, lstm_vals, width, label='Standard LSTM', color='#ff7f0e')
    
    ax4.set_xlabel('Test Station', fontsize=12)
    ax4.set_ylabel('R²', fontsize=12)
    ax4.set_title(f'Per-Station R² (Best Seq Length: {best_seq} days)', fontsize=14)
    ax4.set_xticks(x)
    ax4.set_xticklabels(stations)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim(0, 1.0)
    
    plt.tight_layout()
    
    save_path = output_dir / 'model_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nComparison plot saved to: {save_path}")


def main():
    """Compare EA-LSTM vs Standard LSTM across sequence lengths."""
    
    # Setup
    output_dir = Path("outputs/model_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(str(output_dir), log_level="INFO")
    log = logging.getLogger(__name__)
    
    log.info("="*70)
    log.info("EA-LSTM vs STANDARD LSTM COMPARISON")
    log.info("="*70)
    log.info(f"Sequence lengths: {SEQ_LENGTHS}")
    log.info(f"Model types: {MODEL_TYPES}")
    log.info("")
    log.info("EA-LSTM: Static features control input gate (entity-aware)")
    log.info("Standard LSTM: Static features concatenated as regular input")
    log.info("="*70)
    
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run all experiments
    all_results = []
    
    for seq_len in SEQ_LENGTHS:
        log.info(f"\nSequence Length: {seq_len} days")
        log.info("-" * 40)
        
        for model_type in MODEL_TYPES:
            # Reset seed for fair comparison
            torch.manual_seed(42)
            np.random.seed(42)
            
            summary = run_experiment(model_type, seq_len, log)
            all_results.append(summary)
    
    # Create comparison plots
    create_comparison_plots(all_results, output_dir)
    
    # Print final summary
    print("\n" + "="*90)
    print("MODEL COMPARISON SUMMARY")
    print("="*90)
    print(f"{'Model':<15} {'Seq Len':<10} {'Mean R²':>10} {'Std R²':>10} {'Ne1 R²':>10} {'Ne2 R²':>10} {'Ne3 R²':>10}")
    print("-"*90)
    
    for r in all_results:
        print(f"{r['model_type']:<15} {r['seq_length']:<10} {r['mean_r2']:>10.4f} {r['std_r2']:>10.4f} "
              f"{r['Ne1_r2']:>10.4f} {r['Ne2_r2']:>10.4f} {r['Ne3_r2']:>10.4f}")
    
    print("-"*90)
    
    # Find best for each model
    ealstm_results = [r for r in all_results if r['model_type'] == 'EALSTM']
    lstm_results = [r for r in all_results if r['model_type'] == 'LSTM']
    
    best_ealstm = max(ealstm_results, key=lambda x: x['mean_r2'])
    best_lstm = max(lstm_results, key=lambda x: x['mean_r2'])
    
    print(f"\n🏆 Best EA-LSTM: seq_length={best_ealstm['seq_length']}, Mean R²={best_ealstm['mean_r2']:.4f}")
    print(f"🏆 Best Standard LSTM: seq_length={best_lstm['seq_length']}, Mean R²={best_lstm['mean_r2']:.4f}")
    
    diff = best_ealstm['mean_r2'] - best_lstm['mean_r2']
    if diff > 0:
        print(f"\n✅ EA-LSTM is better by {diff:.4f} R²")
    else:
        print(f"\n❌ Standard LSTM is better by {-diff:.4f} R²")
    
    # Save results to CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / 'model_comparison_results.csv', index=False)
    log.info(f"\nResults saved to: {output_dir / 'model_comparison_results.csv'}")
    
    return all_results


if __name__ == "__main__":
    results = main()

