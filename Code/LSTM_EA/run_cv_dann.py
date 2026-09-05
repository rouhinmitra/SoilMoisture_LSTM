#!/usr/bin/env python
"""
Leave-One-Station-Out Cross-Validation with Domain Adaptation (DANN).

Uses Domain Adversarial Neural Network to learn station-invariant features.
For each fold:
- Source domain: 2 training stations (labeled)
- Target domain: 1 test station (unlabeled during training)

The DANN model learns features that:
1. Are predictive of RZSM (task loss on source)
2. Cannot distinguish between stations (domain loss on both)
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

from src.config import DataConfig, FeatureConfig, TrainingConfig
from src.data_loader import DataProcessor, RZSMDataset
from src.models_da import get_dann_model, EALSTM_DANN
from src.trainer_da import DANNTrainer
from src.evaluator import Evaluator
from utils.logging_utils import setup_logging


class DANNModelWrapper:
    """
    Wrapper to make DANN model compatible with Evaluator.
    The Evaluator calls model(x_d, x_s), so we wrap to call predict().
    """
    def __init__(self, dann_model: EALSTM_DANN):
        self.dann_model = dann_model
    
    def __call__(self, x_d, x_s):
        return self.dann_model.predict(x_d, x_s)
    
    def eval(self):
        self.dann_model.eval()
        return self
    
    def to(self, device):
        self.dann_model.to(device)
        return self

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


@dataclass
class DANNConfig:
    """DANN Cross-validation configuration"""
    data_dir: str = "/Users/rouhinmitra/SM_work/Data/Base/"
    output_dir: str = "outputs/cv_results_dann"
    seq_length: int = 30
    nan_threshold: float = 0.1
    
    # Model hyperparameters
    hidden_dim: int = 128
    dropout: float = 0.5
    
    # Domain adaptation hyperparameters
    domain_hidden_dim: int = 64
    domain_dropout: float = 0.3
    lambda_max: float = 1.0
    lambda_schedule: str = 'gradual'
    domain_loss_weight: float = 0.5
    
    # Training hyperparameters
    batch_size: int = 32
    epochs: int = 150
    learning_rate: float = 0.0005
    early_stopping_patience: int = 20  # Longer for DA training
    weight_decay: float = 0.001
    
    add_temporal: bool = True
    seed: int = 42
    
    # Features
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


def run_dann_fold(
    fold: Dict,
    config: DANNConfig,
    logger: logging.Logger
) -> Dict:
    """
    Run a single DANN CV fold.
    
    For DANN, we need:
    - Source data: training stations (labeled) for task + domain loss
    - Target data: test station (unlabeled) for domain loss only
    """
    fold_name = fold['name']
    train_stations = fold['train']
    test_station = fold['test']
    
    logger.info(f"\n{'='*70}")
    logger.info(f"DANN FOLD: {fold_name}")
    logger.info(f"Source Domain (labeled): {train_stations}")
    logger.info(f"Target Domain (unlabeled): {test_station}")
    logger.info(f"{'='*70}")
    
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
    
    training_config = TrainingConfig(
        batch_size=config.batch_size,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        early_stopping_patience=config.early_stopping_patience,
        weight_decay=config.weight_decay
    )
    
    # Prepare data
    logger.info("Preparing data...")
    processor = DataProcessor(data_config, feature_config)
    data = processor.prepare_data(data_config.train_files, data_config.test_files)
    
    # Create datasets
    # Source: training data (labeled)
    source_dataset = RZSMDataset(data['X_d_train'], data['X_s_train'], data['y_train'])
    
    # Target: test data (unlabeled - labels ignored during training)
    target_dataset = RZSMDataset(data['X_d_test'], data['X_s_test'], data['y_test'])
    
    # Create data loaders
    source_loader = DataLoader(source_dataset, batch_size=config.batch_size, shuffle=True)
    target_loader = DataLoader(target_dataset, batch_size=config.batch_size, shuffle=True)
    
    # Validation loader (from source domain for early stopping)
    val_loader = DataLoader(source_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Test loader (target domain for evaluation)
    test_loader = DataLoader(target_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Get dimensions
    dyn_dim = data['X_d_train'].shape[2]
    stat_dim = data['X_s_train'].shape[2]
    
    # Create DANN model
    logger.info("Creating DANN model...")
    model = get_dann_model(
        dyn_dim=dyn_dim,
        stat_dim=stat_dim,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        domain_hidden_dim=config.domain_hidden_dim,
        domain_dropout=config.domain_dropout
    )
    
    # Setup output directory
    fold_output_dir = Path(config.output_dir) / fold_name
    fold_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = DANNTrainer(
        model=model,
        config=training_config,
        output_dir=str(fold_output_dir),
        device=str(device),
        lambda_max=config.lambda_max,
        lambda_schedule=config.lambda_schedule,
        domain_loss_weight=config.domain_loss_weight
    )
    
    # Train with domain adaptation
    logger.info("Training with domain adaptation...")
    history = trainer.train(source_loader, target_loader, val_loader)
    
    # Evaluate on target domain
    logger.info("Evaluating on target domain...")
    # Wrap the DANN model to be compatible with Evaluator
    model_wrapper = DANNModelWrapper(trainer.model)
    evaluator = Evaluator(model_wrapper, processor.scaler_y, str(fold_output_dir), device=str(device))
    
    # Use the predict method that returns task predictions only
    y_pred, y_true = evaluator.predict(test_loader)
    metrics = evaluator.compute_metrics(y_pred, y_true)
    
    logger.info(f"\nFold {fold_name} Results:")
    logger.info(f"  R²:   {metrics['r2']:.4f}")
    logger.info(f"  RMSE: {metrics['rmse']:.4f}")
    logger.info(f"  MAE:  {metrics['mae']:.4f}")
    
    return {
        'fold_name': fold_name,
        'test_station': test_station,
        'train_stations': train_stations,
        'y_pred': y_pred,
        'y_true': y_true,
        'metrics': metrics,
        'history': history
    }


def create_dann_scatter_plot(results: List[Dict], output_dir: Path):
    """Create scatter plots comparing DANN results across folds."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, (result, ax, color) in enumerate(zip(results, axes, colors)):
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
    
    plt.suptitle('DANN Leave-One-Station-Out Cross-Validation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = output_dir / 'dann_cv_scatter_plots.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"DANN scatter plot saved to: {save_path}")


def create_dann_summary_plot(results: List[Dict], output_dir: Path):
    """Create summary bar plot of R² scores."""
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    stations = [r['test_station'] for r in results]
    r2_scores = [r['metrics']['r2'] for r in results]
    
    x = np.arange(len(stations))
    bars = ax.bar(x, r2_scores, color='steelblue', edgecolor='navy')
    
    for bar, val in zip(bars, r2_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    mean_r2 = np.mean(r2_scores)
    ax.axhline(y=mean_r2, color='red', linestyle='--', linewidth=2, 
               label=f'Mean R² = {mean_r2:.3f}')
    
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_xlabel('Test Station', fontsize=12)
    ax.set_title('DANN Leave-One-Station-Out CV: R² by Test Station', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Test: {s}' for s in stations])
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    save_path = output_dir / 'dann_cv_r2_summary.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"DANN R² summary saved to: {save_path}")


def create_training_history_plot(results: List[Dict], output_dir: Path):
    """Create training history plots showing domain adaptation progress."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for i, result in enumerate(results):
        history = result['history']
        test_station = result['test_station']
        
        # Task and domain loss
        ax1 = axes[0, i]
        epochs = range(1, len(history['train_task_loss']) + 1)
        ax1.plot(epochs, history['train_task_loss'], 'b-', label='Task Loss', linewidth=2)
        ax1.plot(epochs, history['train_domain_loss'], 'r-', label='Domain Loss', linewidth=2)
        if history['val_loss']:
            ax1.plot(epochs, history['val_loss'], 'g--', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'Test: {test_station} - Losses')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Lambda and domain accuracy
        ax2 = axes[1, i]
        ax2.plot(epochs, history['lambda'], 'purple', label='λ (GRL)', linewidth=2)
        ax2_twin = ax2.twinx()
        ax2_twin.plot(epochs, [acc * 100 for acc in history['domain_acc']], 
                      'orange', label='Domain Acc', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Lambda (λ)', color='purple')
        ax2_twin.set_ylabel('Domain Acc (%)', color='orange')
        ax2.set_title(f'Test: {test_station} - DA Progress')
        ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
        ax2_twin.axhline(y=50, color='gray', linestyle=':', alpha=0.5, label='Random (50%)')
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
    
    plt.suptitle('DANN Training History', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = output_dir / 'dann_training_history.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training history saved to: {save_path}")


def main():
    """Run DANN leave-one-station-out cross-validation."""
    
    # Configuration
    config = DANNConfig()
    
    # Setup output directory and logging
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(str(output_dir), log_level="INFO")
    log = logging.getLogger(__name__)
    
    log.info("="*70)
    log.info("DANN LEAVE-ONE-STATION-OUT CROSS-VALIDATION")
    log.info("="*70)
    log.info(f"Stations: {list(STATIONS.keys())}")
    log.info(f"Number of folds: {len(CV_FOLDS)}")
    log.info(f"Domain adaptation settings:")
    log.info(f"  Lambda schedule: {config.lambda_schedule}")
    log.info(f"  Lambda max: {config.lambda_max}")
    log.info(f"  Domain loss weight: {config.domain_loss_weight}")
    
    # Set seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Run all folds
    results = []
    for fold in CV_FOLDS:
        result = run_dann_fold(fold, config, log)
        results.append(result)
    
    # Create visualizations
    log.info("\n" + "="*70)
    log.info("CREATING VISUALIZATIONS")
    log.info("="*70)
    
    create_dann_scatter_plot(results, output_dir)
    create_dann_summary_plot(results, output_dir)
    create_training_history_plot(results, output_dir)
    
    # Print summary
    log.info("\n" + "="*70)
    log.info("DANN CROSS-VALIDATION SUMMARY")
    log.info("="*70)
    
    print("\n" + "="*70)
    print("DANN CROSS-VALIDATION RESULTS")
    print("="*70)
    print(f"{'Test Station':<15} {'Train Stations':<20} {'R²':>10} {'RMSE':>10} {'MAE':>10}")
    print("-"*70)
    
    r2_scores = []
    rmse_scores = []
    
    for result in results:
        test = result['test_station']
        train = '+'.join(result['train_stations'])
        r2 = result['metrics']['r2']
        rmse = result['metrics']['rmse']
        mae = result['metrics']['mae']
        
        r2_scores.append(r2)
        rmse_scores.append(rmse)
        
        print(f"{test:<15} {train:<20} {r2:>10.4f} {rmse:>10.4f} {mae:>10.4f}")
    
    print("-"*70)
    print(f"{'MEAN':<15} {'':<20} {np.mean(r2_scores):>10.4f} {np.mean(rmse_scores):>10.4f}")
    print(f"{'STD':<15} {'':<20} {np.std(r2_scores):>10.4f} {np.std(rmse_scores):>10.4f}")
    print("="*70)
    
    # Save results to CSV
    results_df = pd.DataFrame([
        {
            'test_station': r['test_station'],
            'train_stations': '+'.join(r['train_stations']),
            'r2': r['metrics']['r2'],
            'rmse': r['metrics']['rmse'],
            'mae': r['metrics']['mae'],
            'bias': r['metrics']['bias'],
            'correlation': r['metrics']['correlation'],
            'n_samples': r['metrics']['n_samples']
        }
        for r in results
    ])
    
    results_df.to_csv(output_dir / 'dann_cv_results.csv', index=False)
    log.info(f"\nResults saved to: {output_dir / 'dann_cv_results.csv'}")
    
    log.info(f"\nAll outputs saved to: {output_dir}")
    log.info("DANN CV complete!")
    
    return results


if __name__ == "__main__":
    results = main()

