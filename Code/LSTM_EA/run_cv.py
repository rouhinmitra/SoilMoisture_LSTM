#!/usr/bin/env python
"""
Leave-One-Station-Out Cross-Validation for EA-LSTM.

Trains on 2 stations, tests on the 3rd, for all 3 combinations:
- Fold 1: Train Ne1+Ne2, Test Ne3
- Fold 2: Train Ne1+Ne3, Test Ne2  
- Fold 3: Train Ne2+Ne3, Test Ne1

Generates individual and combined scatter plots with R² scores.
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from dataclasses import dataclass
from typing import List, Dict, Tuple
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


@dataclass
class CVConfig:
    """Cross-validation configuration - TUNED (BEST: Mean R² = 0.67)"""
    data_dir: str = "/Users/rouhinmitra/SM_work/Data/Base/"
    output_dir: str = "outputs/cv_results_tuned"
    seq_length: int = 15
    nan_threshold: float = 0.1
    # TUNED hyperparameters - best overall performance
    hidden_dim: int = 128
    dropout: float = 0.5
    batch_size: int = 32
    epochs: int = 150
    learning_rate: float = 0.0005
    early_stopping_patience: int = 15
    weight_decay: float = 0.001
    add_temporal: bool = True
    seed: int = 42
    model_type: str = "LSTM"  # "EALSTM" or "LSTM"
    num_layers: int = 3  # LSTM layers (only used when model_type="LSTM")
    val_fraction: float = 0.15  # fraction of training data held out for validation (early stopping / LR scheduling)
    year_range: Tuple[int, int] = (2017, 2024)  # restrict experiment data to this year range (inclusive)

    # Presto embeddings as static features (added to Alpha Earth when True)
    use_presto_static: bool = True
    presto_embeddings_path: str = "/Users/rouhinmitra/SM_work/Code/Data/s2_pixels/presto_embeddings_fused_interpolated.csv"
    
    # Features
    dynamic_cols: List[str] = None
    static_cols: List[str] = None
    target_col: str = "RZSM_25_avg"
    
    def __post_init__(self):
        if self.model_type == "LSTM" and self.output_dir == "outputs/cv_results_tuned":
            self.output_dir = "outputs/cv_results_lstm"
        if self.dynamic_cols is None:
            self.dynamic_cols = [
                # Original features
                'SSM', 'SSM_avg', 'SWC_PI_F_2_1_1', 'SWC_PI_F_3_1_1',
                'P_PI_F_1_1_1', 'P_PI_F_2_2_1', 'I', 'TA_1_1_1', 'RH_1_1_1',
                # Sentinel-2 features
                'ndvi', 'b11', 'b12', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b8a'
                # Note: irrigation is NOT in dynamic_cols - it will be added as static feature
                # to control the input gate (similar to Alpha Earth embeddings)
            ]
        if self.static_cols is None:
            # When use_presto_static: daily Presto (emb_0..emb_127) merged in loader + Alpha Earth + precip
            if self.use_presto_static:
                self.static_cols = [f'emb_{k}' for k in range(128)] + [f'A{i:02d}' for i in range(64)] + [
                    'precip_jan_apr', 'precip_may_oct',
                ]
            else:
                self.static_cols = [f'A{i:02d}' for i in range(64)] + [
                    'precip_jan_apr',   # accumulated precip Jan 1–Apr 30 (static / input-gate)
                    'precip_may_oct',   # accumulated precip May–Oct summer (static / input-gate)
                ]


def run_single_fold(
    fold: Dict,
    config: CVConfig,
    logger: logging.Logger
) -> Dict:
    """
    Run a single CV fold.
    
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
    
    # Create configs
    data_config = DataConfig(
        train_files=[STATIONS[s] for s in train_stations],
        test_files=[STATIONS[test_station]],
        data_dir=config.data_dir,
        seq_length=config.seq_length,
        nan_threshold=config.nan_threshold,
        same_year_constraint=True,
        month_range=(1, 12),  # May–October only
        year_range=config.year_range,
    )

    # Resolve Presto embeddings path (relative to LSTM_EA project root)
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
        temporal_features=['doy_sin', 'doy_cos'],
        use_presto_static=config.use_presto_static,
        presto_embeddings_path=presto_path if config.use_presto_static else None,
    )
    
    model_config = ModelConfig(
        model_type=config.model_type,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        num_layers=config.num_layers
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
    full_train_dataset = RZSMDataset(data['X_d_train'], data['X_s_train'], data['y_train'])
    test_dataset = RZSMDataset(data['X_d_test'], data['X_s_test'], data['y_test'])
    
    # Split training data into train/val to prevent data leakage.
    # Early stopping, LR scheduling, and model selection use val_loader
    # (a held-out portion of training stations), NOT the test station.
    val_size = int(len(full_train_dataset) * config.val_fraction)
    train_size = len(full_train_dataset) - val_size
    train_subset, val_subset = random_split(
        full_train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed),
    )
    
    train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    logger.info(f"Train/Val/Test split: {train_size} / {val_size} / {len(test_dataset)} samples")
    
    # Get dimensions
    dyn_dim = data['X_d_train'].shape[2]
    stat_dim = data['X_s_train'].shape[2]
    
    # Create model
    logger.info("Creating model...")
    model = get_model(model_config, dyn_dim, stat_dim)
    
    # Train
    fold_output_dir = Path(config.output_dir) / fold_name
    fold_output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = Trainer(model, training_config, str(fold_output_dir), device=str(device))
    
    logger.info("Training (val_loader from training stations, test station held out)...")
    history = trainer.train(train_loader, val_loader)
    
    # Evaluate
    logger.info("Evaluating...")
    evaluator = Evaluator(trainer.model, processor.scaler_y, str(fold_output_dir), device=str(device))
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
        'history': history,
        'dates_test': data.get('dates_test'),
    }


def create_combined_scatter_plot(results: List[Dict], output_dir: Path, model_type: str = "EALSTM"):
    """Create a combined scatter plot with all folds."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, (result, ax, color) in enumerate(zip(results, axes, colors)):
        y_pred = result['y_pred']
        y_true = result['y_true']
        metrics = result['metrics']
        test_station = result['test_station']
        
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.3, s=10, c=color)
        
        # 1:1 line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 Line')
        
        # Best fit line
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
    
    plt.suptitle(f'Leave-One-Station-Out Cross-Validation: {model_type}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = output_dir / 'cv_scatter_plots.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Combined scatter plot saved to: {save_path}")


def create_summary_plot(results: List[Dict], output_dir: Path):
    """Create a summary bar plot of R² scores."""
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    stations = [r['test_station'] for r in results]
    r2_scores = [r['metrics']['r2'] for r in results]
    rmse_scores = [r['metrics']['rmse'] for r in results]
    
    x = np.arange(len(stations))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, r2_scores, width, label='R²', color='steelblue')
    
    # Add value labels on bars
    for bar, val in zip(bars1, r2_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Add mean line
    mean_r2 = np.mean(r2_scores)
    ax.axhline(y=mean_r2, color='red', linestyle='--', linewidth=2, label=f'Mean R² = {mean_r2:.3f}')
    
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_xlabel('Test Station', fontsize=12)
    ax.set_title('Leave-One-Station-Out CV: R² by Test Station', fontsize=14)
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
    """Run leave-one-station-out cross-validation."""
    
    # Configuration
    config = CVConfig()
    
    # Setup output directory and logging
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(str(output_dir), log_level="INFO")
    log = logging.getLogger(__name__)
    
    log.info("="*70)
    log.info("LEAVE-ONE-STATION-OUT CROSS-VALIDATION")
    log.info("="*70)
    log.info(f"Model: {config.model_type}")
    log.info(f"Stations: {list(STATIONS.keys())}")
    log.info(f"Number of folds: {len(CV_FOLDS)}")
    log.info(f"Year range: {config.year_range[0]}–{config.year_range[1]} (inclusive)")
    log.info("Data: May–November only (month_range=(5, 11))")
    if config.use_presto_static:
        presto_path = Path(config.presto_embeddings_path)
        if not presto_path.is_absolute():
            presto_path = Path(__file__).resolve().parent / presto_path
        log.info("Presto embeddings: ENABLED at daily scale (merged on Date+Site; static = Presto 128-d + AE + precip)")
        log.info(f"  Path: {presto_path}")
    else:
        log.info("Presto embeddings: disabled (static = Alpha Earth A00-A63 + precip only)")
    log.info("Irrigation feature: Used as STATIC feature (input gate control)")
    log.info("  - Similar to Alpha Earth embeddings")
    log.info("  - Extracted from first timestep of each sequence")
    
    # Fail fast if Presto enabled but file missing
    if config.use_presto_static:
        p = Path(config.presto_embeddings_path)
        if not p.is_absolute():
            p = Path(__file__).resolve().parent / p
        if not p.exists():
            raise FileNotFoundError(
                f"Presto embeddings file not found: {p}. "
                "Set use_presto_static=False or provide a valid presto_embeddings_path."
            )

    # Set seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Run all folds
    results = []
    for fold in CV_FOLDS:
        result = run_single_fold(fold, config, log)
        results.append(result)
    
    # Create visualizations
    log.info("\n" + "="*70)
    log.info("CREATING VISUALIZATIONS")
    log.info("="*70)
    
    create_combined_scatter_plot(results, output_dir, model_type=config.model_type)
    create_summary_plot(results, output_dir)
    
    # Print summary
    log.info("\n" + "="*70)
    log.info("CROSS-VALIDATION SUMMARY")
    log.info("="*70)
    
    print("\n" + "="*70)
    print("CROSS-VALIDATION RESULTS")
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
    
    results_df.to_csv(output_dir / 'cv_results.csv', index=False)
    log.info(f"\nResults saved to: {output_dir / 'cv_results.csv'}")
    
    log.info(f"\nOutputs saved to: {output_dir}")
    log.info("CV complete!")
    
    return results


if __name__ == "__main__":
    results = main()

