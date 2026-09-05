#!/usr/bin/env python
"""
Main entry point for EA-LSTM soil moisture prediction.

Usage:
    python main.py --config configs/baseline.yaml
    python main.py --config configs/with_temporal.yaml
"""
import argparse
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import sys
import traceback

from src.config import load_config
from src.data_loader import DataProcessor, RZSMDataset
from src.models import get_model
from src.trainer import Trainer
from src.evaluator import Evaluator
from utils.logging_utils import setup_logging
from utils.experiment_tracker import ExperimentTracker


def main(config_path: str) -> int:
    """
    Main training and evaluation pipeline.
    
    Parameters
    ----------
    config_path : str
        Path to YAML configuration file
    
    Returns
    -------
    int
        Exit code (0 for success, 1 for failure)
    """
    config = None
    
    try:
        # =============================================
        # 1. Load Configuration
        # =============================================
        config = load_config(config_path)
        
        # Setup logging
        logger = setup_logging(config.output_dir, config.log_level)
        import logging
        log = logging.getLogger(__name__)
        
        log.info("="*70)
        log.info(f"Starting Experiment: {config.name}")
        log.info("="*70)
        log.info(f"Configuration loaded from: {config_path}")
        
        # Save config to output directory
        config.save()
        
        # Set random seeds for reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        log.info(f"Random seed: {config.seed}")
        
        # =============================================
        # 2. Data Preparation
        # =============================================
        log.info("\n" + "="*70)
        log.info("DATA PREPARATION")
        log.info("="*70)
        
        data_processor = DataProcessor(config.data, config.features)
        data = data_processor.prepare_data(
            config.data.train_files,
            config.data.test_files
        )
        
        log.info(f"\nFinal Dataset Sizes:")
        log.info(f"  Train: {len(data['y_train'])} samples")
        log.info(f"  Test: {len(data['y_test'])} samples")
        
        # Create datasets and dataloaders
        train_dataset = RZSMDataset(
            data['X_d_train'], 
            data['X_s_train'], 
            data['y_train']
        )
        test_dataset = RZSMDataset(
            data['X_d_test'], 
            data['X_s_test'], 
            data['y_test']
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        
        # =============================================
        # 3. Model Creation
        # =============================================
        log.info("\n" + "="*70)
        log.info("MODEL CREATION")
        log.info("="*70)
        
        # Get feature dimensions
        dyn_dim = data['X_d_train'].shape[2]
        stat_dim = data['X_s_train'].shape[2]
        
        log.info(f"Feature dimensions:")
        log.info(f"  Dynamic: {dyn_dim}")
        log.info(f"  Static: {stat_dim}")
        
        model = get_model(config.model, dyn_dim, stat_dim)
        
        # Check device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log.info(f"Using device: {device}")
        
        # =============================================
        # 4. Training
        # =============================================
        log.info("\n" + "="*70)
        log.info("TRAINING")
        log.info("="*70)
        
        trainer = Trainer(model, config.training, config.output_dir, device=str(device))
        history = trainer.train(train_loader, test_loader)
        
        # =============================================
        # 5. Evaluation
        # =============================================
        log.info("\n" + "="*70)
        log.info("EVALUATION")
        log.info("="*70)
        
        evaluator = Evaluator(
            trainer.model,  # Use the model with best weights loaded
            data_processor.scaler_y,
            config.output_dir,
            device=str(device)
        )
        
        results = evaluator.evaluate(test_loader, config.name)
        
        # Plot training history
        evaluator.plot_training_history(history, config.name)
        
        # =============================================
        # 6. Log to Experiment Tracker
        # =============================================
        tracker = ExperimentTracker(
            Path(config.output_dir).parent / "experiments.csv"
        )
        tracker.log_from_config(config, results['metrics'], history)
        
        # =============================================
        # 7. Summary
        # =============================================
        log.info("\n" + "="*70)
        log.info("EXPERIMENT COMPLETE")
        log.info("="*70)
        log.info(f"Experiment: {config.name}")
        log.info(f"Model: {config.model.model_type}")
        log.info(f"\nFinal Metrics:")
        log.info(f"  R²:   {results['metrics']['r2']:.4f}")
        log.info(f"  RMSE: {results['metrics']['rmse']:.4f}")
        log.info(f"  MAE:  {results['metrics']['mae']:.4f}")
        log.info(f"\nOutputs saved to: {config.output_dir}")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"ERROR: Configuration or data error: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train EA-LSTM for soil moisture prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --config configs/baseline.yaml
    python main.py --config configs/with_temporal.yaml
    
For more information, see the README.md file.
        """
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/baseline.yaml',
        help='Path to YAML configuration file (default: configs/baseline.yaml)'
    )
    
    args = parser.parse_args()
    sys.exit(main(args.config))


