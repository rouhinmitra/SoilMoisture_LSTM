"""
Simple CSV-based experiment tracking for EA-LSTM.
"""
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
import logging
import json

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Simple CSV-based experiment tracking.
    
    Tracks experiment configurations and results for easy comparison.
    
    Parameters
    ----------
    results_file : str or Path
        Path to CSV file for storing results
    """
    
    # CSV columns
    COLUMNS = [
        'timestamp',
        'experiment_name',
        'model_type',
        'hidden_dim',
        'dropout',
        'seq_length',
        'learning_rate',
        'batch_size',
        'epochs_trained',
        'r2',
        'rmse',
        'mae',
        'train_loss',
        'val_loss',
        'config_path',
        'notes'
    ]
    
    def __init__(self, results_file: str = "experiments.csv"):
        self.results_file = Path(results_file)
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """Create CSV file with headers if it doesn't exist"""
        if not self.results_file.exists():
            self.results_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.results_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
                writer.writeheader()
            logger.info(f"Created experiment tracking file: {self.results_file}")
    
    def log_experiment(
        self,
        experiment_name: str,
        model_type: str,
        hidden_dim: int,
        dropout: float,
        seq_length: int,
        learning_rate: float,
        batch_size: int,
        epochs_trained: int,
        metrics: Dict[str, float],
        train_loss: float,
        val_loss: Optional[float] = None,
        config_path: Optional[str] = None,
        notes: str = ""
    ):
        """
        Log an experiment to the tracking file.
        
        Parameters
        ----------
        experiment_name : str
            Name of the experiment
        model_type : str
            Model architecture used
        hidden_dim : int
            Hidden dimension size
        dropout : float
            Dropout rate
        seq_length : int
            Sequence length
        learning_rate : float
            Learning rate
        batch_size : int
            Batch size
        epochs_trained : int
            Number of epochs actually trained
        metrics : Dict[str, float]
            Dictionary containing r2, rmse, mae
        train_loss : float
            Final training loss
        val_loss : float, optional
            Final validation loss
        config_path : str, optional
            Path to config file
        notes : str
            Additional notes
        """
        row = {
            'timestamp': datetime.now().isoformat(),
            'experiment_name': experiment_name,
            'model_type': model_type,
            'hidden_dim': hidden_dim,
            'dropout': dropout,
            'seq_length': seq_length,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'epochs_trained': epochs_trained,
            'r2': f"{metrics.get('r2', 0):.6f}",
            'rmse': f"{metrics.get('rmse', 0):.6f}",
            'mae': f"{metrics.get('mae', 0):.6f}",
            'train_loss': f"{train_loss:.6f}",
            'val_loss': f"{val_loss:.6f}" if val_loss else "",
            'config_path': config_path or "",
            'notes': notes
        }
        
        with open(self.results_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
            writer.writerow(row)
        
        logger.info(f"Logged experiment '{experiment_name}' to {self.results_file}")
    
    def log_from_config(
        self,
        config,  # ExperimentConfig
        metrics: Dict[str, float],
        history: Dict[str, List[float]],
        notes: str = ""
    ):
        """
        Log experiment using config object.
        
        Parameters
        ----------
        config : ExperimentConfig
            Experiment configuration
        metrics : Dict[str, float]
            Evaluation metrics
        history : Dict[str, List[float]]
            Training history
        notes : str
            Additional notes
        """
        self.log_experiment(
            experiment_name=config.name,
            model_type=config.model.model_type,
            hidden_dim=config.model.hidden_dim,
            dropout=config.model.dropout,
            seq_length=config.data.seq_length,
            learning_rate=config.training.learning_rate,
            batch_size=config.training.batch_size,
            epochs_trained=len(history['train_loss']),
            metrics=metrics,
            train_loss=history['train_loss'][-1],
            val_loss=history['val_loss'][-1] if history.get('val_loss') else None,
            config_path=str(Path(config.output_dir) / 'config.yaml'),
            notes=notes
        )
    
    def get_experiments(self) -> List[Dict]:
        """
        Load all experiments from the tracking file.
        
        Returns
        -------
        List[Dict]
            List of experiment records
        """
        experiments = []
        with open(self.results_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                experiments.append(row)
        return experiments
    
    def get_best_experiment(self, metric: str = 'r2', higher_is_better: bool = True) -> Optional[Dict]:
        """
        Get the best experiment based on a metric.
        
        Parameters
        ----------
        metric : str
            Metric to optimize (default: r2)
        higher_is_better : bool
            Whether higher values are better
        
        Returns
        -------
        Dict or None
            Best experiment record
        """
        experiments = self.get_experiments()
        if not experiments:
            return None
        
        # Sort by metric
        try:
            sorted_exp = sorted(
                experiments,
                key=lambda x: float(x.get(metric, 0) or 0),
                reverse=higher_is_better
            )
            return sorted_exp[0]
        except (ValueError, KeyError):
            logger.warning(f"Could not sort experiments by {metric}")
            return None
    
    def print_summary(self, top_n: int = 10):
        """
        Print summary of experiments.
        
        Parameters
        ----------
        top_n : int
            Number of top experiments to show
        """
        experiments = self.get_experiments()
        if not experiments:
            print("No experiments logged yet.")
            return
        
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        print(f"Total experiments: {len(experiments)}")
        
        # Sort by R2 (descending)
        sorted_exp = sorted(
            experiments,
            key=lambda x: float(x.get('r2', 0) or 0),
            reverse=True
        )
        
        print(f"\nTop {min(top_n, len(sorted_exp))} by R²:")
        print("-"*80)
        print(f"{'Name':<30} {'Model':<10} {'R²':>8} {'RMSE':>8} {'MAE':>8}")
        print("-"*80)
        
        for exp in sorted_exp[:top_n]:
            print(
                f"{exp['experiment_name']:<30} "
                f"{exp['model_type']:<10} "
                f"{float(exp['r2'] or 0):>8.4f} "
                f"{float(exp['rmse'] or 0):>8.4f} "
                f"{float(exp['mae'] or 0):>8.4f}"
            )
        print("-"*80)


