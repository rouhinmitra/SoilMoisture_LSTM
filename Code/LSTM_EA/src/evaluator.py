"""
Evaluation and visualization for EA-LSTM models.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Model evaluation and visualization.
    
    Handles:
    - Prediction on test data
    - Metric computation (R2, RMSE, MAE)
    - Visualization (time series, scatter plots)
    - Result saving
    
    Parameters
    ----------
    model : nn.Module
        Trained model
    scaler_y : StandardScaler
        Scaler for inverse transforming predictions
    output_dir : str or Path
        Directory to save outputs
    device : str, optional
        Device for inference
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        scaler_y: StandardScaler,
        output_dir: str,
        device: Optional[str] = None
    ):
        self.model = model
        self.scaler_y = scaler_y
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions on data.
        
        Parameters
        ----------
        data_loader : DataLoader
            Data loader for prediction
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (y_pred, y_true) - Predictions and actual values (inverse transformed)
        """
        self.model.eval()
        predictions = []
        actuals = []
        
        for x_d, x_s, y, g in data_loader:
            x_d = x_d.to(self.device)
            x_s = x_s.to(self.device)
            
            pred = self.model(x_d, x_s)
            predictions.append(pred.cpu().numpy())
            
            # Get last timestep of target
            y_last = y[:, -1:] if y.dim() > 1 else y.unsqueeze(1)
            actuals.append(y_last.numpy())
        
        # Concatenate all batches
        y_pred_scaled = np.concatenate(predictions, axis=0)
        y_true_scaled = np.concatenate(actuals, axis=0)
        
        # Inverse transform to original scale
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        y_true = self.scaler_y.inverse_transform(y_true_scaled)
        
        return y_pred.flatten(), y_true.flatten()
    
    def compute_metrics(
        self, 
        y_pred: np.ndarray, 
        y_true: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Parameters
        ----------
        y_pred : np.ndarray
            Predicted values
        y_true : np.ndarray
            Actual values
        
        Returns
        -------
        Dict[str, float]
            Dictionary of metrics
        """
        # R2 Score
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # RMSE
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        # MAE
        mae = np.mean(np.abs(y_true - y_pred))
        
        # Bias
        bias = np.mean(y_pred - y_true)
        
        # Correlation coefficient
        corr = np.corrcoef(y_pred, y_true)[0, 1]
        
        # NSE (Nash-Sutcliffe Efficiency) - same as R2 for this case
        nse = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
        
        metrics = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'bias': bias,
            'correlation': corr,
            'nse': nse,
            'n_samples': len(y_true)
        }
        
        return metrics
    
    def evaluate(
        self, 
        test_loader: DataLoader, 
        experiment_name: str = "experiment"
    ) -> Dict:
        """
        Full evaluation pipeline.
        
        Parameters
        ----------
        test_loader : DataLoader
            Test data loader
        experiment_name : str
            Name for saving outputs
        
        Returns
        -------
        Dict
            Dictionary containing metrics, predictions, and actuals
        """
        logger.info("="*60)
        logger.info("EVALUATION")
        logger.info("="*60)
        
        # Get predictions
        y_pred, y_true = self.predict(test_loader)
        
        # Compute metrics
        metrics = self.compute_metrics(y_pred, y_true)
        
        # Log metrics
        logger.info(f"\nTest Set Metrics:")
        logger.info(f"  R²:          {metrics['r2']:.4f}")
        logger.info(f"  RMSE:        {metrics['rmse']:.4f}")
        logger.info(f"  MAE:         {metrics['mae']:.4f}")
        logger.info(f"  Bias:        {metrics['bias']:.4f}")
        logger.info(f"  Correlation: {metrics['correlation']:.4f}")
        logger.info(f"  Samples:     {metrics['n_samples']}")
        
        # Create visualizations
        self.plot_time_series(y_pred, y_true, experiment_name)
        self.plot_scatter(y_pred, y_true, metrics, experiment_name)
        
        # Save predictions
        self.save_predictions(y_pred, y_true, experiment_name)
        
        return {
            'metrics': metrics,
            'y_pred': y_pred,
            'y_true': y_true
        }
    
    def plot_time_series(
        self, 
        y_pred: np.ndarray, 
        y_true: np.ndarray,
        name: str,
        max_points: int = 500
    ):
        """
        Plot time series of predictions vs actuals.
        
        Parameters
        ----------
        y_pred : np.ndarray
            Predicted values
        y_true : np.ndarray
            Actual values
        name : str
            Name for saving
        max_points : int
            Maximum points to plot (for readability)
        """
        fig, ax = plt.subplots(figsize=(14, 5))
        
        n_points = min(len(y_true), max_points)
        x = np.arange(n_points)
        
        ax.plot(x, y_true[:n_points], label='Actual', alpha=0.8, linewidth=1.5)
        ax.plot(x, y_pred[:n_points], label='Predicted', alpha=0.8, linewidth=1.5, linestyle='--')
        
        ax.set_xlabel('Sample Index', fontsize=12)
        ax.set_ylabel('RZSM (Volumetric %)', fontsize=12)
        ax.set_title(f'Soil Moisture Prediction: {name}', fontsize=14)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add R2 annotation
        r2 = 1 - np.sum((y_true[:n_points] - y_pred[:n_points])**2) / np.sum((y_true[:n_points] - np.mean(y_true[:n_points]))**2)
        ax.text(0.02, 0.98, f'R² = {r2:.4f}', transform=ax.transAxes, 
               fontsize=12, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        save_path = self.output_dir / f'{name}_timeseries.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Time series plot saved to: {save_path}")
    
    def plot_scatter(
        self, 
        y_pred: np.ndarray, 
        y_true: np.ndarray,
        metrics: Dict[str, float],
        name: str
    ):
        """
        Plot scatter plot with 1:1 line.
        
        Parameters
        ----------
        y_pred : np.ndarray
            Predicted values
        y_true : np.ndarray
            Actual values
        metrics : Dict[str, float]
            Computed metrics
        name : str
            Name for saving
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.3, s=10, c='steelblue')
        
        # 1:1 line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 Line')
        
        # Regression line
        z = np.polyfit(y_true, y_pred, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min_val, max_val, 100)
        ax.plot(x_line, p(x_line), 'g-', linewidth=2, alpha=0.7, label='Best Fit')
        
        ax.set_xlabel('Actual RZSM (Volumetric %)', fontsize=12)
        ax.set_ylabel('Predicted RZSM (Volumetric %)', fontsize=12)
        ax.set_title(f'Prediction Scatter Plot: {name}', fontsize=14)
        ax.legend(loc='upper left', fontsize=10)
        ax.set_aspect('equal', 'box')
        ax.grid(True, alpha=0.3)
        
        # Add metrics annotation
        metrics_text = (
            f"R² = {metrics['r2']:.4f}\n"
            f"RMSE = {metrics['rmse']:.4f}\n"
            f"MAE = {metrics['mae']:.4f}\n"
            f"n = {metrics['n_samples']}"
        )
        ax.text(0.98, 0.02, metrics_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        save_path = self.output_dir / f'{name}_scatter.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Scatter plot saved to: {save_path}")
    
    def plot_training_history(self, history: Dict[str, list], name: str):
        """
        Plot training history.
        
        Parameters
        ----------
        history : Dict[str, list]
            Training history from Trainer
        name : str
            Name for saving
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        ax1 = axes[0]
        epochs = range(1, len(history['train_loss']) + 1)
        ax1.plot(epochs, history['train_loss'], label='Train Loss', linewidth=2)
        if history.get('val_loss'):
            ax1.plot(epochs, history['val_loss'], label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss (MSE)', fontsize=12)
        ax1.set_title('Training & Validation Loss', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Learning rate plot
        ax2 = axes[1]
        ax2.plot(epochs, history['lr'], label='Learning Rate', linewidth=2, color='orange')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Learning Rate', fontsize=12)
        ax2.set_title('Learning Rate Schedule', fontsize=14)
        ax2.set_yscale('log')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.output_dir / f'{name}_training_history.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training history plot saved to: {save_path}")
    
    def save_predictions(
        self, 
        y_pred: np.ndarray, 
        y_true: np.ndarray,
        name: str
    ):
        """
        Save predictions to CSV.
        
        Parameters
        ----------
        y_pred : np.ndarray
            Predicted values
        y_true : np.ndarray
            Actual values
        name : str
            Name for saving
        """
        import pandas as pd
        
        df = pd.DataFrame({
            'actual': y_true,
            'predicted': y_pred,
            'residual': y_true - y_pred
        })
        
        save_path = self.output_dir / f'{name}_predictions.csv'
        df.to_csv(save_path, index=False)
        
        logger.info(f"Predictions saved to: {save_path}")


