"""
Training loop with early stopping and checkpointing for EA-LSTM.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, List
from pathlib import Path
import numpy as np
import logging
import time

from .config import TrainingConfig

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    
    Parameters
    ----------
    patience : int
        Number of epochs to wait for improvement before stopping
    min_delta : float
        Minimum change in monitored quantity to qualify as improvement
    mode : str
        'min' for loss (lower is better), 'max' for metrics (higher is better)
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Parameters
        ----------
        score : float
            Current validation metric
        
        Returns
        -------
        bool
            True if this is the best score so far
        """
        if self.mode == 'min':
            score = -score
            
        if self.best_score is None:
            self.best_score = score
            return True
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:
            self.best_score = score
            self.counter = 0
            return True


class Trainer:
    """
    Training manager for EA-LSTM models.
    
    Handles:
    - Training loop with progress logging
    - Early stopping
    - Learning rate scheduling
    - Model checkpointing
    - Gradient clipping
    
    Parameters
    ----------
    model : nn.Module
        Model to train
    config : TrainingConfig
        Training configuration
    output_dir : str or Path
        Directory to save checkpoints and logs
    device : str
        Device to train on ('cuda' or 'cpu')
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        config: TrainingConfig, 
        output_dir: str,
        device: Optional[str] = None
    ):
        self.model = model
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = self.model.to(self.device)
        logger.info(f"Training on device: {self.device}")
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.scheduler_factor,
            patience=config.scheduler_patience
        )
        
        # Setup loss function
        self.criterion = nn.MSELoss()
        
        # Setup early stopping
        self.early_stopping = None
        if config.early_stopping_patience:
            self.early_stopping = EarlyStopping(
                patience=config.early_stopping_patience,
                mode='min'
            )
        
        # Training history
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'lr': []
        }
        
        # Best model state
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.best_epoch = 0
    
    def _train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Parameters
        ----------
        train_loader : DataLoader
            Training data loader
        
        Returns
        -------
        float
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for x_d, x_s, y in train_loader:
            # Move to device
            x_d = x_d.to(self.device)
            x_s = x_s.to(self.device)
            y = y.to(self.device)
            
            # Forward pass - predict last timestep value
            self.optimizer.zero_grad()
            pred = self.model(x_d, x_s)
            
            # Use last timestep of target for loss
            y_last = y[:, -1:] if y.dim() > 1 else y.unsqueeze(1)
            loss = self.criterion(pred, y_last)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.gradient_clip
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> float:
        """
        Validate model.
        
        Parameters
        ----------
        val_loader : DataLoader
            Validation data loader
        
        Returns
        -------
        float
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for x_d, x_s, y in val_loader:
            x_d = x_d.to(self.device)
            x_s = x_s.to(self.device)
            y = y.to(self.device)
            
            pred = self.model(x_d, x_s)
            y_last = y[:, -1:] if y.dim() > 1 else y.unsqueeze(1)
            loss = self.criterion(pred, y_last)
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: Optional[DataLoader] = None
    ) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Parameters
        ----------
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader, optional
            Validation data loader (used for early stopping)
        
        Returns
        -------
        Dict[str, List[float]]
            Training history
        """
        logger.info("="*60)
        logger.info("STARTING TRAINING")
        logger.info("="*60)
        logger.info(f"Epochs: {self.config.epochs}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Learning rate: {self.config.learning_rate}")
        logger.info(f"Early stopping patience: {self.config.early_stopping_patience}")
        
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            epoch_start = time.time()
            
            # Training
            train_loss = self._train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            
            # Validation
            val_loss = None
            if val_loader is not None:
                val_loss = self._validate(val_loader)
                self.history['val_loss'].append(val_loss)
                
                # Update scheduler
                self.scheduler.step(val_loss)
            
            # Current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['lr'].append(current_lr)
            
            epoch_time = time.time() - epoch_start
            
            # Logging
            if val_loss is not None:
                logger.info(
                    f"Epoch {epoch+1:3d}/{self.config.epochs} | "
                    f"Train Loss: {train_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Time: {epoch_time:.1f}s"
                )
            else:
                logger.info(
                    f"Epoch {epoch+1:3d}/{self.config.epochs} | "
                    f"Train Loss: {train_loss:.6f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Time: {epoch_time:.1f}s"
                )
            
            # Check for best model
            check_loss = val_loss if val_loss is not None else train_loss
            if check_loss < self.best_val_loss:
                self.best_val_loss = check_loss
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self.best_epoch = epoch + 1
                logger.info(f"  → New best model! Loss: {check_loss:.6f}")
                
                # Save checkpoint
                self._save_checkpoint(epoch + 1, check_loss, is_best=True)
            
            # Early stopping
            if self.early_stopping and val_loss is not None:
                self.early_stopping(val_loss)
                if self.early_stopping.early_stop:
                    logger.info(f"\nEarly stopping triggered at epoch {epoch+1}")
                    break
        
        total_time = time.time() - start_time
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"\nLoaded best model from epoch {self.best_epoch}")
        
        logger.info("\n" + "="*60)
        logger.info("TRAINING COMPLETE")
        logger.info("="*60)
        logger.info(f"Total time: {total_time/60:.1f} minutes")
        logger.info(f"Best epoch: {self.best_epoch}")
        logger.info(f"Best loss: {self.best_val_loss:.6f}")
        
        return self.history
    
    def _save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config
        }
        
        # Save latest
        torch.save(checkpoint, self.output_dir / 'checkpoint_latest.pt')
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.output_dir / 'checkpoint_best.pt')
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']

