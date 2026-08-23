"""
Domain-Adaptive Training loop for DANN-based EA-LSTM.
Handles both source (labeled) and target (unlabeled) domain data.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import numpy as np
import logging
import time

from .config import TrainingConfig
from .models_da import EALSTM_DANN

logger = logging.getLogger(__name__)


class DANNTrainer:
    """
    Domain-Adaptive Training manager for EALSTM-DANN models.
    
    Handles:
    - Alternating source/target batch training
    - Task loss (MSE) on source domain
    - Domain loss (BCE) on both domains
    - Lambda scheduling for GRL
    - Early stopping and checkpointing
    
    Parameters
    ----------
    model : EALSTM_DANN
        DANN model to train
    config : TrainingConfig
        Training configuration
    output_dir : str or Path
        Directory to save checkpoints and logs
    device : str
        Device to train on ('cuda' or 'cpu')
    lambda_max : float
        Maximum lambda value for GRL (default: 1.0)
    lambda_schedule : str
        Lambda scheduling strategy ('gradual', 'linear', 'constant')
    domain_loss_weight : float
        Weight for domain loss in combined loss (default: 0.5)
    """
    
    def __init__(
        self,
        model: EALSTM_DANN,
        config: TrainingConfig,
        output_dir: str,
        device: Optional[str] = None,
        lambda_max: float = 1.0,
        lambda_schedule: str = 'gradual',
        domain_loss_weight: float = 0.5
    ):
        self.model = model
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Domain adaptation settings
        self.lambda_max = lambda_max
        self.lambda_schedule = lambda_schedule
        self.domain_loss_weight = domain_loss_weight
        
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
        
        # Setup loss functions
        self.task_criterion = nn.MSELoss()
        self.domain_criterion = nn.BCELoss()
        
        # Early stopping
        self.early_stopping_patience = config.early_stopping_patience
        self.early_stopping_counter = 0
        
        # Training history
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'train_task_loss': [],
            'train_domain_loss': [],
            'val_loss': [],
            'lr': [],
            'lambda': [],
            'domain_acc': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.best_epoch = 0
    
    def _compute_domain_accuracy(
        self,
        domain_pred: torch.Tensor,
        domain_labels: torch.Tensor
    ) -> float:
        """Compute domain classification accuracy"""
        pred_labels = (domain_pred > 0.5).float()
        correct = (pred_labels == domain_labels).sum().item()
        return correct / len(domain_labels)
    
    def _train_epoch(
        self,
        source_loader: DataLoader,
        target_loader: DataLoader,
        current_lambda: float
    ) -> Tuple[float, float, float, float]:
        """
        Train for one epoch with domain adaptation.
        
        Parameters
        ----------
        source_loader : DataLoader
            Source domain data (labeled)
        target_loader : DataLoader
            Target domain data (unlabeled - labels ignored)
        current_lambda : float
            Current GRL lambda value
        
        Returns
        -------
        Tuple[float, float, float, float]
            (total_loss, task_loss, domain_loss, domain_accuracy)
        """
        self.model.train()
        self.model.set_lambda(current_lambda)
        
        total_loss = 0.0
        total_task_loss = 0.0
        total_domain_loss = 0.0
        total_domain_correct = 0
        total_domain_samples = 0
        num_batches = 0
        
        # Create iterators
        source_iter = iter(source_loader)
        target_iter = iter(target_loader)
        
        # Number of batches per epoch (use the smaller of the two)
        n_batches = min(len(source_loader), len(target_loader))
        
        for _ in range(n_batches):
            # Get source batch
            try:
                x_d_src, x_s_src, y_src, _g_src = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                x_d_src, x_s_src, y_src, _g_src = next(source_iter)
            
            # Get target batch
            try:
                x_d_tgt, x_s_tgt, _, _g_tgt = next(target_iter)  # Ignore labels
            except StopIteration:
                target_iter = iter(target_loader)
                x_d_tgt, x_s_tgt, _, _g_tgt = next(target_iter)
            
            # Move to device
            x_d_src = x_d_src.to(self.device)
            x_s_src = x_s_src.to(self.device)
            y_src = y_src.to(self.device)
            x_d_tgt = x_d_tgt.to(self.device)
            x_s_tgt = x_s_tgt.to(self.device)
            
            batch_size_src = x_d_src.size(0)
            batch_size_tgt = x_d_tgt.size(0)
            
            # Create domain labels
            domain_label_src = torch.zeros(batch_size_src, 1, device=self.device)
            domain_label_tgt = torch.ones(batch_size_tgt, 1, device=self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass - source domain (with domain predictions for training)
            task_pred_src, domain_pred_src, _ = self.model(x_d_src, x_s_src, return_domain=True)
            
            # Forward pass - target domain (with domain predictions for training)
            _, domain_pred_tgt, _ = self.model(x_d_tgt, x_s_tgt, return_domain=True)
            
            # Task loss (source only)
            y_src_last = y_src[:, -1:] if y_src.dim() > 1 else y_src.unsqueeze(1)
            task_loss = self.task_criterion(task_pred_src, y_src_last)
            
            # Domain loss (both domains)
            domain_loss_src = self.domain_criterion(domain_pred_src, domain_label_src)
            domain_loss_tgt = self.domain_criterion(domain_pred_tgt, domain_label_tgt)
            domain_loss = (domain_loss_src + domain_loss_tgt) / 2
            
            # Combined loss
            loss = task_loss + self.domain_loss_weight * domain_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            total_task_loss += task_loss.item()
            total_domain_loss += domain_loss.item()
            
            # Domain accuracy
            domain_pred_all = torch.cat([domain_pred_src, domain_pred_tgt])
            domain_labels_all = torch.cat([domain_label_src, domain_label_tgt])
            total_domain_correct += ((domain_pred_all > 0.5).float() == domain_labels_all).sum().item()
            total_domain_samples += batch_size_src + batch_size_tgt
            
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_task_loss = total_task_loss / num_batches
        avg_domain_loss = total_domain_loss / num_batches
        domain_acc = total_domain_correct / total_domain_samples
        
        return avg_loss, avg_task_loss, avg_domain_loss, domain_acc
    
    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> float:
        """
        Validate model (task performance only).
        
        Parameters
        ----------
        val_loader : DataLoader
            Validation data loader
        
        Returns
        -------
        float
            Average validation loss (task loss)
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for x_d, x_s, y, g in val_loader:
            x_d = x_d.to(self.device)
            x_s = x_s.to(self.device)
            y = y.to(self.device)
            
            # Task prediction only
            task_pred = self.model.predict(x_d, x_s)
            y_last = y[:, -1:] if y.dim() > 1 else y.unsqueeze(1)
            loss = self.task_criterion(task_pred, y_last)
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def train(
        self,
        source_loader: DataLoader,
        target_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> Dict[str, List[float]]:
        """
        Full domain-adaptive training loop.
        
        Parameters
        ----------
        source_loader : DataLoader
            Source domain data (labeled training data)
        target_loader : DataLoader
            Target domain data (unlabeled - from test station)
        val_loader : DataLoader, optional
            Validation data loader (typically from source domain)
        
        Returns
        -------
        Dict[str, List[float]]
            Training history
        """
        logger.info("="*60)
        logger.info("STARTING DOMAIN-ADAPTIVE TRAINING (DANN)")
        logger.info("="*60)
        logger.info(f"Epochs: {self.config.epochs}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Learning rate: {self.config.learning_rate}")
        logger.info(f"Lambda schedule: {self.lambda_schedule}")
        logger.info(f"Lambda max: {self.lambda_max}")
        logger.info(f"Domain loss weight: {self.domain_loss_weight}")
        logger.info(f"Early stopping patience: {self.early_stopping_patience}")
        
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            epoch_start = time.time()
            
            # Compute current lambda
            current_lambda = self.model.compute_lambda(
                epoch, self.config.epochs, self.lambda_schedule
            ) * self.lambda_max
            
            # Training
            train_loss, task_loss, domain_loss, domain_acc = self._train_epoch(
                source_loader, target_loader, current_lambda
            )
            
            self.history['train_loss'].append(train_loss)
            self.history['train_task_loss'].append(task_loss)
            self.history['train_domain_loss'].append(domain_loss)
            self.history['lambda'].append(current_lambda)
            self.history['domain_acc'].append(domain_acc)
            
            # Validation
            val_loss = None
            if val_loader is not None:
                val_loss = self._validate(val_loader)
                self.history['val_loss'].append(val_loss)
                self.scheduler.step(val_loss)
            
            # Current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['lr'].append(current_lr)
            
            epoch_time = time.time() - epoch_start
            
            # Logging
            log_msg = (
                f"Epoch {epoch+1:3d}/{self.config.epochs} | "
                f"Task: {task_loss:.4f} | "
                f"Domain: {domain_loss:.4f} | "
                f"D-Acc: {domain_acc:.2%} | "
                f"λ: {current_lambda:.3f}"
            )
            if val_loss is not None:
                log_msg += f" | Val: {val_loss:.4f}"
            log_msg += f" | LR: {current_lr:.2e} | {epoch_time:.1f}s"
            logger.info(log_msg)
            
            # Check for best model
            check_loss = val_loss if val_loss is not None else task_loss
            if check_loss < self.best_val_loss:
                self.best_val_loss = check_loss
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                self.best_epoch = epoch + 1
                self.early_stopping_counter = 0
                logger.info(f"  → New best model! Loss: {check_loss:.6f}")
                self._save_checkpoint(epoch + 1, check_loss, is_best=True)
            else:
                self.early_stopping_counter += 1
            
            # Early stopping
            if self.early_stopping_patience and self.early_stopping_counter >= self.early_stopping_patience:
                logger.info(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
        
        total_time = time.time() - start_time
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"\nLoaded best model from epoch {self.best_epoch}")
        
        logger.info("\n" + "="*60)
        logger.info("DOMAIN-ADAPTIVE TRAINING COMPLETE")
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
            'config': self.config,
            'lambda_schedule': self.lambda_schedule,
            'lambda_max': self.lambda_max,
            'domain_loss_weight': self.domain_loss_weight
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

