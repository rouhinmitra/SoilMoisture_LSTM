"""
Domain Adaptation Models for EA-LSTM.
Implements DANN (Domain Adversarial Neural Network) for station-robust prediction.
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional
import logging
import math

from .models import EALSTMCell

logger = logging.getLogger(__name__)


class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer (GRL) for domain adversarial training.
    
    During forward pass: identity function
    During backward pass: reverses gradients and scales by lambda
    
    This is the core of DANN - it makes the feature extractor learn
    domain-invariant representations by reversing the gradient from
    the domain discriminator.
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        # Reverse the gradient and scale by lambda
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    """
    Wrapper module for Gradient Reversal Function.
    
    Parameters
    ----------
    lambda_ : float
        Scaling factor for gradient reversal (default: 1.0)
    """
    
    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)
    
    def set_lambda(self, lambda_: float):
        """Update lambda value (used for scheduling)"""
        self.lambda_ = lambda_


class DomainDiscriminator(nn.Module):
    """
    Domain Discriminator for DANN.
    
    MLP classifier that distinguishes between source and target domains.
    The gradient reversal layer ensures that the feature extractor
    learns to fool this discriminator.
    
    Parameters
    ----------
    input_dim : int
        Dimension of input features (hidden state from LSTM)
    hidden_dim : int
        Hidden layer dimension (default: 64)
    dropout : float
        Dropout probability (default: 0.3)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
        
        logger.info(f"Created DomainDiscriminator: input_dim={input_dim}, "
                   f"hidden_dim={hidden_dim}, dropout={dropout}")
    
    def _init_weights(self):
        """Initialize weights with Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features, shape (batch, input_dim)
        
        Returns
        -------
        torch.Tensor
            Domain probability (0=source, 1=target), shape (batch, 1)
        """
        return self.discriminator(x)


class EALSTM_DANN(nn.Module):
    """
    Entity-Aware LSTM with Domain Adversarial Training (DANN).
    
    Combines the EA-LSTM feature extractor with:
    1. Task head for RZSM prediction (trained on source domain)
    2. Domain discriminator with GRL (trained on both domains)
    
    The GRL reverses gradients flowing to the feature extractor,
    encouraging domain-invariant feature learning.
    
    Parameters
    ----------
    dyn_dim : int
        Dimension of dynamic features
    stat_dim : int
        Dimension of static features
    hidden_dim : int
        Hidden state dimension
    dropout : float
        Dropout probability for task head
    domain_hidden_dim : int
        Hidden dimension for domain discriminator
    domain_dropout : float
        Dropout probability for domain discriminator
    """
    
    def __init__(
        self,
        dyn_dim: int,
        stat_dim: int,
        hidden_dim: int,
        dropout: float = 0.4,
        domain_hidden_dim: int = 64,
        domain_dropout: float = 0.3
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Feature extractor: EA-LSTM cell
        self.cell = EALSTMCell(dyn_dim, stat_dim, hidden_dim)
        
        # Task head: RZSM prediction
        self.task_dropout = nn.Dropout(dropout)
        self.task_head = nn.Linear(hidden_dim, 1)
        
        # Domain adaptation components
        self.grl = GradientReversalLayer(lambda_=1.0)
        self.domain_discriminator = DomainDiscriminator(
            input_dim=hidden_dim,
            hidden_dim=domain_hidden_dim,
            dropout=domain_dropout
        )
        
        # Track current lambda for logging
        self.current_lambda = 1.0
        
        logger.info(f"Created EALSTM_DANN: dyn_dim={dyn_dim}, stat_dim={stat_dim}, "
                   f"hidden_dim={hidden_dim}, dropout={dropout}")
        
        # Log total parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    def set_lambda(self, lambda_: float):
        """
        Set the lambda value for gradient reversal.
        
        Parameters
        ----------
        lambda_ : float
            New lambda value (typically scheduled from 0 to 1)
        """
        self.grl.set_lambda(lambda_)
        self.current_lambda = lambda_
    
    @staticmethod
    def compute_lambda(epoch: int, total_epochs: int, schedule: str = 'gradual') -> float:
        """
        Compute lambda value based on training progress.
        
        Parameters
        ----------
        epoch : int
            Current epoch (0-indexed)
        total_epochs : int
            Total number of epochs
        schedule : str
            'gradual' - sigmoid schedule (recommended)
            'linear' - linear increase
            'constant' - always 1.0
        
        Returns
        -------
        float
            Lambda value in [0, 1]
        """
        p = epoch / total_epochs  # Training progress [0, 1]
        
        if schedule == 'gradual':
            # Sigmoid schedule from DANN paper
            # Starts slow, increases faster in middle, slows at end
            return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0
        elif schedule == 'linear':
            return p
        elif schedule == 'constant':
            return 1.0
        else:
            raise ValueError(f"Unknown lambda schedule: {schedule}")
    
    def forward(
        self,
        x_dyn_seq: torch.Tensor,
        x_stat: torch.Tensor,
        return_domain: bool = True,
        return_features: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through EALSTM-DANN.
        
        Parameters
        ----------
        x_dyn_seq : torch.Tensor
            Dynamic feature sequence, shape (batch, seq_len, dyn_dim)
        x_stat : torch.Tensor
            Static features, shape (batch, seq_len, stat_dim)
        return_domain : bool
            Whether to return domain predictions (set False for eval/inference)
        return_features : bool
            Whether to return hidden features (for analysis)
        
        Returns
        -------
        If return_domain=True (training):
            Tuple (task_pred, domain_pred, features or None)
        If return_domain=False (inference):
            task_pred only (compatible with base EALSTM)
        """
        batch_size, seq_len, _ = x_dyn_seq.size()
        device = x_dyn_seq.device
        
        # Initialize hidden and cell states
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, device=device)
        
        # Use static features from last timestep
        x_stat_last = x_stat[:, -1, :]
        
        # Process sequence through EA-LSTM
        for t in range(seq_len):
            h, c = self.cell(x_dyn_seq[:, t, :], x_stat_last, h, c)
        
        # Task prediction (from hidden state)
        task_pred = self.task_head(self.task_dropout(h))
        
        # For inference/evaluation, just return task prediction (compatible with Evaluator)
        if not return_domain:
            return task_pred
        
        # Domain prediction (through GRL) - for training
        features_reversed = self.grl(h)
        domain_pred = self.domain_discriminator(features_reversed)
        
        if return_features:
            return task_pred, domain_pred, h
        
        return task_pred, domain_pred, None
    
    def predict(
        self,
        x_dyn_seq: torch.Tensor,
        x_stat: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict RZSM values (inference mode, no domain output).
        
        Parameters
        ----------
        x_dyn_seq : torch.Tensor
            Dynamic feature sequence
        x_stat : torch.Tensor
            Static features
        
        Returns
        -------
        torch.Tensor
            RZSM predictions
        """
        return self.forward(x_dyn_seq, x_stat, return_domain=False)


def get_dann_model(
    dyn_dim: int,
    stat_dim: int,
    hidden_dim: int = 128,
    dropout: float = 0.4,
    domain_hidden_dim: int = 64,
    domain_dropout: float = 0.3
) -> EALSTM_DANN:
    """
    Factory function to create DANN model.
    
    Parameters
    ----------
    dyn_dim : int
        Dimension of dynamic features
    stat_dim : int
        Dimension of static features
    hidden_dim : int
        Hidden state dimension
    dropout : float
        Task head dropout
    domain_hidden_dim : int
        Domain discriminator hidden dimension
    domain_dropout : float
        Domain discriminator dropout
    
    Returns
    -------
    EALSTM_DANN
        Initialized DANN model
    """
    return EALSTM_DANN(
        dyn_dim=dyn_dim,
        stat_dim=stat_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        domain_hidden_dim=domain_hidden_dim,
        domain_dropout=domain_dropout
    )

