"""
Model architectures for soil moisture prediction.
Includes EA-LSTM (Entity-Aware LSTM) and Standard LSTM.
"""
import torch
import torch.nn as nn
from typing import Tuple
import logging

from .config import ModelConfig

logger = logging.getLogger(__name__)


class ContextEncoder(nn.Module):
    """
    D2: encode a longer dynamic history window into a compact context vector z.

    Consumes the FULL context window (batch, context_len, dyn_dim) - which may be
    longer than the prediction window - and returns z of shape (batch, z_dim).
    """

    def __init__(self, dyn_dim: int, z_dim: int, kind: str = "conv", width: int = 32):
        super().__init__()
        self.kind = kind
        if kind == "conv":
            self.net = nn.Sequential(
                nn.Conv1d(dyn_dim, width, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv1d(width, width, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.proj = nn.Linear(width, z_dim)
        elif kind == "gru":
            self.net = nn.GRU(dyn_dim, z_dim, batch_first=True)
            self.proj = None
        else:
            raise ValueError(f"context_encoder_type must be 'conv' or 'gru', got {kind}")

    def forward(self, x_dyn_seq: torch.Tensor) -> torch.Tensor:
        if self.kind == "conv":
            h = self.net(x_dyn_seq.transpose(1, 2)).squeeze(-1)   # (batch, width)
            return self.proj(h)
        _, h_n = self.net(x_dyn_seq)
        return h_n[-1]


class EALSTMCell(nn.Module):
    """
    Entity-Aware LSTM Cell.
    
    The key innovation is that the input gate is controlled by static (entity)
    attributes, while the other gates are driven by dynamic inputs.
    
    This allows the model to learn how different entities (e.g., different
    field sites with different soil properties) should weight new information.
    
    Parameters
    ----------
    input_dim_dyn : int
        Dimension of dynamic input features
    input_dim_stat : int
        Dimension of static input features
    hidden_dim : int
        Dimension of hidden state
    """
    
    def __init__(self, input_dim_dyn: int, input_dim_stat: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Dynamic Gates (Forget, Output, Cell Update) driven by dynamic inputs
        self.weight_ih = nn.Linear(input_dim_dyn, 3 * hidden_dim)
        self.weight_hh = nn.Linear(hidden_dim, 3 * hidden_dim)
        
        # Static Input Gate driven by entity attributes (Alpha Earth features)
        self.weight_sh = nn.Linear(input_dim_stat, hidden_dim)
        
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier initialization"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(
        self, 
        x_dyn: torch.Tensor, 
        x_stat: torch.Tensor, 
        h: torch.Tensor, 
        c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for single timestep.
        
        Parameters
        ----------
        x_dyn : torch.Tensor
            Dynamic input at current timestep, shape (batch, input_dim_dyn)
        x_stat : torch.Tensor
            Static input features, shape (batch, input_dim_stat)
        h : torch.Tensor
            Previous hidden state, shape (batch, hidden_dim)
        c : torch.Tensor
            Previous cell state, shape (batch, hidden_dim)
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (h_next, c_next) - Updated hidden and cell states
        """
        # Dynamic gates: forget, output, cell candidate
        gates = self.weight_ih(x_dyn) + self.weight_hh(h)
        f, o, g = gates.chunk(3, dim=1)
        
        f = torch.sigmoid(f)  # Forget gate
        o = torch.sigmoid(o)  # Output gate
        g = torch.tanh(g)     # Cell candidate
        
        # Static Input Gate (The "Entity-Aware" part)
        # This gate is controlled by static features (e.g., soil properties)
        i = torch.sigmoid(self.weight_sh(x_stat))
        
        # Update cell state: c_next = f * c + i * g
        c_next = f * c + i * g
        
        # Update hidden state: h_next = o * tanh(c_next)
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next


class EALSTM(nn.Module):
    """
    Entity-Aware LSTM Model for soil moisture prediction.
    
    Architecture:
    1. EA-LSTM cell processes sequence with static-aware input gate
    2. Dropout for regularization
    3. Linear head for prediction
    
    Parameters
    ----------
    dyn_dim : int
        Dimension of dynamic features
    stat_dim : int
        Dimension of static features
    hidden_dim : int
        Hidden state dimension
    dropout : float
        Dropout probability
    """
    
    def __init__(self, dyn_dim: int, stat_dim: int, hidden_dim: int, dropout: float = 0.4,
                 context_dim: int = None, context_encoder_type: str = "conv"):
        super().__init__()
        self.hidden_dim = hidden_dim
        # D2: z is concatenated onto the static vector that gates the input gate,
        # so the cell's static input widens by context_dim.
        self.context = (ContextEncoder(dyn_dim, context_dim, context_encoder_type)
                        if context_dim else None)
        stat_dim_eff = stat_dim + (context_dim or 0)
        self.cell = EALSTMCell(dyn_dim, stat_dim_eff, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)
        
        logger.info(f"Created EALSTM: dyn_dim={dyn_dim}, stat_dim={stat_dim}, "
                   f"hidden_dim={hidden_dim}, dropout={dropout}")
        
    def forward(
        self, 
        x_dyn_seq: torch.Tensor, 
        x_stat: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through EA-LSTM.
        
        Parameters
        ----------
        x_dyn_seq : torch.Tensor
            Dynamic feature sequence, shape (batch, seq_len, dyn_dim)
        x_stat : torch.Tensor
            Static features, shape (batch, seq_len, stat_dim)
            Note: We use static features from the last timestep
        
        Returns
        -------
        torch.Tensor
            Predictions, shape (batch, 1)
        """
        batch_size = x_dyn_seq.size(0)
        device = x_dyn_seq.device

        # D2: x_stat carries the prediction-window length; the dynamic tensor may be
        # longer (context).  Encode the full history, then recur over the last L steps.
        pred_len = x_stat.size(1)
        z = self.context(x_dyn_seq) if self.context is not None else None
        x_dyn_pred = x_dyn_seq[:, -pred_len:, :]
        seq_len = x_dyn_pred.size(1)

        # Initialize hidden and cell states
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, device=device)
        
        # Use static features from last timestep (they should be constant anyway)
        x_stat_last = x_stat[:, -1, :]  # (batch, stat_dim)
        if z is not None:
            x_stat_last = torch.cat([z, x_stat_last], dim=-1)
        
        # Process sequence
        for t in range(seq_len):
            h, c = self.cell(x_dyn_pred[:, t, :], x_stat_last, h, c)
        
        # Prediction from final hidden state
        out = self.head(self.dropout(h))
        return out


class StandardLSTM(nn.Module):
    """
    Standard LSTM for comparison with EA-LSTM.
    
    Concatenates static features to dynamic at each timestep.
    
    Parameters
    ----------
    dyn_dim : int
        Dimension of dynamic features
    stat_dim : int
        Dimension of static features
    hidden_dim : int
        Hidden state dimension
    dropout : float
        Dropout probability
    num_layers : int
        Number of LSTM layers
    """
    
    def __init__(
        self, 
        dyn_dim: int, 
        stat_dim: int, 
        hidden_dim: int, 
        dropout: float = 0.4,
        num_layers: int = 1,
        forget_gate_bias_init: float = None,
        context_dim: int = None,
        context_encoder_type: str = "conv"
    ):
        super().__init__()
        
        # Input: concatenated dynamic + static features
        input_dim = dyn_dim + stat_dim
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        # B4: PyTorch bias layout per layer is [i | f | g | o], each of size hidden_dim.
        # bias_ih and bias_hh are summed, so setting ih's forget chunk and zeroing hh's
        # gives a net forget bias of exactly `forget_gate_bias_init`.  Deterministic
        # assignment - consumes no RNG, so seeding is unaffected.
        if forget_gate_bias_init is not None:
            with torch.no_grad():
                for layer in range(num_layers):
                    getattr(self.lstm, f'bias_ih_l{layer}')[hidden_dim:2 * hidden_dim] \
                        .fill_(float(forget_gate_bias_init))
                    getattr(self.lstm, f'bias_hh_l{layer}')[hidden_dim:2 * hidden_dim] \
                        .fill_(0.0)

        # D2: FiLM - predict (gamma, beta) from z and modulate the final hidden state.
        # The film layer is zero-initialised so the module is exactly the identity at
        # step 0 (gamma=0 -> scale 1, beta=0); it can only depart from baseline by learning.
        self.context = (ContextEncoder(dyn_dim, context_dim, context_encoder_type)
                        if context_dim else None)
        if self.context is not None:
            self.film = nn.Linear(context_dim, 2 * hidden_dim)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)
        
        logger.info(f"Created StandardLSTM: input_dim={input_dim}, "
                   f"hidden_dim={hidden_dim}, num_layers={num_layers}, dropout={dropout}, "
                   f"forget_bias_init={forget_gate_bias_init}")
        
    def forward(
        self, 
        x_dyn_seq: torch.Tensor, 
        x_stat: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through Standard LSTM.
        
        Parameters
        ----------
        x_dyn_seq : torch.Tensor
            Dynamic feature sequence, shape (batch, seq_len, dyn_dim)
        x_stat : torch.Tensor
            Static features, shape (batch, seq_len, stat_dim)
        
        Returns
        -------
        torch.Tensor
            Predictions, shape (batch, 1)
        """
        # D2: the dynamic tensor may carry more history than the prediction window;
        # x_stat's length defines the prediction window.
        pred_len = x_stat.size(1)
        z = self.context(x_dyn_seq) if self.context is not None else None
        x_dyn_pred = x_dyn_seq[:, -pred_len:, :]

        # Concatenate dynamic and static features
        x_combined = torch.cat([x_dyn_pred, x_stat], dim=-1)
        
        # LSTM forward pass
        _, (h_n, _) = self.lstm(x_combined)
        
        # Use last layer's hidden state
        h = h_n[-1]
        if z is not None:
            gamma, beta = self.film(z).chunk(2, dim=-1)
            h = (1.0 + gamma) * h + beta
        out = self.head(self.dropout(h))
        return out


def get_model(model_config: ModelConfig, dyn_dim: int, stat_dim: int) -> nn.Module:
    """
    Factory function to create models.
    
    Parameters
    ----------
    model_config : ModelConfig
        Model configuration
    dyn_dim : int
        Dimension of dynamic features
    stat_dim : int
        Dimension of static features
    
    Returns
    -------
    nn.Module
        Initialized model
    """
    if model_config.model_type == "EALSTM":
        model = EALSTM(
            dyn_dim=dyn_dim,
            stat_dim=stat_dim,
            hidden_dim=model_config.hidden_dim,
            dropout=model_config.dropout,
            context_dim=getattr(model_config, 'context_dim', None),
            context_encoder_type=getattr(model_config, 'context_encoder_type', 'conv'),
        )
    elif model_config.model_type == "LSTM":
        model = StandardLSTM(
            dyn_dim=dyn_dim,
            stat_dim=stat_dim,
            hidden_dim=model_config.hidden_dim,
            dropout=model_config.dropout,
            forget_gate_bias_init=getattr(model_config, 'forget_gate_bias_init', None),
            context_dim=getattr(model_config, 'context_dim', None),
            context_encoder_type=getattr(model_config, 'context_encoder_type', 'conv'),
            num_layers=model_config.num_layers
        )
    else:
        raise ValueError(f"Unknown model type: {model_config.model_type}")
    
    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    return model

