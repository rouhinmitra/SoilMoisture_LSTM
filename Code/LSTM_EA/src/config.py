"""
Configuration management for EA-LSTM experiments.
Uses dataclasses for typed configuration with validation.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Sequence
import yaml
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration"""
    train_files: List[str]
    test_files: List[str]
    data_dir: str = "data"
    seq_length: int = 30
    nan_threshold: float = 0.1  # 10% NaN threshold per column per window
    same_year_constraint: bool = True
    # Optional (start_month, end_month) inclusive, 1-12. None = use all months.
    month_range: Optional[Tuple[int, int]] = None
    # Optional (min_year, max_year) inclusive. None = use all years.
    year_range: Optional[Tuple[int, int]] = None
    # A4: drop windows whose calendar span exceeds seq_length days.  Windows are
    # sliced by ROW position, so a gap in the daily record silently produces a
    # window spanning more days than seq_length.  Default False = published behaviour.
    require_contiguous_windows: bool = False
    # A3: impute missing statics AFTER scaling (fill with 0 in z-space = the
    # training mean) instead of before (raw 0.0 -> an arbitrary z-score).
    # Default False = published behaviour.
    impute_statics_after_scaling: bool = False
    # D1: add SWI (exponential-filter soil water index) and API (antecedent precipitation
    # index) channels at these characteristic timescales, in days.  None = published behaviour.
    swi_api_taus: Optional[Sequence[int]] = None
    # D2: emit `context_length` timesteps of dynamic history per window (>= seq_length).
    # Window ACCEPTANCE is unchanged - only the dynamic tensor is extended backwards -
    # so the evaluation set stays identical to the baseline.  None = published behaviour.
    context_length: Optional[int] = None

    def __post_init__(self):
        """Validate configuration"""
        if self.seq_length < 1:
            raise ValueError(f"seq_length must be >= 1, got {self.seq_length}")
        if not 0 <= self.nan_threshold <= 1:
            raise ValueError(f"nan_threshold must be in [0,1], got {self.nan_threshold}")
        if not self.train_files:
            raise ValueError("train_files cannot be empty")
        if not self.test_files:
            raise ValueError("test_files cannot be empty")
        if self.month_range is not None:
            start_m, end_m = self.month_range
            if not (1 <= start_m <= 12 and 1 <= end_m <= 12 and start_m <= end_m):
                raise ValueError(f"month_range must be (start, end) with 1<=start<=end<=12, got {self.month_range}")
        if self.year_range is not None:
            start_y, end_y = self.year_range
            if start_y > end_y:
                raise ValueError(f"year_range must have start <= end, got {self.year_range}")

        # Validate files exist
        data_dir = Path(self.data_dir)
        for file_list in [self.train_files, self.test_files]:
            for file in file_list:
                filepath = data_dir / file
                if not filepath.exists():
                    logger.warning(f"File not found: {filepath}")


@dataclass
class FeatureConfig:
    """Feature definition and engineering configuration"""
    dynamic_cols: List[str]
    static_cols: List[str]
    target_col: str
    add_temporal: bool = False
    temporal_features: List[str] = field(default_factory=lambda: ['doy_sin', 'doy_cos'])
    # Presto (or other geospatial) embeddings as static features instead of Alpha Earth
    use_presto_static: bool = False
    presto_embeddings_path: Optional[str] = None
    # When True, suppress the auto-appended irrigation static feature
    exclude_irrigation_static: bool = False
    
    def __post_init__(self):
        """Validate features"""
        if not self.dynamic_cols:
            raise ValueError("dynamic_cols cannot be empty")
        if not self.target_col:
            raise ValueError("target_col must be specified")
        if self.use_presto_static and not self.presto_embeddings_path:
            raise ValueError("presto_embeddings_path must be set when use_presto_static is True")
        
        # Check for duplicates
        all_features = self.dynamic_cols + self.static_cols + [self.target_col]
        if len(all_features) != len(set(all_features)):
            duplicates = [f for f in all_features if all_features.count(f) > 1]
            raise ValueError(f"Duplicate features found: {set(duplicates)}")
    
    def get_all_dynamic_cols(self) -> List[str]:
        """Get dynamic columns including temporal features if enabled"""
        cols = self.dynamic_cols.copy()
        if self.add_temporal:
            cols.extend(self.temporal_features)
        return cols


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    model_type: str = "EALSTM"  # EALSTM or LSTM
    hidden_dim: int = 32
    dropout: float = 0.4
    num_layers: int = 1
    # B4: initialise the LSTM forget-gate bias to this value (Jozefowicz et al. 2015).
    # None = PyTorch default (uniform), i.e. published behaviour.
    forget_gate_bias_init: Optional[float] = None
    # D2: context/FiLM encoder.  context_dim = width of z; None = encoder disabled.
    context_dim: Optional[int] = None
    context_encoder_type: str = "conv"   # "conv" or "gru"
    
    def __post_init__(self):
        """Validate model config"""
        valid_types = ["EALSTM", "LSTM"]
        if self.model_type not in valid_types:
            raise ValueError(f"model_type must be one of {valid_types}, got {self.model_type}")
        if self.hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {self.hidden_dim}")
        if not 0 <= self.dropout < 1:
            raise ValueError(f"dropout must be in [0,1), got {self.dropout}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")


@dataclass
class TrainingConfig:
    """Training loop configuration"""
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 1e-3
    early_stopping_patience: Optional[int] = 10
    gradient_clip: Optional[float] = 1.0
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    
    def __post_init__(self):
        """Validate training config"""
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
    # V-REx: weight on the variance of per-environment training losses.
    # 0.0 = plain ERM = published behaviour.
    vrex_weight: float = 0.0


@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    name: str
    data: DataConfig
    features: FeatureConfig
    model: ModelConfig
    training: TrainingConfig
    seed: int = 42
    output_dir: str = "outputs"
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Create output directory"""
        output_path = Path(self.output_dir) / self.name
        output_path.mkdir(parents=True, exist_ok=True)
        self.output_dir = str(output_path)
    
    def save(self, filepath: Optional[str] = None):
        """Save configuration to YAML file"""
        if filepath is None:
            filepath = Path(self.output_dir) / "config.yaml"
        
        config_dict = {
            'name': self.name,
            'data': {
                'train_files': self.data.train_files,
                'test_files': self.data.test_files,
                'data_dir': self.data.data_dir,
                'seq_length': self.data.seq_length,
                'nan_threshold': self.data.nan_threshold,
                'same_year_constraint': self.data.same_year_constraint,
                'year_range': self.data.year_range,
            },
            'features': {
                'dynamic_cols': self.features.dynamic_cols,
                'static_cols': self.features.static_cols,
                'target_col': self.features.target_col,
                'add_temporal': self.features.add_temporal,
                'temporal_features': self.features.temporal_features,
            },
            'model': {
                'model_type': self.model.model_type,
                'hidden_dim': self.model.hidden_dim,
                'dropout': self.model.dropout,
                'num_layers': self.model.num_layers,
            },
            'training': {
                'batch_size': self.training.batch_size,
                'epochs': self.training.epochs,
                'learning_rate': self.training.learning_rate,
                'weight_decay': self.training.weight_decay,
                'early_stopping_patience': self.training.early_stopping_patience,
                'gradient_clip': self.training.gradient_clip,
            },
            'seed': self.seed,
            'log_level': self.log_level,
        }
        
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Configuration saved to {filepath}")


def load_config(config_path: str) -> ExperimentConfig:
    """Load and validate configuration from YAML file"""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML config: {e}")
    
    # Handle static_cols generation if specified as range
    if 'features' in config_dict:
        features = config_dict['features']
        static_cols = features.get('static_cols', [])
        use_presto = features.get('use_presto_static', False)
        if use_presto and (static_cols == "A00-A63" or static_cols == ["A00-A63"]):
            # Presto replaces Alpha Earth: static = Presto 128-d + precip only
            features['static_cols'] = ['precip_jan_apr', 'precip_may_oct']
        elif static_cols == "A00-A63" or static_cols == ["A00-A63"]:
            features['static_cols'] = [f'A{i:02d}' for i in range(64)] + [
                'precip_jan_apr', 'precip_may_oct'
            ]
    
    try:
        return ExperimentConfig(
            name=config_dict['name'],
            data=DataConfig(**config_dict['data']),
            features=FeatureConfig(**config_dict['features']),
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            seed=config_dict.get('seed', 42),
            output_dir=config_dict.get('output_dir', 'outputs'),
            log_level=config_dict.get('log_level', 'INFO')
        )
    except KeyError as e:
        raise ValueError(f"Missing required config key: {e}")
    except TypeError as e:
        raise ValueError(f"Invalid config format: {e}")


