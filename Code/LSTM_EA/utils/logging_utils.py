"""
Logging utilities for EA-LSTM experiments.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(
    output_dir: str,
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Parameters
    ----------
    output_dir : str or Path
        Directory to save log files
    log_level : str
        Logging level (DEBUG, INFO, WARNING, ERROR)
    log_to_file : bool
        Whether to log to file
    log_to_console : bool
        Whether to log to console
    
    Returns
    -------
    logging.Logger
        Configured root logger
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"experiment_{timestamp}.log"
    
    # Get numeric log level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Log setup info
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized at level: {log_level}")
    if log_to_file:
        logger.info(f"Log file: {log_file}")
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.
    
    Parameters
    ----------
    name : str
        Logger name (typically __name__)
    
    Returns
    -------
    logging.Logger
        Logger instance
    """
    return logging.getLogger(name)


