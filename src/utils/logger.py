"""
Centralized logging utility for the TinyPythonLLM project.

This module provides a consistent logging interface that writes to both
console and a central log file. All modules in the project should use
this logger for consistent formatting and centralized log management.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "tinyllm",
    log_level: int = logging.INFO,
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    console_output: bool = False  # Changed default to False
) -> logging.Logger:
    """
    Set up a centralized logger with file handler and optional console handler.
    
    Args:
        name: Logger name (default: "tinyllm")
        log_level: Logging level (default: INFO)
        log_file: Path to log file (default: project_root/logs/tinyllm.log)
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        console_output: Whether to also output to console (default: False)
        
    Returns:
        Configured logger instance
    """
    # Get project root directory (3 levels up from this file)
    project_root = Path(__file__).parent.parent.parent
    
    # Create logs directory if it doesn't exist
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Set default log file path
    if log_file is None:
        log_file = logs_dir / "tinyllm.log"
    else:
        log_file = Path(log_file)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Create formatters
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Optional console handler
    if console_output:
        console_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str = "tinyllm", console_output: bool = False) -> logging.Logger:
    """
    Get the centralized logger instance.
    
    Args:
        name: Logger name (should match the module name)
        console_output: Whether to also output to console (default: False)
        
    Returns:
        Logger instance
    """
    # Check if logger already exists and is configured
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Set up logger if not already configured
        return setup_logger(name, console_output=console_output)
    return logger


def configure_external_loggers():
    """Configure external library loggers to reduce verbosity."""
    # Reduce PyTorch verbosity
    logging.getLogger("torch").setLevel(logging.WARNING)
    
    # Reduce other common library verbosity
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


# Default logger instance for convenience
default_logger = get_logger()
