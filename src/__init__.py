"""TinyPythonLLM: A minimal educational language model implementation."""

__version__ = "0.1.0"
__author__ = "TinyLLM Team"

# Core model classes
from .transformer import Transformer, MultiHeadAttention, TransformerBlock
from .character_tokenizer import CharacterTokenizer

# Configuration classes
from .dataclass_config import ModelConfig, TrainingConfig

# Training utilities
from .train_model import run_training, train_epoch, evaluate
from .data_loader import build_dataloaders, load_text, OptimizedTextDataset

# Generation utilities
from .generate_text import load_model, generate_text

# Interactive console
from .interactive import TinyLLMConsole

# Logging utilities
from .logger import get_logger, setup_logger, configure_external_loggers

__all__ = [
    # Core model classes
    "Transformer",
    "MultiHeadAttention", 
    "TransformerBlock",
    "CharacterTokenizer",
    
    # Configuration
    "ModelConfig",
    "TrainingConfig",
    
    # Training
    "run_training",
    "train_epoch",
    "evaluate",
    "build_dataloaders",
    "load_text",
    "OptimizedTextDataset",
    
    # Generation and interaction
    "load_model",
    "generate_text",
    "TinyLLMConsole",
    
    # Utilities
    "get_logger",
    "setup_logger",
    "configure_external_loggers",
]