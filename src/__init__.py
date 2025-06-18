"""TinyPythonLLM: A minimal educational language model implementation."""

__version__ = "0.1.0"
__author__ = "TinyLLM Team"

# Import main classes for easy access
from .transformer import Transformer
from .character_tokenizer import CharacterTokenizer
from .dataclass_config import ModelConfig, TrainingConfig

__all__ = [
    "Transformer",
    "CharacterTokenizer", 
    "ModelConfig",
    "TrainingConfig",
]

"""Model implementations for TinyPythonLLM."""

from .transformer import Transformer

__all__ = ["Transformer"]


"""Tokenization utilities for TinyPythonLLM."""

from .character_tokenizer import CharacterTokenizer

__all__ = ["CharacterTokenizer"]


"""Training utilities for TinyPythonLLM."""

from .train_model import run_training
from .data_loader import build_dataloaders, load_text, OptimizedTextDataset

__all__ = ["run_training", "build_dataloaders", "load_text", "OptimizedTextDataset"]


"""Utility modules for TinyPythonLLM."""

from .dataclass_config import ModelConfig, TrainingConfig
from .logger import get_logger, setup_logger

__all__ = ["ModelConfig", "TrainingConfig", "get_logger", "setup_logger"]


"""Interactive console for TinyPythonLLM."""

from .interactive import TinyLLMConsole

__all__ = ["TinyLLMConsole"]


"""Inference utilities for TinyPythonLLM."""

from .generate_text import load_model, generate_text

__all__ = ["load_model", "generate_text"]


"""Command line scripts for TinyPythonLLM."""
