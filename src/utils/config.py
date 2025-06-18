"""
Configuration classes for TinyPythonLLM.

This module contains configuration dataclasses that define hyperparameters
and settings for training, model architecture, and other components.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    vocab_size: int
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 6
    sequence_length: int = 256
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    batch_size: int = 32
    learning_rate: float = 3e-4
    max_epochs: int = 5
    sequence_length: int = 256
    vocab_size: Optional[int] = None
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.1
    save_every: int = 5
    eval_every: int = 5
    device: str = 'cuda'
    
    # Generation parameters
    temperature: float = 0.7
    max_new_tokens: int = 150
    
    # Console interface parameters
    console_max_tokens: int = 200
    console_temperature: float = 0.8
    console_top_k: int = 50