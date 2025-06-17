"""
Configuration classes for TinyPythonLLM.

This module contains configuration dataclasses that define hyperparameters
and settings for training, model architecture, and other components.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters."""
    
    # Training parameters
    epochs: int = 200  # Increased for full dataset
    batch_size: int = 128  # Increased batch size
    learning_rate: float = 0.0005  # Slightly reduced LR
    
    # Model parameters  
    max_context: int = 256  # Increased context window
    d_model: int = 256     # Larger model
    n_heads: int = 8       # More attention heads
    num_layers: int = 4    # Deeper model
    
    # Training stability
    gradient_clip: float = 1.0  # Gradient clipping for stability
    val_split: float = 0.1      # Validation split ratio
    
    # Logging and monitoring
    sample_frequency: int = 20  # Less frequent sampling
    log_frequency: int = 10     # Log training progress every N epochs
    checkpoint_frequency: int = 50  # Save checkpoints every N epochs
    
    # Generation parameters
    temperature: float = 0.7    # Better for generation
    max_new_tokens: int = 150   # Longer samples

    # Architecture parameters
    vocab_size: int = 10000     # Size of the vocabulary
    pad_token_id: int = 0       # Padding token ID
    eos_token_id: int = 1       # End of sentence token ID
    bos_token_id: int = 2       # Beginning of sentence token ID
    
    # Console interface parameters
    console_max_tokens: int = 200  # Max tokens for console generation
    console_temperature: float = 0.8  # Temperature for console generation
    console_top_k: int = 50     # Top-k sampling for console generation


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
    max_epochs: int = 100
    sequence_length: int = 256
    vocab_size: Optional[int] = None
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.1
    save_every: int = 10
    eval_every: int = 5
    device: str = 'cuda'
