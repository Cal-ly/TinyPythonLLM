"""
Configuration classes for TinyPythonLLM.

This module contains configuration dataclasses that define hyperparameters
and settings for training, model architecture, and other components.
"""

from dataclasses import dataclass


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
    
    # Logging and monitoring
    sample_frequency: int = 20  # Less frequent sampling
    log_frequency: int = 10     # Log training progress every N epochs
    
    # Generation parameters
    temperature: float = 0.7    # Better for generation
    max_new_tokens: int = 150   # Longer samples

    # Architecture parameters
    vocab_size: int = 10000     # Size of the vocabulary
    pad_token_id: int = 0       # Padding token ID
    eos_token_id: int = 1       # End of sentence token ID
    bos_token_id: int = 2       # Beginning of sentence token ID
