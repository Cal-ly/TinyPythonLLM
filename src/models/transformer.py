"""
Minimal Transformer implementation for character-level language modeling.

This module demonstrates how attention mechanisms can be used to process
sequences. Each token attends to all previous tokens, enabling the model
to learn dependencies regardless of distance. The key mathematical ideas
include:
  * **Scaled Dot-Product Attention** - computes attention weights using dot
    products between query and key vectors, scaled by the dimensionality to
    prevent extremely large gradients.
  * **Multi-Head Attention** - splits embeddings into several heads so the
    model can learn different representations at various subspaces.
  * **Positional Encoding** - injects order information into token embeddings
    using sine and cosine waves of varying frequencies.

The model outputs a categorical distribution over the next character for
every position in the sequence. During training we minimize cross entropy
loss between these predictions and the actual next characters.
"""

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 128
    n_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1


class PositionalEncoding(nn.Module):
    """Injects information about token positions using sine/cosine functions."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        # Precompute a matrix of [max_len, d_model] containing positional encodings
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        seq_len = x.size(1)
        # Add positional encoding to embedding
        x = x + self.pe[:, :seq_len]
        return x


class TransformerModel(nn.Module):
    """A tiny Transformer language model.

    Each forward pass predicts the next token for every position in the input
    sequence. This is fundamentally a sequence-to-sequence modeling task where
    the model learns the probability distribution of characters given prior
    context.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embedding converts token ids to vectors of size d_model
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.positional_encoding = PositionalEncoding(config.d_model)

        # TransformerEncoder is a stack of self-attention layers followed by
        # feed-forward networks. Each encoder layer allows the model to attend to
        # previous positions in the sequence, capturing long-range dependencies.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dropout=config.dropout,
            dim_feedforward=config.d_model * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_layers
        )

        # Final projection layer maps the transformer output back to vocabulary size
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)

        logger.debug(f"Initialized TransformerModel with {self._count_parameters()} parameters")

    def _count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning logits and hidden states."""
        # x: [batch, seq_len]
        emb = self.embedding(x)
        emb = self.positional_encoding(emb)
        # Transformer expects [batch, seq_len, d_model]
        hidden = self.transformer(emb)
        logits = self.lm_head(hidden)
        return logits, hidden

    def generate(
        self, input_ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0
    ) -> torch.Tensor:
        """Greedy/temperature-based generation for demo purposes."""
        self.eval()
        logger.debug(f"Starting generation with {max_new_tokens} tokens, temperature={temperature}")

        ids = input_ids
        generated_tokens = 0
        
        with torch.no_grad():  # Add no_grad for inference
            for step in range(max_new_tokens):
                # Limit context window to prevent memory issues
                context = ids[:, -512:] if ids.size(1) > 512 else ids
                
                logits, _ = self.forward(context)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Add small epsilon to prevent NaN issues
                next_token_logits = next_token_logits + 1e-8
                
                # Apply softmax and sample
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Check for valid token - fix the bug here
                if next_token.item() >= self.config.vocab_size:
                    logger.warning(f"Generated invalid token {next_token.item()}, stopping generation")
                    break
                
                ids = torch.cat([ids, next_token], dim=1)
                generated_tokens += 1
                
                if step > 0 and step % 20 == 0:
                    logger.debug(f"Generation step {step}/{max_new_tokens}")

        logger.debug(f"Generation completed, generated {generated_tokens} tokens, total sequence length: {ids.size(1)}")
        return ids
