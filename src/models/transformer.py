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
        ids = input_ids
        for _ in range(max_new_tokens):
            logits, _ = self.forward(ids)
            next_token_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_token], dim=1)
        return ids
