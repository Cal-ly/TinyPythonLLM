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

import os
import sys
import math
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add src to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from utils.logger import get_logger
from utils.config import ModelConfig

logger = get_logger(__name__)

class OptimizedMultiHeadAttention(nn.Module):
    """Optimized Multi-Head Attention for RTX 4070 Mobile"""
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0
        
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout = dropout
        
        # Use single linear layer for efficiency
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        # Compute Q, K, V in one go
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.nhead, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, nhead, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Use Flash Attention if available (PyTorch 2.0+)
        if hasattr(F, 'scaled_dot_product_attention'):
            attn_output = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True if mask is None else False
            )
        else:
            # Fallback implementation
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        return self.out_proj(attn_output)

class OptimizedTransformerBlock(nn.Module):
    """Optimized Transformer block with memory efficiency"""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = OptimizedMultiHeadAttention(d_model, nhead, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Optimized feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),  # More efficient than ReLU for transformers
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm architecture for better training stability
        attn_output = self.attention(self.norm1(x), mask)
        x = x + attn_output
        
        ffn_output = self.ffn(self.norm2(x))
        x = x + ffn_output
        
        return x

class Transformer(nn.Module):
    """Optimized Transformer model for RTX 4070 Mobile"""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 512
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            OptimizedTransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie embeddings for parameter efficiency
        self.head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights with optimal values for training"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        
        # Create position indices
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        # Embeddings
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(position_ids)
        x = self.dropout(token_emb + pos_emb)
        
        # Create causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device))
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Final layer norm and projection
        x = self.ln_final(x)
        logits = self.head(x)
        
        return logits
    
    def generate(
        self, 
        input_ids: torch.Tensor, 
        max_length: int, 
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """Optimized text generation"""
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Only use the last max_seq_length tokens
                if input_ids.size(1) > self.max_seq_length:
                    input_ids = input_ids[:, -self.max_seq_length:]
                
                # Forward pass
                logits = self(input_ids)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering if specified
                if top_k is not None:
                    v, _ = torch.topk(next_token_logits, top_k)
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float('inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
