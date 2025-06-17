"""Utilities for generating text with a trained TinyPythonLLM model."""

from typing import Tuple

import torch

from models.transformer import Transformer
from tokenization.character_tokenizer import CharacterTokenizer


def load_model(model_path: str) -> Tuple[Transformer, CharacterTokenizer]:
    """Load a Transformer model and tokenizer from ``model_path``."""
    checkpoint = torch.load(model_path, map_location="cpu")
    config = checkpoint["config"]
    model = Transformer(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    tokenizer = checkpoint["tokenizer"]
    model.eval()
    return model, tokenizer


def generate_text(
    model: Transformer,
    tokenizer: CharacterTokenizer,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.8,
) -> str:
    """Generate text from ``prompt`` using ``model``."""
    device = next(model.parameters()).device
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    with torch.no_grad():
        generated = model.generate(input_ids, max_length=max_tokens, temperature=temperature)
    return tokenizer.decode(generated[0].tolist())
