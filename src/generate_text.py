"""Utilities for generating text with a trained TinyPythonLLM model."""

from typing import Tuple
from pathlib import Path

import torch

from .transformer import Transformer
from .character_tokenizer import CharacterTokenizer


def load_model(model_path: str) -> Tuple[Transformer, CharacterTokenizer]:
    """Load a Transformer model and tokenizer from ``model_path``."""
    model_file = Path(model_path)
    
    # If directory provided, auto-discover model
    if model_file.is_dir():
        model_files = list(model_file.glob("*_model.pt"))
        if model_files:
            model_file = model_files[0]
        else:
            # Fallback to old naming convention
            fallback = model_file / "shakespeare_model.pt"
            if fallback.exists():
                model_file = fallback
            else:
                raise FileNotFoundError(f"No model files found in {model_path}")
    
    # Load model checkpoint
    # Note: We use weights_only=False because our checkpoints contain
    # CharacterTokenizer objects which are safe (we created them)
    try:
        checkpoint = torch.load(model_file, map_location="cpu", weights_only=False)
    except TypeError:
        # Fallback for older PyTorch versions that don't support weights_only parameter
        checkpoint = torch.load(model_file, map_location="cpu")
    
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