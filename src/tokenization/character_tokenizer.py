"""
Character-based tokenizer.

Tokenization is the first step in any language model pipeline. Here we map
each distinct character to a unique integer, allowing us to represent text
as sequences of numbers that can be fed to the model. Although simplistic,
this mirrors early NLP approaches and keeps the vocabulary small.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict

# Add src to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


class CharacterTokenizer:
    """Character-level tokenizer for text processing."""

    def __init__(self):
        self.char_to_idx: Dict[str, int] = {}
        self.idx_to_char: Dict[int, str] = {}
        self.vocab_size = 0

    def fit(self, text: str) -> None:
        """Build vocabulary from text."""
        unique_chars = sorted(set(text))
        self.char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(unique_chars)}
        self.vocab_size = len(unique_chars)

    def encode(self, text: str) -> List[int]:
        """Convert text to token indices."""
        return [self.char_to_idx.get(char, 0) for char in text]

    def decode(self, tokens: List[int]) -> str:
        """Convert token indices to text."""
        return "".join([self.idx_to_char.get(token, "") for token in tokens])
        encoded = [self.stoi[ch] for ch in text]
        logger.debug(f"Encoded text of length {len(text)} to {len(encoded)} tokens")
        return encoded

    def decode(self, ids: List[int]) -> str:
        """Convert list of token ids back to string."""
        decoded = "".join(self.itos[i] for i in ids)
        logger.debug(f"Decoded {len(ids)} tokens to text of length {len(decoded)}")
        return decoded

    def save_state(self, filepath: str) -> None:
        """Persist tokenizer state as JSON."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"stoi": self.stoi, "itos": self.itos}, f)
        logger.info(f"Saved tokenizer state to {filepath}")

    def load_state(self, filepath: str) -> None:
        """Load tokenizer state from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.stoi = {k: int(v) for k, v in data["stoi"].items()}
        self.itos = data["itos"]
        logger.info(f"Loaded tokenizer state from {filepath} with {len(self.itos)} characters")
