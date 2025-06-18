"""
Character-based tokenizer.

Tokenization is the first step in any language model pipeline. Here we map
each distinct character to a unique integer, allowing us to represent text
as sequences of numbers that can be fed to the model. Although simplistic,
this mirrors early NLP approaches and keeps the vocabulary small.
"""

import json
from typing import List, Dict


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

    def save_state(self, filepath: str) -> None:
        """Persist tokenizer state as JSON."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"char_to_idx": self.char_to_idx, "idx_to_char": self.idx_to_char}, f)

    def load_state(self, filepath: str) -> None:
        """Load tokenizer state from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.char_to_idx = {k: int(v) for k, v in data["char_to_idx"].items()}
        self.idx_to_char = {int(k): v for k, v in data["idx_to_char"].items()}
        self.vocab_size = len(self.idx_to_char)