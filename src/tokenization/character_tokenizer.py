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
    """Character-level tokenizer.

    This simple tokenizer assigns a unique integer index to each character in
    the training corpus. It's equivalent to mapping a vocabulary of characters
    to indices and back. While overly simplistic for real-world language
    modeling, it's a great educational starting point and mirrors how the C#
    version works.
    """

    def __init__(self):
        # mapping from character to integer id
        self.stoi: Dict[str, int] = {}
        # reverse mapping from id to character
        self.itos: List[str] = []

    def fit(self, text: str) -> None:
        """Build vocabulary from given text."""
        unique_chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(unique_chars)}
        self.itos = unique_chars

    def encode(self, text: str) -> List[int]:
        """Convert text to list of token ids."""
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: List[int]) -> str:
        """Convert list of token ids back to string."""
        return "".join(self.itos[i] for i in ids)

    def save_state(self, filepath: str) -> None:
        """Persist tokenizer state as JSON."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"stoi": self.stoi, "itos": self.itos}, f)

    def load_state(self, filepath: str) -> None:
        """Load tokenizer state from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.stoi = {k: int(v) for k, v in data["stoi"].items()}
        self.itos = data["itos"]
