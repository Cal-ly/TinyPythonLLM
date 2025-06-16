"""
Dataset utilities.

The raw text is first normalized and then split into
training and validation sets. We produce short sequences of fixed length
`seq_len`. The model learns to predict the next character for every
position in these sequences.
"""

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from ..tokenization.character_tokenizer import CharacterTokenizer


def load_text(path: Path) -> str:
    """Load raw text file, applying minimal preprocessing."""
    text = path.read_text(encoding="utf-8")
    # Basic normalization to reduce vocabulary size
    text = " ".join(text.split())
    text = text.lower()
    return text


class CharDataset(Dataset):
    """Simple dataset producing sequences of fixed length for language modeling."""

    def __init__(self, data: torch.Tensor, seq_len: int):
        self.data = data
        self.seq_len = seq_len

    def __len__(self) -> int:
        return self.data.size(0) - self.seq_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


def build_dataloaders(
    text: str, tokenizer: CharacterTokenizer, seq_len: int, batch_size: int
) -> Tuple[DataLoader, DataLoader]:
    """Create DataLoaders for training and validation."""
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    # 90/10 train/val split
    split = int(0.9 * len(ids))
    train_ids = ids[:split]
    val_ids = ids[split:]

    train_ds = CharDataset(train_ids, seq_len)
    val_ds = CharDataset(val_ids, seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    return train_loader, val_loader
