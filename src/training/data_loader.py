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
from ..utils.logger import get_logger

logger = get_logger(__name__)


def load_text(path: Path) -> str:
    """Load raw text file, applying minimal preprocessing."""
    logger.info(f"Loading text from {path}")
    text = path.read_text(encoding="utf-8")
    original_length = len(text)

    # Basic normalization to reduce vocabulary size
    text = " ".join(text.split())
    text = text.lower()

    logger.info(
        f"Loaded and normalized text: {original_length} -> {len(text)} characters"
    )
    return text


class CharDataset(Dataset):
    """Simple dataset producing sequences of fixed length for language modeling."""

    def __init__(self, data: torch.Tensor, seq_len: int):
        self.data = data
        self.seq_len = seq_len
        logger.debug(
            f"Created CharDataset with {len(self)} sequences of length {seq_len}"
        )

    def __len__(self) -> int:
        return self.data.size(0) - self.seq_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


def build_dataloaders(
    text: str, tokenizer: CharacterTokenizer, seq_len: int, batch_size: int
) -> Tuple[DataLoader, DataLoader]:
    """Create DataLoaders for training and validation."""
    logger.info(
        f"Building dataloaders with seq_len={seq_len}, batch_size={batch_size}"
    )

    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    logger.info(f"Tokenized text to {len(ids)} tokens")

    # 90/10 train/val split
    split = int(0.9 * len(ids))
    train_ids = ids[:split]
    val_ids = ids[split:]

    logger.info(
        f"Split data: {len(train_ids)} train tokens, {len(val_ids)} val tokens"
    )

    train_ds = CharDataset(train_ids, seq_len)
    val_ds = CharDataset(val_ids, seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    logger.info(
        f"Created dataloaders: {len(train_loader)} train batches, {len(val_loader)} val batches"
    )

    return train_loader, val_loader
