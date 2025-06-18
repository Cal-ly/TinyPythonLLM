"""Dataset utilities for TinyPythonLLM.

This module provides a small wrapper around :class:`torch.utils.data.Dataset`
and :class:`DataLoader`.  Text is tokenized once and split into training
and validation sets.  The dataset simply returns pairs of ``(input, target)``
where ``target`` is the next-token prediction for ``input``.
"""

from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from tokenization.character_tokenizer import CharacterTokenizer
from utils.logger import get_logger

logger = get_logger(__name__)


class OptimizedTextDataset(Dataset):
    """Dataset backed by a single tensor for efficient slicing."""

    def __init__(self, text: str, tokenizer: CharacterTokenizer, seq_len: int) -> None:
        self.seq_len = seq_len
        tokens = tokenizer.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.num_sequences = max(0, len(self.tokens) - seq_len)
        logger.debug("Dataset created with %s sequences", self.num_sequences)

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx
        end = start + self.seq_len + 1
        seq = self.tokens[start:end]
        return seq[:-1], seq[1:]


def load_text(path: str) -> str:
    """Load text from ``path`` using a set of common encodings."""
    encodings = ["utf-8", "latin-1", "cp1252"]
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                data = f.read()
            logger.info("Loaded text using %s encoding", enc)
            return data
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode file {path} with tried encodings")


def build_dataloaders(
    text: str,
    tokenizer: CharacterTokenizer,
    sequence_length: int,
    batch_size: int,
    train_split: float = 0.9,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation loaders from raw text."""

    split_idx = int(len(text) * train_split)
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    train_ds = OptimizedTextDataset(train_text, tokenizer, sequence_length)
    val_ds = OptimizedTextDataset(val_text, tokenizer, sequence_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    logger.info("Train loader: %s batches", len(train_loader))
    logger.info("Val loader: %s batches", len(val_loader))
    return train_loader, val_loader