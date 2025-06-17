"""
Dataset utilities.

The raw text is first normalized and then split into
training and validation sets. We produce short sequences of fixed length
`seq_len`. The model learns to predict the next character for every
position in these sequences.
"""

import os
import sys
from pathlib import Path
from typing import Tuple, List
import random

import torch
from torch.utils.data import Dataset, DataLoader

# Add src to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from tokenization.character_tokenizer import CharacterTokenizer
from ..utils.logger import get_logger

logger = get_logger(__name__)


class OptimizedTextDataset(Dataset):
    """Memory-optimized dataset for RTX 4070 Mobile"""

    def __init__(self, text: str, tokenizer: CharacterTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Tokenize text once and store
        self.tokens = tokenizer.encode(text)

        # Pre-calculate valid starting positions
        self.valid_starts = list(range(0, len(self.tokens) - max_length))

        logger.info(f"Dataset created with {len(self.valid_starts)} sequences")

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start_idx = self.valid_starts[idx]

        # Get input and target sequences
        input_seq = self.tokens[start_idx : start_idx + self.max_length]
        target_seq = self.tokens[start_idx + 1 : start_idx + self.max_length + 1]

        return (
            torch.tensor(input_seq, dtype=torch.long),
            torch.tensor(target_seq, dtype=torch.long),
        )


def load_text(file_path: str) -> str:
    """Load text from file with encoding detection"""
    encodings = ["utf-8", "latin-1", "cp1252"]

    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                text = f.read()
            logger.info(f"Successfully loaded text with {encoding} encoding")
            return text
        except UnicodeDecodeError:
            continue

    raise ValueError(
        f"Could not decode file {file_path} with any of the attempted encodings"
    )


def build_dataloaders(
    text: str,
    tokenizer: CharacterTokenizer,
    max_length: int,
    batch_size: int,
    train_split: float = 0.9,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Build optimized data loaders for RTX 4070 Mobile"""

    # Split text for train/validation
    split_idx = int(len(text) * train_split)
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    # Create datasets
    train_dataset = OptimizedTextDataset(train_text, tokenizer, max_length)
    val_dataset = OptimizedTextDataset(val_text, tokenizer, max_length)

    # Create optimized data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    logger.info(f"Train loader: {len(train_loader)} batches")
    logger.info(f"Val loader: {len(val_loader)} batches")

    return train_loader, val_loader
