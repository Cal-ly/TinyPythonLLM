"""
Training utilities for TinyPythonLLM.

The training loop here follows the classical supervised learning setup:
  1. Encode the dataset into integer tokens.
  2. Feed batches of tokens into a Transformer which predicts the next token.
  3. Compute **cross entropy loss** between predictions and ground truth.
     Cross entropy measures the difference between two probability
     distributions and is commonly used for classification. Minimizing it
     encourages the model to assign high probability to the correct next
     character.
  4. Apply **backpropagation** to update the model weights via gradient descent.

While extremely small, this demonstrates the same core principles used in
training state-of-the-art language models.
"""

from pathlib import Path
from typing import Iterable

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from .data_loader import build_dataloaders, load_text
from ..models.transformer import ModelConfig, TransformerModel
from ..tokenization.character_tokenizer import CharacterTokenizer
from ..utils.config import TrainingConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


def train_epoch(
    model: TransformerModel, data_loader: Iterable, loss_fn, optimizer, device
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = len(data_loader)

    for batch_idx, (x, y) in enumerate(data_loader):
        x = x.to(device)
        y = y.to(device)

        logits, _ = model(x)
        # Shift targets so each position predicts the next token
        loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

        if batch_idx % 50 == 0:
            logger.debug(
                "Training batch %d/%d, loss: %.4f",
                batch_idx,
                num_batches,
                loss.item(),
            )

    avg_loss = total_loss / num_batches
    logger.debug(
        "Training epoch completed, average loss: %.4f",
        avg_loss,
    )
    return avg_loss


def validate(model: TransformerModel, data_loader: Iterable, loss_fn, device) -> float:
    model.eval()
    total_loss = 0.0
    num_batches = len(data_loader)

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            logits, _ = model(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()

    avg_loss = total_loss / num_batches
    logger.debug(
        "Validation completed, average loss: %.4f",
        avg_loss,
    )
    return avg_loss


def run_training(data_path: Path, config: TrainingConfig) -> TransformerModel:
    logger.info("Starting training process")
    logger.info(f"Data path: {data_path}")
    logger.info(
        f"Config: epochs={config.epochs}, batch_size={config.batch_size}, lr={config.learning_rate}"
    )

    # Load text and create tokenizer
    raw_text = load_text(data_path)
    tokenizer = CharacterTokenizer()
    tokenizer.fit(raw_text)

    model_cfg = ModelConfig(vocab_size=len(tokenizer.itos), d_model=128)
    model = TransformerModel(model_cfg)
    logger.info(
        f"Created model with vocab_size={model_cfg.vocab_size}, d_model={model_cfg.d_model}"
    )

    train_loader, val_loader = build_dataloaders(
        raw_text, tokenizer, config.max_context, config.batch_size
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Using device: {device}")

    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    for epoch in range(config.epochs):
        logger.info(f"Starting epoch {epoch + 1}/{config.epochs}")

        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss = validate(model, val_loader, loss_fn, device)
        logger.info(
            f"Epoch {epoch + 1} completed - train loss: {train_loss:.4f}, val loss: {val_loss:.4f}"
        )

        if (epoch + 1) % config.sample_frequency == 0:
            # Generate sample text for monitoring
            logger.info("Generating sample text...")
            prompt = torch.tensor(
                [[tokenizer.stoi[" "]]], dtype=torch.long, device=device
            )
            sample_ids = model.generate(
                prompt, max_new_tokens=50, temperature=config.temperature
            )
            sample_text = tokenizer.decode(sample_ids[0].tolist())
            logger.info(f"Sample: {sample_text}")

    logger.info("Training completed successfully")
    return model
