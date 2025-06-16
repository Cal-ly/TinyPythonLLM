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

import logging
from pathlib import Path
from typing import Iterable

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from .data_loader import build_dataloaders, load_text
from ..models.transformer import ModelConfig, TransformerModel
from ..tokenization.character_tokenizer import CharacterTokenizer
from ..utils.config import TrainingConfig

logger = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename="tinyllm_training.log",
    )


def train_epoch(
    model: TransformerModel, data_loader: Iterable, loss_fn, optimizer, device
) -> float:
    model.train()
    total_loss = 0.0
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)

        logits, _ = model(x)
        # Shift targets so each position predicts the next token
        loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(data_loader)


def validate(model: TransformerModel, data_loader: Iterable, loss_fn, device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            logits, _ = model(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
    return total_loss / len(data_loader)


def run_training(data_path: Path, config: TrainingConfig) -> TransformerModel:
    setup_logging()

    # Load text and create tokenizer
    raw_text = load_text(data_path)
    tokenizer = CharacterTokenizer()
    tokenizer.fit(raw_text)

    model_cfg = ModelConfig(vocab_size=len(tokenizer.itos), d_model=128)
    model = TransformerModel(model_cfg)

    train_loader, val_loader = build_dataloaders(
        raw_text, tokenizer, config.max_context, config.batch_size
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    for epoch in range(config.epochs):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss = validate(model, val_loader, loss_fn, device)
        logger.info(
            "Epoch %d - train loss %.4f - val loss %.4f",
            epoch + 1,
            train_loss,
            val_loss,
        )

        if (epoch + 1) % config.sample_frequency == 0:
            # Generate sample text for monitoring
            prompt = torch.tensor(
                [[tokenizer.stoi[" "]]], dtype=torch.long, device=device
            )
            sample_ids = model.generate(
                prompt, max_new_tokens=50, temperature=config.temperature
            )
            logger.info("Sample: %s", tokenizer.decode(sample_ids[0].tolist()))

    return model
