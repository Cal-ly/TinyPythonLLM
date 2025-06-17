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

import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

# Add src to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from models.transformer import Transformer
from tokenization.character_tokenizer import CharacterTokenizer
from utils.logger import get_logger
from utils.config import TrainingConfig, ModelConfig
from training.data_loader import build_dataloaders, load_text

logger = get_logger(__name__)

def train_epoch(model: Transformer, train_loader: DataLoader, optimizer: optim.Optimizer, 
                criterion: nn.Module, device: str) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Reshape for loss calculation
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 100 == 0:
            logger.info(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    return total_loss / num_batches

def evaluate(model: Transformer, val_loader: DataLoader, criterion: nn.Module, device: str) -> float:
    """Evaluate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

def run_training(data_path: str, config: TrainingConfig, save_dir: Optional[str] = None) -> Transformer:
    """Main training function."""
    # Setup
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    logger.info(f"Loading data from {data_path}")
    text = load_text(data_path)
    logger.info(f"Loaded {len(text)} characters")
    
    # Build tokenizer
    tokenizer = CharacterTokenizer()
    tokenizer.fit(text)
    logger.info(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Update config with vocab size
    config.vocab_size = tokenizer.vocab_size
    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        sequence_length=config.sequence_length,
        dropout=config.dropout
    )
    
    # Build dataloaders
    train_loader, val_loader = build_dataloaders(
        text, tokenizer, config.sequence_length, config.batch_size
    )
    logger.info(f"Created dataloaders: {len(train_loader)} train, {len(val_loader)} val batches")
    
    # Build model
    model = Transformer(model_config).to(device)

    # Optimizer tuned for RTX 4070 Mobile
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.01,
        eps=1e-8,
        betas=(0.9, 0.95),
    )

    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.max_epochs, eta_min=config.learning_rate * 0.1
    )

    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler() if device.type == "cuda" else None

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    metrics = {
        "epoch_times": [],
        "losses": [],
        "val_losses": [],
        "gpu_memory_usage": [],
        "throughput": [],
    }

    logger.info(f"Starting training for {config.max_epochs} epochs...")

    for epoch in range(config.max_epochs):
        epoch_start = time.time()

        # ----- Training -----
        model.train()
        epoch_loss = 0.0
        tokens_processed = 0
        num_batches = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if scaler:
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs.view(-1, tokenizer.vocab_size), targets.view(-1))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs.view(-1, tokenizer.vocab_size), targets.view(-1))
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            tokens_processed += inputs.numel()

            if batch_idx % 10 == 0:
                lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch {epoch + 1}/{config.max_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, LR: {lr:.6f}"
                )

        # ----- Validation -----
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                if scaler:
                    with autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs.view(-1, tokenizer.vocab_size), targets.view(-1))
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs.view(-1, tokenizer.vocab_size), targets.view(-1))

                val_loss += loss.item()
                val_batches += 1

        scheduler.step()

        # Metrics
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / num_batches
        avg_val_loss = val_loss / val_batches if val_batches else 0.0
        throughput = tokens_processed / epoch_time if epoch_time > 0 else 0.0

        if device.type == "cuda":
            gpu_mem = torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()
        else:
            gpu_mem = 0.0

        metrics["epoch_times"].append(epoch_time)
        metrics["losses"].append(avg_loss)
        metrics["val_losses"].append(avg_val_loss)
        metrics["gpu_memory_usage"].append(gpu_mem)
        metrics["throughput"].append(throughput)

        logger.info(
            f"Epoch {epoch + 1} completed in {epoch_time:.2f}s - train: {avg_loss:.4f}, val: {avg_val_loss:.4f}, throughput: {throughput:.0f} tokens/s, GPU mem: {gpu_mem:.1f}GB"
        )

    if save_dir:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "tokenizer": tokenizer,
                "config": model_config,
                "training_metrics": metrics,
            },
            save_path / "shakespeare_model.pt",
        )
        logger.info(f"Training completed! Model saved to {save_path}")

    return model
