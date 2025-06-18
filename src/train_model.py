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

import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from .transformer import Transformer
from .character_tokenizer import CharacterTokenizer
from .logger import get_logger
from .dataclass_config import TrainingConfig, ModelConfig
from .data_loader import build_dataloaders, load_text

logger = get_logger(__name__)


def train_epoch(
    model: Transformer, 
    train_loader: DataLoader, 
    optimizer: optim.Optimizer, 
    criterion: nn.Module, 
    device: torch.device,
    scaler: Optional[GradScaler] = None
) -> float:
    """Train for one epoch with optional mixed precision."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass with optional autocast
        if scaler:
            with autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 1000 == 0:
            lr = optimizer.param_groups[0]["lr"]
            logger.info(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, LR: {lr:.6f}")
    
    return total_loss / num_batches


def evaluate(
    model: Transformer, 
    val_loader: DataLoader, 
    criterion: nn.Module, 
    device: torch.device,
    scaler: Optional[GradScaler] = None
) -> float:
    """Evaluate the model with optional mixed precision."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            if scaler:
                with autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            else:
                outputs = model(inputs)
                loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def run_training(data_path: str, config: TrainingConfig, save_dir: Optional[str] = None) -> Transformer:
    """Main training function."""
    # Setup device and paths
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    save_path = None
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    logger.info(f"Loading data from {data_path}")
    text = load_text(data_path)
    logger.info(f"Loaded {len(text)} characters")
    
    # Extract dataset name from file path for model naming
    dataset_name = Path(data_path).stem  # Gets filename without extension
    
    # Build tokenizer
    tokenizer = CharacterTokenizer()
    tokenizer.fit(text)
    logger.info(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create model config
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
    
    # Build model and training components
    model = Transformer(model_config).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.01,
        eps=1e-8,
        betas=(0.9, 0.95),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.max_epochs, eta_min=config.learning_rate * 0.1
    )
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler('cuda') if device.type == "cuda" else None

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Starting training for {config.max_epochs} epochs...")

    # Training metrics
    metrics = {"losses": [], "val_losses": [], "epoch_times": []}

    for epoch in range(config.max_epochs):
        epoch_start = time.time()

        # Training
        avg_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        
        # Validation
        avg_val_loss = evaluate(model, val_loader, criterion, device, scaler)
        
        # Update scheduler
        scheduler.step()

        # Log metrics
        epoch_time = time.time() - epoch_start
        metrics["losses"].append(avg_loss)
        metrics["val_losses"].append(avg_val_loss)
        metrics["epoch_times"].append(epoch_time)

        logger.info(
            f"Epoch {epoch + 1}/{config.max_epochs} completed in {epoch_time:.2f}s - "
            f"train: {avg_loss:.4f}, val: {avg_val_loss:.4f}"
        )

    # Save model
    if save_path:
        model_filename = f"{dataset_name}_model.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "tokenizer": tokenizer,
                "config": model_config,
                "training_metrics": metrics,
                "dataset_name": dataset_name,
                "data_path": data_path,
            },
            save_path / model_filename,
        )
        logger.info(f"Training completed! Model saved to {save_path / model_filename}")

    return model