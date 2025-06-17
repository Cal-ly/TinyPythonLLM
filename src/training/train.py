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

import os
from pathlib import Path
from typing import Iterable

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler

from .data_loader import build_dataloaders, load_text
from ..models.transformer import ModelConfig, TransformerModel
from ..tokenization.character_tokenizer import CharacterTokenizer
from ..utils.config import TrainingConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


def save_model_artifacts(
    model: TransformerModel, 
    tokenizer: CharacterTokenizer, 
    config: TrainingConfig, 
    save_dir: str = "saved_models"
) -> None:
    """Save model, tokenizer, and configuration."""
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    # Save model state dict
    model_path = save_path / "model.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved model to {model_path}")
    
    # Save tokenizer
    tokenizer_path = save_path / "tokenizer.json"
    tokenizer.save_state(str(tokenizer_path))
    logger.info(f"Saved tokenizer to {tokenizer_path}")
    
    # Save model configuration
    config_path = save_path / "model_config.pt"
    torch.save({
        'vocab_size': model.config.vocab_size,
        'd_model': model.config.d_model,
        'n_heads': model.config.n_heads,
        'num_layers': model.config.num_layers,
        'dropout': model.config.dropout
    }, config_path)
    logger.info(f"Saved model config to {config_path}")
    
    # Save training configuration
    training_config_path = save_path / "training_config.pt"
    torch.save({
        'epochs': config.epochs,
        'batch_size': config.batch_size,
        'learning_rate': config.learning_rate,
        'max_context': config.max_context,
        'd_model': getattr(config, 'd_model', 128),
        'n_heads': getattr(config, 'n_heads', 4),
        'num_layers': getattr(config, 'num_layers', 2),
        'sample_frequency': config.sample_frequency,
        'temperature': config.temperature,
        'max_new_tokens': config.max_new_tokens
    }, training_config_path)
    logger.info(f"Saved training config to {training_config_path}")


def load_model_artifacts(save_dir: str = "saved_models"):
    """Load saved model, tokenizer, and configuration."""
    save_path = Path(save_dir)
    
    # Load model configuration
    config_path = save_path / "model_config.pt"
    model_config_data = torch.load(config_path)
    model_config = ModelConfig(**model_config_data)
    
    # Load tokenizer
    tokenizer = CharacterTokenizer()
    tokenizer_path = save_path / "tokenizer.json"
    tokenizer.load_state(str(tokenizer_path))
    
    # Create and load model
    model = TransformerModel(model_config)
    model_path = save_path / "model.pt"
    model.load_state_dict(torch.load(model_path))
    
    # Load training configuration
    training_config_path = save_path / "training_config.pt"
    training_config_data = torch.load(training_config_path)
    
    logger.info(f"Loaded model artifacts from {save_dir}")
    return model, tokenizer, model_config, training_config_data


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
    logger.info(f"Loaded text with {len(raw_text)} characters")
    
    tokenizer = CharacterTokenizer()
    tokenizer.fit(raw_text)

    # Use config parameters for model architecture
    model_cfg = ModelConfig(
        vocab_size=len(tokenizer.itos), 
        d_model=getattr(config, 'd_model', 128),
        n_heads=getattr(config, 'n_heads', 4),
        num_layers=getattr(config, 'num_layers', 2)
    )
    model = TransformerModel(model_cfg)
    logger.info(
        f"Created model with {sum(p.numel() for p in model.parameters())} parameters"
    )

    train_loader, val_loader = build_dataloaders(
        raw_text, tokenizer, config.max_context, config.batch_size
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Using device: {device}")

    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    
    # Add learning rate scheduler - remove verbose parameter
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=10
    )

    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.epochs):
        # Only log epoch start every 10th epoch
        if (epoch + 1) % 10 == 0:
            logger.info(f"Starting epoch {epoch + 1}/{config.epochs}")

        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss = validate(model, val_loader, loss_fn, device)
        
        # Step scheduler
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            
        # Only log training progress every 10th epoch
        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(
                f"Epoch {epoch + 1} - train: {train_loss:.4f}, val: {val_loss:.4f}, lr: {current_lr:.6f}"
            )

        if (epoch + 1) % config.sample_frequency == 0:
            # Generate sample text for monitoring
            logger.info("Generating sample text...")
            
            try:
                # Use better starting prompts from actual vocabulary
                common_starts = []
                for char in ['t', 'h', 'a', 'w', 'i', 'o', 's', 'b', 'c', 'm']:
                    if char in tokenizer.stoi:
                        common_starts.append(char)
                
                if not common_starts:
                    common_starts = [tokenizer.itos[0]]  # fallback to first char
                
                start_char = common_starts[0]  # Use most common
                prompt = torch.tensor(
                    [[tokenizer.stoi[start_char]]], dtype=torch.long, device=device
                )
                
                # Generate with multiple attempts if needed
                for attempt in range(3):
                    sample_ids = model.generate(
                        prompt, max_new_tokens=50, temperature=0.7
                    )
                    sample_text = tokenizer.decode(sample_ids[0].tolist())
                    
                    # Check if we got meaningful output
                    if len(sample_text.strip()) > len(start_char):
                        logger.info(f"Sample (attempt {attempt+1}): '{sample_text}'")
                        break
                    else:
                        logger.warning(f"Attempt {attempt+1} produced minimal output: '{sample_text}'")
                
            except Exception as e:
                logger.error(f"Failed to generate sample text: {e}")
        
        # Save checkpoints periodically
        if (epoch + 1) % 50 == 0:
            checkpoint_path = f"checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'tokenizer_vocab': tokenizer.itos,
                'config': config
            }, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Early stopping
        if patience_counter > 20:
            logger.info("Early stopping triggered")
            break

    logger.info("Training completed successfully")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    
    # Save final model artifacts
    logger.info("Saving final model artifacts...")
    save_model_artifacts(model, tokenizer, config)
    
    return model
