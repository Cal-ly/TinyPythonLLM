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
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

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
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(config.max_epochs):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Evaluate
        if epoch % config.eval_every == 0:
            val_loss = evaluate(model, val_loader, criterion, device)
            logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if save_dir and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'tokenizer': tokenizer,
                    'config': model_config,
                    'epoch': epoch,
                    'val_loss': val_loss
                }, save_path / 'best_model.pth')
                logger.info(f"Saved best model with val loss: {val_loss:.4f}")
        else:
            logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}")
        
        # Save checkpoint
        if save_dir and epoch % config.save_every == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'tokenizer': tokenizer,
                'config': model_config,
                'epoch': epoch,
                'train_loss': train_loss
            }, save_path / f'checkpoint_epoch_{epoch}.pth')
        
        epoch_time = time.time() - start_time
        logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
    
    return model
    train_loader, val_loader = build_dataloaders(
        text, tokenizer, config['max_seq_length'], 
        config['batch_size'], num_workers=4, pin_memory=True
    )
    
    # Setup optimizer with RTX 4070 Mobile optimized settings
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 0.01),
        eps=1e-8,
        betas=(0.9, 0.95)  # Optimized for transformer training
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['num_epochs'], eta_min=config['learning_rate'] * 0.1
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Training metrics
    training_metrics = {
        'epoch_times': [],
        'losses': [],
        'val_losses': [],
        'gpu_memory_usage': [],
        'throughput': []
    }
    
    print(f"Starting training for {config['num_epochs']} epochs...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(config['num_epochs']):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        tokens_processed = 0
        
        for batch_idx, (input_ids, target_ids) in enumerate(train_loader):
            input_ids = input_ids.to(trainer.device, non_blocking=True)
            target_ids = target_ids.to(trainer.device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            # Mixed precision training
            if trainer.scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids)
                    loss = criterion(outputs.view(-1, tokenizer.vocab_size), target_ids.view(-1))
                
                trainer.scaler.scale(loss).backward()
                trainer.scaler.step(optimizer)
                trainer.scaler.update()
            else:
                outputs = model(input_ids)
                loss = criterion(outputs.view(-1, tokenizer.vocab_size), target_ids.view(-1))
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            tokens_processed += input_ids.numel()
            
            # Log progress
            if batch_idx % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{config['num_epochs']}, "
                      f"Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, LR: {current_lr:.6f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for input_ids, target_ids in val_loader:
                input_ids = input_ids.to(trainer.device, non_blocking=True)
                target_ids = target_ids.to(trainer.device, non_blocking=True)
                
                if trainer.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = model(input_ids)
                        loss = criterion(outputs.view(-1, tokenizer.vocab_size), target_ids.view(-1))
                else:
                    outputs = model(input_ids)
                    loss = criterion(outputs.view(-1, tokenizer.vocab_size), target_ids.view(-1))
                
                val_loss += loss.item()
                val_batches += 1
        
        scheduler.step()
        
        # Calculate metrics
        epoch_time = time.time() - epoch_start_time
        avg_loss = epoch_loss / num_batches
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        throughput = tokens_processed / epoch_time
        
        # GPU memory usage
        if trainer.device.type == 'cuda':
            gpu_memory = torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()
        else:
            gpu_memory = 0
        
        # Store metrics
        training_metrics['epoch_times'].append(epoch_time)
        training_metrics['losses'].append(avg_loss)
        training_metrics['val_losses'].append(avg_val_loss)
        training_metrics['gpu_memory_usage'].append(gpu_memory)
        training_metrics['throughput'].append(throughput)
        
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
        print(f"Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"Throughput: {throughput:.0f} tokens/sec, GPU Memory: {gpu_memory:.1f}GB")
        print("-" * 50)
    
    # Save model and metrics
    save_dir = Path(config.get('save_dir', 'models'))
    save_dir.mkdir(exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer': tokenizer,
        'config': config,
        'training_metrics': training_metrics
    }, save_dir / 'shakespeare_model.pt')
    
    print(f"Training completed! Model saved to {save_dir}")
    print(f"Average throughput: {sum(training_metrics['throughput'])/len(training_metrics['throughput']):.0f} tokens/sec")
    
    return training_metrics

if __name__ == "__main__":
    # ...existing code...
        # Only log training progress every 10th epoch
        if (epoch + 1) % config.log_frequency == 0:
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
        if (epoch + 1) % config.checkpoint_frequency == 0:
            save_checkpoint(model, tokenizer, optimizer, scheduler, epoch, train_loss, val_loss, config)
        
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
