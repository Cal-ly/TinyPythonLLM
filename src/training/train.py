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

# Add src to path if running from project root
current_dir = Path(__file__).parent
src_dir = current_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from models.transformer import Transformer
from tokenization.character_tokenizer import CharacterTokenizer
from .data_loader import build_dataloaders, load_text

class GPUOptimizedTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = self._setup_device()
        self.scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None
        self.compile_model = config.get('compile_model', True)
        
    def _setup_device(self):
        """Setup device with RTX 4070 Mobile optimizations"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            
            # RTX 4070 Mobile specific optimizations
            torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
            torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for better performance
            torch.backends.cudnn.allow_tf32 = True
            
            # Set memory fraction to leave room for system
            torch.cuda.set_per_process_memory_fraction(0.9)
            
            print(f"Using GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            
            return device
        else:
            print("CUDA not available, using CPU")
            return torch.device('cpu')
    
    def _get_optimal_batch_size(self, model, vocab_size: int, seq_length: int):
        """Dynamically determine optimal batch size for RTX 4070 Mobile"""
        if self.device.type != 'cuda':
            return self.config.get('batch_size', 32)
        
        # Start with a reasonable batch size for RTX 4070 Mobile
        test_batch_sizes = [64, 48, 32, 24, 16, 12, 8]
        
        for batch_size in test_batch_sizes:
            try:
                # Test memory usage with dummy data
                dummy_input = torch.randint(0, vocab_size, (batch_size, seq_length)).to(self.device)
                dummy_target = torch.randint(0, vocab_size, (batch_size, seq_length)).to(self.device)
                
                model.train()
                with torch.cuda.amp.autocast():
                    output = model(dummy_input)
                    loss = nn.CrossEntropyLoss()(
                        output.view(-1, vocab_size), 
                        dummy_target.view(-1)
                    )
                
                # Test backward pass
                self.scaler.scale(loss).backward()
                
                # Check memory usage
                memory_used = torch.cuda.memory_allocated() / 1e9
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                
                print(f"Batch size {batch_size}: {memory_used:.1f}GB / {memory_total:.1f}GB used")
                
                if memory_used < memory_total * 0.85:  # Leave 15% headroom
                    model.zero_grad()
                    torch.cuda.empty_cache()
                    return batch_size
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            finally:
                model.zero_grad()
                torch.cuda.empty_cache()
        
        return 8  # Fallback minimum
    
    def _setup_model_optimizations(self, model):
        """Apply RTX 4070 Mobile specific model optimizations"""
        if self.device.type == 'cuda':
            # Enable gradient checkpointing to save memory
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
            
            # Compile model for better performance (PyTorch 2.0+)
            if self.compile_model and hasattr(torch, 'compile'):
                try:
                    model = torch.compile(model, mode='max-autotune')
                    print("Model compiled for optimized performance")
                except Exception as e:
                    print(f"Model compilation failed: {e}")
        
        return model

def create_model(config: Dict[str, Any], vocab_size: int, device: torch.device) -> nn.Module:
    """Create and optimize model for RTX 4070 Mobile"""
    model = Transformer(
        vocab_size=vocab_size,
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config.get('dim_feedforward', 4 * config['d_model']),
        dropout=config.get('dropout', 0.1),
        max_seq_length=config['max_seq_length']
    ).to(device)
    
    return model

def run_training(config: Dict[str, Any]) -> Dict[str, Any]:
    """Main training function with RTX 4070 Mobile optimizations"""
    trainer = GPUOptimizedTrainer(config)
    
    # Load and prepare data
    print("Loading training data...")
    text = load_text(config['data_path'])
    tokenizer = CharacterTokenizer()
    tokenizer.fit(text)
    
    # Build data loaders with optimal batch size
    model = create_model(config, tokenizer.vocab_size, trainer.device)
    
    # Determine optimal batch size
    optimal_batch_size = trainer._get_optimal_batch_size(
        model, tokenizer.vocab_size, config['max_seq_length']
    )
    config['batch_size'] = optimal_batch_size
    print(f"Using optimal batch size: {optimal_batch_size}")
    
    # Apply model optimizations
    model = trainer._setup_model_optimizations(model)
    
    # Build dataloaders with optimized settings
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
