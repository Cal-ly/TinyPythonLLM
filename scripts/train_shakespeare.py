"""
Training script for TinyPythonLLM on Shakespeare dataset.
"""

import os
import sys
from pathlib import Path

# Add src directory to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
src_dir = project_root / "src"

if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from training.train import run_training

def main():
    # RTX 4070 Mobile optimized configuration
    config = {
        # Data settings
        'data_path': project_root / 'data' / 'shakespeare.txt',
        
        # Model architecture - optimized for RTX 4070 Mobile
        'd_model': 512,              # Good balance of capacity and memory usage
        'nhead': 8,                  # Efficient for 512 dimensions
        'num_layers': 8,             # Increased from 6 for better performance
        'dim_feedforward': 2048,     # 4x d_model standard ratio
        'dropout': 0.1,
        'max_seq_length': 256,       # Optimized for memory efficiency
        
        # Training settings
        'num_epochs': 50,
        'learning_rate': 3e-4,       # Optimal for transformer training
        'weight_decay': 0.01,
        'batch_size': 32,            # Will be auto-optimized
        
        # GPU optimizations
        'compile_model': True,       # Enable PyTorch 2.0 compilation
        'mixed_precision': True,     # Enable automatic mixed precision
        
        # Paths
        'save_dir': project_root / 'models',
    }
    
    print("Starting Shakespeare training with RTX 4070 Mobile optimizations...")
    print(f"Configuration: {config}")
    
    try:
        metrics = run_training(config)
        print("Training completed successfully!")
        print(f"Final training loss: {metrics['losses'][-1]:.4f}")
        print(f"Final validation loss: {metrics['val_losses'][-1]:.4f}")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise

if __name__ == "__main__":
    main()