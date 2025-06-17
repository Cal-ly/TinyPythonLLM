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

# Now import from the training module
from training.train import run_training
from utils.config import TrainingConfig
from utils.logger import get_logger

def main():
    # Set up logging
    logger = get_logger(__name__)
    logger.info("Starting Shakespeare training")
    
    # Paths
    data_path = project_root / "data" / "shakespeare.txt"
    save_dir = project_root / "models" / "shakespeare"
    
    # Verify data file exists
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return
    
    # Training configuration
    config = TrainingConfig(
        batch_size=16,
        learning_rate=1e-3,
        max_epochs=50,
        sequence_length=128,
        d_model=256,
        num_heads=8,
        num_layers=4,
        dropout=0.1,
        save_every=5,
        eval_every=2,
        device='cuda' if os.name == 'nt' else 'cpu'  # Use CUDA on Windows if available
    )
    
    logger.info(f"Training config: {config}")
    
    # Run training
    try:
        model = run_training(str(data_path), config, str(save_dir))
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()