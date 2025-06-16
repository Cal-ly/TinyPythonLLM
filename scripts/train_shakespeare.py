"""Train a tiny character-level model on Shakespeare text."""

import sys
from pathlib import Path

# Add the parent directory to Python path so it can import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger, configure_external_loggers
from src.training.train import run_training
from src.utils.config import TrainingConfig


def main():
    """Main entry point for training."""
    # Set up centralized logging
    logger = get_logger("train_shakespeare")
    configure_external_loggers()
    
    logger.info("Starting Shakespeare model training")
    
    # Path to the training data
    data_path = Path("data/shakespeare25k.txt")
    config = TrainingConfig()
    
    try:
        run_training(data_path, config)
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
