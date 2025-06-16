"""Train a tiny character-level model on Shakespeare text."""

import logging
from pathlib import Path

from src.training.train import run_training
from src.utils.config import TrainingConfig


def main():
    logging.basicConfig(level=logging.INFO)
    data_path = Path("data/shakespeare.txt")
    config = TrainingConfig()
    run_training(data_path, config)


if __name__ == "__main__":
    main()
