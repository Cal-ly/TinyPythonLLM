"""Command line interface to train TinyPythonLLM."""

import argparse
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training.train import run_training
from utils.config import TrainingConfig
from utils.logger import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TinyPythonLLM")
    parser.add_argument("data", type=str, help="Path to training text file")
    parser.add_argument(
        "--output_dir", type=str, default="trained_models", help="Directory to save model"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("train_script", console_output=True)

    config = TrainingConfig()
    logger.info("Training config: %s", config)

    repo_root = Path(__file__).resolve().parent.parent
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = repo_root / data_path

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir

    run_training(str(data_path), config, str(output_dir))


if __name__ == "__main__":
    main()
