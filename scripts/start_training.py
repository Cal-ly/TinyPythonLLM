"""Command line interface to train TinyPythonLLM."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train_model import run_training
from src.dataclass_config import TrainingConfig
from src.logger import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TinyPythonLLM")
    parser.add_argument("data", type=str, help="Path to training text file")
    parser.add_argument(
        "--output_dir", type=str, default="trained_models", 
        help="Directory to save model (default: trained_models)"
    )
    parser.add_argument(
        "--epochs", type=int, default=5,
        help="Number of training epochs (default: 5)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Training batch size (default: 32)"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4,
        help="Learning rate (default: 3e-4)"
    )
    parser.add_argument(
        "--sequence_length", type=int, default=256,
        help="Sequence length (default: 256)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("train_script", console_output=True)

    # Create config with command line arguments
    config = TrainingConfig(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        sequence_length=args.sequence_length,
    )

    # Resolve paths
    data_path = Path(args.data).resolve()
    output_dir = Path(args.output_dir).resolve()

    # Check if data file exists
    if not data_path.exists():
        print(f"❌ Error: Data file not found: {data_path}")
        print(f"💡 Available files in data/:")
        data_dir = Path("data")
        if data_dir.exists():
            for file in data_dir.glob("*.txt"):
                print(f"   {file.name}")
        sys.exit(1)

    # Get dataset name for output model
    dataset_name = data_path.stem
    expected_output = output_dir / f"{dataset_name}_model.pt"

    print("🧠 TinyPythonLLM Training")
    print("=" * 40)
    print(f"📁 Input data: {data_path}")
    print(f"📊 Dataset: {dataset_name}")
    print(f"💾 Output model: {expected_output}")
    print(f"⚙️  Training config:")
    print(f"   Epochs: {config.max_epochs}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Sequence length: {config.sequence_length}")
    print("=" * 40)

    logger.info("Training config: %s", config)
    logger.info(f"Data path: {data_path}")
    logger.info(f"Output directory: {output_dir}")

    try:
        run_training(str(data_path), config, str(output_dir))
        print(f"\n🎉 Training complete!")
        print(f"📁 Model saved as: {expected_output}")
        print(f"🎮 Launch console: python scripts/start_console.py")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"❌ Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()