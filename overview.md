# Repository Overview: TinyPythonLLM

TinyPythonLLM is a minimal character-level language model implemented in Python using PyTorch. This project demonstrates how to train and interact with a transformer-based model for educational purposes.

## Directory Structure

```
TinyPythonLLM/
├── data/               # Sample text corpora for training
├── scripts/            # Entry point scripts
│   ├── start_training.py  # Training script with argument parsing
│   └── start_console.py   # Interactive console launcher
├── src/                # Library source code
│   ├── __init__.py     # Package initialization
│   ├── character_tokenizer.py    # Character-level tokenization
│   ├── data_loader.py  # Dataset utilities and data loading
│   ├── dataclass_config.py       # Configuration dataclasses
│   ├── generate_text.py           # Text generation utilities
│   ├── interactive.py  # Interactive console interface
│   ├── logger.py       # Centralized logging utility
│   ├── train_model.py  # Training loop and utilities
│   └── transformer.py  # Core transformer implementation
├── trained_models/     # Saved checkpoints
└── ...
```

## Entry Point Scripts

### `scripts/start_training.py`
Command-line interface for training models with:
* **Argument parsing** – supports all training parameters via command line
* **Path resolution** – handles relative/absolute paths for data files
* **Error handling** – validates data files exist and shows available alternatives
* **Model naming** – automatically names models after dataset (e.g., `shakespeare25k.txt` → `shakespeare25k_model.pt`)
* **Progress display** – shows training configuration and expected output location

Usage patterns:
```bash
python scripts/start_training.py data/shakespeare25k.txt
python scripts/start_training.py data/my_text.txt --epochs 10 --batch_size 64
```

### `scripts/start_console.py` 
Interactive console launcher with intelligent model discovery:
* **Auto-discovery** – finds models in `trained_models/` when no argument provided
* **Smart path resolution** – handles model filenames, full paths, and directory arguments
* **Multi-location search** – searches `trained_models/`, `models/`, current directory
* **Clear feedback** – shows what it's looking for and where it found models
* **Error handling** – graceful failures with helpful error messages

Usage patterns:
```bash
python scripts/start_console.py                    # Auto-discover
python scripts/start_console.py shakespeare25k_model.pt    # By filename
python scripts/start_console.py trained_models/    # By directory
python scripts/start_console.py /path/to/model.pt  # Full path
```

## High Level Components

### `src/character_tokenizer.py`
Defines the **`CharacterTokenizer`** class that maps characters to integer ids and vice versa. Provides methods for `fit`, `encode`, `decode`, `save_state`, and `load_state` to handle character-level tokenization for the language model.

### `src/transformer.py`
Implements the core Transformer architecture with key classes:
* **`MultiHeadAttention`** – scaled dot-product attention with multiple heads and dropout
* **`TransformerBlock`** – combines attention and feed-forward layers with residual connections
* **`Transformer`** – complete model with token/positional embeddings, transformer blocks, and text generation capabilities
Uses causal masking for autoregressive generation and includes optimized generation methods.

### `src/data_loader.py`
Provides data loading utilities including:
* **`OptimizedTextDataset`** – efficient dataset backed by tensors for sequence slicing
* **`build_dataloaders`** – creates training and validation DataLoaders from raw text
* **`load_text`** – robust text loading with multiple encoding fallbacks

### `src/train_model.py`
Contains the complete training pipeline with:
* **`run_training`** – main training function that orchestrates the entire process
* **`train_epoch`** and **`evaluate`** – training and validation loops with mixed precision support
* Model checkpointing, metrics tracking, and learning rate scheduling

### `src/interactive.py`
Provides **`TinyLLMConsole`** class for interactive text generation:
* Loads trained models from multiple possible locations with intelligent path resolution
* Provides a command-line interface with helpful error messages and debugging info
* Supports temperature control, token limits, and various generation parameters
* Interactive commands for adjusting generation settings

### `src/generate_text.py`
Utility functions for programmatic text generation:
* **`load_model`** – loads saved model checkpoints
* **`generate_text`** – generates text from prompts using trained models

### `src/dataclass_config.py`
Defines configuration dataclasses:
* **`ModelConfig`** – model architecture parameters (vocab_size, d_model, num_heads, etc.)
* **`TrainingConfig`** – training hyperparameters and generation settings

### `src/logger.py`
Centralized logging system with:
* **`setup_logger`** and **`get_logger`** – configure rotating file logs and optional console output
* **`configure_external_loggers`** – reduces verbosity of external libraries
* Logs to both files and console with configurable formatting

## Package Structure

The `src/__init__.py` file provides convenient imports for the main classes:
- `Transformer` from transformer module
- `CharacterTokenizer` from character_tokenizer module  
- `ModelConfig` and `TrainingConfig` from dataclass_config module

## Dependencies

The project primarily depends on:
- `torch` – PyTorch for neural network implementation
- `numpy` – numerical computations
- Standard library modules for logging, data structures, and file I/O

## Interactions Between Modules

- **Training Pipeline:**
  1. `scripts/start_training.py` parses arguments and validates data files
  2. `train_model.run_training` loads text via `data_loader.load_text`
  3. Builds a `CharacterTokenizer` and constructs the `Transformer` with `ModelConfig`
  4. `data_loader.build_dataloaders` provides DataLoader objects for training
  5. Training loop uses mixed precision and saves checkpoints with model state, tokenizer, and metrics
  6. Models are automatically named after the dataset (e.g., `shakespeare25k_model.pt`)

- **Interactive Usage:**
  1. `scripts/start_console.py` parses arguments and discovers model locations
  2. `interactive.TinyLLMConsole` loads saved checkpoints from discovered locations
  3. Automatically searches common directories with intelligent path resolution
  4. Uses `CharacterTokenizer` to encode/decode text
  5. Generates text via `Transformer.generate` method

- **Programmatic Generation:**
  - `generate_text` module provides simple functions to load models and generate text
  - Can be used for batch processing or integration into other applications

## Workflow Improvements

The entry point scripts provide:
- **Consistent module loading** – both use the same `sys.path` approach
- **Intelligent path handling** – resolve relative paths, search multiple locations
- **User-friendly feedback** – clear output about what's happening
- **Error recovery** – helpful suggestions when things go wrong
- **Flexible usage** – support both simple and advanced use cases

All modules use the centralized `logger` for consistent output formatting and log management.

---
This overview reflects the improved script architecture and the actual implementation of the TinyPythonLLM project with its intelligent model discovery and user-friendly command-line interface.
