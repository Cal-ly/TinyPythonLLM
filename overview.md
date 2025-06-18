# Repository Overview: TinyPythonLLM

TinyPythonLLM is a minimal character-level language model implemented in Python using PyTorch. The project mirrors the structure of a companion C# version and demonstrates how to train and interact with a transformer-based model.

## Directory Structure

```
TinyPythonLLM/
├── data/               # Sample text corpora for training
├── scripts/            # Command line entry points
├── src/                # Library source code
│   ├── console/        # Interactive console implementation
│   ├── inference/      # Helper for text generation
│   ├── models/         # Transformer architecture
│   ├── tokenization/   # Character tokenizer
│   ├── training/       # Data loader and training loop
│   └── utils/          # Config and logging helpers
├── trained_models/     # Saved checkpoints
└── ...
```

## High Level Components

### `src/tokenization/character_tokenizer.py`
Defines the **`CharacterTokenizer`** class. It maps characters to integer ids and vice versa, with methods `fit`, `encode`, `decode`, `save_state`, and `load_state`【F:src/tokenization/character_tokenizer.py†L14-L48】.
Other modules rely on this tokenizer for preprocessing text.

### `src/models/transformer.py`
Implements a minimal Transformer. Key classes:
* **`MultiHeadAttention`** – scaled dot-product attention with dropout
* **`TransformerBlock`** – attention + feed‑forward with residuals
* **`Transformer`** – token and positional embeddings, a stack of blocks, and generation logic【F:src/models/transformer.py†L26-L183】.
Uses configuration from `utils.config.ModelConfig` and logs via `utils.logger`.

### `src/training/data_loader.py`
Provides `OptimizedTextDataset` and `build_dataloaders` to load text, tokenize it, and return `DataLoader` objects for training and validation【F:src/training/data_loader.py†L18-L89】.

### `src/training/train.py`
Contains the main training routine `run_training`, along with helper functions `train_epoch` and `evaluate` for one epoch and validation respectively【F:src/training/train.py†L16-L203】. The function builds the tokenizer, dataloaders, model, optimizer, and scheduler, then iterates over epochs and saves the trained checkpoint.

### `src/console/interactive.py`
Provides `TinyLLMConsole`, an interactive REPL for generating text with a trained model. It loads a checkpoint, accepts user commands, and prints generated sequences【F:src/console/interactive.py†L1-L188】.

### `src/inference/generate.py`
Utility for programmatic generation. Functions `load_model` and `generate_text` load the checkpoint and produce output given a prompt【F:src/inference/generate.py†L1-L31】.

### `src/utils/config.py`
Defines dataclasses **`ModelConfig`** and **`TrainingConfig`**, specifying hyperparameters such as model depth, sequence length, learning rate, and console settings【F:src/utils/config.py†L1-L46】.

### `src/utils/logger.py`
Centralized logger setup used across the project. `setup_logger`, `get_logger`, and `configure_external_loggers` configure rotating file logs and optional console output【F:src/utils/logger.py†L1-L112】.

## Scripts

* **`scripts/train.py`** – command line interface to start model training using `run_training` and `TrainingConfig`【F:scripts/train.py†L1-L31】.
* **`scripts/console.py`** – entry point that launches the interactive console from `src.console.interactive`【F:scripts/console.py†L1-L13】.

## Data and Checkpoints

The `data/` directory contains sample Shakespeare text files used for experiments, while `trained_models/` stores saved checkpoints such as `shakespeare_model.pt`.

## Dependencies

Listed in `requirements.txt`, the project primarily depends on `torch`, `numpy`, and `tqdm`【F:requirements.txt†L1-L4】. The optional `setup.sh` script creates a virtual environment and installs these packages【F:setup.sh†L1-L10】.

## Interactions Between Modules

- **Training Pipeline:**
  1. `training.train.run_training` loads text via `data_loader.load_text`, builds a `CharacterTokenizer`, and constructs the `Transformer` with a `ModelConfig`.
  2. `data_loader.build_dataloaders` provides `DataLoader` objects used by `train_epoch` and `evaluate`.
  3. After training, the checkpoint (model state, tokenizer, config) is saved to `trained_models/` and later consumed by the console or inference utilities.

- **Console and Inference:**
  - `console.interactive.TinyLLMConsole` and `inference.generate` both load checkpoints, decode user prompts with `CharacterTokenizer`, and generate tokens via `Transformer.generate`.

Logging is handled through `utils.logger`, which is imported by most modules to keep output consistent.

---
This overview outlines the purpose of each major file and how they connect, giving both developers and language models a quick reference to navigate the repository.
