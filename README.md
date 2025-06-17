# 🧠 Python Project: `tinyllm_py`

A small-scale educational language model implementation in **Python + PyTorch**, built to mirror the structure and learning goals of the C# version.

---

## ⚙️ Training Configuration

### `TrainingConfig` (in `src/utils/config.py`)

Use `@dataclass` to define all hyperparameters:

```python
@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    max_context: int
    gradient_clip: float
    val_split: float
    sample_frequency: int
    temperature: float
    console_max_tokens: int
    console_temperature: float
    console_top_k: int
```

## 🔁 Training Loop

### Core Loop (`src/training/train.py`)

* For each epoch:
  * Iterate over batches
  * Compute loss (`torch.nn.CrossEntropyLoss`)
  * Apply gradient clipping
  * Backpropagation (`loss.backward()`)
  * Optimizer step (`torch.optim.Adam`)
* Validation split for monitoring
* Sample generation at intervals
* Checkpointing and early stopping
* Save best model artifacts

## 🎮 Interactive Console

### Features
* Real-time text generation
* Adjustable parameters (temperature, max tokens, top-k)
* Command interface for configuration
* Model loading from saved artifacts

### Usage
```bash
python scripts/console.py [model_directory]
```

## 🚀 Getting Started

1. **Train on Shakespeare dataset:**
   ```bash
   python scripts/train_shakespeare.py
   ```

2. **Launch interactive console:**
   ```bash
   python scripts/console.py
   ```

## 📁 Project Structure

```
TinyPythonLLM/
├── src/
│   ├── console/          # Interactive console
│   ├── models/           # Transformer implementation
│   ├── tokenization/     # Character tokenizer
│   ├── training/         # Training utilities
│   └── utils/           # Configuration and logging
├── scripts/             # Training and console scripts
├── data/               # Training datasets
└── logs/               # Training logs
```

---

## 📦 Dependencies

**Minimum Python version**: `>=3.9`

Add the following to `requirements.txt`:

```txt
torch>=2.0
numpy>=1.23
tqdm>=4.65        # Optional, for progress bars
```

---

## 🔤 Tokenization

### `CharacterTokenizer`

* Maps each character to a unique integer and vice versa.
* Supports:

  * `fit(text: str)`
  * `encode(text: str) -> List[int]`
  * `decode(ids: List[int]) -> str`
  * `save_state(filepath)`
  * `load_state(filepath)`

Mirrors the C# `CharacterTokenizer` for interoperability and conceptual parity.

---

## 📚 Dataset Handling

### Dataset Loader (in `src/training/data_loader.py`)

* Loads and preprocesses raw text:

  * Lowercasing
  * Whitespace normalization
* Splits data into training and validation sets.
* Supports:

  * Context windows
  * Batching (via `torch.utils.data.Dataset` + `DataLoader` or generator)

---

## 🧠 Model

### `TransformerModel` (in `src/models/`)

* Minimal transformer architecture:

  * Token embedding
  * Positional encoding
  * N transformer blocks
  * Multi-head attention
  * Feed-forward layers
  * Output projection
* Inspired by `ModelFactory.CreateTinyModel` (C# equivalent)
* Configurable via passed-in `TrainingConfig` or JSON

---

## ⚙️ Training Configuration

### `TrainingConfig` (in `src/utils/config.py`)

Use `@dataclass` to define all hyperparameters:

```python
@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    max_context: int
    gradient_clip: float
    val_split: float
    sample_frequency: int
    temperature: float
    console_max_tokens: int
    console_temperature: float
    console_top_k: int
```

## 🔁 Training Loop

### Core Loop (`src/training/train.py`)

* For each epoch:
  * Iterate over batches
  * Compute loss (`torch.nn.CrossEntropyLoss`)
  * Apply gradient clipping
  * Backpropagation (`loss.backward()`)
  * Optimizer step (`torch.optim.Adam`)
* Validation split for monitoring
* Sample generation at intervals
* Checkpointing and early stopping
* Save best model artifacts

## 🎮 Interactive Console

### Features
* Real-time text generation
* Adjustable parameters (temperature, max tokens, top-k)
* Command interface for configuration
* Model loading from saved artifacts

### Usage
```bash
python scripts/console.py [model_directory]
```

## 🚀 Getting Started

1. **Train on Shakespeare dataset:**
   ```bash
   python scripts/train_shakespeare.py
   ```

2. **Launch interactive console:**
   ```bash
   python scripts/console.py
   ```

## 📁 Project Structure

```
TinyPythonLLM/
├── src/
│   ├── console/          # Interactive console
│   ├── models/           # Transformer implementation
│   ├── tokenization/     # Character tokenizer
│   ├── training/         # Training utilities
│   └── utils/           # Configuration and logging
├── scripts/             # Training and console scripts
├── data/               # Training datasets
└── logs/               # Training logs
```

---
