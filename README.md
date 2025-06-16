# 🧠 Python Project: `tinyllm_py`

A small-scale educational language model implementation in **Python + PyTorch**, built to mirror the structure and learning goals of the C# version.

---

## 📁 Project Structure

```
tinyllm_py/
├── data/                        # Raw and preprocessed datasets (e.g., shakespeare.txt)
├── src/
│   ├── tokenization/            # CharacterTokenizer (fit, encode, decode, save/load state)
│   ├── models/                  # TransformerModel and layers (PyTorch modules)
│   ├── training/                # Training loop, optimizer setup, validation
│   ├── utils/                   # Logging, config loading, checkpointing
├── scripts/
│   └── train_shakespeare.py     # Main entry point for training
├── tests/                       # Unit tests for tokenizer, model, training loop
├── requirements.txt
└── README.md
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
    val_frequency: int
    sample_frequency: int
    temperature: float
    metrics_csv_enabled: bool
```

---

## 🔁 Training Loop

### Core Loop (`src/training/train.py`)

* For each epoch:

  * Iterate over batches
  * Compute loss (`torch.nn.CrossEntropyLoss`)
  * Backpropagation (`loss.backward()`)
  * Optimizer step (`torch.optim.Adam`)
* Sample generation at intervals (controlled by config)
* Validation using a fixed set of batches
* Optional early stopping
* Save best model via checkpoints

---

## 🧾 Logging

* Uses Python's built-in `logging` module
* Logs to `tinyllm_training.log` by default
* Optional `MetricsLogger` writes CSV-formatted metrics for plotting/tracking

---

## ✍️ Text Generation

* After training epochs (or per config), generate text from a fixed prompt
* Sample using softmax temperature scaling
* Used to monitor training quality in human-readable form

---

## 🚀 Training Script

### `scripts/train_shakespeare.py`

* Loads dataset from `data/`
* Builds tokenizer and model
* Initializes training config
* Starts training loop
* Logs results, saves model, and prints sample outputs

---

## 📖 Documentation

### `README.md` should include:

* Setup instructions:

  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```
* How to run training:

  ```bash
  python scripts/train_shakespeare.py
  ```
* Where to find logs:

  * Training: `tinyllm_training.log`
  * Samples: printed to stdout or file
  * Metrics: optional `metrics.csv`
* Expected runtimes:

  * Character-level model with Shakespeare: \~30–60 min per epoch
  * Code datasets (future): \~1–3 hours depending on size

---

## 🔮 Future Extensions

* Add Byte Pair Encoding (BPE) or WordPiece tokenizer
* Quantization for model compression
* FP16 training
* Larger transformer variants
* Fine-tuning with your own data

---

Would you like a cookiecutter template or zipped repo skeleton based on this layout?
