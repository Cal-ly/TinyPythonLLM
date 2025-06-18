# 🧠 TinyPythonLLM

A minimal educational language model implementation in **Python + PyTorch**, built to demonstrate transformer architecture and training principles.

## 🚀 Quick Start

### Option 1: Install as Package (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/TinyPythonLLM.git
cd TinyPythonLLM

# Install in development mode
pip install -e .

# Train a model
tinyllm-train data/shakespeare6k.txt

# Launch interactive console
tinyllm-console
```

### Option 2: Run Scripts Directly

```bash
# Clone the repository
git clone https://github.com/yourusername/TinyPythonLLM.git
cd TinyPythonLLM

# Install dependencies
pip install -r requirements.txt

# Train a model
python scripts/train.py data/shakespeare6k.txt

# Launch interactive console
python scripts/console.py
```

## 📦 Installation

### Dependencies

**Minimum Python version**: `>=3.9`

Install dependencies:

```bash
pip install torch>=2.0 numpy>=1.23 tqdm>=4.65
```

Or from requirements.txt:

```bash
pip install -r requirements.txt
```

### Development Installation

For development, install in editable mode:

```bash
pip install -e .
```

This allows you to modify the code and see changes immediately without reinstalling.

## 🎯 Usage Examples

### Training a Model

```bash
# Basic training
tinyllm-train data/shakespeare6k.txt

# Custom parameters
tinyllm-train data/shakespeare6k.txt --epochs 10 --batch_size 64 --learning_rate 1e-3

# Custom output directory
tinyllm-train data/shakespeare6k.txt --output_dir my_models
```

### Interactive Console

```bash
# Use default model location (trained_models/)
tinyllm-console

# Specify custom model directory
tinyllm-console path/to/my/model

# Or using Python directly
python -m tinyllm.console.interactive
```

### Programmatic Usage

```python
import torch
from tinyllm import Transformer, CharacterTokenizer, ModelConfig

# Load a trained model
from tinyllm.inference import load_model, generate_text

model, tokenizer = load_model("trained_models/shakespeare_model.pt")
generated = generate_text(model, tokenizer, "To be or not to be", max_tokens=100)
print(generated)
```

## ⚙️ Configuration

### Training Configuration

Customize training via `TrainingConfig`:

```python
from tinyllm.utils import TrainingConfig

config = TrainingConfig(
    batch_size=32,
    learning_rate=3e-4,
    max_epochs=10,
    sequence_length=256,
    d_model=512,
    num_heads=8,
    num_layers=6,
    dropout=0.1,
    device='cuda'  # or 'cpu'
)
```

### Model Architecture

Configure model via `ModelConfig`:

```python
from tinyllm.utils import ModelConfig

config = ModelConfig(
    vocab_size=65,  # Set automatically from data
    d_model=512,
    num_heads=8,
    num_layers=6,
    sequence_length=256,
    dropout=0.1
)
```

## 📁 Project Structure

```
TinyPythonLLM/
├── setup.py                 # Package installation
├── requirements.txt         # Dependencies
├── README.md                # This file
├── scripts/                # Backwards-compatible entry points
│   ├── train.py            # Training script wrapper
│   └── console.py          # Console script wrapper
├── src/                    # Main package source
│   ├── __init__.py         # Package exports
│   ├── console/            # Interactive console
│   │   ├── __init__.py
│   │   └── interactive.py
│   ├── inference/          # Text generation utilities
│   │   ├── __init__.py
│   │   └── generate.py
│   ├── models/             # Transformer implementation
│   │   ├── __init__.py
│   │   └── transformer.py
│   ├── scripts/            # CLI entry points
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── console.py
│   ├── tokenization/       # Character tokenizer
│   │   ├── __init__.py
│   │   └── character_tokenizer.py
│   ├── training/           # Training utilities
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── data_loader.py
│   └── utils/              # Configuration and logging
│       ├── __init__.py
│       ├── config.py
│       └── logger.py
├── data/                   # Training datasets
├── trained_models/         # Saved model checkpoints
└── logs/                   # Training logs
```

## 🔤 Tokenization

The `CharacterTokenizer` maps each character to a unique integer:

```python
from tinyllm import CharacterTokenizer

tokenizer = CharacterTokenizer()
tokenizer.fit("Hello world!")

# Encode text to integers
tokens = tokenizer.encode("Hello")  # [72, 101, 108, 108, 111]

# Decode integers to text
text = tokenizer.decode([72, 101, 108, 108, 111])  # "Hello"
```

## 🧠 Model Architecture

The `Transformer` implements a minimal GPT-style architecture:

- **Token embedding**: Maps characters to dense vectors
- **Positional encoding**: Adds position information
- **Multi-head attention**: Allows tokens to attend to previous tokens
- **Feed-forward layers**: Non-linear transformations
- **Layer normalization**: Stabilizes training
- **Causal masking**: Ensures autoregressive generation

## 🎮 Interactive Console Features

- **Real-time generation**: Type prompts and get completions
- **Adjustable parameters**: Change temperature and max tokens on the fly
- **Command interface**: Built-in help and configuration commands
- **Easy model switching**: Load different trained models

### Console Commands

```
/help      - Show available commands
/temp 0.8  - Set temperature (0.1-2.0)
/tokens 150 - Set max tokens (1-500)
/quit      - Exit console
```

## 🔧 Troubleshooting

### Import Errors

If you get import errors:

1. **Install the package**: `pip install -e .`
2. **Check Python path**: Ensure you're in the project directory
3. **Use fallback scripts**: `python scripts/train.py` instead of `tinyllm-train`

### CUDA Issues

For GPU training:

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU if needed
tinyllm-train data/shakespeare6k.txt --device cpu
```

### Path Issues

Models are saved to `trained_models/` by default. Ensure this directory exists or specify a custom path:

```bash
tinyllm-train data/shakespeare6k.txt --output_dir /path/to/models
tinyllm-console /path/to/models
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.