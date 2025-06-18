# 🧠 TinyPythonLLM

A minimal educational language model implementation in Python + PyTorch.

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install torch numpy tqdm

# 2. Train a model
python scripts/start_training.py data/shakespeare25k.txt

# 3. Launch console (auto-discovers models)
python scripts/start_console.py
```

## 📁 File Structure

```
TinyPythonLLM/
├── src/                    # All Python source code
├── data/                   # Training text files
│   ├── shakespeare25k.txt  # Main dataset
│   └── shakespeare6k.txt   # Smaller for testing
├── trained_models/         # Auto-generated model files
├── scripts/                # Entry point scripts
│   ├── start_training.py   # Train models
│   └── start_console.py    # Interactive console
└── logs/                   # Training logs
```

## 🎯 Usage Examples

### Training Models

```bash
# Train with your main dataset
python scripts/start_training.py data/shakespeare25k.txt

# Train with custom parameters
python scripts/start_training.py data/shakespeare25k.txt --epochs 10 --batch_size 64

# Train with any text file
python scripts/start_training.py data/my_text.txt
```

**Output:** Creates `trained_models/shakespeare25k_model.pt` (or `my_text_model.pt`)

### Using the Console

```bash
# Auto-discover the most recent model
python scripts/start_console.py

# Use a specific model file
python scripts/start_console.py shakespeare25k_model.pt

# Use model with full path
python scripts/start_console.py trained_models/shakespeare25k_model.pt

# Use model from different directory
python scripts/start_console.py models/
```

**The console will:**
- 🔍 Auto-discover models in `trained_models/` if no argument provided
- 📁 Search common locations (`trained_models/`, `models/`, current directory) for model files
- 🎯 Handle both model filenames and directory paths intelligently

### With Make (optional)

```bash
make train                    # Train with shakespeare25k.txt
make console                  # Launch console (auto-discover)
make console-model MODEL=shakespeare25k_model.pt  # Specific model
make list-models             # Show available models
make clean                   # Clean up cache files
```

## ⚙️ Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 5 | Number of training epochs |
| `--batch_size` | 32 | Training batch size |
| `--learning_rate` | 3e-4 | Learning rate |
| `--sequence_length` | 256 | Input sequence length |
| `--output_dir` | trained_models | Where to save models |

## 🎮 Console Commands

```
/help      - Show available commands
/temp 0.8  - Set temperature (0.1-2.0)
/tokens 150 - Set max tokens (1-500)
/quit      - Exit console
```

## 📊 Model Naming & Discovery

Models are automatically named after your dataset:

- `data/shakespeare25k.txt` → `shakespeare25k_model.pt`
- `data/my_novel.txt` → `my_novel_model.pt`
- `data/code_samples.txt` → `code_samples_model.pt`

The console intelligently finds models by:
1. 🔍 Auto-discovering in `trained_models/` directory
2. 📂 Searching `trained_models/`, `models/`, current directory for specific files
3. 🎯 Handling both filenames and directory paths

## 🔧 Dependencies

- Python 3.9+
- PyTorch
- NumPy
- tqdm (for progress bars)

Install with: `pip install torch numpy tqdm`

## 🐛 Troubleshooting

**No model found:**
```bash
# Check what models exist
python scripts/start_console.py  # Will show available models
# Or train a new one
python scripts/start_training.py data/shakespeare25k.txt
```

**Import errors:**
- Make sure you're in the project root directory
- Check that `src/` contains all the Python files

**CUDA issues:**
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

**Model path issues:**
```bash
# The console will show you what it's looking for
python scripts/start_console.py your_model.pt
# Output shows: 📁 Looking for model: your_model.pt
#               📂 Model directory: trained_models
```

That's it! Simple and flexible. 🎯