#!/bin/bash
# setup_dev.sh - Development setup script for TinyPythonLLM

set -e  # Exit on any error

echo "🧠 Setting up TinyPythonLLM development environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.9"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)" 2>/dev/null; then
    echo "Error: Python 3.9+ required, found Python $python_version"
    exit 1
fi

echo "Python version check passed: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install package in development mode
echo "Installing TinyPythonLLM in development mode..."
pip install -e .

# Create necessary directories
echo "Creating directories..."
mkdir -p data
mkdir -p trained_models
mkdir -p logs

# Download sample data if not exists
if [ ! -f "data/shakespeare6k.txt" ]; then
    echo "Sample data not found. You can add training data to the data/ directory."
    echo "For example, save text files like data/shakespeare.txt"
fi

echo ""
echo "  Setup complete! You can now:"
echo ""
echo "   1. Activate the environment: source venv/bin/activate"
echo "   2. Train a model: tinyllm-train data/your_text_file.txt"
echo "   3. Use the console: tinyllm-console"
echo ""
echo "   Or run directly:"
echo "   python scripts/train.py data/your_text_file.txt"
echo "   python scripts/console.py"
echo ""
echo "💡 Add your training data to the data/ directory to get started!"