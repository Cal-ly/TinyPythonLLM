# Makefile for TinyPythonLLM development

.PHONY: help install clean test train console setup lint format

# Default target
help:
	@echo "TinyPythonLLM Development Commands:"
	@echo ""
	@echo "  setup     - Set up development environment"
	@echo "  install   - Install package in development mode"
	@echo "  clean     - Clean up generated files"
	@echo "  train     - Train model with sample data"
	@echo "  console   - Launch interactive console"
	@echo "  lint      - Run code linting (if tools available)"
	@echo "  format    - Format code (if tools available)"
	@echo "  test      - Run tests (placeholder)"
	@echo ""

# Development setup
setup:
	@echo "🧠 Setting up development environment..."
	@python3 -m venv venv || echo "Virtual environment already exists"
	@. venv/bin/activate && pip install --upgrade pip
	@. venv/bin/activate && pip install -e .
	@mkdir -p data trained_models logs
	@echo "✅ Setup complete! Run 'source venv/bin/activate' to activate."

# Install package in development mode
install:
	pip install -e .

# Clean up generated files
clean:
	@echo "🧹 Cleaning up..."
	@rm -rf build/
	@rm -rf dist/
	@rm -rf *.egg-info/
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@echo "✅ Cleanup complete!"

# Train model (requires data file)
train:
	@if [ -f "data/shakespeare6k.txt" ]; then \
		echo "🚀 Training model..."; \
		python scripts/train.py data/shakespeare6k.txt; \
	else \
		echo "❌ No training data found. Add a text file to data/ directory."; \
		echo "   For example: data/shakespeare6k.txt"; \
	fi

# Launch console
console:
	@if [ -f "trained_models/shakespeare_model.pt" ]; then \
		echo "🎮 Launching console..."; \
		python scripts/console.py; \
	else \
		echo "❌ No trained model found. Run 'make train' first."; \
	fi

# Linting (if flake8 is available)
lint:
	@if command -v flake8 >/dev/null 2>&1; then \
		echo "🔍 Running linter..."; \
		flake8 src/ scripts/ --max-line-length=100; \
	else \
		echo "💡 Install flake8 for linting: pip install flake8"; \
	fi

# Code formatting (if black is available)
format:
	@if command -v black >/dev/null 2>&1; then \
		echo "✨ Formatting code..."; \
		black src/ scripts/ --line-length=100; \
	else \
		echo "💡 Install black for formatting: pip install black"; \
	fi

# Placeholder for tests
test:
	@echo "🧪 No tests implemented yet."
	@echo "   Add tests to tests/ directory and update this target."

# Quick development cycle
dev: clean install
	@echo "🔄 Development environment refreshed!"