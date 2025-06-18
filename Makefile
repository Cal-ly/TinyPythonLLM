# Makefile for TinyPythonLLM

.PHONY: help train console clean list-models

help:
	@echo "TinyPythonLLM Commands:"
	@echo ""
	@echo "  train              - Train with shakespeare25k.txt"
	@echo "  train-small        - Train with shakespeare6k.txt (faster)"
	@echo "  console            - Launch interactive console (auto-discover models)"
	@echo "  console-model      - Launch console with specific model"
	@echo "  list-models        - Show available trained models"
	@echo "  clean              - Clean up cache files"
	@echo ""
	@echo "Custom training:"
	@echo "  make train DATA=data/your_file.txt"
	@echo "  make train DATA=data/your_file.txt EPOCHS=10"
	@echo ""
	@echo "Console with specific model:"
	@echo "  make console-model MODEL=shakespeare25k_model.pt"

# Default training with your main dataset
train:
	python scripts/start_training.py data/shakespeare25k.txt

# Quick training for testing
train-small:
	python scripts/start_training.py data/shakespeare6k.txt --epochs 3

# Custom training with parameters
train-custom:
	python scripts/start_training.py $(DATA) --epochs $(EPOCHS)

# Launch console (auto-discover models)
console:
	python scripts/start_console.py

# Launch console with specific model
console-model:
	python scripts/start_console.py $(MODEL)

# List available models
list-models:
	@echo "Available models:"
	@if exist "trained_models" ( \
		dir /b trained_models\*.pt 2>nul || echo   No models found \
	) else ( \
		echo   trained_models\ directory not found \
	)

# Clean up
clean:
	@echo 🧹 Cleaning up...
	@for /r %%i in (*.pyc) do @del "%%i" 2>nul
	@for /d /r %%i in (__pycache__) do @rd /s /q "%%i" 2>nul
	@echo ✅ Cleanup complete

# Check dependencies
check:
	@python -c "import torch, numpy; print('✅ Dependencies OK')" || \
		echo "❌ Missing dependencies. Run: pip install torch numpy tqdm"