"""
Launch interactive console for TinyPythonLLM.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import TinyLLMConsole

def main():
    """Launch the interactive console."""
    model_dir = "trained_models"
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        # If argument is a .pt file or contains 'model', treat as filename
        if arg.endswith('.pt') or 'model' in arg:
            # Check if it's an absolute path or just filename
            if Path(arg).is_absolute() or '/' in arg or '\\' in arg:
                model_dir = str(Path(arg).parent) if Path(arg).parent != Path('.') else "."
            else:
                # Just a filename, look in common locations
                for possible_dir in ["trained_models", ".", "models"]:
                    if Path(possible_dir, arg).exists():
                        model_dir = possible_dir
                        break
        else:
            # Treat as directory
            model_dir = arg
    
    print(f"Looking for model in directory: {model_dir}")
    console = TinyLLMConsole(model_dir)
    console.run_interactive()

if __name__ == "__main__":
    main()
