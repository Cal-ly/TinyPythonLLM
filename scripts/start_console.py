"""
Launch interactive console for TinyPythonLLM.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import TinyLLMConsole

def parse_arguments():
    """Parse command line arguments and return model directory and name."""
    if len(sys.argv) <= 1:
        return "trained_models", None
    
    arg = sys.argv[1]
    
    # If argument is a .pt file or contains 'model', treat as filename
    if arg.endswith('.pt') or 'model' in arg:
        model_name = arg
        # Check if it's an absolute path or contains path separators
        if Path(arg).is_absolute() or '/' in arg or '\\' in arg:
            return str(Path(arg).parent), Path(arg).name
        else:
            # Just a filename, use default directory
            return "trained_models", model_name
    else:
        # Treat as directory
        return arg, None

def find_model_path(model_dir, model_name=None):
    """Find the actual model path, checking common locations if needed."""
    # If we have a specific model name, try to find it
    if model_name and not model_name.endswith('.pt'):
        # Just a filename without directory, search common locations
        for possible_dir in ["trained_models", ".", "models"]:
            if Path(possible_dir, model_name).exists():
                return possible_dir
    
    # Return the specified directory (or validate it exists)
    if Path(model_dir).exists():
        return model_dir
    
    # Fallback to default
    return "trained_models"

def main():
    """Launch the interactive console."""
    print("🧠 TinyPythonLLM Console Launcher")
    print("=" * 40)

    # Parse command line arguments
    model_dir, model_name = parse_arguments()
    
    # Display what we're looking for
    if model_name:
        print(f"📁 Looking for model: {model_name}")
    else:
        print("🔍 Auto-discovering models...")
    
    # Find the actual model directory
    final_model_dir = find_model_path(model_dir, model_name)
    print(f"📂 Model directory: {final_model_dir}")
    
    # Launch console
    try:
        console = TinyLLMConsole(final_model_dir)
        console.run_interactive()
    except Exception as e:
        print(f"❌ Error launching console: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()