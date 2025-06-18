"""
Interactive console interface for TinyPythonLLM.
"""

import torch
from pathlib import Path
from typing import Optional

from .transformer import Transformer
from .character_tokenizer import CharacterTokenizer
from .logger import get_logger

logger = get_logger(__name__)


class TinyLLMConsole:
    """Interactive console for TinyPythonLLM."""
    
    def __init__(self, model_dir: str = "trained_models"):
        """Initialize console with trained model."""
        self.model = None
        self.tokenizer = None
        self.config = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self.load_model(model_dir)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            print(f"Failed to load model: {e}")
            self.model = None
    
    def load_model(self, model_dir: str):
        """Load trained model and tokenizer."""
        logger.info(f"Loading model from {model_dir}")
        print(f"Looking for model in: {model_dir}")

        # Try different model file locations
        possible_paths = [
            Path(model_dir) / "shakespeare_model.pt",
            Path(model_dir),  # If model_dir is actually a file path
            Path("trained_models") / "shakespeare_model.pt",
            Path(".") / "shakespeare_model.pt",
            Path("models") / "shakespeare_model.pt"
        ]
        
        model_path = None
        for path in possible_paths:
            if path.exists() and path.is_file():
                model_path = path
                break
        
        if model_path is None:
            # List available files for debugging
            search_dirs = [model_dir, "trained_models", ".", "models"]
            print("Model not found. Searched in:")
            for search_dir in search_dirs:
                dir_path = Path(search_dir)
                if dir_path.exists() and dir_path.is_dir():
                    files = list(dir_path.glob("*.pt"))
                    print(f"  {search_dir}: {[f.name for f in files] if files else 'no .pt files'}")
                else:
                    print(f"  {search_dir}: directory not found")
            
            raise FileNotFoundError(f"Model not found. Tried: {[str(p) for p in possible_paths]}")
        
        print(f"Loading model from: {model_path}")
        
        # Handle PyTorch 2.6+ security changes - try weights_only=False for custom classes
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        except Exception as e:
            logger.error(f"Failed to load with weights_only=False: {e}")
            # If that fails, try the old default behavior
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
            except Exception as e2:
                raise RuntimeError(f"Failed to load model: {e2}")
        
        # Reconstruct model
        self.config = checkpoint['config']
        self.model = Transformer(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.tokenizer = checkpoint['tokenizer']
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("Model loaded successfully!")
        print("Model loaded successfully!")
        print(f"Vocabulary size: {self.tokenizer.vocab_size}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def generate_text(self, prompt: str, max_tokens: int = 100, temperature: float = 0.8) -> str:
        """Generate text from prompt."""
        if self.model is None:
            return "Error: No model loaded"
        
        try:
            # Encode prompt
            prompt_tokens = self.tokenizer.encode(prompt)
            if not prompt_tokens:
                prompt_tokens = [0]  # fallback
            
            input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
            
            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids, 
                    max_length=max_tokens,
                    temperature=temperature
                )
            
            # Decode
            generated_text = self.tokenizer.decode(generated_ids[0].tolist())
            return generated_text
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"Error during generation: {e}"
    
    def run_interactive(self):
        """Run interactive console loop."""
        if self.model is None:
            print("Error: No model loaded. Please train a model first.")
            return
        
        print("\n🧠 TinyPythonLLM Interactive Console")
        print("=" * 50)
        print("Enter text prompts to generate continuations.")
        print("Commands:")
        print("  /help    - Show this help")
        print("  /temp X  - Set temperature to X (0.1-2.0)")
        print("  /tokens X - Set max tokens to X")
        print("  /quit    - Exit")
        print("=" * 50)
        
        # Current settings
        max_tokens = 100
        temperature = 0.8
        
        while True:
            try:
                prompt = input(f"\n[temp={temperature:.1f}, tokens={max_tokens}] > ").strip()
                
                if not prompt:
                    continue
                
                # Handle commands
                if prompt.startswith('/'):
                    if self._handle_command(prompt, max_tokens, temperature):
                        break
                    continue
                
                # Generate text
                print("\nGenerating...")
                generated = self.generate_text(prompt, max_tokens, temperature)
                print(f"\n{generated}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

    def _handle_command(self, prompt: str, max_tokens: int, temperature: float) -> bool:
        """Handle console commands. Returns True if should exit."""
        cmd_parts = prompt.split()
        cmd = cmd_parts[0].lower()
        
        if cmd == '/quit':
            print("Goodbye!")
            return True
        elif cmd == '/help':
            print("\nCommands:")
            print("  /help    - Show this help")
            print("  /temp X  - Set temperature to X (0.1-2.0)")
            print("  /tokens X - Set max tokens to X")
            print("  /quit    - Exit")
        elif cmd == '/temp' and len(cmd_parts) > 1:
            try:
                new_temp = float(cmd_parts[1])
                if 0.1 <= new_temp <= 2.0:
                    temperature = new_temp
                    print(f"Temperature set to {temperature}")
                else:
                    print("Temperature must be between 0.1 and 2.0")
            except ValueError:
                print("Invalid temperature value")
        elif cmd == '/tokens' and len(cmd_parts) > 1:
            try:
                new_tokens = int(cmd_parts[1])
                if 1 <= new_tokens <= 500:
                    max_tokens = new_tokens
                    print(f"Max tokens set to {max_tokens}")
                else:
                    print("Max tokens must be between 1 and 500")
            except ValueError:
                print("Invalid token count")
        else:
            print("Unknown command. Type /help for available commands.")
        
        return False


def main():
    """Main console function for direct script execution."""
    import sys
    
    model_dir = "trained_models"
    model_name = None
    
    if len(sys.argv) > 1:
        if sys.argv[1].endswith('.pt') or '_model' in sys.argv[1]:
            # Specific model file provided
            model_name = sys.argv[1]
        else:
            # Directory provided
            model_dir = sys.argv[1]
    
    if len(sys.argv) > 2:
        model_name = sys.argv[2]

    console = TinyLLMConsole(model_dir, model_name)
    console.run_interactive()


if __name__ == "__main__":
    main()