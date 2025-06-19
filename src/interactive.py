"""
Interactive console interface for TinyPythonLLM.
"""

import torch
from pathlib import Path
from typing import Optional, List

from .transformer import Transformer
from .character_tokenizer import CharacterTokenizer
from .logger import get_logger

logger = get_logger(__name__)


class TinyLLMConsole:
    """Interactive console for TinyPythonLLM."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize console with trained model."""
        self.model = None
        self.tokenizer = None
        self.config = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self.load_model(model_path)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            print(f"Failed to load model: {e}")
            self.model = None
    
    def find_model_files(self, search_dir: str = "trained_models") -> List[Path]:
        """Find all model files in directory."""
        search_path = Path(search_dir)
        model_files = []
        
        if search_path.exists() and search_path.is_dir():
            # Look for any .pt files
            model_files.extend(search_path.glob("*_model.pt"))
            model_files.extend(search_path.glob("*.pt"))
        
        return sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True)  # Most recent first
    
    def auto_discover_model(self) -> Optional[Path]:
        """Auto-discover the most recent model file."""
        search_dirs = ["trained_models", ".", "models"]
        
        for search_dir in search_dirs:
            model_files = self.find_model_files(search_dir)
            if model_files:
                print(f"Found {len(model_files)} model(s) in {search_dir}:")
                for i, model_file in enumerate(model_files):
                    print(f"  {i+1}. {model_file.name}")
                
                # Return the most recent one
                return model_files[0]
        
        return None
    
    def load_model(self, model_path: Optional[str] = None):
        """Load trained model and tokenizer."""
        if model_path is None:
            print("No model path specified, auto-discovering...")
            model_file = self.auto_discover_model()
            if model_file is None:
                raise FileNotFoundError("No model files found. Train a model first!")
        else:
            model_file = Path(model_path)
            
            # If it's a directory, find models in it
            if model_file.is_dir():
                model_files = self.find_model_files(str(model_file))
                if not model_files:
                    raise FileNotFoundError(f"No model files found in {model_file}")
                model_file = model_files[0]  # Most recent
                print(f"Found model: {model_file.name}")
            
            # If it's a file path but doesn't exist, try to find it
            elif not model_file.exists():
                # Maybe they gave just a filename
                for search_dir in ["trained_models", ".", "models"]:
                    candidate = Path(search_dir) / model_file.name
                    if candidate.exists():
                        model_file = candidate
                        break
                else:
                    raise FileNotFoundError(f"Model file not found: {model_path}")
        
        logger.info(f"Loading model from {model_file}")
        print(f"Loading model: {model_file}")
        
        # Load model checkpoint
        # Note: We use weights_only=False because our checkpoints contain
        # CharacterTokenizer objects which are safe (we created them)
        try:
            checkpoint = torch.load(model_file, map_location=self.device, weights_only=False)
        except TypeError:
            # Fallback for older PyTorch versions that don't support weights_only parameter
            checkpoint = torch.load(model_file, map_location=self.device)
        
        # Reconstruct model
        self.config = checkpoint['config']
        self.model = Transformer(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.tokenizer = checkpoint['tokenizer']
        
        self.model.to(self.device)
        self.model.eval()
        
        # Print model info
        dataset_name = checkpoint.get('dataset_name', 'unknown')
        print(f"✅ Model loaded successfully!")
        print(f"📊 Dataset: {dataset_name}")
        print(f"🔤 Vocabulary size: {self.tokenizer.vocab_size}")
        print(f"🧠 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        logger.info("Model loaded successfully!")
    
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
            print("❌ Error: No model loaded. Please train a model first.")
            print("💡 Run: python scripts/start_training.py data/your_file.txt")
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
                    cmd_parts = prompt.split()
                    cmd = cmd_parts[0].lower()
                    
                    if cmd == '/quit':
                        print("Goodbye!")
                        break
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


def main():
    """Main console function for direct script execution."""
    import sys
    
    model_path = None
    if len(sys.argv) > 1:
        model_path = sys.argv[1]

    console = TinyLLMConsole(model_path)
    console.run_interactive()


if __name__ == "__main__":
    main()