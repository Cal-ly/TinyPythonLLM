"""
Interactive console interface for TinyPythonLLM.
"""

import os
import sys
import torch
from pathlib import Path
from typing import Optional

# Add src to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from models.transformer import Transformer
from tokenization.character_tokenizer import CharacterTokenizer
from utils.config import ModelConfig
from utils.logger import get_logger

logger = get_logger(__name__)

class TinyLLMConsole:
    """Interactive console for TinyPythonLLM."""
    
    def __init__(self, model_dir: str = "models"):
        """Initialize console with trained model."""
        self.model = None
        self.tokenizer = None
        self.config = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self.load_model(model_dir)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
    
    def load_model(self, model_dir: str):
        """Load trained model and tokenizer."""
        logger.info(f"Loading model from {model_dir}")
        
        model_path = Path(model_dir) / "best_model.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Reconstruct model
        self.config = checkpoint['config']
        self.model = Transformer(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.tokenizer = checkpoint['tokenizer']
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("Model loaded successfully!")
        logger.info(f"Vocabulary size: {self.tokenizer.vocab_size}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
    
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
    """Main console function."""
    import sys
    
    model_dir = "models"
    if len(sys.argv) > 1:
        model_dir = sys.argv[1]
    
    console = TinyLLMConsole(model_dir)
    console.run_interactive()

if __name__ == "__main__":
    main()
