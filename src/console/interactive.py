"""
Interactive console interface for TinyPythonLLM.
"""

import torch
from pathlib import Path
from typing import Optional

from ..training.train import load_model_artifacts
from ..models.transformer import TransformerModel, ModelConfig
from ..tokenization.character_tokenizer import CharacterTokenizer
from ..utils.config import TrainingConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)

class TinyLLMConsole:
    """Interactive console for TinyPythonLLM."""
    
    def __init__(self, model_dir: str = "saved_models"):
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
        
        model, tokenizer, model_config, training_config = load_model_artifacts(model_dir)
        
        self.model = model
        self.tokenizer = tokenizer
        self.config = training_config
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("Model loaded successfully!")
        logger.info(f"Vocabulary size: {len(self.tokenizer.itos)}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
    
    def generate_text(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None
    ) -> str:
        """Generate text from prompt."""
        if self.model is None:
            return "Error: No model loaded"
        
        # Use default values from config if not specified
        max_tokens = max_tokens or self.config.get('console_max_tokens', 100)
        temperature = temperature or self.config.get('console_temperature', 0.8)
        top_k = top_k or self.config.get('console_top_k', 50)
        
        try:
            # Encode prompt
            prompt_tokens = [self.tokenizer.stoi.get(char, 0) for char in prompt]
            if not prompt_tokens:
                prompt_tokens = [0]  # fallback
            
            input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
            
            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids, 
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k
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
        print("  /config  - Show current configuration")
        print("  /temp X  - Set temperature to X (0.1-2.0)")
        print("  /tokens X - Set max tokens to X")
        print("  /topk X  - Set top-k to X")
        print("  /quit    - Exit")
        print("=" * 50)
        
        # Current settings
        max_tokens = self.config.get('console_max_tokens', 100)
        temperature = self.config.get('console_temperature', 0.8)
        top_k = self.config.get('console_top_k', 50)
        
        while True:
            try:
                prompt = input(f"\n[temp={temperature:.1f}, tokens={max_tokens}, top_k={top_k}] > ").strip()
                
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
                        print("  /config  - Show current configuration")
                        print("  /temp X  - Set temperature to X (0.1-2.0)")
                        print("  /tokens X - Set max tokens to X")
                        print("  /topk X  - Set top-k to X")
                        print("  /quit    - Exit")
                    elif cmd == '/config':
                        print(f"\nCurrent configuration:")
                        print(f"  Temperature: {temperature}")
                        print(f"  Max tokens: {max_tokens}")
                        print(f"  Top-k: {top_k}")
                        print(f"  Device: {self.device}")
                        print(f"  Vocabulary size: {len(self.tokenizer.itos)}")
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
                    elif cmd == '/topk' and len(cmd_parts) > 1:
                        try:
                            new_topk = int(cmd_parts[1])
                            if 1 <= new_topk <= 100:
                                top_k = new_topk
                                print(f"Top-k set to {top_k}")
                            else:
                                print("Top-k must be between 1 and 100")
                        except ValueError:
                            print("Invalid top-k value")
                    else:
                        print("Unknown command. Type /help for available commands.")
                    
                    continue
                
                # Generate text
                print("\nGenerating...")
                generated = self.generate_text(prompt, max_tokens, temperature, top_k)
                print(f"\n{generated}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

def main():
    """Main console function."""
    import sys
    
    model_dir = "saved_models"
    if len(sys.argv) > 1:
        model_dir = sys.argv[1]
    
    console = TinyLLMConsole(model_dir)
    console.run_interactive()

if __name__ == "__main__":
    main()
