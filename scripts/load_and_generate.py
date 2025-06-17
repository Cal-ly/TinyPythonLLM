"""Load a saved model and generate text."""

import sys
import torch
from pathlib import Path

# Add the parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.train import load_model_artifacts
from src.utils.logger import get_logger

logger = get_logger("load_and_generate")


def generate_text(prompt: str = "the", max_tokens: int = 200, temperature: float = 0.8):
    """Generate text using a saved model."""
    try:
        # Load model artifacts
        model, tokenizer, model_config, training_config = load_model_artifacts()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        logger.info(f"Loaded model with {sum(p.numel() for p in model.parameters())} parameters")
        logger.info(f"Vocabulary size: {len(tokenizer.itos)}")
        
        # Validate prompt characters
        if not all(c in tokenizer.stoi for c in prompt):
            invalid_chars = [c for c in prompt if c not in tokenizer.stoi]
            logger.warning(f"Invalid characters in prompt: {invalid_chars}")
            prompt = "the"  # fallback
        
        # Encode prompt
        prompt_ids = torch.tensor(
            [tokenizer.encode(prompt)], dtype=torch.long, device=device
        )
        
        logger.info(f"Generating text starting with '{prompt}'...")
        
        # Generate text
        with torch.no_grad():
            generated_ids = model.generate(
                prompt_ids, 
                max_new_tokens=max_tokens, 
                temperature=temperature
            )
        
        # Decode result
        generated_text = tokenizer.decode(generated_ids[0].tolist())
        
        logger.info("Generated text:")
        logger.info(f"'{generated_text}'")
        print(f"\nGenerated text:\n{generated_text}\n")
        
        return generated_text
        
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        raise


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate text using saved model")
    parser.add_argument("--prompt", default="the", help="Starting prompt")
    parser.add_argument("--max_tokens", type=int, default=200, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    
    args = parser.parse_args()
    
    generate_text(args.prompt, args.max_tokens, args.temperature)


if __name__ == "__main__":
    main()
