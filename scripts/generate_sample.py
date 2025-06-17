"""Generate sample text from a trained model."""

import sys
from pathlib import Path
import torch

# Add the parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger
from src.models.transformer import ModelConfig, TransformerModel
from src.tokenization.character_tokenizer import CharacterTokenizer
from src.training.data_loader import load_text

logger = get_logger("generate_sample", console_output=True)


def generate_samples():
    """Generate sample text using the trained model."""
    logger.info("Loading data and setting up tokenizer...")
    
    # Load the same data used for training to build tokenizer
    data_path = Path("data/shakespeare25k.txt")
    raw_text = load_text(data_path)
    
    tokenizer = CharacterTokenizer()
    tokenizer.fit(raw_text)
    
    # Create model (same config as training)
    model_cfg = ModelConfig(vocab_size=len(tokenizer.itos), d_model=128)
    model = TransformerModel(model_cfg)
    
    # For demo purposes, we'll use the untrained model
    # In practice, you'd load saved weights here
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    logger.info(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    # Try different starting prompts
    prompts = ["the", "and", "to", "of", "a"]
    
    for prompt_text in prompts:
        if all(c in tokenizer.stoi for c in prompt_text):
            logger.info(f"\nGenerating text starting with '{prompt_text}':")
            
            # Encode prompt
            prompt_ids = torch.tensor(
                [tokenizer.encode(prompt_text)], dtype=torch.long, device=device
            )
            
            # Generate
            with torch.no_grad():
                generated = model.generate(
                    prompt_ids, 
                    max_new_tokens=200, 
                    temperature=0.8
                )
            
            # Decode and display
            generated_text = tokenizer.decode(generated[0].tolist())
            logger.info(f"Generated: {generated_text}")
        else:
            logger.warning(f"Prompt '{prompt_text}' contains unknown characters")


if __name__ == "__main__":
    generate_samples()
