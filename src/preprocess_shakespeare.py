import re
import os
from pathlib import Path

# Set root directory to two levels up from this file
# This allows the script to be run from anywhere in the project
root = Path(__file__).parent.parent

def clean_shakespeare_text(text):
    """
    Clean Shakespeare text by removing stage directions while keeping character names.
    Standardize formatting for LLM training.
    """
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
            
        # Remove stage directions (text in brackets, parentheses, and specific patterns)
        # Remove [Enter...], [Exit...], [Exeunt...], etc.
        if re.match(r'^\[.*\]$', line):
            continue
        if re.match(r'^Enter\s', line, re.IGNORECASE):
            continue
        if re.match(r'^Exit', line, re.IGNORECASE):
            continue
        if re.match(r'^Exeunt', line, re.IGNORECASE):
            continue
        if re.match(r'^Flourish', line, re.IGNORECASE):
            continue
        if re.match(r'^Alarum', line, re.IGNORECASE):
            continue
        if re.match(r'^Scene\s', line, re.IGNORECASE):
            continue
        if re.match(r'^ACT\s', line, re.IGNORECASE):
            continue
        if re.match(r'^SCENE', line, re.IGNORECASE):
            continue
        
        # Remove stage directions in parentheses
        line = re.sub(r'\([^)]*\)', '', line)
        
        # Remove stage directions in square brackets
        line = re.sub(r'\[[^\]]*\]', '', line)
        
        # Skip dramatis personae and other metadata
        if 'DRAMATIS PERSONAE' in line.upper():
            continue
        if 'Dramatis Personae' in line:
            continue
        if line.upper().startswith('SCENE:'):
            continue
        if line.upper().startswith('THE END'):
            continue
        if re.match(r'^\d{4}$', line):  # Skip years
            continue
        if line.startswith('by William Shakespeare'):
            continue
        if 'filepath:' in line:
            continue
            
        # Clean up character names and dialogue
        # Character names are typically in ALL CAPS followed by a period or colon
        if re.match(r'^[A-Z][A-Z\s]+[.:]\s*', line):
            # Standardize character name format
            parts = re.split(r'[.:]', line, 1)
            if len(parts) == 2:
                character_name = parts[0].strip()
                dialogue = parts[1].strip()
                if dialogue:
                    line = f"{character_name}: {dialogue}"
                else:
                    line = f"{character_name}:"
            else:
                line = parts[0].strip() + ":"
        
        # Clean up extra whitespace
        line = re.sub(r'\s+', ' ', line).strip()
        
        # Skip very short lines that are likely formatting artifacts
        if len(line) < 2:
            continue
            
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def create_dataset_versions(text, output_dir):
    """
    Create different versions of the dataset with specified character counts.
    """
    # Target character counts
    targets = {
        'shakespeare25k': 25000,
        'shakespeare50k': 50000,
        'shakespeare100k': 100000,
        'shakespeare250k': 250000,
        'shakespeare1mil': 1000000,
        'shakespeare_complete': len(text)
    }
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for name, target_chars in targets.items():
        if target_chars >= len(text):
            # Use complete text
            subset_text = text
        else:
            # Take text from the beginning up to target character count
            # Try to break at a reasonable point (end of line)
            subset_text = text[:target_chars]
            
            # Find the last complete line within the character limit
            last_newline = subset_text.rfind('\n')
            if last_newline > target_chars * 0.9:  # If we're close to target, use it
                subset_text = subset_text[:last_newline]
        
        # Save to file
        output_path = Path(output_dir) / f"{name}.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(subset_text)
        
        print(f"Created {name}.txt - {len(subset_text):,} characters")

def main():
    """
    Main preprocessing function.
    """
    # Input and output paths
    input_file = Path('data/data_raw/shakespeare.txt')
    output_dir = Path('data')
    
    # Check if input file exists
    if not input_file.exists():
        print(f"Error: Input file {input_file} not found!")
        return
    
    print("Reading Shakespeare dataset...")
    
    # Read the original file
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    
    print(f"Original text: {len(raw_text):,} characters")
    
    # Clean the text
    print("Cleaning text...")
    cleaned_text = clean_shakespeare_text(raw_text)
    
    print(f"Cleaned text: {len(cleaned_text):,} characters")
    
    # Create different versions
    print("Creating dataset versions...")
    create_dataset_versions(cleaned_text, output_dir)
    
    print("\nPreprocessing complete!")
    print(f"Output files saved to: {output_dir}")
    
    # Show sample of cleaned text
    print("\nSample of cleaned text:")
    print("-" * 50)
    print(cleaned_text[:500] + "...")

if __name__ == "__main__":
    main()
