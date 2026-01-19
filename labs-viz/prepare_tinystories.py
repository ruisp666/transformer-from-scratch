import os
from tqdm import tqdm
from datasets import load_dataset # pip install datasets

def prepare_tinystories():
    print("Downloading TinyStories (this might take a minute)...")
    
    # Load the dataset (streaming mode to save RAM)
    # We use the 'roneneldan/TinyStories' dataset
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    
    output_file = "data/tinystories_input.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # We want roughly 200MB of text to start.
    # 1 char ~= 1 byte. So 200,000,000 chars.
    target_size = 200 * 1024 * 1024 
    current_size = 0
    
    print(f"Writing to {output_file}...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        # Iterate through the stream
        for item in tqdm(dataset):
            text = item['text']
            
            # Simple cleaning (TinyStories is usually clean, but good habit)
            text = text.strip() + "\n\n<|endoftext|>\n\n"
            
            f.write(text)
            current_size += len(text)
            
            if current_size >= target_size:
                break
                
    print(f"\nSuccess! Saved {current_size / 1024 / 1024:.2f} MB to {output_file}")
    print("You can now point your training config to this file.")

if __name__ == "__main__":
    prepare_tinystories()