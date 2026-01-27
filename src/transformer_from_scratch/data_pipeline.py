import torch
import tiktoken
import requests
import os
from torch.utils.data import DataLoader

from transformer_from_scratch.data_generators import WikiAutoRegressive

def get_shakespeare_loaders(batch_size, seq_len):
    """
    Downloads TinyShakespeare, encodes it, and returns DataLoaders.
    """
    # 1. Prepare Data
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    input_file_path = os.path.join(data_dir, 'input.txt')
    
    # Download TinyShakespeare (approx 1MB)
    if not os.path.exists(input_file_path):
        print("Downloading TinyShakespeare dataset...")
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        with open(input_file_path, 'w') as f:
            f.write(requests.get(data_url).text)
            
    with open(input_file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # 2. Split Train/Val (90/10 split)
    n = len(text)
    split_idx = int(n * 0.9)
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    # 3. Tokenizer
    # We use GPT-2's tokenizer. It's fast and standard.
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab # 50257

    print(f"Data Loaded. Vocab Size: {vocab_size}")
    print(f"Train Tokens: {len(enc.encode(train_text)):,}")
    print(f"Val Tokens: {len(enc.encode(val_text)):,}")

    # 4. Create Datasets
    train_ds = WikiAutoRegressive(train_text, seq_len, enc)
    val_ds = WikiAutoRegressive(val_text, seq_len, enc)

    # 5. Create Loaders
    # shuffle=True is crucial for training to break correlations
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader, vocab_size