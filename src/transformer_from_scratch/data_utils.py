import torch
import tiktoken
import os
from torch.utils.data import DataLoader
from transformer_from_scratch.data_generators import WikiAutoRegressive

def get_text_loaders(file_path, batch_size, seq_len, split_ratio=0.9):
    """
    Generic loader for any text file.
    Args:
        file_path (str): Path to the .txt file (e.g., 'data/tinystories_input.txt')
        batch_size (int): Batch size for training
        seq_len (int): Context length (block_size)
        split_ratio (float): Ratio of data to use for training (default 0.9)
    """
    # 1. Validation
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find data file at: {file_path}. Did you run the download script?")

    print(f"Loading text from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # 2. Split Train/Val
    n = len(text)
    split_idx = int(n * split_ratio)
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    # 3. Tokenizer Setup
    # Using GPT-2 tokenizer (standard BPE)
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab 

    print(f"--- Data Stats ---")
    print(f"File Size: {n/1024/1024:.2f} MB")
    print(f"Vocab Size: {vocab_size}")
    print(f"Train Tokens: {len(enc.encode(train_text, allowed_special={'<|endoftext|>'})):,}")
    print(f"Val Tokens:   {len(enc.encode(val_text, allowed_special={'<|endoftext|>'})):,}")
    print(f"------------------")

    # 4. Create Datasets
    # Using your existing WikiAutoRegressive class
    train_ds = WikiAutoRegressive(train_text, seq_len, enc)
    val_ds = WikiAutoRegressive(val_text, seq_len, enc)

    # 5. Create Loaders
    # num_workers=0 is safer for simple debugging on Mac/Windows
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=True,
        drop_last=True
    )

    return train_loader, val_loader, vocab_size