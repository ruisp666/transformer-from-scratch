
import torch
from transformer_from_scratch.transformer_blk import TransformerBlock
from transformer_from_scratch.positional_embeddings import SinusoidalPositionalEncoding
from torch import nn
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class ReversalDataset(Dataset):
    def __init__(self, num_samples, seq_len, vocab_size):
        # We generate the "Golden Source" once
        self.x = torch.randint(0, vocab_size, (num_samples, seq_len))
        self.y = torch.flip(self.x, dims=[1])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]



def generate_data(batch_size, seq_len, vocab_size):
    # Random sequences of integers
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    # Target is the reverse of x
    y = torch.flip(x, dims=[1])
    return x, y

def train_simple_sequence(d_model=32, seq_len=10, n_epochs=20):
    n_layers = 15
    vocab_size = 10
    batch_size = 8
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    # Create the data
    full_dataset = ReversalDataset(num_samples=256, seq_len=seq_len, vocab_size=vocab_size)   
    train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
    transformer_pre_ln =  nn.Sequential(nn.Embedding(vocab_size, d_model),
                                       *[TransformerBlock(d_model,4,0.1, 'Pre-Ln') for _ in range(n_layers)],
                                       nn.Linear(d_model, vocab_size))
    transformer_pre_ln_with_positional_enc = nn.Sequential(nn.Embedding(vocab_size, d_model),
                                                            SinusoidalPositionalEncoding(seq_len, d_model),
                                                            *[TransformerBlock(d_model,4,0.1, 'Pre-Ln') for _ in range(n_layers)],
                                                            nn.Linear(d_model, vocab_size))
    loss_model = nn.CrossEntropyLoss()
    for model in [transformer_pre_ln, transformer_pre_ln_with_positional_enc]:
        print('*'*50)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
        model.to(device)
        for n in range(n_epochs):
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                output = model(x_batch)
                L = loss_model(output, y_batch)
                print(f'Loss: {L.item()}: epoch {n+1}/{n_epochs}')
                L.backward()
                optimizer.step()
                optimizer.zero_grad()
            

if __name__ == '__main__':
    train_simple_sequence(32,10)