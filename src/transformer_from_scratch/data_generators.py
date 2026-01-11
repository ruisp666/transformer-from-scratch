import torch
from torch.utils.data import Dataset

class ReversalDataset(Dataset):
    """
    A synthetic dataset where the target is the reverse of the input.
    Used for validating positional embeddings.
    """
    def __init__(self, num_samples, seq_len, vocab_size):
        # We generate the "Golden Source" once
        self.x = torch.randint(0, vocab_size, (num_samples, seq_len))
        self.y = torch.flip(self.x, dims=[1])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    

class RepeatCopyDataset(Dataset):
    """
    The Relative Position Task.
    Input:  [A, B, C, 0, 0, 0]
    Target: [0, 0, 0, A, B, C]
    
    The model must learn to 'look back' by exactly seq_len/2 steps.
    """
    def __init__(self, num_samples, seq_len, vocab_size):
        half_len = seq_len // 2
        
        # 1. Generate random patterns for the first half
        patterns = torch.randint(1, vocab_size, (num_samples, half_len))
        
        # 2. Create Inputs: Pattern + Zeros
        zeros = torch.zeros((num_samples, seq_len - half_len), dtype=torch.long)
        self.x = torch.cat([patterns, zeros], dim=1)

        # 3. Create Targets: Zeros + Pattern (The Copy)
        # We use 0 as the 'ignore' token for the first half of the target
        self.y = torch.cat([zeros, patterns], dim=1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    