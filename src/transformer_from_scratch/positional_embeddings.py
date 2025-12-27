from torch import nn
import math
import matplotlib.pyplot as plt
import torch


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        base_vector = torch.exp(torch.arange(0,d_model, step=2) * - math.log(10000)/d_model)
        pos_vector = torch.arange(seq_len).float().unsqueeze(1)
        angles = pos_vector * base_vector
        sp = torch.zeros(seq_len, d_model)

        # 4. Fill the PE matrix
        sp[:, 0::2] = torch.sin(angles) # Even indices
        sp[:, 1::2] = torch.cos(angles) # Odd indices

        self.register_buffer('sp', sp.unsqueeze(0))


    def forward(self, X):
        return X + self.sp[:, :X.size(1), :]
    
def visualize_pe_slices(pe_module):
    # Get the raw matrix (seq_len, d_model)
    pe_matrix = pe_module.sp.squeeze(0).cpu().numpy()
    seq_len = pe_matrix.shape[0]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Slice 1: Dimensions 1-5 (High Frequency / Fast Clock)
    # We include index 0 to 6 to capture the first few sine/cosine pairs
    im1 = axes[0].imshow(pe_matrix[:, 0:6], aspect='auto', cmap='RdBu', extent=[30, 50, seq_len, 0])
    axes[0].set_title("Dimensions 0-5 (Fast Rotation)")
    axes[0].set_xlabel("d_model index")
    axes[0].set_ylabel("Position (Time)")
    fig.colorbar(im1, ax=axes[0])

    # Slice 2: Dimensions 30-50 (Lower Frequency / Slower Clock)
    im2 = axes[1].imshow(pe_matrix[:, 30:50], aspect='auto', cmap='RdBu')
    axes[1].set_title("Dimensions 30-50 (Slow Rotation)")
    axes[1].set_xlabel("d_model index")
    axes[1].set_ylabel("Position (Time)")
    fig.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.show()

def test_similarities(pe_module):
    # Ensure we have a (seq_len, d_model) matrix
    pe_matrix = pe_module.sp.squeeze(0).cpu()
    
    # Calculate dot product of position 0 with ALL other positions
    # Shape: (seq_len, d_model) @ (d_model, 1) -> (seq_len)
    pos0 = pe_matrix[0].unsqueeze(1)
    similarities = torch.mm(pe_matrix, pos0).flatten() 

    plt.figure(figsize=(10, 4))
    plt.plot(similarities.numpy(), color='black', linewidth=1.5)
    plt.fill_between(range(len(similarities)), similarities.numpy(), alpha=0.1)
    plt.title("Dot Product Similarity: Position 0 vs. All Other Positions")
    plt.xlabel("Distance from Position 0")
    plt.ylabel("Similarity (Energy)")
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__=='__main__':
    model = SinusoidalPositionalEncoding(seq_len=100, d_model=128)
    visualize_pe_slices(model)
    test_similarities(model)
