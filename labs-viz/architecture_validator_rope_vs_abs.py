import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

from transformer_from_scratch.transformer_blk import TransformerBlock
from transformer_from_scratch.positional_embeddings import SinusoidalPositionalEncoding
from transformer_from_scratch.multi_head_attention_rope import AttentionRoPE

class RoPETransformerBlock(TransformerBlock):
    """A wrapper to inject RoPEAttention into the standard block"""
    def __init__(self, d_model, n_heads, drop_prob, norm_style, seq_len):
        super().__init__(d_model, n_heads, drop_prob, norm_style)
        # Override standard self.attention with RoPEAttention
        self.attention = AttentionRoPE(d_model, seq_len, n_heads, drop_prob)

def effective_dimension(s):
    eigs = s**2
    return (eigs.sum()**2) / (eigs.pow(2).sum())

def get_pca_explained_variance(s):
    variance = s**2
    return (variance / variance.sum()).detach().numpy()

def compare_absolute_rope():
    # Configuration
    d_model = 64
    n_heads = 4
    n_layers = 15 
    seq_len = 128
    
    # Model 1: Absolute PE (Additive) + Pre-LN
    # Structure: [Input + Pos] -> [Block] -> [Block]...
    model_abs = nn.Sequential(
        SinusoidalPositionalEncoding(seq_len, d_model),
        *[TransformerBlock(d_model, n_heads, 0.0, 'Pre-Ln') for _ in range(n_layers)]
    )

    # Model 2: RoPE (Multiplicative) + Pre-LN
    # Structure: [Input] -> [RoPE Block] -> [RoPE Block]...
    # Note: No initial PE layer. Position is injected INSIDE the attention.
    model_rope = nn.Sequential(
        *[RoPETransformerBlock(d_model, n_heads, 0.0, 'Pre-Ln', seq_len) for _ in range(n_layers)]
    )

    X_input = torch.randn(1, seq_len, d_model)

    print(f"Running Forward Pass (Depth: {n_layers})...")
    with torch.no_grad():
        y_abs = model_abs(X_input)
        y_rope = model_rope(X_input)

    # Compute SVDs
    s_abs = torch.linalg.svdvals(y_abs.squeeze(0))
    s_rope = torch.linalg.svdvals(y_rope.squeeze(0))

    # Metrics
    ed_abs = effective_dimension(s_abs)
    ed_rope = effective_dimension(s_rope)

    print(f"\n--- Geometric Impact Results ---")
    print(f"Effective Dimension (Absolute/Additive):  {ed_abs:.2f} / {d_model}")
    print(f"Effective Dimension (RoPE/Multiplicative): {ed_rope:.2f} / {d_model}")
    
    # Plotting
    cum_var_abs = np.cumsum(get_pca_explained_variance(s_abs))
    cum_var_rope = np.cumsum(get_pca_explained_variance(s_rope))

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, d_model + 1), cum_var_abs, marker='o', markersize=4, label='Absolute PE (Additive)', color='red')
    plt.plot(range(1, d_model + 1), cum_var_rope, marker='x', markersize=4, label='RoPE (Orthogonal)', color='green')
    
    plt.axhline(y=0.90, color='gray', linestyle='--', label='90% Variance Threshold')
    plt.title(f"Manifold Spectrum: Additive vs Multiplicative (Depth {n_layers})")
    plt.xlabel("Principal Component Index")
    plt.ylabel("Cumulative Variance Explained")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__=='__main__':
    compare_absolute_rope()