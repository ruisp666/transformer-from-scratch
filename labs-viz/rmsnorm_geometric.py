

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Paste your corrected RMSNorm class here or import it
# from src.normalization import RMSNorm 
from transformer_from_scratch.rms_norm import RMSNorm

def compare_norms():
    d_model = 64
    seq_len = 1000
    
    # 1. Create "Bad" Data (High Mean, High Variance)
    # This simulates a deep network where activations have drifted
    X_raw = torch.randn(seq_len, d_model) * 5.0 + 20.0 
    
    # 2. Apply Both Norms
    rms_layer = RMSNorm(d_model)
    ln_layer = nn.LayerNorm(d_model, elementwise_affine=True) # Standard Torch LayerNorm
    
    # Initialize LayerNorm weights to 1/0 to be fair
    nn.init.ones_(ln_layer.weight)
    nn.init.zeros_(ln_layer.bias)

    with torch.no_grad():
        X_rms = rms_layer(X_raw)
        X_ln = ln_layer(X_raw)

    # --- Visualization ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    
    # Plot 1: Activation Distribution (The Histogram)
    sns.kdeplot(X_raw.flatten(), ax=axes[0], color='gray', fill=True, label='Raw Input', alpha=0.3)
    sns.kdeplot(X_rms.flatten(), ax=axes[0], color='green', fill=False, linewidth=2, label='RMSNorm')
    sns.kdeplot(X_ln.flatten(), ax=axes[0], color='red', linestyle='--', linewidth=2, label='LayerNorm')
    axes[0].set_title("Activation Distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: L2 Norm per Token (The "Sphere")
    # Both should do roughly the same job here
    axes[1].plot(torch.norm(X_rms, dim=-1)[:100], color='green', label='RMSNorm', alpha=0.8)
    axes[1].plot(torch.norm(X_ln, dim=-1)[:100], color='red', linestyle='--', label='LayerNorm', alpha=0.8)
    axes[1].axhline(y=d_model**0.5, color='gray', linestyle=':', label=f'sqrt({d_model})')
    axes[1].set_title("L2 Norm per Token (Radially)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Mean Value per Token (THE DIFFERENCE)
    # LayerNorm explicitly centers to 0. RMSNorm does not.
    axes[2].plot(X_rms.mean(dim=-1)[:100], color='green', label='RMSNorm (Preserves Drift)')
    axes[2].plot(X_ln.mean(dim=-1)[:100], color='red', linestyle='--', label='LayerNorm (Forced Zero)')
    axes[2].axhline(y=0, color='black', linewidth=1)
    axes[2].set_title("Mean Value per Token (Centering)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('src/transformer_from_scratch/labs-viz//rmsnorm_vs_layernorm.png', dpi=300) 
    print("Figure saved to labs-viz/plots/rmsnorm_vs_layernorm.png")
    plt.show()

    # --- Stats ---
    print(f"LayerNorm Mean (Should be 0.0): {X_ln.mean().item():.5f}")
    print(f"RMSNorm Mean (Should be > 0.0): {X_rms.mean().item():.5f}")

if __name__=='__main__':
    compare_norms()