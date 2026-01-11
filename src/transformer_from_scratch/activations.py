"""
EDUCATIONAL IMPLEMENTATION: SwiGLU & FFN
========================================

This module implements the SwiGLU activation and Feed-Forward Network 
mechanics from scratch using raw `nn.Parameter` and explicit initialization.

PEDAGOGICAL NOTE: "Split" vs "Fused" Implementation
---------------------------------------------------
In this file, we separate the logic into two classes:
1. `SwiGlu`: Handles the gating mechanism (Up-projection & Gate-projection).
2. `FFNSwiGlu`: Handles the down-projection back to the residual stream.

**Production vs. Learning:**
- **Learning (This Code):** Classes are separated to visualize the distinct 
  mathematical operations (Gate vs Value) and parameter management.
- **Production (e.g., xformers, Llama 2/3):** These operations are "fused" 
  into a single CUDA kernel. The Gate, Value, and Output matrices are often 
  loaded simultaneously to minimize VRAM I/O (Memory Bandwidth is the bottleneck, 
  not compute).

Reference: "GLU Variants Improve Transformer" (Shazeer, 2020)
"""


import torch
from torch import nn
import torch.nn.functional as F
import math

class SwiGlu(nn.Module):
    def __init__(self, in_dim, out_dim):
        """I
        Implements SwiGlu gate.
        SwiGLU(x,W,V)= (x @ W) * /sigma((x @ W)) * (x @ V))
        """
        super().__init__()
        self.W_gate = nn.Parameter(data=torch.empty(in_dim, out_dim))
        self.W_value = nn.Parameter(data=torch.empty(in_dim, out_dim))
        self.hidden_dim = out_dim

        self.reset_parameters()

    def forward(self, X):
        # Compute gate projection
        gate_proj = X @ self.W_gate

        # Apply swish to the gate projection
        swish = gate_proj * F.sigmoid(gate_proj)

        return swish * (X @ self.W_value)
    
    def reset_parameters(self):
        """
        Custom initialization logic.
        Swish is ReLU-like, so we generally use Kaiming (He) Initialization.
        """
        # Fan_in mode preserves magnitude variance in the forward pass
        nn.init.kaiming_uniform_(self.W_gate, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_value, a=math.sqrt(5))
        
        print(f"Initialized SwiGLU: {self.W_gate.shape} (Hidden: {self.hidden_dim})")

class FFNSwiGlu(nn.Module):
    """Implements the FFN using SwiGlu as the first layer, and
    adjusts the dimensions as per the paper -
      (8/3) makes the number of parameters roughly equivalent to the standard
      with 2 matrices."""
    def __init__(self, d_model, expansion_factor=8/3):
        super().__init__()
        self.swiglu_layer = SwiGlu(d_model,int(expansion_factor*d_model))

        # This is how nn.Linear is implemented
        self.W_2 = nn.Parameter(torch.empty(int(expansion_factor*d_model), d_model))
        self.reset_paramters()

    def reset_paramters(self):
        # For the output projection, Xavier/Glorot is often safer to keep gradients stable
        nn.init.xavier_uniform_(self.W_2)
        
    def forward(self, x):
        return self.swiglu_layer(x) @ self.W_2
    
if __name__=='__main__':
    d_model = 6
    X = torch.ones(2,5,d_model)
    SwishGluFFN = FFNSwiGlu(d_model)

    # Forward pass
    out = SwishGluFFN(X)

    print(f"\nInput Shape: {X.shape}")
    print(f"Output Shape: {out.shape}")
    print(f"Output Mean: {out.mean().item():.4f}")
