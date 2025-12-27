from transformer_from_scratch.multi_head_attention import MultiHeadAttention
from transformer_from_scratch.positional_embeddings import SinusoidalPositionalEncoding
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np  

class TransformerBlock(nn.Module):
    """A Transformer Block implementation
      for Pre Layer Normalization and Post Layer Normalization Architecture"""
    def __init__(self, d_model, n_heads, dropout=0.1, style='Pre-Ln'):
        super().__init__()

        # MHA Layer
        self.mha = MultiHeadAttention(d_model, n_heads, dropout)

        # First LayerNorm layer
        self.ln_1 = nn.LayerNorm(d_model)

        # Simple Feed Forward Network (FFN) for the second sublayer
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        
        # Second LayerNorm lauyer
        self.ln_2 = nn.LayerNorm(d_model)
        self.style = style

    def forward(self, X):
        if self.style == 'Pre-Ln':
            # Take the multi-head attention of the layer normalized X
            att = self.mha(self.ln_1(X))

            # Comoute the residual block # 1
            res_1 = X + att

            # return the ffn of the layernorm of res_1 + the residual
            return self.ffn(self.ln_2(res_1)) + res_1
        elif self.style == 'Post-Ln':
            # Compute the multihead attention of the raw X
            att = self.mha(X)

            # Take the layernom of the residual
            res_1 = self.ln_1(att + X)

            # Compute the residual of the ffn and return the layer normalzation of res_2
            res_2 = self.ffn(res_1) + res_1
            return self.ln_2(res_2)
        else:
            ValueError('Please select either Post-Ln or Pre-Ln architecture')
