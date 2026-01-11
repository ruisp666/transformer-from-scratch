from transformer_from_scratch.multi_head_attention import MultiHeadAttention
from transformer_from_scratch.multi_head_attention_rope import MultiHeadAttentionROPE
from transformer_from_scratch.rms_norm import RMSNorm
from transformer_from_scratch.activations import FFNSwiGlu
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np  


class TransformerBlock(nn.Module):
    """A Transformer Block implementation:
      for Pre Layer Normalization and Post Layer Normalization Architecture"""
    def __init__(self, d_model, n_heads, dropout=0.1, style='Pre-Ln',
                    norm='layer'):
        super().__init__()

        # MHA Layer
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
      
        # Layers for LayerNorm or Layers for RMSNNorm
        if norm == 'layer':
            self.norm_layer_1 = nn.LayerNorm(d_model)
            self.norm_layer_2 = nn.LayerNorm(d_model)
        elif norm=='rms':
            self.norm_layer_1 = RMSNorm(d_model)
            self.norm_layer_2 = RMSNorm(d_model)
        else:
            ValueError('Please select either LayerNorm or RMSNorm architecture')

        # Simple Feed Forward Network (FFN) for the second sublayer
        self.ffn = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.ReLU(),
                nn.Linear(4 * d_model, d_model)
            )
        self.style = style
    

    def forward(self, X):
        if self.style == 'Pre-Ln':
            # Take the multi-head attention of the normalized X

            att = self.attention(self.norm_layer_1(X))

            # Compute the residual block # 1
            res_1 = X + att

            # return the ffn of the norm of res_1 + the residual
            return self.ffn(self.norm_layer_2(res_1)) + res_1
        elif self.style == 'Post-Ln':
            # Compute the multihead attention of the raw X
            att = self.attention(X)

            # Take the norm of the residual
            res_1 = self.norm_layer_1(att + X)

            # Compute the residual of the ffn and return the layer normalzation of res_2
            res_2 = self.ffn(res_1) + res_1
            return self.norm_layer_2(res_2)
        else:
            ValueError('Please select either Post-Ln or Pre-Ln architecture')


class RoPETransformerBlock(TransformerBlock):
    """A wrapper to inject RoPEAttention into the standard block"""
    def __init__(self, d_model, n_heads, dropout, style, norm, seq_len):
        super().__init__(d_model=d_model, n_heads=n_heads, dropout=dropout, style=style, norm=norm)
        # Override standard self.attention with RoPEAttention
        self.attention = MultiHeadAttentionROPE(d_model=d_model,seq_len=seq_len, num_heads=n_heads, dropout=dropout)

class RoPESwiGluTransformerBlock(RoPETransformerBlock):
    """A wrapper to inject SwiGlu FFN into the RoPE Block"""
    def __init__(self, d_model, n_heads, dropout, style, norm, seq_len, expansion_factor):
        super().__init__(d_model=d_model, n_heads=n_heads, dropout=dropout, style=style, norm=norm, seq_len=seq_len)
        # Override ffn with SwiGlu FFN
        self.ffn = FFNSwiGlu(d_model, expansion_factor=8/3)
