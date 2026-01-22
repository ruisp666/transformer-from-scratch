import pytest
import torch
import torch.nn as nn
from transformer_from_scratch.multi_head_attention_rope import MultiHeadAttentionROPE


class TestGQA:
    """Tests for the Multi-Head Attention with RoPE."""

    def test_mha_rope_forward(self, model_params, input_tensor):
        mha = MultiHeadAttentionROPE(
            d_model=model_params['d_model'],
            seq_len=model_params['seq_len'],
            num_heads=model_params['n_heads'],
            dropout=model_params['dropout']
        )
        
        output = mha(input_tensor, n_groups=4)
        assert output.shape == input_tensor.shape
        
   