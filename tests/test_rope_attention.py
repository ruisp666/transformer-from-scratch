import pytest
import torch
import torch.nn as nn
from transformer_from_scratch.rotary_positional_embeddings import RoPE
from transformer_from_scratch.multi_head_attention_rope import MultiHeadAttentionROPE

class TestRoPEGeometry:
    """Tests strictly for the Rotary Positional Embeddings class."""
    
    def test_rope_initialization(self, model_params):
        head_dim = model_params['d_model'] // model_params['n_heads']
        rope = RoPE(seq_len=model_params['seq_len'], d_model=head_dim)
        
        # Check buffer registration
        assert hasattr(rope, 'freq_bank_rope')
        # Check shape: (1, 1, Seq_len, Head_Dim/2) or similar depending on your implementation
        assert rope.freq_bank_rope.shape[2] == model_params['seq_len']

    def test_rope_norm_preservation(self, model_params):
        """RoPE uses rotation, so the vector norm should remain constant."""
        head_dim = model_params['d_model'] // model_params['n_heads']
        rope = RoPE(seq_len=model_params['seq_len'], d_model=head_dim)
        
        # Create (Batch, Seq, Heads, Head_Dim)
        x = torch.randn(2, 4, 10, head_dim)
        x_rotated = rope(x)
        
        # Norm difference should be negligible (< 1e-5)
        diff = torch.abs(torch.norm(x) - torch.norm(x_rotated))
        assert diff < 1e-5, f"RoPE destroyed vector norm! Diff: {diff}"

class TestMHARoPE:
    """Tests for the Multi-Head Attention with RoPE."""

    def test_mha_rope_forward(self, model_params, input_tensor):
        mha = MultiHeadAttentionROPE(
            d_model=model_params['d_model'],
            seq_len=model_params['seq_len'],
            num_heads=model_params['n_heads'],
            dropout=model_params['dropout']
        )
        
        output = mha(input_tensor)
        assert output.shape == input_tensor.shape
        
    def test_odd_head_dim_assertion(self):
        """
        Verify that the code raises an AssertionError if head_dim is odd.
        This confirms your 'interview question' assert logic works.
        """
        # 33 is not divisible by 2 (Complex number pairing impossible)
        d_model_bad = 33 
        n_heads = 1
        
        with pytest.raises(AssertionError) as excinfo:
            MultiHeadAttentionROPE(d_model=d_model_bad, seq_len=10, num_heads=n_heads)
        
        assert "should be an even number" in str(excinfo.value)

    def test_divisibility_assertion(self):
        """Verify d_model must be divisible by n_heads."""
        d_model = 32
        n_heads = 5 # 32 % 5 != 0
        
        with pytest.raises(AssertionError) as excinfo:
             MultiHeadAttentionROPE(d_model=d_model, seq_len=10, num_heads=n_heads)
             
        assert "should be a multiple" in str(excinfo.value)