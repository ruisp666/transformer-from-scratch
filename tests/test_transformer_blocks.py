import pytest
import torch
from transformer_from_scratch.transformer_blk import (
    TransformerBlock, 
    RoPETransformerBlock, 
    RoPESwiGluTransformerBlock
)

class TestTransformerArchitectures:
    
    @pytest.mark.parametrize("style", ["Pre-Ln", "Post-Ln"])
    @pytest.mark.parametrize("norm", ["layer", "rms"])
    def test_base_block_combinations(self, model_params, input_tensor, style, norm):
        """
        Test the flexible Base Block with different configs.
        """
        block = TransformerBlock(
            d_model=model_params['d_model'],
            n_heads=model_params['n_heads'],
            style=style,
            norm=norm
        )
        output = block(input_tensor)
        assert output.shape == input_tensor.shape
        
        # Verify Norm Types
        if norm == 'rms':
            # Check if it actually instantiated RMSNorm (by checking class name)
            assert "RMSNorm" in str(type(block.norm_layer_1))

    def test_rope_swiglu_block(self, model_params, input_tensor):
        """
        Test the modern Llama-style block.
        """
        block = RoPESwiGluTransformerBlock(
            d_model=model_params['d_model'],
            n_heads=model_params['n_heads'],
            dropout=0.1,
            style="Pre-Ln",
            norm="rms",
            seq_len=model_params['seq_len'],
            expansion_factor=8/3
        )
        
        output = block(input_tensor)
        assert output.shape == input_tensor.shape

        # Verify SwiGLU initialization (Hidden dim should be ~2.66x d_model)
        # 32 * 8/3 = 85.33 -> 85
        # Check internal SwiGLU dimension
        expected_hidden = int(model_params['d_model'] * 8/3)
        actual_hidden = block.ffn.swiglu_layer.W_gate.shape[1] # Output dim of gate
        
        assert actual_hidden == expected_hidden

    def test_overfitting_sanity_check(self, model_params):
        """
        A 'smoke test' to see if the block can learn (gradients flow).
        We try to overfit a single batch to zero loss.
        """
        block = RoPESwiGluTransformerBlock(
            d_model=model_params['d_model'],
            n_heads=model_params['n_heads'],
            dropout=0.0, # Disable dropout for deterministic overfitting
            style="Pre-Ln",
            norm="rms",
            seq_len=model_params['seq_len'],
            expansion_factor=8/3
        )
        
        # Input and Target
        x = torch.randn(1, model_params['seq_len'], model_params['d_model'])
        target = torch.randn(1, model_params['seq_len'], model_params['d_model'])
        
        optimizer = torch.optim.Adam(block.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()
        
        # Run 50 steps
        initial_loss = criterion(block(x), target).item()
        
        for _ in range(50):
            optimizer.zero_grad()
            output = block(x)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
        final_loss = loss.item()
        
        # Loss should decrease significantly
        assert final_loss < initial_loss
        assert final_loss < 0.1 # Arbitrary threshold for "learning happened"