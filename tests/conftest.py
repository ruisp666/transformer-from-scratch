import pytest
import torch

@pytest.fixture(scope="module")
def model_params():
    """Standard hyperparameters for testing."""
    return {
        "d_model": 32,
        "n_heads": 8,
        "seq_len": 10,
        "batch_size": 8,
        "dropout": 0.1
    }

@pytest.fixture
def input_tensor(model_params):
    """A standard random input tensor (Batch, Seq, d_model)."""
    return torch.randn(
        model_params['batch_size'], 
        model_params['seq_len'], 
        model_params['d_model']
    )

@pytest.fixture
def casual_mask(model_params):
    """A standard look-ahead mask."""
    seq_len = model_params['seq_len']
    return torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()