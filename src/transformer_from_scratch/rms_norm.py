import torch.nn as nn
import torch
import torch.functional as F
import torch.linalg as LA

class RMSNorm(nn.Module):
    """
    RMS Norm.
    
    Parameters
    ----------
        d_model: Total dimension of the model (e.g. 512).
        num_heads: Number of attention heads (e.g. 8)
        dropout: Dropout probability (default 0.1)
    """
    def __init__(self, dim_model, eps=1e-7) -> torch.Tensor:
        super().__init__()
        self.dampener = nn.Parameter(torch.ones(dim_model))
        self.d = dim_model
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes rms norm
        Parameters
        ----------
            x: (batch, seq_len, d_model)

        Returns
        -------
            output: (batch, seq_len, d_model)
            
        """
        d = x.shape[-1]
        # Used LA.norm(x, ord=2, dim=-1) ** 2 which is ineffeciant
        norm_scaled = (x.pow(2).mean(dim=-1) + self.eps )** .5
        # Unsqueeze so that we get vectors divided by scalers at the last dimension 
        # In practice norm_scaled goes from (Batch,Seq_len) to (Batch,Seq_len,1)
        norm_scaled = norm_scaled.unsqueeze(-1)
        x_rms = x / norm_scaled * self.dampener
        return x_rms


def manifold_test(norm_layer, d_model, seq_len):
    
    # Create a random input
    X = torch.randn(2, seq_len, d_model) # (Batch, Seq, Heads, d_model)
    Y = torch.ones_like(X)
    return norm_layer(Y)
    
if __name__=='__main__':
    d_model = 5
    seq_len = 3
    norm_layer = RMSNorm(d_model)
    normed = manifold_test(norm_layer,d_model,seq_len)
    print(normed)
