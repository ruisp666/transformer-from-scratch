import torch.nn as nn
import torch
from transformer_from_scratch.scaled_attention import scaled_dot_attention

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.
    
    Parameters
    ----------
        d_model: Total dimension of the model (e.g. 512).
        num_heads: Number of attention heads (e.g. 8)
        dropout: Dropout probability (default 0.1)
    """
    def __init__(self, d_model: int,
                  num_heads: int, dropout: float = 0.1) -> torch.Tensor:
        super().__init__()

        # ToDo 1: Verify d_model is divisible by num_heads
        assert d_model%num_heads==0 , f'The dimension {d_model} should be a multiple of {num_heads}'

        # ToDo 2: calculate head dimension
        head_dim = d_model//num_heads

        # ToDo 3: Create projection matrices (no bias)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # ToDo 4: Dropout Layer

        self.dropout = nn.Dropout(p=dropout)

        # ToDo 5: Store hyperparams
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split the last dimension into (num_heads, head_dim).
        Transpose to get dimensions (batch, num_heads, seq_len, head_dim)
        
        Parameters
        ---------
            x: (batch, seq_len, d_model)
        
        Returns
        -------
            (batch, num_heads, seq_len, head_dim)
        """

        # Split the last dimension in self.num_heads
        return x.reshape(-1, x.shape[1], self.num_heads, self.head_dim).permute(0,2,1,3)
    
    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Merge back the heads into (n_dim).
        
        Parameters
        --------
            x: (batch, num_heads, seq_len, head_dim)
            
        Returns
        -------
            (batch, seq_len, d_model)
        """
        # could use .permute(0,2,1,3)
        return x.transpose(2,1).reshape(x.shape[0], -1, self.d_model )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Parameters
        ----------
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len, seq_len) or (seq_len, seq_len)

        Returns
        -------
            output: (batch, seq_len, d_model)
            
        """
        batch_size, seq_len, d_model = x.shape

        # ToDo 7: Project to Q, K and V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # ToDo 8: Split into heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # Shape (batch, seq_len, d_dimenion)
        multihead_attention = self.combine_heads(scaled_dot_attention(Q,K,V,mask=None)[0])

        # Output
        return self.W_o(self.dropout(multihead_attention))


