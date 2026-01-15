import torch.nn as nn
import torch
import torch.functional as F
from transformer_from_scratch.multi_head_attention import scaled_dot_attention
from transformer_from_scratch.rotary_positional_embeddings import RoPE


class MultiHeadAttentionROPE(nn.Module):
    """
    Multi-head attention mechanism.
    
    Parameters
    ----------
        d_model: Total dimension of the model (e.g. 512).
        num_heads: Number of attention heads (e.g. 8)
        dropout: Dropout probability (default 0.1)
    """
    def __init__(self, d_model: int, seq_len: int,
                  num_heads: int, dropout: float = 0.1) -> torch.Tensor:
        super().__init__()

        # Verify d_model is divisible by num_heads
        assert d_model%num_heads==0 , f'The dimension {d_model} should be a multiple of {num_heads}'

        # Calculate  head dimension
        head_dim = d_model//num_heads
        assert head_dim%2==0, f'The head dimension {d_model//num_heads} should be an even number'


        # Create projection matrices (no bias)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # RoPE Layer
        self.rope = RoPE(seq_len=seq_len, d_model=head_dim)

        # Dropout Layer
        self.dropout = nn.Dropout(p=dropout)


        # Store hyperparams
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

        # ToDo 7: Project to Q, K and V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # ToDo 8: Split into heads and apply RoPE
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # Apply Rope to softmax of Q and Rope to Exp of K (as per RoFormer approach)
        Q_rope = self.rope(Q)
        K_rope = self.rope(K)
        

        # Todo 9: Compute scaled dot with rope
        multihead_scaled_rope_attention = self.combine_heads(scaled_dot_attention(Q_rope,K_rope,V,mask=mask)[0])

        # Output
        return self.W_o(self.dropout(multihead_scaled_rope_attention))


