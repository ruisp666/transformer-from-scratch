import torch.nn as nn
import torch
import torch.functional as F
from transformer_from_scratch.multi_head_attention import scaled_dot_attention
from transformer_from_scratch.rotary_positional_embeddings import RoPE


class MultiHeadAttentionROPE(nn.Module):
    """
    Multi-head attention mechanism.
    It implements GQA
    When converting a multi-head checkpoint to a GQA checkpoint,
      we construct each group key and value head by mean- pooling all
        the original heads within that group. Recall you 
        pool at the dimension level so across heads.
    
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
        return x.reshape(-1, x.shape[1], self.num_heads, self.head_dim).permute(0,2,1,3).contiguous()
    
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
        return x.transpose(2,1).contiguous().view(x.shape[0], -1, self.d_model )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, n_groups=None) -> torch.Tensor:
        """
        Parameters
        ----------
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len, seq_len) or (seq_len, seq_len)
            gqa_g: the number of groups as per Grouped Query Attention

        Returns
        -------
            output: (batch, seq_len, d_model)
            
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K and V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Split into heads and apply RoPE shape:  (batch, num_heads, seq_len, head_dim)
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        
        # Compute GQA
        if n_groups is not None:
            # (batch_size, num_heads, seq_len, head_dim) -> (batch_size, n_groups, heads_per_group, seq_len, head_dim)
            K = K.view(batch_size, n_groups, self.num_heads//n_groups, seq_len, self.head_dim)

            # (batch_size, n_groups, heads_per_group, seq_len, head_dim) -> (batch_size, n_groups, 1, seq_len, head_dim)
            K = K.mean(dim=2, keepdim=True)

            # Squeeze, rope it, and then unsqueeze for broadcast with Q
            K_rope = self.rope(K.squeeze(2))
            K_rope = K_rope.unsqueeze(2)

            # V same treatment as K
            V = V.view(batch_size, n_groups,  self.num_heads//n_groups, seq_len, self.head_dim).mean(dim=2, keepdim=True)

            # (batch_size, num_heads, seq_len, head_dim) - > (batch_size, n_groups, heads_per_group, seq_len, head_dim)
            Q_rope = self.rope(Q)
            Q_rope = Q_rope.view(batch_size,  n_groups, self.num_heads//n_groups, seq_len, self.head_dim)
    

            # scaled_dot_attention returns (attention, attention_weights)
            mha = (scaled_dot_attention(Q_rope,K_rope,V,mask=mask)[0]).view(batch_size, self.num_heads, seq_len, self.head_dim)
        else:

            # Apply Rope to softmax of Q and Rope to Exp of K (as per RoFormer approach)
            Q_rope = self.rope(Q)
            K_rope = self.rope(K)
            mha = scaled_dot_attention(Q_rope,K_rope,V,mask=mask)[0]

        # Compute scaled dot with rope and combine heads acros head dimension
        multihead_scaled_rope_attention = self.combine_heads(mha)

        # Output
        return self.W_o(self.dropout(multihead_scaled_rope_attention))



