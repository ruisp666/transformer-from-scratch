import torch.nn as nn
import torch
import torch.nn.functional as F


def delta_chunk_attention(Q: torch.tensor, K: torch.tensor, V: torch.tensor, beta, C, mask=None):
    """
    Implements delta chunk attention
    
    Parameters
    ----------
    Q : torch.Tensor
        Query tensor of shape (..., Seq_Len_Q, d_k).
        The leading dimensions (...) typically represent (Batch, Heads).
    K : torch.Tensor
        Key tensor of shape (..., Seq_Len_K, d_k).
        The d_k dimension must match Q.
    V : torch.Tensor
        Value tensor of shape (..., Seq_Len_K, d_v).
        The Seq_Len_K dimension must match K.
    mask : torch.Tensor, optional
        Mask tensor of shape (seq_len, seq_len). 
        Positions with False/0 will be masked (set to -inf)
    beta: torch.Tensor
        Controls forgetting, it should be (batch, num_heads_C,C, C)
    C: int
        Size of the chunk in tokens

    
    Returns
    -------
    output : torch.Tensor
        Attention output of shape matching V
    attention_weights : torch.Tensor
        delta attention weights of shape (seq_len, seq_len)
    
    Notes
    -----
    Implements chunked delta gate
    """
    # Normalize L-2 for stability "We don't have softmax to save us anymore"
    K = F.normalize(K,p=2,dim=-1)
    V = F.normalize(V,p=2,dim=-1)
    B, H, seq_len,dim = Q.shape
    if seq_len % C != 0:
        raise ValueError('The chunk size should be a factor of seq_len')
    S = torch.zeros(B, H, dim, dim, device=Q.device, dtype=Q.dtype)
    O = torch.zeros(B, H, seq_len, dim, device=Q.device, dtype=Q.dtype)
    causal_mask = torch.tril(torch.ones(C, C, device=Q.device, dtype=Q.dtype))
    eye = torch.eye(C, device=Q.device, dtype=Q.dtype).unsqueeze(0).unsqueeze(0)
    eye_S = torch.eye(dim,  device=Q.device, dtype=Q.dtype).unsqueeze(0).unsqueeze(0)
    for i in range(seq_len// C):
        start = i*C
        end = (i + 1) * C
        Q_chunk = Q[:,:,start:end,:]
        K_chunk = K[:,:,start:end,:]
        V_chunk = V[:,:,start:end,:]
        beta_chunk = beta[:,:,start:end ]
        # beta_chunk shape: (B, H, C). We unsqueeze to (B, H, C, 1) to broadcast over dim
        beta_K = beta_chunk.unsqueeze(-1) * K_chunk
        # A is (C,D)
        A = torch.tril(beta_K @ K_chunk.transpose(-1,-2),diagonal=-1)

        # This inverse here (1-A)^-1 is the same as when you compute total effect from a causal adjency matrix A
        # think about it as the sum of the powers of A (Neumann series)
        T = torch.linalg.solve_triangular(eye - A, eye, upper=False)
        # This is literally a scalar multiplication on the dim vector
        T_pre = T * beta_chunk.unsqueeze(-2)
        W = T_pre @ K_chunk
        U = T_pre @ V_chunk
        # Need to compute output on the input S
        # O_chunk is (B,H, C, dim)
        O_chunk  = Q_chunk @ S.transpose(-2,-1) + (Q_chunk @ K_chunk.transpose(-1,-2) * causal_mask) @ (U - W@ S.transpose(-2,-1))

        # S is (dim,dim) so the identity needs to be (dim,dim)
        S = S @ (eye_S -  W.transpose(-2,-1) @ K_chunk) + U.transpose(-2,-1) @ K_chunk
        O[:,:,start:end,:] = O_chunk
    return O

    

    


class DeltaNetLayer(nn.Module):
    """
    Delta net Layer
    
    Parameters
    ----------
        d_model: Total dimension of the model (e.g. 512).
        num_heads: Number of attention heads (e.g. 8)
        dropout: Dropout probability (default 0.1)
    """
    def __init__(self, d_model: int,
                  num_heads: int, chunk_size: int, dropout: float = 0.1) -> torch.Tensor:
        super().__init__()

        # Verify d_model is divisible by num_heads
        assert d_model%num_heads==0 , f'The dimension {d_model} should be a multiple of {num_heads}'

        # Calculate head dimension
        head_dim = d_model//num_heads

        # Create projection matrices (no bias)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Beta for the writing intensity (think multi-head, the State should be though about as per head)
        # This has bias
        self.W_beta = nn.Linear(d_model, num_heads, bias=True)

        # ToDo 4: Dropout Layer

        self.dropout = nn.Dropout(p=dropout)

        # ToDo 5: Store hyperparams
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size

    
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

        # Project to Q, K and V and beta,split heads, and permute so that is per head per token 
        split_shape = (x.shape[0], x.shape[1], self.num_heads, self.head_dim)
        Q = self.W_q(x).view(*split_shape).permute(0, 2, 1, 3)
        K = self.W_k(x).view(*split_shape).permute(0, 2, 1, 3)
        V = self.W_v(x).view(*split_shape).permute(0, 2, 1, 3)

        # Transform into 0-1 space
        beta  = torch.sigmoid(self.W_beta(x)).permute(0,2,1)

        gated_delta_output = self.combine_heads(delta_chunk_attention(Q,K,V,beta,self.chunk_size, mask=None))

        # Output
        return self.W_o(self.dropout(gated_delta_output))


if __name__=='__main__':
    d_model = 12
    num_heads = 3
    chunk_size = 2
    seq_len = 16

    X = torch.randn(5,seq_len,d_model)
    gated_linear = DeltaNetLayer(d_model, num_heads,chunk_size)
    test = gated_linear(X)
    print(test.shape)
    print('OK')