import torch
import torch.nn.fuctional as F



def scaled_dot_attention(Q: torch.tensor, K: torch.tensor, V: torch.tensor, mask):
    """
    Implements scaled dot-product attention.
    
    Parameters
    ----------
    Q : torch.Tensor
        Query tensor of shape (batch, seq_len, d_k) or (seq_len, d_k)
    K : torch.Tensor
        Key tensor of shape (batch, seq_len, d_k) or (seq_len, d_k)
    V : torch.Tensor
        Value tensor of shape (batch, seq_len, d_v) or (seq_len, d_v)
    mask : torch.Tensor, optional
        Mask tensor of shape (seq_len, seq_len). 
        Positions with False/0 will be masked (set to -inf)
    
    Returns
    -------
    output : torch.Tensor
        Attention output of shape matching V
    attention_weights : torch.Tensor
        Attention weights of shape (seq_len, seq_len)
    
    Notes
    -----
    Implements: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
    """
    d = Q.shape[-1]
    query_times_k_scaled = torch.matmul(Q, K.transpose(-2,-1))* d**(-1/2)
    attention = torch.matmul(F.softmax(query_times_k_scaled, dim=-1), V)
    return d, query_times_k_scaled, attention



