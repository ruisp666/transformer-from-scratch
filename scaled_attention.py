import torch
import torch.nn.fuctional as F



def scaled_dot_attention(Q: torch.tensor, K: torch.tensor, V: torch.tensor, mask):
    """
    Implements scaled dot-product attention.
    
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



