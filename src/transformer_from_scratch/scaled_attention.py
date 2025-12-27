import torch
import torch.nn.functional as F



def scaled_dot_attention(Q: torch.tensor, K: torch.tensor, V: torch.tensor, mask=None):
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
    if mask is not None:
        query_times_k_scaled = query_times_k_scaled.masked_fill(mask==0, -1e-9)
    attention_w = F.softmax(query_times_k_scaled, dim=-1)
    output = torch.matmul(attention_w, V)
    return output, attention_w



