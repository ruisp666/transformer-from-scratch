import torch
import torch.nn.fuctional as F



def scaled_dot_attention(Q: torch.tensor, K: torch.tensor, V: torch.tensor):
    '''Implements scaled dot attention.
    ---------
    Parameters
    ------------
    Q: torch.tensor
        - The query tensor
    K: torch.tensor
        - The key tensor
    V: torch.tensor
        - the value tensor
    ---------
    Returns
    d, query_times_k_scaled, attention: int, torch.tensor, torch.tensor
      - Dimension, Query times Key transposed (for testing), and scaled dot attention
    '''
    d = Q.shape[-1]
    query_times_k_scaled = torch.matmul(Q, K.transpose(-2,-1))* d**(-1/2)
    attention = torch.matmul(F.softmax(query_times_k_scaled, dim=-1), V)
    return d, query_times_k_scaled, attention



