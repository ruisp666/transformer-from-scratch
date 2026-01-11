from transformer_from_scratch.multi_head_attention import MultiHeadAttention
from transformer_from_scratch.scaled_attention import scaled_dot_attention
import torch

def tests_attention():
    batch_size = 5
    dim_model = 12
    num_heads = 4
    seq_len = 7
    shape_1 = batch_size, seq_len, dim_model
    Q = torch.randn(batch_size, seq_len, dim_model)
    K = torch.randn(batch_size, seq_len, dim_model)
    V = torch.randn(batch_size, seq_len, dim_model )
    output, attention_w = scaled_dot_attention(Q,K,V,None)
    print(attention_w.sum(dim=-1))
    print(torch.ones((batch_size, seq_len)))
    print(torch.allclose(attention_w.sum(dim=-1), torch.ones(batch_size, seq_len), 1e-5))


def test_shapes_multihead():
    batch_size = 5
    dim_model = 24
    num_heads = 4
    seq_len = 5
    shape_1 = batch_size, seq_len, dim_model

    x  = torch.randn(shape_1)
    mhda = MultiHeadAttention(dim_model, num_heads)
    output = mhda(x)
    assert output.shape == (batch_size, seq_len, dim_model), f"Expected ({batch_size}, {seq_len}, {dim_model}), got {output.shape}"
    

if __name__ == '__main__':
    test_shapes_multihead()

