from torch import nn
import math
import matplotlib.pyplot as plt
import torch


class RoPE(nn.Module):
    """Implements Rotary Position Embeddings.
    Critically it uses rotations instead of 
    translations as in the original transformer architecture.
    I"""
    def __init__(self, seq_len, d_model, theta=1e4):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        # Shape is (d//2)
        freq_vector = torch.exp(torch.arange(0,d_model, step=2) * - math.log(theta)/d_model)

        # Unsqueeze pos_vector to (seq_len, 1) so that we can do the outer product
        pos_vector = torch.arange(seq_len).float().unsqueeze(1)

        # Shape is (seq_len, d//2)
        angles = pos_vector * freq_vector

        # Rotate
        freq_bank_rope = torch.exp(torch.tensor(1j) * angles) 

        # Adjust for the (batch, head,seq_len,dim)
        self.register_buffer('freq_bank_rope', freq_bank_rope.unsqueeze(0).unsqueeze(0))


    def forward(self, X, style='interleaved', offset=None):
         # Introduced this stuff for inference to work
         if style == 'interleaved':

            # Interleave, complexify, and rotate
            new_shape = (*X.size()[:-1],-1, 2)
            X_complex = torch.view_as_complex(X.reshape(new_shape))
            if offset is not None:
                X_rotated = X_complex * self.freq_bank_rope[:,:,offset,:]
            else:
                seq_len = X_complex.shape[2]
                X_rotated = X_complex * self.freq_bank_rope[:, :, :seq_len, :]
            # Move back to real and reshape as original 
            return torch.view_as_real(X_rotated).reshape_as(X) 
         elif style == 'half-split':
            # Half-split, complexify, and rotate
            # Use the actual shape of x
            half_split = X.shape[1]//2
            x1 = X[..., :half_split]
            x2 = X[..., half_split:]
            X_complex = torch.complex(x1, x2)
            X_rotated = X_complex * self.freq_bank_rope

            # Extract real and imag and concat at hidden dim
            return torch.cat([X_rotated.real, X_rotated.imag], dim=-1)
         
def manifold_test(rope_layer, d_model, seq_len):
# Create a random input
    X = torch.randn(2, seq_len, 8, d_model) # (Batch, Seq, Heads, d_model)
    
    # Test 1: Norm Preservation (Interleaved)
    X_rot_inter = rope_layer(X, style='interleaved')
    norm_diff_inter = torch.abs(torch.norm(X) - torch.norm(X_rot_inter))
    
    # Test 2: Norm Preservation (Half-split)
    X_rot_half = rope_layer(X, style='half-split')
    norm_diff_half = torch.abs(torch.norm(X) - torch.norm(X_rot_half))
    
    print(f"Norm Difference (Interleaved): {norm_diff_inter.item():.2e}")
    print(f"Norm Difference (Half-split):  {norm_diff_half.item():.2e}")
    
    # Test 3: Relative Distance Property
    # Dot product of position 1 and 2 should be the same as 3 and 4 
    # if the content is the same, because distance is constant (1)
    q = torch.ones(1, seq_len, 1, d_model)
    k = torch.ones(1, seq_len, 1, d_model)
    
    q_rot = rope_layer(q)
    k_rot = rope_layer(k)
    
    # Attention score pos 1 to 2
    score_1_2 = torch.matmul(q_rot[:, 1, :, :], k_rot[:, 2, :, :].transpose(-1, -2))
    # Attention score pos 3 to 4
    score_3_4 = torch.matmul(q_rot[:, 3, :, :], k_rot[:, 4, :, :].transpose(-1, -2))
    
    dist_diff = torch.abs(score_1_2 - score_3_4).max()
    print(f"Relative Distance Variance:    {dist_diff.item():.2e}")

    
if __name__=='__main__':
    # Run it
    d_model = 16
    seq_len = 10
    rope = RoPE(seq_len, d_model)
    manifold_test(rope, d_model, seq_len)

