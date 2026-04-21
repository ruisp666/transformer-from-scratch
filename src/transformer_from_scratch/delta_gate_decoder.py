from transformer_from_scratch.activations import FFNSwiGlu 
from transformer_from_scratch.rms_norm import RMSNorm
from transformer_from_scratch.delta_gate_layer import DeltaNetLayer
from torch import nn
import torch


class DeltaNetDecoder(nn.Module):
    """
    A Modern Delta Net Decoder-Only Block.

    This block serves as a drop-in replacement for a standard Transformer 
    decoder block. It swaps out standard Multi-Head Attention for a 
    Chunkwise Delta Gate Layer (Linear Attention), while maintaining the 
    standard Pre-Norm residual architecture and SwiGLU FFN.
    
    Because the recurrent memory state (S) is encapsulated entirely within 
    the DeltaNetLayer, this block maintains the standard sequential shape 
    contract, making it perfectly suited for Hybrid Transformer architectures.
    Parameters
    ----------
    d_model : int
        The total dimension of the residual stream (model width).
    n_heads : int
        The number of attention heads. d_model must be divisible by n_heads.
    dropout : float
        The dropout probability applied after the delta attention projection.
    expansion_factor : float
        The multiplier for the SwiGLU hidden dimension (typically 8/3).
    chunk_size : int
        The number of tokens per chunk for parallelized WY representation 
        updates. The sequence length must be cleanly divisible by this value.

    Shape
    -----
    - Input: (Batch, Seq_Len, d_model)
    - Output: (Batch, Seq_Len, d_model)
    
    References
    ----------
    - Yang et al. (2024), "Parallelizing Linear Transformers with the Delta 
      Rule over Sequence Length"
    - Yang et al. (2024), "Gated Delta Networks: Improving Mamba2 with 
      Delta Rule"
    """

    def __init__(self, d_model, n_heads, dropout, expansion_factor, chunk_size):
        super().__init__()
        
        # 1. Delta Net (the masks are inside the delta gate function)
        self.delta_attention = DeltaNetLayer(d_model,  n_heads, chunk_size, dropout)
        
        # 2. Feed Forward Network (The "Memory")
        # Uses SwiGLU (Swish Gated Linear Unit) for better performance.
        self.swigluffn = FFNSwiGlu(d_model, expansion_factor)
        
        # 3. RMSNorm (The "Stabilizer")
        # Pre-Norm architecture requires two norm layers per block.
        self.norm_1 = RMSNorm(d_model)
        self.norm_2 = RMSNorm(d_model)
        
       
        
    def forward(self, x):

        # 1. Pre-Norm -> Causal Attention -> Residual
        # Note: We normalize x BEFORE passing it to attention (Pre-Norm)
        norm_x = self.norm_1(x)
        delta_attention = self.delta_attention(norm_x)
        res_1 = x + delta_attention

        # 2. Pre-Norm -> SwiGLU FFN -> Residual
        # We normalize the residual from step 1 before passing to FFN
        norm_res_1 = self.norm_2(res_1)
        out = self.swigluffn(norm_res_1)
        
        return out + res_1

if __name__=="__main__":
   if __name__=="__main__":
    # Quick sanity check
    batch_size = 5
    d_model = 16
    n_heads = 2
    dropout = 0.1
    seq_len = 8      # Changed to 8 so it is cleanly divisible by chunk_size
    chunk_size = 4   
    expansion_factor = 8/3
    
    model = DeltaNetDecoder(d_model, n_heads, dropout, expansion_factor, chunk_size)
    print("DeltaNetDecoder initialized successfully.")
    
    # Create dummy input
    dummy_x = torch.randn(batch_size, seq_len, d_model)
    
    # Run forward pass
    out = model(dummy_x)
    print(f"Input shape:  {dummy_x.shape}")
    print(f"Output shape: {out.shape}")