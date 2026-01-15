from transformer_from_scratch.activations import FFNSwiGlu 
from transformer_from_scratch.multi_head_attention_rope import MultiHeadAttentionROPE
from transformer_from_scratch.rms_norm import RMSNorm
from torch import nn
import torch


class ModernDecoder(nn.Module):
    """
    A Modern Decoder-Only Block (Llama/Mistral Style).
    
    References:
    ----------
    - Architecture (Llama): "LLaMA: Open and Efficient Foundation Language Models" (Touvron et al., 2023)
    - Normalization (RMSNorm): "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)
    - Positional Embeddings (RoPE): "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
    - Activation (SwiGLU): "GLU Variants Improve Transformer" (Shazeer, 2020)
    
    Key Architectural Differences from Vaswani et al. (2017):
    1. Pre-Normalization: RMSNorm is applied BEFORE Attention and FFN.
    2. No Cross-Attention: Self-Attention handles all context.
    3. RoPE: Rotates Q and K vectors instead of adding absolute position embeddings.
    4. SwiGLU: Gated Linear Unit activation replaces standard ReLU/GELU.
    """
    def __init__(self, d_model, n_heads, dropout, seq_len, expansion_factor):
        super().__init__()
        
        # 1. Causal Self-Attention (The "Brain")
        # Uses RoPE internally to handle position information.
        self.attention_causal = MultiHeadAttentionROPE(d_model, seq_len, n_heads, dropout)
        
        # 2. Feed Forward Network (The "Memory")
        # Uses SwiGLU (Swish Gated Linear Unit) for better performance.
        self.swigluffn = FFNSwiGlu(d_model, expansion_factor)
        
        # 3. RMSNorm (The "Stabilizer")
        # Pre-Norm architecture requires two norm layers per block.
        self.norm_1 = RMSNorm(d_model)
        self.norm_2 = RMSNorm(d_model)
        
        # -----------------------------------------------------------------------
        # CAUSAL MASK IMPLEMENTATION DETAILS
        # -----------------------------------------------------------------------
        # We create a BOOLEAN mask where:
        #   True (1)  = Visible/Allowed (The Past and Present)
        #   False (0) = Hidden/Blocked (The Future)
        #
        # Shape: Lower Triangular Matrix (tril)
        # [[1, 0, 0],  <- Token 1 sees only itself
        #  [1, 1, 0],  <- Token 2 sees Token 1 & 2
        #  [1, 1, 1]]  <- Token 3 sees everything so far
        # -----------------------------------------------------------------------
        causal_mask = torch.tril(torch.ones(seq_len, seq_len)) == 1
        
        # We register it as a buffer so it becomes part of the state_dict (saved with model)
        # but is NOT a learnable parameter (gradients won't update it).
        self.register_buffer('causal_mask', causal_mask)

    def forward(self, x):
        # x shape: (Batch, Current_Seq_Len, D_Model)
        
        # -----------------------------------------------------------------------
        # DYNAMIC MASK SLICING
        # -----------------------------------------------------------------------
        # We slice the pre-computed mask to match the current batch's sequence length.
        # This allows the model to handle:
        #   1. Training: Fixed length (e.g., 256) -> uses full mask.
        #   2. Inference: Variable length (e.g., 5) -> uses 5x5 sub-mask.
        # -----------------------------------------------------------------------
        batch_seq = x.shape[1]
        sliced_mask = self.causal_mask[:batch_seq, :batch_seq]

        # -----------------------------------------------------------------------
        # CRITICAL DEPENDENCY: Attention Implementation
        # -----------------------------------------------------------------------
        # Because we pass a BOOLEAN mask (True/False), the underlying 
        # 'scaled_dot_attention' function MUST implement the masking logic as:
        #
        #    scores.masked_fill(mask == 0, float('-inf'))
        #
        # ALTERNATIVE APPROACH (The "Additive" Mask):
        # If your attention function expects to ADD the mask (scores + mask),
        # you would define the mask as Upper Triangular (triu) with -inf values:
        #    mask = torch.triu(torch.ones(...) * float('-inf'), diagonal=1)
        # -----------------------------------------------------------------------
        
        # 1. Pre-Norm -> Causal Attention -> Residual
        # Note: We normalize x BEFORE passing it to attention (Pre-Norm)
        norm_x = self.norm_1(x)
        causal_att = self.attention_causal(norm_x, mask=sliced_mask)
        res_1 = x + causal_att

        # 2. Pre-Norm -> SwiGLU FFN -> Residual
        # We normalize the residual from step 1 before passing to FFN
        norm_res_1 = self.norm_2(res_1)
        out = self.swigluffn(norm_res_1)
        
        return out + res_1

if __name__=="__main__":
    # Quick sanity check
    d_model = 16
    n_heads = 2
    dropout = 0.1
    seq_len = 3
    expansion_factor = 8/3
    
    model = ModernDecoder(d_model, n_heads, dropout, seq_len, expansion_factor)
    print("ModernDecoder initialized successfully.")
    print(f"Mask Shape: {model.causal_mask.shape}")