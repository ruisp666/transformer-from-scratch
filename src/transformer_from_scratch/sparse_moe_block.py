from torch import nn
import torch

from transformer_from_scratch.multi_head_attention_rope import MultiHeadAttentionROPE
from transformer_from_scratch.rms_norm import RMSNorm
from transformer_from_scratch.decoder import ModernDecoder
from transformer_from_scratch.moe import MoELayer 

class SparseMoEBlock(ModernDecoder):
    """
    A Sparse MoE Transformer Block (Mixtral/Grok Style).
    
    This replaces the standard MLP/FFN with a Sparse Mixture-of-Experts layer.
    
    Key Changes from ModernDecoder:
    1. Replaced FFNSwiGlu with MoELayer.
    2. Forward pass returns (output, aux_loss) tuple instead of just output.
    """
    def __init__(self, d_model, n_heads, n_experts, capacity_factor, dropout, seq_len):
        # 1. Initialize Parent
        # We pass expansion_factor=0 or any dummy value since we'll replace the FFN immediately.
        super().__init__(d_model, n_heads, dropout, seq_len, expansion_factor=4)
        
        # 2. Remove the Dense FFN (Save memory/params)
        del self.swigluffn
        
        # 3. Add the Sparse MoE
        self.moe = MoELayer(d_model, n_experts, capacity_factor)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, D_Model)
        
        # Slice mask for current batch length
        batch_seq = x.shape[1]
        sliced_mask = self.causal_mask[:batch_seq, :batch_seq]

        # -------------------------------------------------------------------
        # PART 1: Attention (Standard)
        # -------------------------------------------------------------------
        # Norm -> Attention -> Add Residual
        norm_x = self.norm_1(x)
        causal_att = self.attention_causal(norm_x, mask=sliced_mask)
        res_1 = x + causal_att

        # -------------------------------------------------------------------
        # PART 2: Sparse MoE (The Change)
        # -------------------------------------------------------------------
        # Norm -> MoE -> Add Residual
        norm_res_1 = self.norm_2(res_1)
        
        # The MoE layer returns both the content and the load balancing loss
        moe_out, aux_loss = self.moe(norm_res_1)
        
        # CRITICAL: We perform the residual connection here.
        # If a token was dropped by MoE, moe_out is 0. 
        # So: x + 0 = x (The token skips the layer unchanged).
        out = res_1 + moe_out
        
        # We must return the aux_loss so it can be bubbled up to the training loop
        return out, aux_loss
    
if __name__ == "__main__":
    # Sanity Check
    d_model = 64
    n_heads = 4
    n_experts = 4
    capacity_factor = 1.0
    dropout = 0.1
    seq_len = 10
    
    block = SparseMoEBlock(d_model, n_heads, n_experts, capacity_factor, dropout, seq_len)
    x = torch.randn(2, seq_len, d_model)
    
    out, loss = block(x)
    print(f"Block Output: {out.shape}")
    print(f"Aux Loss: {loss.item():.4f}")