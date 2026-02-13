import torch
from torch import nn
from transformer_from_scratch.rms_norm import RMSNorm
# Import your new block
from sparse_moe_block import SparseMoEBlock

class SparseMoETransformer(nn.Module):
    """
    A Sparse Mixture-of-Experts Transformer (Mixtral Style).
    
    Key Changes from ModernTransformer:
    1. Uses SparseMoEBlock instead of ModernDecoder.
    2. Manual forward loop to handle (output, loss) tuples.
    3. Returns both Logits and Total Auxiliary Loss.
    """
    def __init__(self, vocab_size, n_layers, d_model, n_heads, n_experts, capacity_factor, dropout, seq_len):
        super().__init__()
        
        # 1. Embedding Layer
        self.embedding_layer = nn.Embedding(vocab_size, d_model)
        
        # 2. The Decoder Stack (The "Brain")
        # We usage nn.ModuleList because we need to handle the tuple return (x, loss) manually.
        self.layers = nn.ModuleList([
            SparseMoEBlock(d_model, n_heads, n_experts, capacity_factor, dropout, seq_len)
            for _ in range(n_layers)
        ])
        
        # 3. Final Normalization
        self.final_norm = RMSNorm(d_model)
        
        # 4. The Language Model Head
        self.linear = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        # x shape: (Batch, Seq_Len)
        
        # 1. Embed tokens
        x = self.embedding_layer(x)
        
        # 2. Run through MoE Stack
        total_aux_loss = 0.0
        
        for layer in self.layers:
            # Pass input through the block
            x, layer_loss = layer(x)
            
            # Accumulate the load balancing loss from this layer
            total_aux_loss += layer_loss
            
        # 3. Final Norm
        x = self.final_norm(x)
        
        # 4. Project to Vocabulary
        logits = self.linear(x)
        
        # Return both the predictions and the structural cost
        return logits, total_aux_loss

if __name__ == "__main__":
    # Tiny Config for Verification
    vocab_size = 1000
    n_layers = 2
    d_model = 64
    n_heads = 4
    n_experts = 4
    capacity_factor = 1.0
    dropout = 0.1
    seq_len = 10
    
    model = SparseMoETransformer(vocab_size, n_layers, d_model, n_heads, n_experts, capacity_factor, dropout, seq_len)
    
    # Fake Input (Batch=2, Seq=10)
    x = torch.randint(0, vocab_size, (2, seq_len))
    
    logits, total_loss = model(x)
    
    print(f"Logits Shape: {logits.shape}") # Should be (2, 10, 1000)
    print(f"Total Load Balancing Loss: {total_loss.item():.4f}")