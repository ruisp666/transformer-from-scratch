from transformer_from_scratch.delta_gate_decoder import DeltaNetDecoder
from transformer_from_scratch.decoder import ModernDecoder
from transformer_from_scratch.rms_norm import RMSNorm
from torch import nn

class HybridTransformer(nn.Module):
    """
    A Hybrid Decoer-Only Transformer
    
    This architecture allows for the combination of linear delta net (G) and standard 
    multi-head attention (A) in a single stack
    
    Architecture Flow:
    1. Embedding: Integers -> Vectors
    2. Stack: N x Blocks (Hybrid sequence of ModernDecoder & DeltaNetDecoder)
    3. Final Norm: RMSNorm (Stability)
    4. Head: Vectors -> Logits (Prediction)
    
    Attributes:
        embedding_layer (nn.Embedding): Learnable lookup table for tokens.
        decoders (nn.Sequential): The stack of Transformer blocks.
        final_norm (RMSNorm): Pre-normalization before the final projection.
        linear (nn.Linear): Projecting embedding dimension back to vocabulary size.
    """
    def __init__(self, vocab_size, d_model, n_heads, dropout, seq_len, expansion_factor,
                  chunk_size, layer_pattern='AAAG'):
        """
        Args:
            vocab_size (int): Size of the vocabulary (e.g., 50257 for GPT-2).
            decoder_n (int): Number of stacked decoder blocks (Depth).
            d_model (int): Dimension of the embedding vector (Width).
            n_heads (int): Number of attention heads.
            dropout (float): Dropout probability.
            seq_len (int): Maximum sequence length (context window).
            expansion_factor (float): Expansion factor for the SwiGLU FFN (usually 8/3).
            chunk_size (int): The chunk size required for the Delta Gate layers.
            layer_pattern (str): A string defining the stack architecture. 
                                 'A' = Standard Attention (ModernDecoder)
                                 'G' = Gated Delta (DeltaNetDecoder)
        """
        super().__init__()
        
        # 1. Embedding Layer
        # Converts token indices (e.g., 502) to dense vectors (e.g., [0.1, -0.5, ...])
        self.embedding_layer = nn.Embedding(vocab_size, d_model)
       # 2. The Decoder Stack (The "Hybrid Brain")
        # We parse the layer_pattern string to dynamically build the network depth.
        blocks = []
        for i, layer_type in enumerate(layer_pattern.upper()):
            if layer_type == 'A':
                blocks.append(ModernDecoder(d_model, n_heads, dropout, seq_len, expansion_factor))
            elif layer_type == 'G':
                blocks.append(DeltaNetDecoder(d_model, n_heads, dropout, expansion_factor, chunk_size))
            else:
                raise ValueError(f"Unrecognized layer type at index {i}: '{layer_type}'. Use 'A' for Attention or 'G' for Delta Gate.")
                
        # nn.Sequential seamlessly routes the forward pass through the mixed blocks
        self.decoders = nn.Sequential(*blocks)
        
        # 3. Final Normalization (Crucial for Pre-Norm Architectures)
        # In Pre-Norm (Llama/GPT-3), the output of the last block is un-normalized.
        # We must normalize it before the final projection to prevent instability.
        self.final_norm = RMSNorm(d_model)
        
        # 4. The Language Model Head (The "Voice")
        # Projects vectors back to the vocabulary size to get logits.
        # Note: We use bias=False, standard for modern LLMs.
        self.linear = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, X):
        embeddings = self.embedding_layer(X)
        last_pre_norm = self.decoders(embeddings)
        return self.linear(self.final_norm(last_pre_norm))

if __name__ == "__main__":
    import torch
    
    # 1. Define tiny model hyperparameters for the sanity check
    vocab_size = 1000
    d_model = 64
    n_heads = 4
    dropout = 0.1
    seq_len = 16
    chunk_size = 4
    expansion_factor = 8/3
    layer_pattern = "AAG"  # 2 Standard Attention layers, 1 Delta Gate layer

    # Parameter counts easy: 12d^2
    
    # 2. Instantiate the Hybrid Transformer
    model = HybridTransformer(
        vocab_size=vocab_size, 
        d_model=d_model, 
        n_heads=n_heads, 
        dropout=dropout, 
        seq_len=seq_len, 
        expansion_factor=expansion_factor,
        chunk_size=chunk_size,
        layer_pattern=layer_pattern
    )
    
    # 3. Architectural Parameter Counts
    # Remember the total parameters is 
    # approximately 12d^2 * L (Layers * (MHA + FFN)) + V*d (embeddings)
    def print_param_counts(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print("\n--- Architectural Parameter Counts ---")
        print(f"Layer Pattern:     {layer_pattern}")
        print(f"Total Parameters:  {total_params:,}")
        print(f"Trainable Params:  {trainable_params:,}")
        
        # Breakdown by component
        embedding_params = sum(p.numel() for p in model.embedding_layer.parameters())
        head_params = sum(p.numel() for p in model.linear.parameters())
        stack_params = sum(p.numel() for p in model.decoders.parameters())
        
        print(f"Embedding Layer:   {embedding_params:,}")
        print(f"Decoder Stack:     {stack_params:,}")
        print(f"LM Head:           {head_params:,}")
        print("--------------------------------------\n")

    print_param_counts(model)
    
    # 4. Simple Shape Check
    batch_size = 2
    
    # Create dummy integer token IDs
    dummy_x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print("--- Shape Check ---")
    print(f"Input tokens shape: {dummy_x.shape} -> (Batch, Seq_Len)")
    
    # Run the forward pass
    logits = model(dummy_x)
    
    print(f"Output logits shape: {logits.shape} -> (Batch, Seq_Len, Vocab_Size)")
    assert logits.shape == (batch_size, seq_len, vocab_size), "Shape mismatch in forward pass!"
    print("Shape check passed successfully! All components are wired correctly.")
