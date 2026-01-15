from transformer_from_scratch.decoder import ModernDecoder
from transformer_from_scratch.rms_norm import RMSNorm
from torch import nn

class ModernTransformer(nn.Module):
    """
    A Modern Decoder-Only Transformer (GPT/Llama style).
    
    This architecture serves as the "Chassis" that holds the components together.
    It transforms integer token IDs into probability distributions over the vocabulary.
    
    Architecture Flow:
    1. Embedding: Integers -> Vectors
    2. Stack: N x ModernDecoder Blocks (Processing)
    3. Final Norm: RMSNorm (Stability)
    4. Head: Vectors -> Logits (Prediction)
    
    Attributes:
        embedding_layer (nn.Embedding): Learnable lookup table for tokens.
        decoders (nn.Sequential): The stack of Transformer blocks.
        final_norm (RMSNorm): Pre-normalization before the final projection.
        linear (nn.Linear): Projecting embedding dimension back to vocabulary size.
    """
    def __init__(self, vocab_size, decoder_n, d_model, n_heads, dropout, seq_len, expansion_factor):
        """
        Args:
            vocab_size (int): Size of the vocabulary (e.g., 50257 for GPT-2).
            decoder_n (int): Number of stacked decoder blocks (Depth).
            d_model (int): Dimension of the embedding vector (Width).
            n_heads (int): Number of attention heads.
            dropout (float): Dropout probability.
            seq_len (int): Maximum sequence length (context window).
            expansion_factor (float): Expansion factor for the SwiGLU FFN (usually 8/3).
        """
        super().__init__()
        
        # 1. Embedding Layer
        # Converts token indices (e.g., 502) to dense vectors (e.g., [0.1, -0.5, ...])
        self.embedding_layer = nn.Embedding(vocab_size, d_model)
        
        # 2. The Decoder Stack (The "Brain")
        # We use nn.Sequential because ModernDecoder handles its own causal masking.
        # This simplifies the forward pass significantly.
        self.decoders = nn.Sequential(*[
            ModernDecoder(d_model, n_heads, dropout, seq_len, expansion_factor) 
            for i in range(decoder_n)
        ])
        
        # 3. Final Normalization (Crucial for Pre-Norm Architectures)
        # In Pre-Norm (Llama/GPT-3), the output of the last block is un-normalized.
        # We must normalize it before the final projection to prevent instability.
        self.final_norm = RMSNorm(d_model)
        
        # 4. The Language Model Head (The "Voice")
        # Projects vectors back to the vocabulary size to get logits.
        # Note: We use bias=False, standard for modern LLMs.
        self.linear = nn.Linear(d_model, vocab_size, bias=False)