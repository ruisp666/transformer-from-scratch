import torch.nn as nn
import torch
import torch.functional as F
from transformer_from_scratch.multi_head_attention import scaled_dot_attention
from transformer_from_scratch.rotary_positional_embeddings import RoPE

class KVCache(nn.Module):
    """
    Implements KVCache.
    Shape: (2, Max_Size, D_Model) -> (Key/Value, Time, Features)
    """
    def __init__(self, d_model: int, max_size: int):
        super().__init__()
        # The first dimension has (K,V)
        self.register_buffer('kvcache', torch.zeros(2, max_size, d_model))

    def insert_at_idx(self,  k=None, v=None, idx=1):
            """Inserts V and K at a given idx"""
            # Helper: Ensure we write flat vectors
            seq_len = k.shape[-2]
            self.kvcache[0, idx: idx + seq_len] = k.view(seq_len, -1)
            self.kvcache[1, idx: idx + seq_len] = v.view(seq_len, -1)

    def retrieve_cache(self, idx_end):
        return self.kvcache[0, :idx_end + 1, :], self.kvcache[1, :idx_end + 1, :]

class MultiHeadAttentionROPE(nn.Module):
    def __init__(self, d_model: int, seq_len: int,
                  num_heads: int, dropout: float = 0.1, max_tokens=1024) -> torch.Tensor:
        super().__init__()

        assert d_model % num_heads == 0
        head_dim = d_model // num_heads
        assert head_dim % 2 == 0

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.rope = RoPE(seq_len=seq_len, d_model=head_dim)
        self.dropout = nn.Dropout(p=dropout)

        # FIX: Use d_model passed in init, not hardcoded 512
        self.KVcache = KVCache(d_model, max_tokens)

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(-1, x.shape[1], self.num_heads, self.head_dim).permute(0,2,1,3).contiguous()
    
    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(2,1).contiguous().view(x.shape[0], -1, self.d_model )
    
    def _process_heads(self, q, k, v, n_groups, batch_size, seq_len):
        """
        Handles Splitting, Reshaping, and GQA Pooling.
        Returns heads ready for RoPE.
        """
        # 1. Standard Split: (Batch, Heads, Seq, Dim)
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # 2. If GQA is active, Pool K and V
        if n_groups is not None:
            # Reshape to (Batch, Groups, Heads_Per_Group, Seq, Dim)
            # Then Mean Pool -> (Batch, Groups, 1, Seq, Dim)
            # Then Squeeze -> (Batch, Groups, Seq, Dim)
            k = k.view(batch_size, n_groups, -1, seq_len, self.head_dim).mean(dim=2).squeeze(2)
            v = v.view(batch_size, n_groups, -1, seq_len, self.head_dim).mean(dim=2).squeeze(2)
        
        return q, k, v
    

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, n_groups=None, kvcache=None, idx=None) -> torch.Tensor:
        """
        Parameters
        ----------
            x: (batch, seq_len, d_model), or (1, seq_len, d_model) if in inference
            mask: (batch, seq_len, seq_len) or (seq_len, seq_len)
            n_groups: the number of groups for Grouped Query Attention
            kvcache: The KV Cache object
            idx: The current token position (required for inference/cache)

        Returns
        -------
            output: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # 1. Project to Q, K and V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 2. Split into heads and apply RoPE shape: (batch, num_heads, seq_len, head_dim)
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # Determine RoPE offset (if inference, we start at 'idx', else 0 or None)
        rope_offset = idx if kvcache is not None else None

        # 3. Handle GQA vs Standard MHA
        if n_groups is not None:
            # --- GQA PATH ---
            
            # (batch_size, num_heads, seq_len, head_dim) -> (batch_size, n_groups, heads_per_group, seq_len, head_dim)
            K = K.view(batch_size, n_groups, self.num_heads//n_groups, seq_len, self.head_dim)
            V = V.view(batch_size, n_groups, self.num_heads//n_groups, seq_len, self.head_dim)

            # (batch_size, n_groups, heads_per_group, seq_len, head_dim) -> (batch_size, n_groups, 1, seq_len, head_dim)
            K_pooled = K.mean(dim=2, keepdim=True)
            V_pooled = V.mean(dim=2, keepdim=True) # Keep V as (B, G, 1, S, D)

            # Squeeze to (B, n_groups, seq_len, head_dim) for RoPE
            # We need this shape for the KV Cache update as well!
            K_squeezed = K_pooled.squeeze(2)
            
            # Apply RoPE (Rotated Key)
            K_rope = self.rope(K_squeezed, offset=rope_offset)
            
            # V treatment: Squeeze to (B, n_groups, seq_len, head_dim) for Cache compatibility
            V = V_pooled.squeeze(2)

            # Handle Q: (batch_size, num_heads, seq_len, head_dim) -> (batch_size, n_groups, heads_per_group, seq_len, head_dim)
            Q_rope = self.rope(Q, offset=rope_offset)
            Q_rope = Q_rope.view(batch_size, n_groups, self.num_heads//n_groups, seq_len, self.head_dim)

        else:
            # --- STANDARD MHA PATH ---
            # Apply Rope to Q and K
            Q_rope = self.rope(Q, offset=rope_offset)
            K_rope = self.rope(K, offset=rope_offset)
            # V remains as is

        # 4. KV Cache Interaction (Inference)
        if kvcache is not None:
            # Insert the current NEW tokens (K_rope and V are already projected, grouped, and rotated)
            # Shapes here are (Batch, Num_Groups/Heads, Seq_Len, Head_Dim)
            kvcache.insert_at_idx(K_rope, V, idx=idx)

            # Retrieve FULL history
            # Returns flattened (1, Total_Seq, Total_Dim)
            K_hist, V_hist = kvcache.retrieve_cache(idx + seq_len -1)
            
            # Reshape History back to 4D for attention
            # We assume Batch=1 for inference
            target_heads = n_groups if n_groups is not None else self.num_heads
            
            # View as (1, Seq, Heads, Dim) -> Transpose to (1, Heads, Seq, Dim)
            K_rope = K_hist.view(1, -1, target_heads, self.head_dim).transpose(1, 2)
            V = V_hist.view(1, -1, target_heads, self.head_dim).transpose(1, 2)

        # 5. Final Broadcast Prep for GQA
        # If we are in GQA mode, K_rope/V are (B, Groups, Seq, Dim).
        # We need to unsqueeze them back to (B, Groups, 1, Seq, Dim) so they broadcast against Q's heads.
        if n_groups is not None:
            K_rope = K_rope.unsqueeze(2)
            V = V.unsqueeze(2)

        # 6. Attention
        # scaled_dot_attention returns (attention, attention_weights)
        mha = scaled_dot_attention(Q_rope, K_rope, V, mask=mask)[0]

        # If GQA, we need to flatten the groups/heads back
        if n_groups is not None:
             mha = mha.view(batch_size, self.num_heads, seq_len, self.head_dim)

        # Compute scaled dot with rope and combine heads across head dimension
        multihead_scaled_rope_attention = self.combine_heads(mha)

        # Output
        return self.W_o(self.dropout(multihead_scaled_rope_attention))


if __name__=='__main__':
    torch.manual_seed(42)

    # Create Model
    # Note: max_tokens must be large enough to hold prompt + generation
    model = MultiHeadAttentionROPE(d_model=512, seq_len=50, num_heads=8, max_tokens=100)

    # --- 1. PRE-FILL (Prompt) ---
    # Input: 2 tokens ("Hello World")
    x_prompt = torch.randn(1, 2, 512)

    # We pass the model's internal cache and start writing at index 0
    out_1 = model(x_prompt, kvcache=model.KVcache, idx=0)

    print(f"Pre-fill Input:  {x_prompt.shape}")
    print(f"Pre-fill Output: {out_1.shape}") # Should be (1, 2, 512)

    # --- 2. DECODE (Generation Step 1) ---
    # Input: 1 new token
    x_gen = torch.randn(1, 1, 512)

    # We write at index 2 (because 0 and 1 are occupied by prompt)
    out_2 = model(x_gen, kvcache=model.KVcache, idx=2)

    print(f"Decode Input:    {x_gen.shape}")
    print(f"Decode Output:   {out_2.shape}") # Should be (1, 1, 512)

    # --- 3. DECODE (Generation Step 2) ---
    x_gen_2 = torch.randn(1, 1, 512)
    out_3 = model(x_gen_2, kvcache=model.KVcache, idx=3)
    print(f"Decode Step 2:   {out_3.shape}")