import torch


# --- 1. The Dummy Model ---
class DummyModel(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, idx):
        """
        Mimics a GPT forward pass.
        Input:  (Batch, Time)
        Output: (Batch, Time, Vocab_Size)
        """
        B, T = idx.shape
        # Just return random logits (noise)
        # This means the model will predict random garbage, which is fine for testing the LOOP.
        return torch.randn(B, T, self.vocab_size)

def get_next_token_id(logits, temperature=1.0, top_k=None):
    # 0. Handle "Greedy" Decoding (Temp = 0) explicitly
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1).item()
        
    # 1. Apply Temperature
    logits_temperature = logits / temperature
    
    # 2. Apply Top-K Filtering
    if top_k is not None:
        v, i = torch.topk(logits_temperature, k=top_k, dim=-1)
        probs = torch.nn.functional.softmax(v, dim=-1)
        sample_idx_in_top_k = torch.multinomial(probs, 1)
        # [CRITICAL FIX] Map back to the original vocabulary index
        # We use gather to pick the index from 'i' corresponding to 'sample_idx'
        # gather(input, dim, index)
        next_token_idx = torch.gather(i, -1, sample_idx_in_top_k).item()
        
    else:
        # Standard sampling from the full distribution
        probs = torch.nn.functional.softmax(logits_temperature, dim=-1)
        # FIX: Added .item() back so this returns an int, not a Tensor
        next_token_idx = torch.multinomial(probs, 1).item()
        
    return next_token_idx


def crop_context(tokens, block_size):
    """
    Args:
        tokens (torch.Tensor): Shape (1, current_seq_len)
        block_size (int): Max context length
    """
    current_seq_len = tokens.shape[1]
    
    # If the context is shorter than the max block size, return it as is
    if current_seq_len <= block_size:
        return tokens
    else:
        # Otherwise, take the last 'block_size' tokens
        return tokens[:, -block_size:]

def generate(model, idx, max_new_tokens, block_size, temperature=1.0, top_k=None):
    """
    idx: (B, T) tensor of indices in the current context
    """
    for _ in range(max_new_tokens):
        # 1. CROP
        # If idx is too long, crop it so the model doesn't crash
        idx_cond = crop_context(idx, block_size)
        
        # 2. FORWARD
        # Get predictions
        with torch.no_grad():
            logits = model(idx_cond)
        
        # 3. PLUCK (The most common error source)
        # The model outputs logits for EVERY token in the sequence: (B, T, V)
        # We only care about the prediction for the LAST token.
        # TODO: Slice logits to get shape (B, V) corresponding to the last step.
        logits_last_step = logits[:,-1,:]
        
        # 4. SAMPLE (Assume get_next_token_id handles the B=1 case for now)
        # For this exercise, just pick the max (Greedy) to keep it simple
        next_token_id= get_next_token_id(logits_last_step, temperature, top_k)
        
        # Convert the int back to a tensor on the correct device
        next_token_tensor = torch.tensor([[next_token_id]], device=idx.device)
        # TODO: Add the new index to the growing sequence
        idx = torch.cat((idx, next_token_tensor), dim=1)

    
    return idx
    


if __name__=='__main__':
    # Setup
    vocab_size = 10
    block_size = 5
    max_new_tokens = 4
    # --- Test 1: Greedy ---
    test_logits = torch.tensor([9.0, 2001.0, 1.0, 2000.0, 6.0, 0.5])
    print(f"Temp 0 (Greedy): {get_next_token_id(test_logits, temperature=0)}")
    # Output: 2001 (Index 1)

    # --- Test 2: Top-K ---
    # With Top-K=2, valid indices are 1 (val 2001) and 3 (val 2000). 
    # Index 0 (val 9.0) is ignored despite being positive.
    print(f"Temp 1.0, Top-K 2: {get_next_token_id(test_logits, temperature=1.0, top_k=2)}")

    # --- Test 3: Context Cropping ---
    long_text = torch.arange(0, 10, 1).unsqueeze(0) # Shape [1, 10]: [[0, 1, ... 9]]
    block_size = 5
    cropped_text = crop_context(long_text, block_size)
    
    print("\n--- Cropping Test ---")
    print(f"Original: {long_text}")
    print(f"Cropped:  {cropped_text}")
    # Expected: tensor([[5, 6, 7, 8, 9]])

    print("\n--Looping---")

    # Initialize the "Fake" Model
    model = DummyModel(vocab_size)
    
    # Create a starting sequence (Batch=1, Length=3)
    # E.g., [1, 2, 3]
    start_context = torch.tensor([[1, 2, 3]])
    
    print(f"Starting context: {start_context}")
    
    # Run Generation
    output = generate(model, start_context, max_new_tokens, block_size)
    
    print(f"Final output shape:     {output.shape}")
    print(f"Generated sequence:     {output}")
    
    # Verification
    expected_length = start_context.shape[1] + max_new_tokens
    if output.shape[1] == expected_length:
        print("SUCCESS: Output length is correct!")
    else:
        print("FAIL: Output length mismatch.")
