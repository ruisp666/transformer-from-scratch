import torch
import tiktoken
import sys
import os

# --- 1. Path Setup ---
# Add the 'src' directory to the python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from transformer_from_scratch.transformer import ModernTransformer
from transformer_from_scratch.config import TrainingConfig
from transformer_from_scratch.inference import generate

def load_model(checkpoint_path, device):
    """
    Loads the model architecture and weights from a checkpoint.
    """
    # 1. Load Config
    # We use 'nano' because that matches your trained checkpoints.
    # If you switch to 'base' training later, update this line!
    config = TrainingConfig.tinystories() 
    
    # 2. Initialize Model
    print(f"Initializing model with {config.run_name} config...")
    model = ModernTransformer(
        vocab_size=config.vocab_size,
        decoder_n=config.n_layers,
        d_model=config.d_model,
        n_heads=config.n_heads,
        dropout=config.dropout,
        seq_len=config.seq_len,
        expansion_factor=config.expansion_factor
    )
    
    # 3. Load Weights
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    print(f"Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract state_dict and fix any compile prefixes
    state_dict = checkpoint['model_state_dict']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model, config

def main():
    # --- Settings ---
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else "cpu"

    # --- Robust Path Construction ---
    # Get the directory where THIS script lives (labs-viz/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go up one level to root, then down into checkpoints
    # This works regardless of where you run the python command from
    project_root = os.path.join(script_dir, '..')
    checkpoint_dir = os.path.join(project_root, 'checkpoints')
    
    # EXACT filename (Check your folder to be sure!)
    checkpoint_name = "tinystories-base-70M_latest.pt" 
    
    CHECKPOINT_PATH = os.path.join(checkpoint_dir, checkpoint_name)
    
    print(f"Looking for checkpoint at: {os.path.abspath(CHECKPOINT_PATH)}")

    # --- Load ---
    try:
        model, config = load_model(CHECKPOINT_PATH, device)
    except FileNotFoundError as e:
        print(e)
        return

    # --- Run Inference ---
    enc = tiktoken.get_encoding("gpt2")
    
    while True:
        print("\n" + "="*50)
        start_text = input("Prompt (or 'q' to quit): ")
        if start_text.lower() in ['q', 'quit']:
            break
        
        if not start_text.strip():
            start_text = "The King"

        print("-" * 50)
        print("Generating...", end="", flush=True)

        # Encode
        start_ids = enc.encode(start_text)
        x = torch.tensor([start_ids], dtype=torch.long, device=device)

        # Generate
        y = generate(
            model, 
            x, 
            max_new_tokens=150, 
            block_size=config.seq_len, 
            temperature=0.8, 
            top_k=25
        )

        # Decode
        output_text = enc.decode(y[0].tolist())
        print("\r" + output_text)
        print("="*50)

if __name__ == "__main__":
    main()