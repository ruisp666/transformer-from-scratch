
import torch
from transformer_from_scratch.transformer_blk import TransformerBlock, RoPESwiGluTransformerBlock
from transformer_from_scratch.positional_embeddings import SinusoidalPositionalEncoding
from transformer_from_scratch.utils import plot_training_results
from transformer_from_scratch.data_generators import ReversalDataset, RepeatCopyDataset
from torch import nn
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader



def train_model(model, train_loader, vocab_size, n_epochs=20, lr=1e-3, device="cpu"):
    """
    A generic training loop that accepts any model and any dataloader.
    Returns the loss history.
    """
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_model = nn.CrossEntropyLoss()
    
    loss_history = []
    
    print(f"Starting training on {device}...")
    
    for epoch in range(n_epochs):
        epoch_loss = 0
        batch_count = 0
        
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(x_batch)
            
            # Loss Calculation (Flattening Trick)
            # Flatten predictions to (Batch*Seq, Vocab) and targets to (Batch*Seq)
            output_flat = output.view(-1, vocab_size)
            y_flat = y_batch.view(-1)
            
            L = loss_model(output_flat, y_flat)
            L.backward()
            optimizer.step()
            
            loss_history.append(L.item())
            epoch_loss += L.item()
            batch_count += 1
            
        avg_loss = epoch_loss / batch_count
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs} | Avg Loss: {avg_loss:.4f}")
            
    return loss_history

# --- The Main Loop Experiment ---
def run_experiment_loop(task_name="repeat_copy"):
    # 1. Hyperparameters
    d_model = 64
    seq_len = 30
    vocab_size = 20
    batch_size = 32
    n_epochs = 15
    n_layers = 10
    device = "cuda" if torch.cuda.is_available() else "mps" if  torch.backends.mps.is_available()  else "cpu"
    lr = 1e-3
    
    print(f"--- Running Experiment: {task_name.upper()} ---")

    # 2. Select Dataset
    if task_name == "repeat_copy":
        # RoPE should win here (Relative pattern)
        dataset = RepeatCopyDataset(2000, seq_len, vocab_size)
        plot_title = "Architecture Battle: Pattern Copying (Relative Task)"
    elif task_name == "reversal":
        # Standard should win here (Absolute Index Task)
        dataset = ReversalDataset(2000, seq_len, vocab_size)
        plot_title = "Architecture Battle: Sequence Reversal (Absolute Task)"
    else:
        raise ValueError("Unknown task")
        
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 3. Define Models in a Dictionary (The Loop Target)
    models_config = {
        "1. Baseline (No Pos)": nn.Sequential(
            nn.Embedding(vocab_size, d_model),
            *[TransformerBlock(d_model, n_heads=4, dropout=0.0, style='Pre-Ln') for _ in range(n_layers)],
            nn.Linear(d_model, vocab_size)
        ),
        "2. Standard (Absolute)": nn.Sequential(
            nn.Embedding(vocab_size, d_model),
            SinusoidalPositionalEncoding(seq_len, d_model),
            *[TransformerBlock(d_model, n_heads=4, dropout=0.0, style='Pre-Ln', norm='layer') for _ in range(n_layers)],
            nn.Linear(d_model, vocab_size)
        ),
        "3. Modern (RoPE + SwiGLU)": nn.Sequential(
            nn.Embedding(vocab_size, d_model),
            # RoPE is internal to the block
            *[RoPESwiGluTransformerBlock(d_model, n_heads=4, dropout=0.0, style='Pre-Ln', norm='rms', seq_len=seq_len, expansion_factor=8/3) for _ in range(n_layers)],
            nn.Linear(d_model, vocab_size)
        )
    }

    # 4. The Loop
    histories = {}
    for name, model in models_config.items():
        print(f"Training: {name}...")
        histories[name] = train_model(model, loader, vocab_size, n_epochs, lr, device)

    # 5. Plot
    plot_training_results(histories, title=plot_title, save_path=f"labs-viz/plots/{task_name}_results.png")

if __name__ == '__main__':
    run_experiment_loop(task_name="reversal")
    run_experiment_loop(task_name="repeat_copy")

