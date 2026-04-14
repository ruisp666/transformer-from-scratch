import torch
import os
from transformer_from_scratch.config import TrainingConfig
from transformer_from_scratch.trainer import Trainer
from transformer_from_scratch.sparse_moe_transformer import SparseMoETransformer 
from transformer_from_scratch.data_utils import get_text_loaders

def main():
    # -------------------------------------------------------------------------
    # 1. Configuration
    # -------------------------------------------------------------------------
    cfg = TrainingConfig.tinystories()

    # Apply MoE Overrides
    cfg.n_experts = 4
    cfg.capacity_factor = 1.15
    cfg.aux_loss_coef = 0.05
    cfg.batch_size = 32
    cfg.epochs = 3  # Run 3 total epochs
    
    # Update Run Name for clarity in WandB
    cfg.run_name = "tinystories-moe-4exp-v2-non-ovelapping"

    print(f"--- MoE Training Configuration ---")
    print(f"Experts: {cfg.n_experts} | Capacity Factor: {cfg.capacity_factor}")
    print(f"Aux Loss Coef: {cfg.aux_loss_coef}")
    print(f"Batch Size: {cfg.batch_size}")
    print(f"Epochs: {cfg.epochs}")

    # -------------------------------------------------------------------------
    # 2. Data Pipeline
    # -------------------------------------------------------------------------
    train_loader, val_loader, vocab_size = get_text_loaders(
        file_path=cfg.input_file_path,
        batch_size=cfg.batch_size,
        seq_len=cfg.seq_len
    )

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    model = SparseMoETransformer(
        vocab_size=vocab_size, 
        n_layers=cfg.n_layers,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_experts=cfg.n_experts,
        capacity_factor=cfg.capacity_factor,
        dropout=cfg.dropout,
        seq_len=cfg.seq_len
    )
    
    # Device Placement
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Moving model to {device}...")
    model.to(device)

    # -------------------------------------------------------------------------
    # 4. Optimizer
    # -------------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg.lr, 
        weight_decay=cfg.weight_decay
    )

    # -------------------------------------------------------------------------
    # 5. Load Checkpoint
    # -------------------------------------------------------------------------
    checkpoint_path = "checkpoints/moe-non-overlapping-run/tinystories-moe-4exp-v2-non-ovelapping_latest.pt"
    start_step = 0
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint.get('step', 0)  
        print(f"✓ Resumed from step {start_step}")
    else:
        print(f"No checkpoint found at {checkpoint_path}, starting fresh")
    
    # -------------------------------------------------------------------------
    # 6. Training Loop
    # -------------------------------------------------------------------------
    trainer = Trainer(model, optimizer, train_loader, val_loader, cfg)

    # Restore step counter if resuming
    if start_step > 0:
        trainer.step_num = start_step
        trainer.tokens_processed = start_step * cfg.batch_size * cfg.seq_len
        print(f"Continuing training from step {start_step}")
    
    trainer.train()

if __name__=='__main__':
    main()