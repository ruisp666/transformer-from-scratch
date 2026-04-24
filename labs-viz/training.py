import argparse
import torch
from transformer_from_scratch.config import TrainingConfig
from transformer_from_scratch.trainer import Trainer
from transformer_from_scratch.transformer import ModernTransformer
from transformer_from_scratch.hybridtransformer import HybridTransformer
from transformer_from_scratch.data_utils import get_text_loaders

def parse_args():
    parser = argparse.ArgumentParser(description="Train Hybrid Ablations")
    parser.add_argument(
        "--pattern", 
        type=str, 
        default="AAAAAA", 
        help="Layer pattern string (e.g., AAAAAA, GAAAAA, AAAGGG)"
    )
    parser.add_argument(
        "--chunksize", 
        type=int, 
        default=16, 
        help="Chunk Size"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Load Config (Dynamically inject the pattern from CLI)
    cfg = TrainingConfig.tinystories_hybrid(pattern=args.pattern, chunk_size=args.chunksize)
    print(f"--- Starting Experiment: {cfg.run_name} ---")

    # 2. Data
    train_loader, val_loader, vocab_size = get_text_loaders(
        file_path=cfg.input_file_path,
        batch_size=cfg.batch_size,
        seq_len=cfg.seq_len
    )
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # 3. Model Routing
    # If the string is pure 'A's, use the standard baseline
    if set(cfg.layer_pattern) == {'A'}:
        print("Routing to pure ModernTransformer...")
        model = ModernTransformer(
            vocab_size=cfg.vocab_size,
            d_model=cfg.d_model,
            decoder_n=cfg.n_layers,
            n_heads=cfg.n_heads,
            dropout=cfg.dropout,
            seq_len=cfg.seq_len,
            expansion_factor=cfg.expansion_factor
        ).to(device)
    else:
        print("Routing to HybridTransformer...")
        model = HybridTransformer(
            vocab_size=cfg.vocab_size,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            dropout=cfg.dropout,
            seq_len=cfg.seq_len,
            expansion_factor=cfg.expansion_factor,
            chunk_size=cfg.chunk_size,
            layer_pattern=cfg.layer_pattern
        ).to(device)

    # 4. Optimizer & Train
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    trainer = Trainer(model, optimizer, train_loader, val_loader, cfg)
    trainer.train()

if __name__=='__main__':
    main()