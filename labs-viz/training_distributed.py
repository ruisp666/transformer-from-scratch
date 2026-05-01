import argparse
import torch
import os
from transformer_from_scratch.config import TrainingConfig
from transformer_from_scratch.trainer import Trainer
from transformer_from_scratch.transformer import ModernTransformer
from transformer_from_scratch.hybridtransformer import HybridTransformer
from transformer_from_scratch.data_utils import get_text_loaders
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


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

    # 1.1 Dist or single GPU?
    is_distributed = "WORLD_SIZE" in os.environ
    if is_distributed:
        # Here initiate the gpu links (nccl)
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


    # Make the guard
    is_main = rank == 0

     # 1. Load Config (Dynamically inject the pattern from CLI)
    cfg = TrainingConfig.tinystories_hybrid(pattern=args.pattern, chunk_size=args.chunksize)
    if is_main:
        print(f"--- Starting Experiment: {cfg.run_name} ---")
        
    # 2. Data
    train_loader, val_loader, _ = get_text_loaders(
        file_path=cfg.input_file_path,
        batch_size=cfg.batch_size,
        seq_len=cfg.seq_len,
        rank=rank,
        world_size=world_size
    )
    

    # 3. Model Routing
    # If the string is pure 'A's, use the standard baseline
    if set(cfg.layer_pattern) == {'A'}:
        if is_main:
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
        if is_main:
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
    if is_distributed:
        # This not just moving to device, it deals with the sync across backward pass
        # And manages comms
        model = DDP(model, device_ids=[local_rank])


    # 4. Optimizer & Train
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    trainer = Trainer(model, optimizer, train_loader, val_loader, cfg, rank=rank)
    trainer.train()
    if is_distributed:
        dist.destroy_process_group()

if __name__=='__main__':
    main()