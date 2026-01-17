import torch
from transformer_from_scratch.config import TrainingConfig
from transformer_from_scratch.trainer import Trainer
from transformer_from_scratch.transformer import ModernTransformer
from transformer_from_scratch.data_pipeline import get_shakespeare_loaders


def main():
    # Config
    cfg = TrainingConfig.base()

    train_loader, val_loader, vocab_size = get_shakespeare_loaders(batch_size=cfg.batch_size,seq_len=cfg.seq_len)

    # 3. Model
    model = ModernTransformer(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        decoder_n=cfg.n_layers,
        n_heads=cfg.n_heads,
        dropout=cfg.dropout,
        seq_len=cfg.seq_len,
        expansion_factor=cfg.expansion_factor
    ).to("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # 4. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    
    # 5. Kick-off training
    trainer = Trainer(model, optimizer, train_loader, val_loader, cfg)
    trainer.train()

if __name__=='__main__':
    main()