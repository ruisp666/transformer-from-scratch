import torch
import tiktoken
import os
from torch.utils.data import DataLoader, DistributedSampler
from transformer_from_scratch.data_generators import WikiAutoRegressive


def get_text_loaders(
    file_path: str,
    batch_size: int,
    seq_len: int,
    split_ratio: float = 0.9,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[DataLoader, DataLoader, int]:
    """
    Build train and validation DataLoaders from a raw text file.

    Supports both single-process and distributed (multi-GPU) training.
    In distributed mode (world_size > 1), a DistributedSampler is used
    to partition the dataset across ranks so each GPU sees a distinct,
    non-overlapping shard. In single-process mode the behaviour is
    identical to the original implementation.

    Important: in distributed mode, call
        train_loader.sampler.set_epoch(epoch)
    at the start of each epoch so the sampler re-shuffles differently
    each epoch rather than repeating the same shard assignment.

    Args:
        file_path: Path to the raw .txt data file.
        batch_size: Per-GPU batch size. Effective global batch size is
            batch_size * world_size.
        seq_len: Context length (block size) in tokens.
        split_ratio: Fraction of data used for training. Default 0.9.
        rank: Global rank of this process. 0 in single-process mode.
        world_size: Total number of processes. 1 in single-process mode.

    Returns:
        train_loader: DataLoader for the training split.
        val_loader:   DataLoader for the validation split.
        vocab_size:   Vocabulary size of the GPT-2 tokeniser.

    Raises:
        FileNotFoundError: If file_path does not exist.
    """
    distributed = world_size > 1

    # pin_memory is a CUDA concept — not supported on MPS.
    # Enable only when CUDA is available.
    use_pin_memory = torch.cuda.is_available()

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Could not find data file at: {file_path}. "
            "Did you run the download script?"
        )

    # Suppress duplicate prints — all ranks read the same file,
    # only rank 0 needs to report stats.
    if rank == 0:
        print(f"Loading text from {file_path}...")

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # --- Train / val split (character-level, before tokenisation) ---
    n = len(text)
    split_idx = int(n * split_ratio)
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    # --- Tokeniser ---
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab

    if rank == 0:
        print("--- Data Stats ---")
        print(f"File Size:     {n / 1024 / 1024:.2f} MB")
        print(f"Vocab Size:    {vocab_size}")
        print(
            f"Train Tokens:  "
            f"{len(enc.encode(train_text, allowed_special={'<|endoftext|>'})):,}"
        )
        print(
            f"Val Tokens:    "
            f"{len(enc.encode(val_text, allowed_special={'<|endoftext|>'})):,}"
        )
        print("------------------")

    # --- Datasets ---
    train_ds = WikiAutoRegressive(train_text, seq_len, enc)
    val_ds = WikiAutoRegressive(val_text, seq_len, enc)

    # --- Samplers ---
    if distributed:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,   # shuffled; re-seed each epoch via set_epoch()
            drop_last=True, # avoid uneven batch sizes across ranks,
                            # which cause NCCL to hang
        )
        val_sampler = DistributedSampler(
            val_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,  # deterministic val evaluation
            drop_last=True,
        )
    else:
        train_sampler = None
        val_sampler = None

    # --- DataLoaders ---
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        # shuffle and sampler are mutually exclusive:
        # when a sampler is provided, it owns the ordering
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=2,          # 2 per GPU is a reasonable default;
                                # increase if data loading is the bottleneck
        pin_memory=use_pin_memory,
        drop_last=True,         # consistent batch sizes across steps
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=2,
        pin_memory=use_pin_memory,
        drop_last=True,
    )

    return train_loader, val_loader, vocab_size