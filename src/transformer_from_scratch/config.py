from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # Model Architecture
    vocab_size: int = 50304 # GPT-2 vocab size (rounded to nearest multiple of 64 for efficiency)
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    seq_len: int = 256  # Context window
    dropout: float = 0.1
    expansion_factor=8/3
    
    # Training Hyperparameters
    epochs: int = 1
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-1
    
    # Scheduling (Cosine Warmup)
    warmup_steps: int = 100
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    
    # Logging & Checkpoints
    run_name: str = "default_run"
    project_name: str = "llama-scratch-prod"
    log_interval: int = 10
    save_interval: int = 500
    eval_interval: int = 500  # Evaluate every 500 steps
    
    # Data Paths (NEW)
    input_file_path: str = "data/input.txt" # Default fallback
    
    # --- PRESETS -------------------------------------------------------------
    
    @classmethod
    def nano(cls):
        """
        Tiny config for debugging on CPU/MPS.
        Target: TinyShakespeare (1MB)
        """
        return cls(
            d_model=64,
            n_layers=4,
            n_heads=4,
            seq_len=64,
            batch_size=32,
            lr=1e-3,
            run_name="nano-shakespeare",
            input_file_path="data/shakespeare_input.txt" # Explicit path
        )

    @classmethod
    def base(cls):
        """
        The 'Base' Llama-style model (approx 70M params).
        Target: TinyShakespeare (1MB) - Prone to overfitting!
        """
        return cls(
            d_model=512,
            n_layers=6,
            n_heads=8,
            seq_len=256,
            batch_size=32, # Kept low for memory safety
            lr=3e-4,
            run_name="base-shakespeare",
            input_file_path="data/shakespeare_input.txt"
        )

    @classmethod
    def tinystories(cls):
        """
        Configuration for the 200MB TinyStories dataset.
        Architecture: Same as 'Base' (70M params).
        Target: TinyStories (200MB) - The real pre-training run.
        """
        return cls(
            # Architecture (Matches Base)
            d_model=512,
            n_layers=6,
            n_heads=8,
            seq_len=256,
            dropout=0.1,
            
            # Training
            epochs=1,     # 1 epoch on 200MB is ALOT of steps (~50k steps)
            batch_size=32,
            lr=3e-4,
            
            # Run Identity
            run_name="tinystories-base-70M",
            project_name="llama-scratch-prod",
            
            # The Critical Path
            input_file_path="data/tinystories_input.txt"
        )