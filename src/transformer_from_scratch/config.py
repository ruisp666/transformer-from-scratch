from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # --- Model Architecture ---
    vocab_size: int = 50304
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    seq_len: int = 256
    dropout: float = 0.1
    expansion_factor: float = 8/3  # Standard for SwiGLU
    
    # --- MoE Hyperparameters (NEW) ---
    # Defaults set for a small MoE run
    n_experts: int = 4             # If 1, acts as dense model (conceptually)
    capacity_factor: float = 1.25  # Buffer capacity (1.0 = strict, >1.0 = slack)
    aux_loss_coef: float = 0.01    # Load balancing loss weight
    
    # --- Training Hyperparameters ---
    epochs: int = 1
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-1
    
    # --- Scheduling ---
    warmup_steps: int = 100
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    
    # --- Logging & Checkpoints ---
    run_name: str = "default_run"
    project_name: str = "llama-scratch-prod"
    log_interval: int = 10
    save_interval: int = 500
    eval_interval: int = 500
    
    # --- Data Paths ---
    input_file_path: str = "data/input.txt"

    # -------------------------------------------------------------------------
    # PRESETS
    # -------------------------------------------------------------------------
    
    @classmethod
    def nano(cls):
        """Debug run on CPU/MPS."""
        return cls(
            d_model=64, n_layers=4, n_heads=4, seq_len=64, 
            batch_size=32, lr=1e-3, 
            run_name="nano-debug",
            input_file_path="data/shakespeare_input.txt",
            n_experts=2  # Test MoE logic even on nano
        )

    @classmethod
    def base(cls):
        """Standard Llama-style Dense Model (70M)."""
        return cls(
            d_model=512, n_layers=6, n_heads=8, seq_len=256,
            batch_size=32, lr=3e-4,
            run_name="base-dense-70M",
            input_file_path="data/shakespeare_input.txt",
            n_experts=1 # Effectively dense
        )

    @classmethod
    def tinystories(cls):
        """The Dense Baseline for TinyStories."""
        return cls(
            d_model=512, n_layers=6, n_heads=8, seq_len=256,
            epochs=1, batch_size=32, lr=3e-4,
            run_name="tinystories-dense-70M",
            input_file_path="data/tinystories_input.txt",
            n_experts=1 
        )

    @classmethod
    def tinystories_moe(cls):
        """
        The Sparse MoE Run for TinyStories.
        Same width/depth as dense, but with 4 Experts.
        """
        return cls(
            # Architecture
            d_model=512, n_layers=6, n_heads=8, seq_len=256,
            
            # MoE Specifics
            n_experts=4,
            capacity_factor=1.25,
            aux_loss_coef=0.01,
            
            # Training
            epochs=1, batch_size=32, lr=3e-4,
            
            # Run Identity
            run_name="tinystories-moe-4exp",
            input_file_path="data/tinystories_input.txt"
        )