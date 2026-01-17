from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # --- Model Architecture ---
    vocab_size: int = 50257  # Standard GPT-2/Tiktoken size
    d_model: int = 256       # Embedding dimension
    n_layers: int = 4        # Depth of the network
    n_heads: int = 4         # Number of attention heads
    seq_len: int = 128       # Context window (Context Length)
    dropout: float = 0.1
    expansion_factor: float = 8/3 # Standard Llama SwiGLU ratio

    # --- Optimization ---
    batch_size: int = 32
    lr: float = 3e-4
    epochs: int = 1
    
    # --- Infrastructure ---
    log_interval: int = 10
    save_interval: int = 500
    project_name: str = "llama-scratch"
    run_name: str = "run_default"

    def __post_init__(self):
        """
        Validates the configuration mathematically after initialization.
        """
        # 1. Check Division
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        
        # 2. Check RoPE Compatibility (Head Dim must be Even)
        # RoPE rotates pairs of numbers (x, y) as complex numbers.
        head_dim = self.d_model // self.n_heads
        if head_dim % 2 != 0:
            raise ValueError(
                f"Head dimension ({head_dim}) must be even for RoPE. "
                f"Current config: d_model={self.d_model}, n_heads={self.n_heads}"
            )

    @classmethod
    def nano(cls):
        """
        Returns a 'Nano' configuration for rapid debugging on a laptop.
        Sanity Check mode: Fast training, guarantees pipeline works.
        """
        return cls(
            d_model=128,
            n_layers=2,
            n_heads=4,
            seq_len=128,
            batch_size=32,
            lr=1e-4,
            run_name="nano-debug",
            project_name="llama-scratch-debug"
        )

    @classmethod
    def base(cls):
        """
        Returns a 'Base' configuration closer to a small GPT-2.
        Use this for actual training runs overnight.
        """
        return cls(
            d_model=512,
            n_layers=6,
            n_heads=8,
            seq_len=256,
            batch_size=32,
            epochs=10
            lr=3e-4,
            run_name="base-training",
            project_name="llama-scratch-prod"
        )