import torch
import time
import os 
import tiktoken # <--- NEW: Needed for visualizing text
from pathlib import Path

try:
    import wandb
except ImportError:
    wandb = None

class Trainer:
    def __init__(self, model, optimizer, train_loader, val_loader, config):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = config
        
        # Setup Device (GPU/MPS/CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)
        
        self.step_num = 0
        self.tokens_processed = 0
        self.start_time = None

        # Initialize WandB if configured
        if wandb and self.cfg.project_name:
            # Check if we are logged in, otherwise this might wait for input
            wandb.init(
                project=self.cfg.project_name, 
                name=self.cfg.run_name, 
                config=self.cfg.__dict__
            )

    def train(self):
        """
        The Orchestrator: Manages epochs, validation, and saving.
        """
        print(f"--- Starting Training on {self.device} ---")
        print(f"Config: {self.cfg.run_name} | Params: {sum(p.numel() for p in self.model.parameters()):,}")
        
        self.start_time = time.time()
        
        # [WandB] Watch the model to see gradients and topology
        if wandb and wandb.run:
            wandb.watch(self.model, log="all", log_freq=100)
        
        for epoch in range(self.cfg.epochs):
            print(f"\nEpoch {epoch + 1}/{self.cfg.epochs}")
        
            # 1. Train one full epoch
            self.train_epoch()
            
            # 2. Validate at end of epoch
            val_loss = self.evaluate()
            print(f"End of Epoch {epoch+1} | Val Loss: {val_loss:.4f}")
            
            # [PRO MOVE] Log Validation Loss to WandB
            if wandb and wandb.run:
                wandb.log({
                    "val_loss": val_loss, 
                    "epoch": epoch + 1
                })
            
            # 3. Save Checkpoint
            self.save_checkpoint(f"epoch_{epoch+1}")

    def train_epoch(self):
        """
        The Loop: Iterates over the dataloader.
        """
        self.model.train()

        for batch_idx, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)
            loss, grad_norm = self.train_step(x, y)

            # --- Logging ---
            self.step_num += 1
            self.tokens_processed += x.numel()
            
            if self.step_num % self.cfg.log_interval == 0:
                self.log_metrics(loss, grad_norm)
            
            # [NEW] Log a text sample every 500 steps to see progress
            if self.step_num % 500 == 0:
                self.log_sample_text()

            if self.step_num % self.cfg.save_interval == 0:
                self.save_checkpoint(f"step_{self.step_num}")

    def train_step(self, x, y):
        """
        The Atomic Unit: Forward -> Backward -> Update.
        Returns: loss (float), grad_norm (float)
        """
        # 1. Forward
        logits = self.model(x)

        # 2. Compute Loss 
        # Reshape from (B, T, V) -> (B*T, V) for CrossEntropy
        B, T, V = logits.shape
        loss = torch.nn.functional.cross_entropy(logits.view(B*T, V), y.view(B*T))

        # 3. Backward
        self.optimizer.zero_grad()
        loss.backward()

        # 4. Gradient Clipping (Crucial for Transformers)
        # Prevents "exploding gradients" which cause NaNs
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # 5. Update
        self.optimizer.step()

        return loss.item(), grad_norm.item()
    
    def evaluate(self):
        """
        Runs validation on the hold-out set.
        """
        self.model.eval()
        total_loss = 0
        steps = 0
        
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                B, T, V = logits.shape
                loss = torch.nn.functional.cross_entropy(logits.view(B*T, V), y.view(B*T))
                total_loss += loss.item()
                steps += 1
        
        # Avoid division by zero if loader is empty
        return total_loss / steps if steps > 0 else 0.0
    
    def log_metrics(self, loss, grad_norm):
        """
        Calculates throughput and logs to console/WandB.
        """
        elapsed = time.time() - self.start_time
        tps = self.tokens_processed / elapsed
        
        print(f"Step {self.step_num} | Loss: {loss:.4f} | GradNorm: {grad_norm:.2f} | TPS: {tps:.0f}")
        
        if wandb and wandb.run:
            wandb.log({
                "train_loss": loss,
                "grad_norm": grad_norm,
                "tokens_per_sec": tps,
                "step": self.step_num
            })

    def log_sample_text(self):
        """
        [NEW] Generates a text sample and logs it to WandB.
        This lets you watch the model learn grammar, then spelling, then meaning.
        """
        if not (wandb and wandb.run): return
        
        # 1. Switch to Eval Mode (disable dropout)
        self.model.eval()
        
        # 2. Setup (Tokenize 'The King said')
        enc = tiktoken.get_encoding("gpt2")
        start_text = "The King said"
        start_ids = enc.encode(start_text)
        x = torch.tensor([start_ids], dtype=torch.long, device=self.device)
        
        # 3. Quick Greedy Generation (50 tokens)
        # We assume inference.py logic here but kept simple for the trainer
        for _ in range(50):
            with torch.no_grad():
                logits = self.model(x)
                # Crop context if needed (handled by model usually, but simple safety)
                # Just pluck last token logits
                logits = logits[:, -1, :] 
                # Greedy sample (argmax) is best for checking training progress
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                x = torch.cat((x, next_token), dim=1)
        
        # 4. Decode and Log
        generated_text = enc.decode(x[0].tolist())
        
        # Create a WandB Table
        columns = ["Step", "Generated Text"]
        data = [[self.step_num, generated_text]]
        table = wandb.Table(data=data, columns=columns)
        
        wandb.log({"samples": table})
        
        # 5. Switch back to Train Mode!
        self.model.train()

    def save_checkpoint(self, tag):
        """
        Saves model state and config.
        """
        Path("checkpoints").mkdir(exist_ok=True)
        
        path = f"checkpoints/{self.cfg.run_name}_{tag}.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.cfg,
            'step': self.step_num
        }, path)
        print(f"Saved checkpoint: {path}")