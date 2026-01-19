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
            
            # 1. Fast Logs (Every 10 steps) - Just print speed/loss
            if self.step_num % self.cfg.log_interval == 0:
                self.log_metrics(loss, grad_norm)
            
            # 2. Heavy Logs (Every eval_interval, e.g. 500 steps)
            # This is the "Industry Standard" check-up
            if self.step_num % self.cfg.eval_interval == 0:
                print(f"\n[Step {self.step_num}] Running Mid-Epoch Evaluation...")
                
                # A. Validate (Check against data the model hasn't seen)
                val_loss = self.evaluate()
                print(f"Step {self.step_num} | Val Loss: {val_loss:.4f}")

                # B. Generate Text (Visual check in wandb)
                self.log_sample_text()

                is_milestone = (self.step_num % 5000 == 0)
                
                # C. Save Checkpoint (Safety)
                self.save_checkpoint(f"step_{self.step_num}", is_permanent=is_milestone)

                # D. Log everything to WandB
                if wandb and wandb.run:
                    wandb.log({
                        "val_loss": val_loss,
                        "step": self.step_num
                    })

                # Switch back to train mode after evaluation!
                self.model.train()

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
    
    def evaluate(self, max_batches=100):
        """
        Runs validation on the hold-out set.
        """
        self.model.eval()
        total_loss = 0
        steps = 0
        
        with torch.no_grad():
            for i, (x, y) in enumerate(self.val_loader):

                if i > max_batches:
                    break
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

def save_checkpoint(self, tag, is_permanent=False):
        """
        Saves model state. 
        - Always overwrites 'latest.pt' (for crash recovery).
        - Only creates a timestamped file if is_permanent=True.
        """
        Path("checkpoints").mkdir(exist_ok=True)
        
        # Data to save
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.cfg,
            'step': self.step_num,
            'val_loss': self.best_val_loss if hasattr(self, 'best_val_loss') else None
        }
        
        # 1. Always save/overwrite 'latest.pt'
        latest_path = f"checkpoints/{self.cfg.run_name}_latest.pt"
        torch.save(checkpoint, latest_path)
        
        # 2. If it's a special milestone, save a permanent copy
        if is_permanent:
            archive_path = f"checkpoints/{self.cfg.run_name}_{tag}.pt"
            torch.save(checkpoint, archive_path)
            print(f"Saved ARCHIVE checkpoint: {archive_path}")
        else:
            print(f"Updated latest checkpoint: {latest_path}")