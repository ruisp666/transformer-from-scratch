import torch
import time
import os 
import tiktoken
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
        
        # Setup Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)
        
        self.step_num = 0
        self.tokens_processed = 0
        self.start_time = None
        
        # MoE specific config (default to 0.01 if not set)
        self.aux_loss_coef = getattr(config, 'aux_loss_coef', 0.01)

        if wandb and self.cfg.project_name:
            wandb.init(project=self.cfg.project_name, name=self.cfg.run_name, config=self.cfg.__dict__)

    def _forward(self, x):
        """
        Helper to normalize model output.
        Returns: (logits, aux_loss)
        - If Dense: returns (logits, 0.0)
        - If MoE: returns (logits, aux_loss)
        """
        output = self.model(x)
        
        # Check if the model returned a tuple (MoE style)
        if isinstance(output, tuple):
            logits, aux_loss = output
        else:
            logits = output
            aux_loss = torch.tensor(0.0, device=self.device)
            
        return logits, aux_loss

    def train(self):
        print(f"--- Starting Training on {self.device} ---")
        print(f"Config: {self.cfg.run_name} | Params: {sum(p.numel() for p in self.model.parameters()):,}")
        
        self.start_time = time.time()
        if wandb and wandb.run:
            wandb.watch(self.model, log="all", log_freq=100)
        
        for epoch in range(self.cfg.epochs):
            print(f"\nEpoch {epoch + 1}/{self.cfg.epochs}")
            self.train_epoch()
            
            # Validation at end of epoch
            val_loss, val_aux_loss = self.evaluate()
            print(f"End of Epoch {epoch+1} | Val Loss: {val_loss:.4f} | Val Aux Loss: {val_aux_loss:.4f}")
            
            self.save_checkpoint(f"epoch_{epoch+1}")

    def train_epoch(self):
        self.model.train()
        for batch_idx, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)
            
            # Use the generic step function (no flags needed)
            loss, aux_loss, grad_norm = self.train_step(x, y)

            self.step_num += 1
            self.tokens_processed += x.numel()
            
            if self.step_num % self.cfg.log_interval == 0:
                self.log_metrics(loss, aux_loss, grad_norm)
            
            if self.step_num % self.cfg.eval_interval == 0:
                print(f"\n[Step {self.step_num}] Running Evaluation...")
                val_loss, val_aux_loss = self.evaluate()
                print(f"Step {self.step_num} | Val Loss: {val_loss:.4f} | Aux Loss: {val_aux_loss:.4f}")

                self.log_sample_text()
                
                # Checkpointing
                is_milestone = (self.step_num % 5000 == 0)
                self.save_checkpoint(f"step_{self.step_num}", is_permanent=is_milestone)

                if wandb and wandb.run:
                    wandb.log({
                        "val_loss": val_loss,
                        "val_aux_loss": val_aux_loss,
                        "step": self.step_num
                    })
                
                self.model.train()

    def train_step(self, x, y):
        # 1. Forward (Generic)
        logits, aux_loss = self._forward(x)

        # 2. Compute Total Loss
        B, T, V = logits.shape
        # Flatten for CrossEntropy: (B*T, V) vs (B*T)
        ce_loss = torch.nn.functional.cross_entropy(logits.view(-1, V), y.view(-1))
        
        # Total Loss = Main Task + (Coef * Load Balancing)
        total_loss = ce_loss + (self.aux_loss_coef * aux_loss)

        # 3. Backward
        self.optimizer.zero_grad()
        total_loss.backward()

        # 4. Gradient Clipping (Crucial for Transformers)
        # Prevents "exploding gradients" which cause NaNs
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return total_loss.item(), aux_loss.item(), grad_norm.item()
    
    def evaluate(self, max_batches=50):
        self.model.eval()
        total_ce_loss = 0
        total_aux_loss = 0
        steps = 0
        
        with torch.no_grad():
            for i, (x, y) in enumerate(self.val_loader):
                if i >= max_batches: break
                x, y = x.to(self.device), y.to(self.device)
                
                # Use helper
                logits, aux_loss = self._forward(x)
                
                B, T, V = logits.shape
                ce_loss = torch.nn.functional.cross_entropy(logits.view(-1, V), y.view(-1))
                
                total_ce_loss += ce_loss.item()
                total_aux_loss += aux_loss.item()
                steps += 1
        
         # Avoid division by zero if loader is empty
        if steps == 0: return 0.0, 0.0
        return total_ce_loss / steps, total_aux_loss / steps

    def log_metrics(self, loss, aux_loss, grad_norm):
        elapsed = time.time() - self.start_time
        tps = self.tokens_processed / elapsed if elapsed > 0 else 0
        
        print(f"Step {self.step_num} | Loss: {loss:.4f} (Aux: {aux_loss:.4f}) | Norm: {grad_norm:.2f} | TPS: {tps:.0f}")
        
        if wandb and wandb.run:
            wandb.log({
                "train_loss": loss,
                "aux_loss": aux_loss,
                "grad_norm": grad_norm,
                "tokens_per_sec": tps,
                "step": self.step_num
            })

    def log_sample_text(self):
        """
        Generates text. Handles tuple returns gracefully.
        """
        self.model.eval()
        enc = tiktoken.get_encoding("gpt2")
        # Ensure start_ids is 2D: (1, Seq_Len)
        start_ids = enc.encode("The King said")
        x = torch.tensor([start_ids], dtype=torch.long, device=self.device)
        
        for _ in range(50): 
            with torch.no_grad():
                # Use helper! We only care about logits here.
                logits, _ = self._forward(x)
                
                # Standard generation logic
                logits = logits[:, -1, :] 
                probs = torch.nn.functional.softmax(logits / 0.8, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                x = torch.cat((x, next_token), dim=1)
        
        generated_text = enc.decode(x[0].tolist())
        print(f"\n--- SAMPLE (Step {self.step_num}) ---\n{generated_text}\n---------------------------------------")
        
        # 5. Log to WandB
        if wandb and wandb.run:
            wandb.log({"sample_text": wandb.Html(f"<pre>{generated_text}</pre>"), "step": self.step_num})
        
        self.model.train()

    def save_checkpoint(self, tag, is_permanent=False):
        Path("checkpoints").mkdir(exist_ok=True)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.cfg,
            'step': self.step_num
        }
        torch.save(checkpoint, f"checkpoints/{self.cfg.run_name}_latest.pt")
        if is_permanent:
            torch.save(checkpoint, f"checkpoints/{self.cfg.run_name}_{tag}.pt")