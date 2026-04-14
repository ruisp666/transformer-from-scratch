import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import tiktoken
from transformer_from_scratch.transformer import ModernTransformer  # Your dense model
from transformer_from_scratch.sparse_moe_transformer import SparseMoETransformer
from transformer_from_scratch.config import TrainingConfig

class CheckpointAnalyzer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                   "mps" if torch.backends.mps.is_available() else "cpu")
        self.enc = tiktoken.get_encoding("gpt2")
        
    def load_checkpoint_info(self, checkpoint_path):
        """Extract metadata without loading full model"""
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        return {
            'step': ckpt.get('step', 0),
            'config': ckpt.get('config'),
            'val_loss': ckpt.get('val_loss', None)
        }
    
    def load_model_from_checkpoint(self, checkpoint_path, is_moe=False):
        """Load a model from checkpoint"""
        print(f"Loading {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        cfg = ckpt['config']

         # Detect actual vocab size from saved weights (handles mismatch between config and actual model)
        actual_vocab_size = ckpt['model_state_dict']['embedding_layer.weight'].shape[0]
        print(f"DEBUG: Config vocab_size={cfg.vocab_size}, Actual weights vocab_size={actual_vocab_size}")
        
        if is_moe:
            print(f"DEBUG: Creating MoE with vocab_size={cfg.vocab_size}")  # ADD THIS
            model = SparseMoETransformer(
                vocab_size=actual_vocab_size,
                n_layers=cfg.n_layers,
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                n_experts=cfg.n_experts,
                capacity_factor=cfg.capacity_factor,
                dropout=cfg.dropout,
                seq_len=cfg.seq_len
            )
        else:
            # Dense model
            model = ModernTransformer(
                vocab_size=actual_vocab_size,
                decoder_n=cfg.n_layers,  # Note: uses decoder_n not n_layers
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                dropout=cfg.dropout,
                seq_len=cfg.seq_len,
                expansion_factor=cfg.expansion_factor
            )
        
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        model.to(self.device)
        model.eval()
        
        return model, cfg, ckpt.get('step', 0)
    
    def generate_sample(self, model, prompt="Once upon a time", max_tokens=100, temperature=0.8):
        """Generate text from a model"""
        model.eval()
        start_ids = self.enc.encode(prompt)
        x = torch.tensor([start_ids], dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            for _ in range(max_tokens):
                # Handle both dense and MoE outputs
                output = model(x)
                logits = output[0] if isinstance(output, tuple) else output
                
                logits = logits[:, -1, :]
                probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                x = torch.cat((x, next_token), dim=1)
        
        return self.enc.decode(x[0].tolist())
    
    def evaluate_perplexity(self, model, test_data, max_batches=50):
        """Calculate perplexity on test data"""
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for i, (x, y) in enumerate(test_data):
                if i >= max_batches:
                    break
                    
                x, y = x.to(self.device), y.to(self.device)
                
                # Handle both dense and MoE outputs
                output = model(x)
                logits = output[0] if isinstance(output, tuple) else output
                
                B, T, V = logits.shape
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, V), 
                    y.view(-1),
                    reduction='sum'
                )
                total_loss += loss.item()
                total_tokens += B * T
        
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        return perplexity, avg_loss
    
    def compare_checkpoints(self, checkpoint_paths, labels, test_loader):
        """Compare multiple checkpoints"""
        results = []
        
        for path, label in zip(checkpoint_paths, labels):
            is_moe = 'moe' in label.lower()
            model, cfg, step = self.load_model_from_checkpoint(path, is_moe=is_moe)
            
            # Evaluate
            ppl, loss = self.evaluate_perplexity(model, test_loader)
            
            # Generate samples
            sample1 = self.generate_sample(model, "Once upon a time")
            sample2 = self.generate_sample(model, "The little girl")
            
            results.append({
                'label': label,
                'step': step,
                'perplexity': ppl,
                'loss': loss,
                'sample1': sample1,
                'sample2': sample2,
                'params': sum(p.numel() for p in model.parameters())
            })
            
            print(f"\n{'='*60}")
            print(f"{label} (Step {step})")
            print(f"{'='*60}")
            print(f"Perplexity: {ppl:.2f}")
            print(f"Loss: {loss:.4f}")
            print(f"Parameters: {results[-1]['params']:,}")
            print(f"\nSample 1:\n{sample1[:200]}...")
            print(f"\nSample 2:\n{sample2[:200]}...")
            
            # Clean up memory
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
        return results
    
    def plot_comparison(self, results):
        """Plot comparison metrics"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        labels = [r['label'] for r in results]
        perplexities = [r['perplexity'] for r in results]
        losses = [r['loss'] for r in results]
        
        # Perplexity comparison
        axes[0].bar(labels, perplexities, color=['#2ecc71', '#e74c3c', '#3498db'])
        axes[0].set_ylabel('Perplexity (lower is better)')
        axes[0].set_title('Model Comparison: Perplexity')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Loss comparison
        axes[1].bar(labels, losses, color=['#2ecc71', '#e74c3c', '#3498db'])
        axes[1].set_ylabel('Cross-Entropy Loss (lower is better)')
        axes[1].set_title('Model Comparison: Loss')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('checkpoint_comparison.png', dpi=150, bbox_inches='tight')
        print("\n✓ Saved plot to checkpoint_comparison.png")
        plt.show()


# Usage Example
if __name__ == '__main__':
    from transformer_from_scratch.data_utils import get_text_loaders
    
    # Setup
    analyzer = CheckpointAnalyzer()
    
    # Load test data (use validation set)
    cfg = TrainingConfig.tinystories()
    _, test_loader, _ = get_text_loaders(
        file_path=cfg.input_file_path,
        batch_size=32,
        seq_len=256
    )
    
    # =========================================================================
    # COMPARISON 1: Dense vs MoE (Non-Overlapping)
    # =========================================================================
    print("\n" + "="*80)
    print("COMPARISON 1: Dense vs MoE (Both Non-Overlapping)")
    print("="*80)
    
    comparison1_checkpoints = [
        "checkpoints/dense-sliding-window-run/tinystories-base-70M_step_75000.pt",
        "checkpoints/moe-non-overlapping-run/tinystories-moe-4exp-v2-non-ovelapping_step_15000.pt"
    ]
    comparison1_labels = [
        "Dense (75k steps)",
        "MoE Non-Overlapping (15k steps)"
    ]
    
    results1 = analyzer.compare_checkpoints(comparison1_checkpoints, comparison1_labels, test_loader)
    
    # =========================================================================
    # COMPARISON 2: MoE Sliding vs Non-Overlapping
    # =========================================================================
    print("\n" + "="*80)
    print("COMPARISON 2: MoE Sliding Window vs Non-Overlapping")
    print("="*80)
    
    comparison2_checkpoints = [
        "checkpoints/moe-sliding-window-run/tinystories-moe-4exp-v1_step_60000.pt",
        "checkpoints/moe-non-overlapping-run/tinystories-moe-4exp-v2-non-ovelapping_step_15000.pt"
    ]
    comparison2_labels = [
        "MoE Sliding (60k steps)",
        "MoE Non-Overlapping (15k steps)"
    ]
    
    results2 = analyzer.compare_checkpoints(comparison2_checkpoints, comparison2_labels, test_loader)
    
    # Plot both comparisons
    analyzer.plot_comparison(results1 + results2)