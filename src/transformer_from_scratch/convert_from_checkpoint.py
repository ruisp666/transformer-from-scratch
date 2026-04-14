import torch
from safetensors.torch import save_file
from pathlib import Path

def convert_checkpoint_to_safetensors(old_checkpoint_path, run_name):
    """
    Converts a PyTorch .pt checkpoint to safetensors format.
    Safetensors is faster to load, safer (no pickle), and 
    compatible with HuggingFace model hub.
    """
    print(f"Loading old checkpoint: {old_checkpoint_path}")
    checkpoint = torch.load(old_checkpoint_path, weights_only=False, map_location='cpu')
    
    # Extract components
    model_state = checkpoint['model_state_dict']
    optimizer_state = checkpoint['optimizer_state_dict']
    step = checkpoint.get('step', 0)
    
    # Save model weights as safetensors
    model_path = f"checkpoints/{run_name}_latest_model.safetensors"
    metadata = {
        'step': str(step),
        'run_name': run_name,
    }
    save_file(model_state, model_path, metadata=metadata)
    print(f"✓ Saved model to: {model_path}")
    
    # Save optimizer state as .pt (still pickle, but safer)
    optimizer_path = f"checkpoints/{run_name}_latest_optimizer.pt"
    torch.save({
        'optimizer_state_dict': optimizer_state,
        'step': step
    }, optimizer_path)
    print(f"✓ Saved optimizer to: {optimizer_path}")
    
    print("Conversion complete!")

if __name__ == '__main__':
    # Update these paths
    old_checkpoint = "checkpoints/tinystories-moe-4exp-v2-non-ovelapping_latest.pt"
    new_run_name = "tinystories-moe-4exp-v2-non-ovelapping"
    
    convert_checkpoint_to_safetensors(old_checkpoint, new_run_name)