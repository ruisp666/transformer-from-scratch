import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_training_results(histories, save_path="labs-viz/architecture_comparison.png", title='name of the dataset'):
    """
    Plots validation loss curves for multiple models.
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))
    
    colors = {
        "Baseline": "red",
        "Standard": "orange",
        "RoPE+SwiGLU": "green"
    }
    
    for name, loss_history in histories.items():
        # Match partial keys to colors
        color = 'blue' # default
        for key in colors:
            if key in name: 
                color = colors[key]
                
        plt.plot(loss_history, label=name, linewidth=2, alpha=0.8, color=color)

    plt.title(title, fontsize=16)
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("CrossEntropy Loss", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to: {os.path.abspath(save_path)}")
    plt.show()