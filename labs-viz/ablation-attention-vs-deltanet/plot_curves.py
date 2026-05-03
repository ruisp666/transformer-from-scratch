import os
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

def plot_ablation_metrics(entity_name, project_name):
    save_dir = "lab-viz/ablation-attention-vs-deltanet"
    os.makedirs(save_dir, exist_ok=True)
    
    api = wandb.Api()
    runs = api.runs(f"{entity_name}/{project_name}")
    
    target_prefix = "tinystories-hybrid-"
    
    # Metrics we want to plot based on your exact WandB logs
    metrics_to_plot = {
        "train_loss": {"title": "Training Loss Convergence", "ylabel": "Train Loss"},
        "tokens_per_sec": {"title": "Hardware Throughput", "ylabel": "Tokens per Second"}
    }
    
    colors = sns.color_palette("husl", 6)
    
    # We will fetch data for all runs first to avoid redundant API calls
    print("Fetching data from WandB...")
    run_data = []
    for run in runs:
        if run.name.startswith(target_prefix):
            print(f"Downloading history for: {run.name}")
            # Pulling step, train_loss, and tokens_per_sec
            history = run.history(keys=["step", "train_loss", "tokens_per_sec"])
            if not history.empty:
                pattern_label = run.name.replace(target_prefix, "")
                run_data.append({"label": pattern_label, "history": history})

    # Now generate a separate plot for each metric
    for metric_key, config in metrics_to_plot.items():
        plt.figure(figsize=(12, 8))
        sns.set_theme(style="whitegrid")
        color_idx = 0
        
        for data in run_data:
            df = data["history"]
            # Ensure the metric actually exists in this run's history
            if metric_key in df.columns and not df[metric_key].dropna().empty:
                sns.lineplot(
                    x=df["step"], 
                    y=df[metric_key], 
                    label=data["label"],
                    color=colors[color_idx],
                    linewidth=2.5
                )
            color_idx = (color_idx + 1) % len(colors)
        
        plt.title(config["title"], fontsize=16, fontweight='bold')
        plt.xlabel("Training Steps", fontsize=14)
        plt.ylabel(config["ylabel"], fontsize=14)
        plt.legend(title="Layer Pattern", title_fontsize='13', fontsize='12')
        
        save_path = os.path.join(save_dir, f"{metric_key}_overlay.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to: {save_path}")
        plt.close()

if __name__ == "__main__":
    WANDB_USERNAME = "your_username" 
    PROJECT_NAME = "llama-scratch-prod"
    
    plot_ablation_metrics(WANDB_USERNAME, PROJECT_NAME)