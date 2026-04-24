# Ablation Study: Attention vs. Delta Network Topologies

This directory contains the experimental setup and results for evaluating **Hybrid Transformer Architectures**. Specifically, we ablate the placement and ratio of standard $O(N^2)$ Multi-Head Attention (MHA) layers against $O(N)$ Chunkwise Delta Gate (Linear Attention) layers.

## 🎯 Objective
To empirically determine how lossy memory compression (Delta Networks) affects learning dynamics, hardware throughput, and final validation loss depending on *where* it is injected into the model depth.

## 🧬 Architectural Patterns

Our baseline model is a 6-layer architecture trained from scratch on the TinyStories dataset. We represent standard Attention layers as `A` and Delta Gate layers as `G`. 

We test six specific topologies to cover the three major design philosophies of hybrid LLMs:

| Pattern | Description | Strategy |
| :--- | :--- | :--- |
| **`AAAAAA`** | **Pure Attention** | The "Gold Standard" baseline. Maximum exact-recall capacity, but heaviest on compute and memory footprint. |
| **`GGGGGG`** | **Pure Delta** | The "Efficiency Standard" baseline. Fastest throughput and lowest memory footprint, forcing $O(1)$ state compression everywhere. |
| **`GGGAAA`** | **Delta Early (Bottom-Heavy)** | Forces compression of low-level lexical features (grammar, syntax) early. Saves exact-match reasoning for high-level semantic routing at the end. |
| **`AAAGGG`** | **Delta Late (Top-Heavy)** | Builds rich, perfect-recall representations early, but forces a massive compression bottleneck right before the LM head predictions. |
| **`GAGAGA`** | **Interleaved (Delta-First)** | A rhythm of compression and exact-matching, applying compression to the raw embeddings immediately. |
| **`AGAGAG`** | **Interleaved (Attention-First)** | Similar to modern models like Jamba. Builds exact local relationships before compressing them sequentially. |

### 🧪 Core Hypotheses
1. **Hardware Throughput (`tokens_per_sec`):** We expect a strict hierarchy where `GGGGGG` > Hybrids > `AAAAAA`. Replacing `A` with `G` will linearly decrease the memory footprint and increase tokens/sec.
2. **Convergence (Loss):** `AAAAAA` will reach the lowest absolute validation loss.
3. **The Bottleneck:** `AAAGGG` will likely suffer a "loss floor" (plateau) compared to `GGGAAA`, proving that attention is most critical in the final layers of a network.

## 🚀 How to Run the Experiments

Our training infrastructure uses a dynamic string parser. You can run any topology without modifying the source code by passing the pattern via the CLI.

```bash
# 1. Run the Baselines
uv run python training.py --pattern AAAAAA
uv run python training.py --pattern GGGGGG

# 2. Run the Bipartite Splits
uv run python training.py --pattern GGGAAA
uv run python training.py --pattern AAAGGG

# 3. Run the Interleaved Topologies
uv run python training.py --pattern GAGAGA
uv run python training.py --pattern AGAGAG
```

**Dataset Configuration:**
*   Dataset: TinyStories
*   Sequence Length: 256
*   Chunking Strategy: Strict Non-overlapping (stride = sequence length)
*   Delta Chunk Size: 16

## 📊 Logging & Visualization

All metrics are automatically synced to Weights & Biases under the `llama-scratch-prod` project. The `Trainer` dynamically logs the run name based on the CLI pattern (e.g., `tinystories-hybrid-GGGAAA`).

**Key Metrics Tracked:**
*   `train_loss` & `val_loss` (Learning dynamics)
*   `tokens_per_sec` (Hardware throughput verification)
*   Memory allocation (VRAM/Unified Memory footprint)

### Generating Local Plots
Once the runs are complete, you can generate high-resolution, publication-ready overlay plots to visualize the results:

```bash
# Pulls data from WandB and generates PNG overlays
uv run python lab-viz/ablation-attention-vs-deltanet/plot_curves.py
```
This will output `train_loss_overlay.png` and `tokens_per_sec_overlay.png` directly to your local visualization folder.

***
