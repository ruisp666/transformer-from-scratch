

# Week 4 Summary: The Inference Engine & Scaling Up

**Objective:** Finalize the Modern Architecture wrapper, transition from static loss-calculation to dynamic text-generation, and launch the first "Base" scale training run (70M params).

## 1. Daily Breakdown

### Thursday (Jan 15) - "The Architecture Bridge"

**Task: Finalizing the Model Wrapper**

* **Concept:** Implemented `ModernTransformer` as the top-level orchestrator.
* **Efficiency:** Implemented **Weight Tying** (sharing weights between the Embedding layer and the final Linear head). This reduced parameter count and improved training stability by ensuring the input and output embeddings share the same latent space.
* **Structure:** Composed the final hierarchy: `Embedding`  `Stack of ModernDecoders`  `RMSNorm`  `Head`.

### Friday (Jan 16) - "The Engine Start"

**Task: The Inference Loop**
**Challenge 4.1 - Autoregression**

* **Concept:** Implemented `generate.py` (The loop: `idx`  `logits`  `next_token`  `idx`).
* **Logic:** Implemented `get_next_token_id` supporting:
* **Greedy Decoding:** (`argmax`) for deterministic testing.
* **Sampling:** (`multinomial`) with Temperature scaling.



### Saturday (Jan 17) - "The Heavy Lift" (6+ hours)

**Task: Debugging "The Tensor Reality" & Scaling**
**Challenge 4.2 - Shape Mismatches & Observability**

* **The RoPE Crisis:** Debugged the 5-token inference crash.
* *Root Cause:* Pre-computed frequencies were fixed at `Max_Len` (128), but inference input was variable (`T=5`).
* *Solution:* Implemented dynamic slicing in the RoPE forward pass.


* **The Sampling Bug:** Fixed the `RuntimeError` in top-k sampling.
* *Root Cause:* Advanced indexing collapsed the batch dimension.
* *Solution:* Switched to `torch.gather` to preserve tensor shapes.


* **Observability Upgrade:** Integrated **Weights & Biases (WandB)**.
* Implemented `log_sample_text()` to visualize "The King said..." in real-time during training.


* **Scaling to "Base":**
* Scaled up to **70M Parameters** (`d_model=512`, `n_layers=6`).
* **Optimization:** Diagnosed and fixed a 20GB Swap Memory crash by tuning Batch Size (64  32) and leveraging MPS.



---

## 2. Technical Commits (Week 4)

### `61ff45b` (Jan 17) - Feat: Switch to Base model config and add standalone generation scripts

* **Config:** Added `TrainingConfig.base()`: 70M parameters (6 layers, 8 heads, 512 dim). Tuned batch size to 32 to fit within consumer hardware memory constraints.
* **Scripts:** Added `generate.py` (root) and `labs-viz/simple_inference.py`. These scripts implement robust path handling to load checkpoints from any directory and perform interactive inference without padding.

### `3568f24` (Jan 17) - Feat: Add WandB integration and real-time text sampling during training

* **Trainer Upgrade:** Imported `tiktoken` and added `log_sample_text()` method.
* **Logic:** Every 500 steps, the model switches to `eval()` mode, generates 50 tokens using greedy decoding ("The King said..."), logs the result to a WandB Table, and resumes training.

### `b10a145` (Jan 17) - Fix: Add dynamic slicing to RoPE and fix tensor shape bug in inference

* **Rotary Embeddings:** Modified `forward` pass to slice `freq_bank_rope` based on the input sequence length `T`.
* **Inference Sampling:** Replaced direct indexing with `torch.gather(i, -1, sample_idx)` inside `get_next_token_id`. This ensures the indices selected from Top-K align correctly with the batch dimension.

### `6ae33e1` (Jan 15) - feat: implement ModernTransformer class with Weight Tying

* **Wrapper:** Added `ModernTransformer` (transformer.py) as the top-level Model Wrapper.
* **Composition:** Defined the forward pass flow: `Embedding`  `Stack of ModernDecoders`  `RMSNorm`  `Head`.
* **Efficiency:** Implemented Weight Tying: `self.lm_head.weight = self.token_embedding.weight`. This significantly reduces total parameters while improving semantic alignment.
* **Documentation:** Added docstrings detailing architecture flow and academic references.

---

## 3. Engineering Insights

### Inductive Bias: The "Greedy" Trap

We observed that early in training (Step 500-1000), the model outputs repetitive loops ("he, he, he") when using Greedy Decoding.

* **Insight:** The model minimizes loss by predicting the statistically "safest" next token. Without temperature-based sampling to introduce noise, it falls into a local minimum of repeating high-probability function words.

### The Memory Wall

We hit the physical limits of the hardware when scaling to 70M parameters.

* **Discovery:** Parameter count is not the bottleneck; **Activations** are.
* **Metric:** A Batch Size of 64 with 512 dimensions created intermediate states larger than physical RAM (20GB+ Swap), forcing the OS to thrash. Reducing Batch Size to 32 stabilized throughput at ~11,500 tokens/sec.

---

### Review Checklist

* [x] **Verify Inference:** Confirmed `generate.py` works on both CPU and MPS devices.
* [x] **Fix RoPE:** Ensured Rotary Embeddings handle inputs shorter than `block_size` without crashing.
* [x] **Architecture Stable:** The 70M parameter model is stable (Gradient Norm ~0.5) and learning rapidly.

**Next Step:** Move to **Week 5: Data Science & Tokenization**, where we pivot from "Engineering" to "Curriculum," replacing TinyShakespeare with **TinyStories** to teach the model actual reasoning capabilities.