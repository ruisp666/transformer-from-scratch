

Friday (2 hours) - COMPLETED
Task: Signal Stability & Gating
Challenge 3.1 - Modern Components
Plan:
Implement RMSNorm (Scale Invariance)
Implement SwiGLU (Gated Linear Units)
**Key Focus:** Manual parameter control (`nn.Parameter`) vs. using "black box" layers.

Saturday (4 hours) - COMPLETED
Task: Integration & The Architecture Battle
Challenge 3.2 - Putting it together
Plan:
Build `TransformerBlock` hierarchy (Base vs. RoPE+SwiGLU)
**Debug:** The RoPE Broadcasting Crash (Heads vs SeqLen mismatch)
**Benchmark:** Train on Reversal Task (Absolute) vs. Repeat Copy (Relative)
Visualize the "Inductive Bias" trade-off.

Deliverable: `architecture_comparison.png` and `repeat_copy_results.png`

Sunday (1 hour) - COMPLETED
Task: Week 3 Review
Plan:
Refactor into `data_generators` and `utils`
Add Unit Tests (`pytest`) with fixtures
Write summary: "Inductive Bias & Tensor Shapes"
Prep for Inference Week

---

# Week 3 Summary: The Modern Architecture (Llama-fication)

**Objective:** Evolve the architecture from "Standard Transformer" (ReLU + LayerNorm) to "Modern LLM" (SwiGLU + RMSNorm) and empirically prove the trade-offs of Relative Embeddings.

## 1. Core Implementation

### RMSNorm & SwiGLU (Efficiency > Tradition)

* **RMSNorm Discovery:** We abandoned `LayerNorm`'s mean-centering. By only normalizing variance (), we preserved signal magnitude while saving compute. It is mathematically equivalent to projecting input onto a hypersphere.
* **SwiGLU Discovery:** We replaced the "dumb" on/off ReLU switch with a "smart" Dimmer Switch.
* **The Dimensionality Trap:** We learned that switching from 2 matrices (FFN) to 3 matrices (SwiGLU) requires resizing the hidden dim by  to keep parameter counts fair.
* **Initialization Control:** We manually handled `nn.Parameter` allocation (`torch.empty` + `kaiming_uniform`) rather than relying on `nn.Linear`, giving us "Quant-level" control over the model's starting state.



### Tensor Fluency: The "Broadcasting" War

* **The Broadcasting Crash:** The biggest hurdle was the 4D mismatch between the Attention output (`Batch, Heads, Seq, Dim`) and the RoPE frequency cache.
* *The Fix:* We stopped fighting the dimensions manually and optimized the `RoPE` init to create a shape of `(1, 1, Seq, Dim)`.


* **The "0 and -1" Rule:** We enforced a strict coding standard: never hardcode intermediate dimensions (like `dim=2`). Always anchor operations to the Batch (`dim=0`) or the Feature (`dim=-1`). This makes the code robust whether the input is a single token or a massive batch.

---

## 2. Geometric Awareness: Inductive Bias

* **The Hypothesis:** "RoPE is better" is a dangerous simplification. RoPE is better *at specific things*.
* **The Experiment:** We ran a controlled "Architecture Battle" on two distinct synthetic tasks.

| Task | Regime | Winner | Why? |
| --- | --- | --- | --- |
| **Sequence Reversal** | Absolute Indexing () | **Standard (Abs PE)** | The task requires a fixed coordinate system (0,0). RoPE struggled to "infer" absolute position from relative neighbors. |
| **Repeat Copy** | Relative Pattern () | **Modern (RoPE)** | RoPE won decisively (converged 3x faster). The rule "copy from 15 steps back" is a single relative rotation, whereas Abs PE had to memorize 15 separate index associations. |

* **The "Grokking" Cliff:** We observed the Standard model flatlining for 180 epochs on the Reversal task before suddenly plummeting to zero loss. This confirmed the model was memorizing noise before "grokking" the mathematical rule.

---

## 3. Engineering Rigor & Testing

* **The "Senior Quant" Assertion:** We added `assert d_model // n_heads % 2 == 0`. This protects against the mathematical impossibility of pairing features for complex numbers if the head dimension is odd—a silent failure in most open-source implementations.
* **Separation of Concerns:** We refactored "Scripting" into "Engineering" by separating `data_generators.py` (The Source), `utils.py` (The Viz), and `transformer_blk.py` (The Model).
* **MPS Acceleration:** We successfully targeted the Mac M-series GPU, reducing training time for a 15-layer deep network to just 40 seconds.

---

### Sunday Review Checklist

* [x] **Verify SwiGLU Math:** Confirmed that `F.silu(x)` is numerically stable compared to `x * sigmoid(x)`.
* [x] **Check RoPE Integration:** Ensured RoPE is applied *after* splitting heads but *before* attention scores.
* [x] **Inductive Bias:** Concluded that modern architectures trade "Absolute Position Capabilities" for "Long-Context Relative Capabilities."

**Next Step:** Move to **Week 4: Inference Engine**, where we build the KV-Cache and Autoregressive Loop to make this model actually speak.