# PyTorch Cheatsheet
### This can help us save hours in debugging

---

# Week 1: The Semantic & Temporal Manifold
**Objective:** Build a position-aware Transformer Encoder from scratch and analyze its geometric properties.

## 1. Core Implementation

### The Attention Mechanism
The heart of the model is the **Scaled Dot-Product Attention**.
* **Discovery:** The $\sqrt{d_k}$ scaling factor is a "gradient stabilizer." Without it, the softmax saturates, gradients vanish, and the manifold loses the ability to learn.
* **Mechanism:** It acts as a dynamic weighted average where tokens "pull" information from others based on their similarity in a projected space.

### Multi-Head Attention (MHA)
* **Discovery:** Tensor orientation is everything. We use `permute(0, 2, 1, 3)` during `split_heads` to ensure that the batch and head dimensions are isolated. This allows the last two dimensions (Sequence and Head Dim) to "agree" for matrix multiplication.
* **Discovery:** Attention at a specific position is a linear combination of signals, effectively acting as a composition of information from across the sequence.

### The Transformer Block (Residuals & Norm)
* **Discovery:** The **Residual Sign (+)** is a safeguard. In backpropagation, it ensures an "Identity" gradient of **1.0** always flows back, preventing the signal from dying even if the sub-layer saturates.
* **Discovery:** **Pre-LN** is superior for deep architectures. By normalizing *before* the MHA, we keep the gradient bounded and protect the "Effective Dimension" of the manifold. **Post-LN** often leads to rank collapse and requires aggressive learning-rate warmups.

## 2. Temporal Awareness: Positional Encoding
* **Discovery:** Sinusoidal encodings aren't just "adding noise"; they provide a **Coordinate System**.
* **Linear Rotation Property:** We proved that $PE_{pos+k}$ can be represented as a linear rotation of $PE_{pos}$. This allows the model to perceive *relative* distance through phase shifts.
* **The Frequency Bank:** Using a log-space frequency bank creates "clocks" that run at different speeds across $d_{model}$, allowing the model to capture both local and global sequence order.

## 3. Empirical Discoveries: Debugging Triggers

| Observation | Discovery |
| :--- | :--- |
| **Rank Collapse** | Without residuals or with Post-LN, the "Effective Dimension" drops toward 1.0, turning the manifold into a 1D line. |
| **Permutation Invariance** | Transformers are "Bag of Words" by default. Without PE, a model cannot distinguish between a sequence and its reverse. |
| **Gradient Flow** | When calculating $\frac{\partial L}{\partial x}$ in a residual layer, the gradient always includes the identity term. |
| **Tensor Geometry** | The last two dimensions are the "active" dimensions in PyTorch; `unsqueeze` is the primary tool for creating outer products. |

---

# Week 2: The Rotary Revolution (RoPE)
**Objective:** Upgrade the encoder from absolute "time-stamps" to relative "rotational alignment" using Complex Number theory.

## 1. Core Implementation

### The RoPE Mechanism (Geometry > Algebra)

* **Discovery:** RoPE is a **Basis Transformation**, not an additive signal. We inject position by rotating the basis of Queries and Keys.
* **The "Euler Shortcut":** We replaced massive block-diagonal rotation matrices with simple **Complex Multiplication**. By treating pairs of dimensions as complex numbers ($x + iy$), the rotation becomes a fast element-wise operation: `X_complex * freqs_cis`.

### Tensor Fluency & Robustness
* **Discovery:** **Rank Agnosticism** is vital. Using `*X.shape[:-1]` instead of hardcoded dimensions allows the RoPE layer to handle 3D (inference), 4D (training), or 5D (video) tensors.
* **Discovery:** The **"Last Two Dimensions" Rule** reigns supreme. Whether using `interleaved` (pairs) or `half-split` (chunks), the rotation only operates on the final feature manifold, leaving Batch/Head dimensions untouched via broadcasting.

## 2. Geometric Awareness: Relative vs. Absolute
* **The Fundamental Shift:** Applying rotations $R_{pos}$ and $R_{pos+k}$ forces the dot product to depend **only** on the relative distance $k$.
* **The "Cancellation" Effect:** The absolute indices $m$ and $n$ vanish, leaving only the relative offset $m-n$.
* **The Multi-Scale Clock:** A log-spaced frequency bank creates a "time piece" with hands moving at different speeds to capture neighbors (high freq) vs. long-range dependencies (low freq).

## 3. Lab Observations: Debugging RoPE

| Observation | Discovery |
| :--- | :--- |
| **Unitarity** | RoPE is an **Orthogonal Transformation**. `norm(RoPE(x)) == norm(x)`. It injects position without altering energy. |
| **Interleaved vs. Half-Split** | `Half-Split` (Llama style) is faster on GPUs due to contiguous slicing. Mathematically, they yield identical attention scores. |
| **Broadcasting Crash** | To make the `freqs_cis` cache compatible with MHA, we must introduce "dummy" dimensions (`unsqueeze`). |
| **Integration Point** | RoPE must be applied **after** Linear Projections but **before** Softmax. Never apply to Values ($V$). |

---

# Week 3: The Modern Architecture (Llama-fication)
**Objective:** Evolve to "Modern LLM" standards (SwiGLU + RMSNorm) and prove Inductive Bias trade-offs.

## 1. Core Implementation

### RMSNorm & SwiGLU (Efficiency > Tradition)
* **RMSNorm Discovery:** Abandoning `LayerNorm`'s mean-centering preserves signal magnitude while saving compute. It projects input onto a hypersphere.
* **SwiGLU Discovery:** Replaces "dumb" ReLU with a "smart" Gated Linear Unit.
    * **The Dimensionality Trap:** Switching from 2 matrices (FFN) to 3 matrices (SwiGLU) requires resizing hidden dim by $\approx \frac{8}{3}$ to maintain parameter parity.
    * **Initialization:** Manual `nn.Parameter` allocation gives "Quant-level" control over starting states.

### Tensor Fluency: The "Broadcasting" War
* **The Broadcasting Crash:** The 4D mismatch between Attention output (`Batch, Heads, Seq, Dim`) and RoPE cache is the most common bug.
    * *The Fix:* Optimize `RoPE` init to create a shape of `(1, 1, Seq, Dim)` to allow broadcasting against heads.
* **The "0 and -1" Rule:** Never hardcode intermediate dimensions (like `dim=2`). Always anchor operations to Batch (`dim=0`) or Feature (`dim=-1`).

## 2. Geometric Awareness: Inductive Bias

* **The Hypothesis:** RoPE is better *at specific things*, not everything.
* **The Experiment:**
    * **Sequence Reversal (Absolute Task):** **Standard (Abs PE)** wins. RoPE struggles to "infer" absolute 0,0 coordinates.
    * **Repeat Copy (Relative Task):** **Modern (RoPE)** wins. It converges 3x faster because the rule "copy from 15 steps back" is a single rotation.
* **The "Grokking" Cliff:** Standard models may flatline for epochs (memorizing noise) before suddenly discovering the mathematical rule.

## 3. Engineering Rigor Checklists
* **The "Senior Quant" Assertion:** Always assert `d_model // n_heads % 2 == 0`. Odd head dimensions make complex number pairing impossible.
* **SwiGLU Math:** `F.silu(x)` is numerically stable; `x * sigmoid(x)` is not.
* **RoPE Integration:** RoPE goes *after* split heads, *before* attention scores.
* **MPS Acceleration:** Targeting Mac M-series GPUs can reduce training time from minutes to seconds for deep networks.