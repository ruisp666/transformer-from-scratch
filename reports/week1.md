

# Week 1 Summary: The Semantic & Temporal Manifold

**Objective:** Build a position-aware Transformer Encoder from scratch and analyze its geometric properties.

## 1. Core Implementation

### The Attention Mechanism

The heart of the model is the **Scaled Dot-Product Attention**.

* **Discovery:** The  scaling factor is a "gradient stabilizer." Without it, the softmax saturates, and the manifold loses the ability to learn.
* **Mechanism:** It acts as a dynamic weighted average where tokens "pull" information from others based on their similarity in a projected space.

### Multi-Head Attention (MHA)

* **Discovery:** Tensor orientation is everything. We use `permute(0, 2, 1, 3)` during `split_heads` to ensure that the batch and head dimensions are isolated, allowing the last two dimensions (Sequence and Head Dim) to "agree" for matrix multiplication.
* **Discovery:** Attention at a specific position is a linear combination of signals, effectively acting as a composition of information from across the sequence.

### The Transformer Block (Residuals & Norm)

* **Discovery:** The **Residual Sign ()** is a safeguard. In backpropagation, it ensures an "Identity" gradient of **1.0** always flows back, preventing the signal from dying even if the sub-layer saturates.
* **Discovery:** **Pre-LN** is superior for deep architectures. By normalizing *before* the MHA, we keep the gradient bounded and protect the "Effective Dimension" of the manifold. **Post-LN** often leads to rank collapse and requires aggressive learning-rate warmups.

---

## 2. Temporal Awareness: Positional Encoding

* **Discovery:** Sinusoidal encodings aren't just "adding noise"; they provide a **Coordinate System**.
* **Linear Rotation Property:** I proved that  can be represented as a linear rotation of . This allows the model to perceive *relative* distance through phase shifts.
* **The Frequency Bank:** Using a log-space frequency bank creates "clocks" that run at different speeds across , allowing the model to capture both local and global sequence order.

---

## 3. Empirical Discoveries & Lab Observations

| Observation | Discovery |
| --- | --- |
| **Rank Collapse** | Without residuals or with Post-LN, the "Effective Dimension" (Participation Ratio) drops toward 1.0, turning the manifold into a 1D line. |
| **Permutation Invariance** | Transformers are "Bag of Words" by default. Without PE, a model cannot distinguish between a sequence and its reverse. |
| **Gradient Flow** | When calculating  in a residual layer, the gradient always includes the identity term, stabilizing deep networks. |
| **Tensor Geometry** | The last two dimensions are the "active" dimensions in PyTorch; `unsqueeze` is the primary tool for creating outer products for the frequency bank. |

---

## 4. Final Validation: The Reversal Task

We tested a **15-layer Pre-LN Transformer** on a sequence reversal task.

* **The Blind Model:** Failed to break a loss of **~2.1**, proving that self-attention alone cannot perceive order.
* **The Aware Model:** Using the Sinusoidal PE implementation, the loss plummeted to **0.0038**, proving the model successfully "unlocked" the temporal manifold.

---

### Sunday Review Checklist

* [ ] **Verify Scale:** Ensure `d**(-1/2)` is applied to avoid softmax saturation.
* [ ] **Consolidate Notes:** Finalize the "Probing Task" intuition (predicting position from embeddings).
* [ ] **Code Polish:** Transition `masked_fill` from `'-inf'` to `-1e9` for numerical stability.

