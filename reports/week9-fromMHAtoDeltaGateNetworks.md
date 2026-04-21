# Research Log: Gated Delta Networks

## From KV Cache to Fixed-Size State — Three Steps

**Date:** April 20-21, 2026
**Topic:** Linear Recurrence, Delta Rule, Chunkwise Parallelisation
**Code:** `delta_gate_layer.py`

---

## TL;DR: The Conceptual Shift

Before diving into the math, the architectural shift from standard Multi-Head Attention (MHA) to Linear Attention (like Delta Networks) fundamentally changes how information travels through time:

*   **Standard MHA is direct (*token-to-token*):** It operates as a single-hop routing mechanism. Token $t_{100}$ looks directly at token $t_2$, grabs exactly what it needs, and moves on. This grants perfect, lossless recall but requires an expensive $O(N^2)$ complete graph of connections.
*   **Linear Attention is cascaded (*token-to-token-to-token*):** It operates as a multi-hop reasoning mechanism. Token $t_{100}$ cannot see $t_2$ directly. Instead, $t_2$ writes into a shared memory state $S$, $t_3$ reads and modifies that state, and so on until $t_{100}$ reads the heavily compounded result. This drops the complexity to $O(N)$ and unlocks deep sequential reasoning, but forces information to survive a continuous game of telephone through a fixed-size state matrix.

## In Practice: The Hardware Reality (Why This Matters)

This mathematical shift is not just a theoretical flex; it is a strict hardware requirement for modern long-context models. To understand the stakes, consider the memory footprint of a single layer in a base model like Qwen 3.5 ($d_\text{model} = 4096$, $16$ heads, $d_h = 256$) processing a $32,768$ (32K) token sequence:

*   **Standard Attention (The KV Cache):** Must store a perfect record of 2 vectors ($K$ and $V$) per token. 
    *   $32,768 \text{ tokens} \times 4,096 \text{ dim} \times 2 = \mathbf{268 \text{ million elements}}$.
    *   In FP16, that is **~536 MB** of VRAM *per layer, per sequence*. At inference, memory grows linearly and compute slows down quadratically ($O(N^2)$) as it scans this massive cache.
*   **Delta Networks (The Fixed State):** Compresses history into a fixed $d_h \times d_h$ matrix per head.
    *   $16 \text{ heads} \times 256 \times 256 = \mathbf{1.04 \text{ million elements}}$.
    *   In FP16, this is strictly **~2 MB** of VRAM. It remains exactly 2 MB whether generating token 100 or token 100,000. Inference compute is constant time ($O(1)$) and exceptionally fast because the state easily fits in GPU SRAM.

**The Trade-off:** The Delta Gate provides a staggering **256x reduction** in memory at 32K context, but it comes at the cost of a 256:1 lossy compression ratio. The "surgical eraser" mechanism of the Delta Rule is doing the heavy lifting to ensure only the most critical information survives that compression.


## Step 1 — Why Softmax Attention Needs the Cache

At step $i$, a query must attend to all past keys inside a softmax:

$$\text{Attention}(q_i, K, V) = \text{softmax}\left(\frac{q_i K^\top}{\sqrt{d}}\right) V$$

The output at step $i$ with a generic kernel function $f$:

$$o_i = \sum_{j=1}^{t} f(q_i, k_j) \cdot v_j$$

If $f$ is linear — meaning $f(q_i, k_j) = \phi(q_i)^\top \phi(k_j)$ for some feature map $\phi$ — then:

$$o_i = \sum_{j=1}^{t} \phi(q_i)^\top \phi(k_j) v_j^\top$$

Pull $\phi(q_i)^\top$ outside the sum — it does not depend on $j$:

$$o_i = \phi(q_i)^\top \underbrace{\sum_{j=1}^{t} \phi(k_j) v_j^\top}_{S_t \in \mathbb{R}^{d \times d}}$$

The state $S_t$ accumulates all past keys and values. Retrieval is a single matrix-vector product $o_i = \phi(q_i)^\top S_t$, updated recurrently as each new token arrives.

With softmax this factorisation is impossible — the normalisation denominator $\sum_j \exp\left(q_i \cdot k_j / \sqrt{d}\right)$ couples all terms together, so $\phi(q_i)^\top$ cannot be pulled outside the sum. The full KV history must be materialised.

---

## Step 2 — Linear Attention as Associative Memory

Remove the softmax. The state becomes:

$$S_t = \sum_{i=1}^{t} v_i k_i^\top \in \mathbb{R}^{d \times d}$$

This is a fixed-size associative memory — a $d \times d$ matrix that compresses all past tokens. Retrieval is just $o_t = S_t q_t$. Constant memory, linear compute.

The trade-off: the state is lossy. All past tokens compete for the same $d \times d$ matrix with equal write strength. There is no mechanism to decide which tokens matter — every token writes unconditionally.

---

## Step 3 — Delta Rule as Innovation-Gated Writing

Introduce two ideas simultaneously: write selectively, and write only what is new.

**The state update:**

$$S_t = S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$$

**The innovation:** $v_t - S_{t-1} k_t$ — what the state currently predicts for key $k_t$ versus the actual value $v_t$. If the state already knows the answer, the update vanishes. Writing is self-limiting for well-learned content.

**The gate $\beta_t$:** controls write strength per token. Computed as:

$$\beta_t = \sigma(W_\beta x_t)$$

where $W_\beta$ is a learned linear layer. $\beta_t$ is not itself a parameter — it is a deterministic function of the input, computed fresh per token. This is the same pattern as LSTM gates.

**Design decision — per-head $\beta$:** The paper defines $\beta$ as a scalar per token. In the multi-head setting, the natural extension is one scalar per head per token — different heads can specialise their write behaviour independently. $W_\beta: (d_\text{model} \to n_\text{heads})$, applied before `split_heads`.

** Design decision **: We apply L2 normalization to Q and K as per the Delta Rule paper. This ensures the update acts as a stable Householder projection, preventing the recurrent state from exploding, which is critical since we no longer have a softmax to bound our magnitudes.


---

## The Parallelisation Problem and Solution

### Failed attempt — parallel scan

The recurrence can be written as:

$$S_t = S_{t-1} M_t + X_t$$

where $M_t = I - \beta_t k_t k_t^\top$ and $X_t = \beta_t v_t k_t^\top$. This matches the first-order recurrence form — parallel scan applies. But products of $M_t$ matrices grow from $O(1)$ rank-1 terms to $O(L)$ terms at each scan level, requiring materialisation of $d \times d$ matrices at every step. Memory cost: $O(Ld^2)$. Impractical.

### The WY Representation

The transition matrices are rank-1 perturbations of the identity — structurally similar to Householder reflectors. Their cumulative product collapses:

$$\prod_{i=1}^{t} \left(I - \beta_i k_i k_i^\top\right) = I - WK^\top$$

where $W$ and $K$ are thin $t \times d$ matrices — two vectors stacked, not a product of $t$ dense matrices. This is the WY representation (Bischof & Van Loan, 1987).

**Key insight:** $(I - A)^{-1}$ where $A = \text{tril}(\text{diag}(\beta) K K^\top, -1)$ is the total effect matrix from the causal adjacency graph — entry $[r, i]$ is the sum of weights of all paths from position $i$ to position $r$. By the Neumann series:

$$(I - A)^{-1} = I + A + A^2 + A^3 + \cdots$$

This terminates in finite steps because $A$ is strictly lower triangular — nilpotent. No approximation, exact computation. The connection to causal inference is direct: this is do-calculus total effect estimation applied to the attention mechanism's internal dependency graph.

### Chunkwise Algorithm

Divide the sequence into chunks of size $C$. For chunk $i$, store only the $d \times d$ state at chunk boundaries $S_{[i]}$. Inside each chunk, run the UT transform,
using broadcasted multiplication for efficiency (avoiding dense diagonal matrices):

```python
# instead of diag_embed
beta_K = beta_chunk.unsqueeze(-1) * K_chunk
A = torch.tril(beta_K @ K_chunk.transpose(-1, -2), diagonal=-1)

T = torch.linalg.solve_triangular(eye - A, eye, upper=False)  # (I-A)⁻¹
T_pre = T * beta_chunk.unsqueeze(-2) 
W = T_pre @ K_chunk   # (B, H, C, head_dim)
U = T_pre @ V_chunk   # (B, H, C, head_dim) — symmetric structure, β absorbed
```
**Output equation:**

$$O = Q S^{-\top} + (Q K^\top \odot M)(U - W S^{-\top})$$

- First term: what the past already learned — inter-chunk memory retrieval
- Second term: what is new in this chunk — intra-chunk innovation, causally masked by $\odot M$
- $\beta$ has disappeared — absorbed into $W$ and $U$ by the UT transform

**Critical ordering — output before state update:**

```python
# S⁻ is the incoming state — must be used for O_chunk first
O_chunk = Q_chunk @ S.transpose(-2,-1) + (Q_chunk @ K_chunk.transpose(-1,-2) * causal_mask) @ (U - W @ S.transpose(-2,-1))
# Only then update S for the next chunk
S = S @ (eye - W.transpose(-2,-1) @ K_chunk) + U.transpose(-2,-1) @ K_chunk
```

Reversing this order uses $S_\text{new}$ in the output equation — mathematically wrong, silent bug. No shape error will surface.

---

## PyTorch Tips and Tricks

**`@` vs `*`:** `@` is `torch.matmul` — aggregates across a dimension, shape shrinks. `*` is element-wise — shape unchanged. In the output equation both appear in the same line: `@` for score computation, `*` for causal masking.

**`torch.diag_embed`:** Broadcasting vs. diag_embed:
Multiplying by a diagonal matrix is mathematically elegant but computationally wasteful in PyTorch. Instead of torch.diag_embed(beta_chunk) to create a (B,H,C,C) matrix, unsqueeze the last dimension beta_chunk.unsqueeze(-1) and use element-wise multiplication * against your target tensor. This saves memory and drastically speeds up the FLOPs.

**`solve_triangular` vs `inv`:** $(I - A)$ is lower triangular with ones on the diagonal by construction. `torch.linalg.solve_triangular(..., upper=False)` exploits this via forward substitution — $O(C^2)$ not $O(C^3)$. Always prefer over `torch.linalg.inv` when structure is known.

**Unpacking view shapes:**
```python
split_shape = (B, seq_len, num_heads, head_dim)
Q = self.W_q(x).view(*split_shape).permute(0, 2, 1, 3)
```
Define the shape tuple once, unpack with `*` — cleaner than repeating dimensions for $Q$, $K$, $V$.

**Move constants outside the loop:** `eye` and `causal_mask` are identical every iteration. Compute once before the loop, reuse inside.

**Device and dtype propagation:**
```python
S = torch.zeros(B, H, dim, dim, device=Q.device, dtype=Q.dtype)
```
Always match device and dtype to the input tensor — avoids silent CPU/GPU transfer errors.

---

## Shape Contract

| Tensor | Shape | Notes |
|---|---|---|
| $x$ | $(B, L, d_\text{model})$ | Input |
| $Q, K, V$ | $(B, H, L, d_h)$ | Post `split_heads` |
| $\beta$ | $(B, H, L)$ | Post `permute(0,2,1)` |
| $S$ | $(B, H, d_h, d_h)$ | State — constant size |
| $Q, K, V$ chunk | $(B, H, C, d_h)$ | Per chunk slice |
| $\beta$ chunk | $(B, H, C)$ | Per chunk slice |
| $A$ | $(B, H, C, C)$ | Strictly lower triangular |
| $T$ | $(B, H, C, C)$ | $(I-A)^{-1}$ — lower triangular |
| $W, U$ | $(B, H, C, d_h)$ | UT transform outputs |
| causal mask $M$ | $(C, C)$ | Lower triangular ones — chunk-local |
| $O$ chunk | $(B, H, C, d_h)$ | Chunk output |
| $O$ | $(B, H, L, d_h)$ | Full output — fed to `combine_heads` |

---

## Code

See `delta_gate_layer.py`.

---

## 5 Questions to Test Your Understanding

**Q1: The softmax barrier**
Why does removing softmax enable a fixed-size state? What property of softmax prevents compression into a recurrent form?

**Q2: Self-limiting writes**
What happens to the state update when $S_{t-1}$ already perfectly predicts $v_t$ from $k_t$? What does this mean for tokens the model has seen many times?

**Q3: The solver choice**
Why is `solve_triangular` preferred over `torch.linalg.inv` for computing $(I-A)^{-1}$? What property of $A$ makes this possible?

**Q4: Output before state**
Why must $O_\text{chunk}$ be computed using $S^-$ before $S$ is updated? Explain why there would be no shape error but still a silent mathematical bug if the order were reversed.

**Q5: Per-head $\beta$**
The paper defines $\beta$ as a scalar per token. Justify the decision to use $W_\beta: (d_\text{model} \to n_\text{heads})$ instead of $W_\beta: (d_\text{model} \to 1)$. When would the scalar version be preferable?


## Literature
**Paper 1 — The foundation**
Yang, S., Wang, B., Zhang, Y., Shen, Y., & Kim, Y. (2024). *Parallelizing Linear Transformers with the Delta Rule over Sequence Length.* NeurIPS 2024. arXiv:2406.06484.

**Paper 2 — The gating extension**
Yang, S., et al. (2024). *Gated Delta Networks: Improving Mamba2 with Delta Rule.* arXiv:2412.06464.

**Blog — derivation of the parallelisation trick**
Yang, S. (2024). *DeltaNet Explained (Part II): An algorithm that parallelizes DeltaNet computation across the sequence length dimension.* https://sustcsonglin.github.io/blog/2024/deltanet-2/

**Background — WY representation**
Bischof, C., & Van Loan, C. (1987). *The WY representation for products of Householder matrices.* SIAM Journal on Scientific and Statistical Computing, 8(1), s2–s13.

**Background — Neumann series / total effects in causal graphs**
Pearl, J. (2009). *Causality: Models, Reasoning, and Inference.* 2nd ed. Cambridge University Press. — Chapter 3 for do-calculus and total effects via $(I - A)^{-1}$.

