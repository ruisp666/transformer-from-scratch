
# Weeks 5-8: Vectorized Sparse Mixture-of-Experts (MoE)

**Date:** January 19-feb 13, 2026


**Goal:** Implement a GShard-style Top-2 Gating MoE layer without using Python loops, focusing on tensor efficiency and capacity constraints.

### 1. The Core Problem

Standard Transformers use a dense FeedForward Network (FFN) applied to every token.

* **Dense:** 
* **MoE:** 

The challenge is **Routing**. In a system with 8 experts and 1M tokens, we cannot iterate through tokens to assign them to experts. We need to "teleport" tokens into expert buffers in parallel.

### 2. The Vectorization Strategy

We successfully replaced iterative logic with **Masked Matrix Multiplication**.

#### A. The Capacity Constraint

Experts have a fixed buffer size .
If more tokens want an expert than capacity allows, the overflow tokens are **dropped** (passed through via residual).

#### B. The "Sort" via Cumsum

To determine which token goes to which slot in the buffer, we utilized `cumsum` on One-Hot vectors.

* Let  be the matrix of requested experts.
* 
* This gives us a "Ticket Number" for every request in  parallel time.

#### C. The Dispatch Tensor

We constructed a binary tensor .

*  iff Token  is assigned to Expert  at Slot .
* This allowed us to move data using `einsum`:



### 3. Key Implementation Details

| Component | Technique | Insight |
| --- | --- | --- |
| **Gating** | `TopK(2)` + `Scatter` | We map the Top-2 probabilities (e.g., 0.7, 0.3) back to a sparse expert grid using `scatter_`. |
| **Routing** | `Cumsum` + `Mask` | Calculating `cumsum` on the one-hot selections gives us the buffer index without sorting. |
| **Compute** | 3D Weights | Experts are stored as `(N_Exp, In, Out)` parameters to allow `einsum` to process all experts simultaneously. |
| **Residual** | `output + x` | **Critical:** Tokens that are dropped due to capacity constraints effectively have `output=0`. The residual ensures they survive to the next layer. |
|**Parameters**| `nn.Parameters`| Implements the initialization from scracth using 3d parameters rather than linear with reshape|

### 4. The "Aha" Moment: The Bitmask

The breakthrough was realizing that the **Router** is just a machine that produces two tensors:

1. **Dispatch Mask (Binary):** "Who goes where?" (Used for `x` input).
2. **Combine Weights (Float):** "How much does it count?" (Used for `y` output).

By separating the **Control Plane** (calculating masks) from the **Data Plane** (moving tensors), we kept the forward pass clean and differentiable.

### 5. Next Steps

* **Load Balancing Loss:** Currently, the router might collapse to a single expert. We need to add the auxiliary loss: `Loss = N_experts * sum(gates * experts)`.
* **Expert Parallelism:** In a real cluster, the `(Experts, Capacity, Dim)` tensor would be split across GPUs.