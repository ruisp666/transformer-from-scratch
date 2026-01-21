

# Research Log: Implementing Grouped Query Attention (GQA)

**Date:** January 21, 2026
**Topic:** MHA to GQA Conversion & Tensor Broadcasting Logic
**Goal:** Convert a 6-head MHA model to a 2-group GQA model using Mean Pooling.

## 1. Experimental Setup & Parameters

Defining the dimensions for the 70M parameter model test.

```python
import torch

# Dimensions
batch_num = 20
num_head = 6       # Original MHA heads
seq_len = 5
head_dim = 16
n_groups = 2       # Target GQA groups

# Target Structure
# We want 2 groups, meaning 3 Query heads will share 1 Key/Value head.
# Ratio: 3:1

```

## 2. Phase 1: The Mean-Pooling Strategy

**Objective:** Compress existing MHA Key/Value weights into GQA heads without losing information.
**Method:** Average the weights of the heads within a group.

```python
# Simulate MHA Projection
multi_head_projection = torch.randn(batch_num, num_head, seq_len, head_dim)

# 1. Single Token Verification (Manual)
idx = 2
batch_idx = 0
single_token_hidden = multi_head_projection[batch_idx, :, idx, :].unsqueeze(0).unsqueeze(2)

# Manual pooling of a specific token
single_token_hidden_gqa_1 = single_token_hidden.mean(dim=1)

# 2. Tensor-wide Pooling
# This effectively converts (Batch, 6, Seq, Dim) -> (Batch, 1, Seq, Dim) if averaging all
# NOTE: We need to group first before averaging for GQA.

```

## 3. Phase 2: The Bottleneck (RuntimeError)

**The Issue:** Attempting to multiply the 6-head Query tensor directly against the 2-head pooled Key tensor fails due to dimension mismatch.

```python
# The Setup
Q = torch.randn(batch_num, num_head, seq_len, head_dim) # Shape: [20, 6, 5, 16]

# Creating the pooled Key (Simulated)
# We reshape to separate groups, then mean-pool across the sub-heads
grouped_reshape = multi_head_projection.view(batch_num, num_head//n_groups, n_groups, seq_len, head_dim)
grouped_reshape_pool = grouped_reshape.mean(dim=1).unsqueeze(1) 
# Resulting Key Shape: [20, 1, 2, 5, 16] (Matches 2 groups)

# The Crash
try:
    # Attempting [20, 6, ...] @ [20, 1, 2, ...].T
    torch.matmul(Q, grouped_reshape_pool.transpose(-2, -1))
except RuntimeError as e:
    print(f"Error: {e}")

# Error Log:
# "The size of tensor a (6) must match the size of tensor b (2) at non-singleton dimension 2"

```

**Constraint:** We have 2 pooled Key groups and 6 Query heads. We need to multiply each group of 3 Query heads by the *same* Key head.

## 4. Phase 3: The Solution (5D Tensor Broadcasting)

**The Unlock:** Reshape the tensors into 5 dimensions to leverage PyTorch broadcasting.
**Mental Model:** Treat the last two dimensions (Seq, Dim) as a "square blob" (matrix math zone) and the earlier dimensions as routing coordinates.

### The Logic

1. **Query Reshape:** `[Batch, Groups, Heads_Per_Group, Seq, Dim]`
2. **Key Reshape:** `[Batch, Groups, 1, Seq, Dim]`
3. **Action:** PyTorch broadcasts the `1` in the Key tensor to match the `Heads_Per_Group` in the Query tensor.

```python
# 1. Reshape Q to isolate groups
# Shape becomes: [Batch, 2, 3, Seq, Dim]
Q_head_grouped = Q.view(batch_num, n_groups, num_head//n_groups, seq_len, head_dim)

print(f"Grouped Query Shape: {Q_head_grouped.shape}") 
# Output: torch.Size([20, 2, 3, 5, 16])

# 2. The Key is already pooled to [Batch, Groups, 1, Seq, Dim] via the mean operation
# Note: Ensure the singleton dimension is in the right place for broadcasting

```

## 5. Verification ("The Crazy Test")

Confirming that the spatial broadcasting yields the exact same results as manual loop computation.

```python
# Simulation Data
A = torch.randn(6, 3, 2, 5, 16) # Query: [Batch, Heads_Per_Group, Groups, Seq, Dim] (Permuted for test)
B = torch.ones(6, 1, 2, 5, 16)  # Key:   [Batch, 1, Groups, Seq, Dim]

# --- Manual Check (Specific Slice) ---
# Pick Group 1, Head 1
pooled_key_single = B[2, 0, 1, ...] # Shape: (Seq, Dim)
grouped_queries = A[2, :, 1, ...]   # Shape: (Heads_Per_Group, Seq, Dim)

# Manual Dot Product
manual_calc = grouped_queries @ pooled_key_single.transpose(-1, -2)

# --- Broadcast Check ---
# Multiply the whole tensors
attention_weights = A @ B.transpose(-2, -1)

# Compare
print(f"Broadcast Result (First Head): \n{attention_weights[2, 0, 1]}")
print(f"Manual Result: \n{manual_calc[0]}")

# Result: It's the same.

```

## Summary of Architecture Change

| Feature | Original MHA | GQA (Implementation) |
| --- | --- | --- |
| **KV Heads** | 6 (Matches Query) | 2 (Pooled) |
| **Tensor Rank** | 4D `(B, H, S, D)` | 5D `(B, G, H/G, S, D)` |
| **Broadcasting** | Direct matching | `1` broadcasts to `H/G` |
| **Memory Impact** | High VRAM | Reduced VRAM (3x savings on KV) |