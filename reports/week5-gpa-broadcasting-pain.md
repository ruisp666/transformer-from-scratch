

# Research Log: Implementing Grouped Query Attention (GQA)

**Date:** January 19-22, 2026
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

```

## 3. Phase 2: The Bottleneck (RuntimeError)

**The Issue:** Attempting to multiply the 6-head Query tensor directly against the 2-head pooled Key tensor fails due to dimension mismatch.

```python
# The Crash
try:
    # Attempting [20, 6, ...] @ [20, 1, 2, ...].T
    torch.matmul(Q, grouped_reshape_pool.transpose(-2, -1))
except RuntimeError as e:
    print(f"Error: {e}")

# Error Log:
# "The size of tensor a (6) must match the size of tensor b (2) at non-singleton dimension 2"

```

## 4. Phase 3: The Solution (5D Tensor Broadcasting)

**The Unlock:** Reshape the tensors into 5 dimensions to leverage PyTorch broadcasting.
**Mental Model:** Treat the last two dimensions (Seq, Dim) as a "square blob" (matrix math zone) and the earlier dimensions as routing coordinates.

### The Logic

1. **Query Reshape:** `[Batch, Groups, Heads_Per_Group, Seq, Dim]`
2. **Key Reshape:** `[Batch, Groups, 1, Seq, Dim]`
3. **Action:** PyTorch broadcasts the `1` in the Key tensor to match the `Heads_Per_Group` in the Query tensor.

## 5. Phase 4: Implementation Realities ("The Pain of Today")

**Date:** January 22, 2026

Moving from tensor theory to a working `nn.Module` revealed three critical engineering hurdles.

### A. The Contiguity Trap

**The Error:** `RuntimeError: view size is not compatible with input tensor's size and stride`.
**The Cause:** Our `split_heads` function used `.permute(0, 2, 1, 3)` to rearrange heads. This changes the *stride* but not the physical memory layout.
**The Fix:** You cannot `.view()` a permuted tensor without enforcing memory order first.

```python
# BAD
return x.permute(0, 2, 1, 3).view(...) 

# GOOD
return x.permute(0, 2, 1, 3).contiguous().view(...)

```

### B. The RoPE Optimization

**Observation:** Applying Rotary Embeddings to 6 Key heads and *then* averaging them is mathematically identical to averaging them first and *then* applying RoPE.
**The Win:** By pooling first, we apply RoPE to only 2 Key heads instead of 6. This saves **66% of the trigonometric operations** for the Key/Value cache processing.
**Logic:**

```python
# 1. Pool First
K_pooled = K.mean(dim=2, keepdim=True) 
# 2. RoPE Second (Optimized)
K_rope = self.rope(K_pooled)

```


---

## Summary of Architecture Change

| Feature | Original MHA | GQA (Implementation) |
| --- | --- | --- |
| **KV Heads** | 6 (Matches Query) | 2 (Pooled) |
| **Tensor Rank** | 4D `(B, H, S, D)` | 5D `(B, G, H/G, S, D)` |
| **Broadcasting** | Direct matching | `1` broadcasts to `H/G` |
| **RoPE Ops** | Applied to 6 Heads | **Applied to 2 Groups (Optimized)** |

---

### 5 Questions to Test Your Understanding

**Q1: The Compression Ratio**
You have a model with 32 Query Heads. You implement GQA with `n_groups=4`.
How many "Heads per Group" do you have, and what is the factor of memory reduction for your KV Cache compared to MHA?

**Q2: The Broadcasting Logic**
In our 5D setup `(Batch, Groups, Heads_Per_Group, Seq, Dim)`, why is the `Heads_Per_Group` dimension set to **1** for the Key tensor but **3** (or N) for the Query tensor? What would happen if the Key tensor had `3` in that dimension instead of `1`?

**Q3: RoPE Mathematics**
Why are we allowed to pool the heads *before* applying RoPE? If our positional embedding was a non-linear operation (like a standard feed-forward layer with ReLU), would this optimization still work?

**Q4: The "Online" vs "Offline" Distinction**
Our current implementation performs "Online Pooling" (averaging weights during the forward pass). If we wanted to perform "Offline Weight Folding," what specifically would we change in the `__init__` method and the `checkpoint` file?

**Q5: The Contiguity Check**
If you run `x.transpose(1, 2)` and then immediately print `x.is_contiguous()`, what will it return? Why does `.view()` crash if you skip the `.contiguous()` call after a transpose?

