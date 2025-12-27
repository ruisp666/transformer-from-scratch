Plan Week 2: RoPE - Rotary Position Embeddings
Monday (2 hours) - NOT STARTED
Task: RoPE Theory
Paper: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
Plan:

 Read RoPE paper sections 1-3
 Understand rotation mechanism using complex numbers
 Work through math: why it preserves relative position
 Create diagrams explaining the concept

Deliverable: Notes on RoPE with mathematical understanding

Tuesday (2 hours) - NOT STARTED
Task: Frequency Computation
Challenge 2.1 - Core implementation
Plan:

 Implement precompute_freqs_cis() function
 Test unit magnitude property (|cis| = 1)
 Verify frequency ranges
 Understand theta=10000 choice

Deliverable: Working frequency precomputation with tests

Wednesday (2 hours) - NOT STARTED
Task: Rotation Application - Part 1
Challenge 2.2 - First half
Plan:

 Implement apply_rotary_emb() for single tensor
 Convert real to complex, apply rotation, convert back
 Test L2 norm preservation
 Verify on simple inputs

Deliverable: Basic rotation working

Thursday (2 hours) - NOT STARTED
Task: Rotation Application - Part 2
Challenge 2.2 - Second half
Plan:

 Handle batching and multi-head dimensions
 Implement relative position property test
 Verify: dot(rotate(v,i), rotate(v,j)) depends only on (i-j)
 Test edge cases

Deliverable: Complete apply_rotary_emb() with verification

Friday (2 hours) - NOT STARTED
Task: RoPE Attention Module
Challenge 2.3 - Setup
Plan:

 Build RoPEAttention class structure
 Integrate rotation into attention flow
 Test basic functionality
 Compare to standard attention

Deliverable: RoPEAttention module passing basic tests

Saturday (4 hours) - NOT STARTED
Task: Comparison & Analysis
Challenge 2.4 - Empirical comparison
Plan:

 Complete RoPEAttention with causal masking (1 hour)
 Set up arithmetic task dataset (1 hour)
 Train both models: sinusoidal vs RoPE (1.5 hours)
 Evaluate on different lengths, create plots (0.5 hours)

Deliverable: Training curves and extrapolation analysis

Sunday (1 hour) - NOT STARTED
Task: Week 2 Review
Plan:

 Test RoPE with edge cases
 Review why RoPE extrapolates better
 Write summary: "Absolute vs Relative Position Encoding"
 Prep for normalization week


 #####################