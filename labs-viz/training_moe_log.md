# Training Log - TinyStories MoE

Update: All the experiments below with losses below 7 would be considered normal, as we have found out on experiment 6
that the aux_loss is actually an accumulation over layers and the model under training has 6 layers, which produces a theoretical minimum loss of 6 (1 per layer).


## Run 002: The "Router Collapse" Baseline
**Date:** 2026-02-13
**Config:**
- Model: SparseMoETransformer (108M Params)
- Experts: 4 (Top-2 Gating)
- Batch Size: 32
- LR: 3e-4
- **Aux Loss Coef:** 0.01 (Default) 

**Observations:**
- Training started successfully on MPS.
- Throughput: ~5900 TPS (Surprisingly fast!).
- **Issue:** Aux Loss started high (~6.8) and *increased* to ~7.4 by step 60.
- **Diagnosis:** The router collapsed. The penalty (0.01) was too weak (0.07 total loss contribution) compared to the main CrossEntropy loss (5.0). The model ignored the load balancing to focus on the easy next-token prediction.

**Outcome:** Killed run at Step 60.

---

## Run 003: The "Force Balance" Experiment
**Date:** 2026-02-13
**Changes:**
- **Aux Loss Coef:** 0.1 (Increased 10x)
- **Goal:** Force the optimizer to care about expert utilization.

**Hypothesis:**
- Aux Loss should drop rapidly towards 2.0 (theoretical minimum for balanced Top-2 gating).
- Main Loss might start slightly higher (since we are constraining the model), but should eventually overtake the baseline as experts specialize.

**Results:**
- Seems to decrease a bit but then collapses again (norm is 42 on step 40)


## Run 004: The "Force even more balance" Exepriment
**Changes**
- **Aux Loss Coef:** 0.3 (Increased 10x3)

**Results:**
- Explodes. We need to use more batches.

## Run 005: Increase Batch to 64

**Results:**
- OOM

## Run 004: The "Force even more balance" Exepriment
**Changes**
- **Aux Loss Coef:** 0.3 (Increased 10x3)

**Results:**
- Explodes. We need to use more batches.

## Run 005: Increase capacity to 3

**Results:**
- stabilizes around 12 and doesn't go down


## Run 006: The "Z-Loss" Stability Test (with 0.001 router z score)
**Date:** 2026-02-16
**Changes:**
- **Implemented Router Z-Loss** (Penalizing large logits).
- **Capacity Factor:** 1.15 (Tightened from 1.25 to test precision).
- **Aux Loss Coef:** 0.01 (Reverted to standard).

**Hypothesis:**
- Z-Loss will prevent the "logit explosion" that causes router collapse.
- Even with a weak coefficient (0.01) and tight capacity (1.15), the router should remain balanced.

## Run 006: The "Z-Loss" Stability Test (Aborted)
**Date:** 2026-02-16
**Changes:**
- **Implemented Router Z-Loss** (Penalizing large logits).
- **Capacity Factor:** 1.15 (Tightened from 1.25 to test precision).
- **Aux Loss Coef:** 0.01 (Reverted to standard).

**Hypothesis:**
- Z-Loss will prevent the "logit explosion" that causes router collapse.
- Even with a weak coefficient (0.01) and tight capacity (1.15), the router should remain balanced.

## Run 007: The "Balanced" Run
**Date:** 2026-02-16
**Status:** HEALTHY
**Metrics (Step 100):**
- **Main Loss:** 4.55 (Rapid convergence).
- **Aux Loss:** ~7.06 Total (1.17 per layer).
- **Interpretation:** Experts are perfectly balanced. No collapse.

**Events:**
- **Step 90:** Massive Gradient Spike (Norm 19.93).
- **Resolution:** Gradient Clipper saved the run. Loss continued to drop (4.55) immediately after.
- **Throughput:** ~5800 TPS.

**Action:** Continue training.








