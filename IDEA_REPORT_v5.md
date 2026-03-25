# Idea Discovery Report v5: DriftCIL

**Direction**: Cross-domain HSI+LiDAR CIL — Benchmark + Drift-Aware Method + Statistics Memory
**Date**: 2026-03-25
**Pipeline**: novelty-check → idea-discovery (research-lit → idea-creator → novelty-check → research-review → refine)

## Executive Summary

**DriftCIL = AnchorLoRA + DARA + CSM**

三合一方法：在 S²CM backbone 上，用 **DARA (Drift-Aware Rank Allocation)** 自动分配 per-branch per-layer LoRA rank，用 **CSM (Cross-modal Statistics Memory)** 存储旧类统计量做伪特征回放。同时发布 **首个跨域多模态 HSI+LiDAR CIL benchmark**。

**Target**: TGRS (7.5/10 评估) 或 TIP (6.5/10 评估)
**量化目标**: 75.7% → 78-80%，Houston forgetting -24.6pp → -18pp 以下

---

## Method Overview

### Component 1: DARA (Drift-Aware Rank Allocation)

**Problem**: 手动设 r_HSI=4, r_LiDAR=8 是不可泛化的调参。

**Solution**: 每个 task 开始前，用 rank=1 的 probe LoRA 跑 K=16 步，测量每个 branch 每个 layer 的 cosine drift score。在固定总预算下按 drift 比例分配 rank。

**Algorithm**:
```
1. Attach probe LoRA (rank=1) to all 8 units (4 HSI blocks + 4 LiDAR blocks)
2. Run K=16 minibatch steps with CE + KD loss
3. Measure drift: d_u = mean(1 - cosine(anchor_feat, probe_feat)) per unit
4. Two-stage allocation:
   a. Branch level: split free budget between HSI and LiDAR by total drift
   b. Layer level: within each branch, split by per-layer drift
5. r_u = r_base + allocated_extra, clipped to [0, r_max]
6. Re-initialize proper LoRA with assigned ranks, train normally
```

**Budget**: 32 rank units per task (equivalent to 4+4 per block, same total as uniform r=4)

**Novelty vs prior work**:
- vs PEARL: uses parameter distance, not feature drift
- vs FM-LoRA: uses task similarity, not measured per-branch drift
- vs CoDyRA: uses sparsity regularization, not online estimation
- vs OA-Adapter: generic budget rule, not multimodal drift-guided
- **None of these do per-modality per-layer drift-guided rank for multimodal CL**

### Component 2: CSM (Cross-modal Statistics Memory)

**Problem**: Exemplar-free 方法无法回访旧类数据，prototype drift 导致遗忘。

**Solution**: 存储每个旧类的 per-branch + fused-space 均值和标准差（~32KB for 32 classes）。训练时从 Gaussian 采样伪特征做 prototype 补充 + statistics matching loss。

**Storage**: 32 classes × 4 spaces × 64 dims × 2 (mean+std) × 2 bytes (fp16) ≈ 32KB

**What to store**:
- spec: mean + std (64-d)
- hsi_spa: mean + std (64-d)
- lid_spa: mean + std (64-d)
- fused: mean + std (64-d)

**Usage during training**:
- Sample pseudo-features from N(mu, noise_scale * sigma) for old classes
- Use pseudo prototypes for cosine NCM loss on old classes
- Statistics matching loss: MSE(current_mean, stored_mean) + 0.1 * MSE(log_std, log_stored_std)

### Component 3: Benchmark

**First standardized cross-domain HSI+LiDAR CIL benchmark**

**Datasets**: Trento (6cls), Houston2013 (15cls), MUUFL (11cls), optionally Houston2018 (20cls)

**Protocols**:
- 9-task marathon: Trento [2,2,2] → Houston [5,5,5] → MUUFL [4,4,3]
- 3-task simplified: each dataset = 1 task
- 6+ orderings: THM, HTM, MTH, HMT, TMH, MHT
- Modality-stress orders (to be defined)

**Metrics**: Avg TAg, per-domain accuracy, forgetting rate, BWT, parameter growth, memory overhead

**Baselines**: 12+ methods (frozen, EWC, LwF, iCaRL, LUCIR, analytic, DCPRN, prompt tuning, uniform LoRA, AnchorLoRA, DriftCIL variants)

---

## Loss Function

```
L = L_CE_branch + 0.5 * L_CE_fused + λ_kd * L_kd + λ_ortho * L_ortho + 0.1 * L_stats_match
```

---

## Experiment Plan

### Phase 1: Quick Pilot (1-2 GPU hours)
THM order, max_tasks=4, reduced epochs (warmup=8, lora=12, probe=8)

| Run | Config | Purpose |
|-----|--------|---------|
| 1 | fixed r=4/8 (current best) | Regression check |
| 2 | uniform budget=48 | Budget-matched baseline |
| 3 | drift budget=48 | DARA only |
| 4 | drift budget=48 + CSM | DARA + CSM |

**Success**: Run 3 > Run 2, Run 4 reduces Houston forgetting vs Run 3

### Phase 2: Full Validation (4-6 GPU hours)
THM order, full 9-task, normal epochs (warmup=15, lora=25, probe=16)

| Run | Method | Purpose |
|-----|--------|---------|
| 1 | AnchorLoRA (fixed r=4/8) | Baseline |
| 2 | Uniform budget=48 | Budget control |
| 3 | Fixed r=4/8 + CSM | CSM isolated |
| 4 | DARA only (budget=48) | DARA isolated |
| 5 | DriftCIL full (DARA + CSM) | Full method |
| 6 | DriftCIL (budget=72) | Fairer budget match |

### Phase 3: Benchmark + Ablations (8-12 GPU hours)

Benchmark table: {THM, HTM, MTH} × {AnchorLoRA, DriftCIL} × seed=0 = 6 runs

Ablations on THM:
- uniform vs drift allocation
- drift only vs drift + stats
- branch-only vs branch+layer allocation
- budget 48 vs 72
- n_pseudo = 8 vs 16
- cosine drift vs L2 drift

---

## Implementation Plan

### New files:
- `code/driftcil_experiment.py` (~600-800 new lines, copied from anchor_lora_experiment.py)
- `code/stats_memory.py` (optional, ~120 lines)

### Key new components:
1. `AdaptiveTaskLoRABank` — accepts per-block rank list
2. `forward_features(return_block_feats=True)` — expose intermediate block features
3. `estimate_drift_and_allocate()` — probe + allocation
4. `largest_remainder_allocate()` — budget distribution
5. `StatsMemory` — store/sample per-class statistics
6. `build_mixed_prototypes()` — combine real + pseudo features
7. `train_driftcil_task()` — new training loop with DARA + CSM
8. `run_driftcil_marathon()` — marathon runner

### Coding order:
1. Copy baseline → driftcil_experiment.py
2. Implement AdaptiveTaskLoRABank with rank lists
3. Add block feature capture + probe drift
4. Get rank_mode=uniform working first
5. Turn on rank_mode=drift
6. Add StatsMemory
7. Add fused-memory loss
8. Benchmark protocol registry

---

## Risk Assessment

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Drift tracks adaptation pressure, not forgetting | HIGH | Show drift-forgetting correlation; add retention weighting |
| Branch-wise stats too weak for joint structure | MEDIUM | Add fused-space statistics |
| Probe phase overhead | LOW | Only 16 steps, <5% of total training |
| DARA gains < 1pp | MEDIUM | Fall back to benchmark-first paper |
| Stats memory = compressed replay, not exemplar-free | LOW | Storage is 32KB, clearly not "exemplars" |

---

## GPT-5.4 Review Summary

- TGRS: 7.5/10, TIP: 6.5/10
- Benchmark is the strongest contribution
- DARA needs to be formalized, not just "adaptive hyperparameter"
- CSM needs concrete mechanism (pseudo-feature sampling), not vague "alignment"
- Must show drift predicts forgetting, not just adaptation magnitude
- Target: +2.5-4pp over AnchorLoRA, Houston forgetting -24.6pp → -18pp
