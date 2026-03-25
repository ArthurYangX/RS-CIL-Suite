# Project Stage Summary & Next Phase Plan

**Date**: 2026-03-24
**Project**: AnchorLoRA — Multimodal HSI+LiDAR Cross-Domain Class-Incremental Learning

---

## Stage 1 Summary: Method Development & Validation (Completed)

### What we built
- **S2CM backbone**: Spectral-Spatial Contrastive Mamba (3-branch: spectral + HSI-spatial + LiDAR-spatial)
- **AnchorLoRA CIL**: Spectral freeze after warmup + unconstrained LoRA on spatial branches + asymmetric rank (HSI r=4, LiDAR r=8)
- **Marathon CIL protocol**: 3 domains × 9 tasks × 32 classes (Trento→Houston→MUUFL)

### Key results (17 experiments)
- **Best: 75.7% Avg TAg** (mean of 2 seeds), +7.0pp over SHINE
- Ordering-robust: 3 orderings all show +4-8pp improvement
- Method simplified through ablation: no ortho, no KD, no dual head, no routing needed

### Ideas tested and eliminated
| Idea | Result | Reason |
|------|--------|--------|
| DualSpeed-S²CM (dual head) | 72.3% (-3.1pp) | Dual head over-complicates |
| SpecRoute (spectral routing) | 72.1% (-3.3pp) | Routing worse than accumulation |
| Orthogonal constraint | 74.4% (-1.0pp) | Too restrictive for small rank |
| Domain-selective LoRA | 70.8% (-4.6pp) | Cross-domain diversity helps |
| Auto-freeze (drift-based) | N/A | Spectral drift never converges |
| Spectral KD | 75.2% (-0.2pp) | Negligible effect |

### External review (GPT-5.4)
- **Score: 5/10 — Not ready for TIP**
- Top issues: insufficient baselines, overclaimed novelty, Houston forgetting, evaluation rigor

### Code assets
| File | Purpose | Status |
|------|---------|--------|
| `anchor_lora_experiment.py` | Main method (optimized, bug-fixed) | ✅ Ready |
| `dualspeed_experiment.py` | DualSpeed (eliminated) | ✅ Done |
| `specroute_experiment.py` | SpecRoute (eliminated) | ✅ Done |
| `baselines_experiment.py` | 9 CIL baselines on S2CM | ✅ Ready to run |
| `vit_baselines_experiment.py` | 3 CIL methods on Dual-Stream LSGAVIT | ✅ Ready to run |
| `sart_v4_experiment.py` | Previous SART method (baseline) | ✅ Has results |

---

## Stage 2 Plan: Baseline Comparison & Evaluation Rigor

### Goal
Raise review score from 5/10 → 7.5/10 by addressing all reviewer concerns.

### Phase 2.1: Baseline Experiments (需要 GPU, ~3 小时)

#### S2CM Backbone Track (9 baselines)

| Batch | Methods | Type | 并发 | 预计时间 |
|-------|---------|------|------|---------|
| 1 | frozen, analytic, pscen | Exemplar-free, 无/轻量训练 | 3 并发 | ~20 min |
| 2 | ewc, lwf, foster | Exemplar-free, fine-tune/adapter | 3 并发 | ~40 min |
| 3 | dcprn, icarl, lucir | Pseudo-replay / exemplar-based | 3 并发 | ~40 min |

**关键观察点**: 看 EWC/LwF/iCaRL 的 Houston 遗忘率。如果所有方法都 >-20pp，说明遗忘是 benchmark 特性。

#### ViT Backbone Track (3 methods)

| Batch | Methods | 并发 | 预计时间 |
|-------|---------|------|---------|
| 4 | ViT-frozen, ViT-L2P, ViT-AnchorLoRA | 3 并发 | ~60 min |

**注意**: ViT 线没有预训练 checkpoint，需要 warmup 从零训练。第一次会慢。

### Phase 2.2: 统计可靠性 (需要 GPU, ~4 小时)

| 实验 | 配置 | 预计时间 |
|------|------|---------|
| 5 seeds (best config) | seed 0-4, no ortho, r4/8, THM | 5 runs × ~30 min = ~2.5h (分批) |
| 6 orderings | THM, THM, HMT, HTM, MTH, MHT | 6 runs (3 已有) × ~30 min = ~1.5h |

### Phase 2.3: 分析实验 (需要 GPU, ~1 小时)

| 实验 | 目的 |
|------|------|
| Branch-wise drift 分析 | 量化证明 spectral < spatial drift |
| Per-class confusion matrix | 分析 Houston 遗忘的具体类别 |
| Params/FLOPs/latency 统计 | 效率对比表 |
| LiDAR adapter ablation | 测试 learned vs zero-padding |

### Phase 2.4: Narrative 调整 (不需要 GPU)

| 改动 | 内容 |
|------|------|
| 收窄 claim | 不说 "first Mamba+LoRA+CIL"，改为 "modality-asymmetric PEFT inductive bias" |
| 重新定位贡献 | (1) Marathon CIL protocol (2) Modality-asymmetric PEFT (3) Systematic ablation |
| Houston 遗忘 | 如果 baseline 都遗忘严重 → benchmark 特性。否则 → 加 feature regularization |
| 物理动机 | 用 drift 分析数据支撑 spectral 更稳定的 claim |

### Phase 2.5: Re-review (不需要 GPU)

带上完整 baseline 表 + 5 seeds + 6 orderings + drift 分析，重新提交 GPT-5.4 review。
目标 ≥ 7.5/10。

---

## 预期最终结果表格

### Table 1: Marathon CIL Main Results (S2CM Backbone)

| Method | Type | Trento | Houston | MUUFL | Avg TAg |
|--------|------|--------|---------|-------|---------|
| Frozen NCM | EF | ~83% | ~52% | ~53% | ~58% |
| SHINE | EF | ~87% | ~62% | ~60% | ~66% |
| EWC | EF | ? | ? | ? | ? |
| LwF | EF | ? | ? | ? | ? |
| Analytic | EF | ? | ? | ? | ? |
| PSCEN | EF | ? | ? | ? | ? |
| FOSTER | EF | ? | ? | ? | ? |
| DCPRN | PR | ? | ? | ? | ? |
| iCaRL (20/cls) | EB | ? | ? | ? | ? |
| LUCIR (20/cls) | EB | ? | ? | ? | ? |
| SART+SHINE | EF | ~87% | ~62% | ~66% | ~74% |
| **AnchorLoRA+SHINE** | **EF** | **~92%** | **~70%** | **~75%** | **~75.7%** |

### Table 2: Cross-Backbone Comparison (ViT vs S2CM)

| Backbone | CIL Method | Avg TAg |
|----------|-----------|---------|
| Dual-LSGAVIT | Frozen NCM | ? |
| Dual-LSGAVIT | L2P | ? |
| Dual-LSGAVIT | AnchorLoRA | ? |
| **S2CM** | Frozen NCM | 58.0% |
| **S2CM** | **AnchorLoRA+SHINE** | **75.7%** |

### Table 3: Ablation Study

| Component | Avg TAg | Δ |
|-----------|---------|---|
| Full (best config) | 75.4% | — |
| + ortho constraint | 74.4% | -1.0 |
| + spectral KD | 75.2% | -0.2 |
| Symmetric rank (r4/4) | 71.5% | -3.9 |
| warmup=2 vs 3 | 74.3% vs 74.4% | -0.1 |
| Larger rank (r8/16) | 74.5% | +0.1 |

### Table 4: Ordering Robustness (5 seeds mean±std)

| Ordering | Avg TAg | Δ SHINE |
|----------|---------|---------|
| THM | ?±? | +?pp |
| HMT | ?±? | +?pp |
| HTM | ?±? | +?pp |
| MTH | ?±? | +?pp |
| MHT | ?±? | +?pp |
| TMH | ?±? | +?pp |

---

## 执行时间线

| 阶段 | 内容 | 预计时间 | 需要 GPU? |
|------|------|---------|----------|
| 2.1 | Baseline experiments | ~3h | 是 |
| 2.2 | 5 seeds + 6 orderings | ~4h | 是 |
| 2.3 | 分析实验 | ~1h | 是 |
| 2.4 | Narrative 调整 | ~2h | 否 |
| 2.5 | Re-review | ~30min | 否 |
| **Total** | | **~10h GPU + 2.5h 写作** | |

开卡后建议按 2.1 → 2.3 → 2.2 → 2.4 → 2.5 顺序执行。先跑 baseline 看 Houston 遗忘情况再决定后续策略。
