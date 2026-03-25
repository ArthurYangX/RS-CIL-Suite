# Methods Inventory: Everything We Tried & What It Taught Us

**Project:** Cross-Domain Multimodal (HSI+LiDAR) Class-Incremental Learning
**Duration:** 2025-12 → 2026-03 (~4 months)
**Total experiments:** 30+ method variants, 50+ runs

---

## 1. Project Timeline

```
2025-12  ──  Phase 0: HyperKD Framework
             HyperMamba backbone, single-dataset CIL (Trento only)
             Exemplar-based methods, KD variants, fusion ablations
             → Learned: single-dataset CIL is "solved" (~89%)

2026-03  ──  Phase 1: S2CM + Marathon Protocol Discovery
     early   S2CM backbone (spectral + HSI-spatial + LiDAR-spatial)
             Cross-dataset evaluation → 发现 Houston 崩溃到 44%
             10+ post-hoc classifiers → 全部失败 → SHINE (z-score whitening)
             → Learned: frozen backbone 是 representation bottleneck

2026-03  ──  Phase 2: Feature Transport
     mid     SART v1-v4: spectral-conditioned spatial feature correction
             → Learned: SART v2 actually beat exemplar methods (82%)
                        但 v4 (exemplar-free) 降到 78%

2026-03  ──  Phase 3: PEFT Methods
     late    AnchorLoRA, DualSpeed, SpecRoute, SDBT, SAFC
             大量 ablation (rank, ortho, KD, routing, drift compensation)
             → Learned: 简单方法 (unconstrained LoRA) 优于复杂设计

2026-03  ──  Phase 4: Latent Replay + Baselines
     latest  Pre-LoRA latent replay, 9 classical CIL baselines, ViT cross-backbone
             → Learned: replay 是最有效的单一策略 (80.4%)
```

---

## 2. Complete Method Inventory

### Phase 0: HyperKD Era (Trento, 3-task single-dataset CIL)

| # | Method | TAg Avg | Verdict | Problem Exposed |
|---|--------|---------|---------|-----------------|
| 1 | Baseline (Plateau scheduler) | 87.7% | OK | — |
| 2 | Baseline (Cosine scheduler) | **89.9%** | Best | 调度器比方法更重要 |
| 3 | Baseline (AdvCosine) | 89.7% | ≈Cosine | 高级调度没有额外收益 |
| 4 | MMCS selection (herding minmax) | 86.7% | Worse | 精心选样本 < 简单 random + 好调度 |
| 5 | MCF fusion (multi-scale cross-fusion) | 86.8% | Worse | 更复杂的融合没有帮助 |
| 6 | LoLA (low-rank lateral adapter) | 81.4% | Bad | Adapter 在小数据集上过拟合 |
| 7 | LoLA+MCF | 78.9% | Worst | 两个不 work 的模块叠加更差 |
| 8 | RKD Loss (relational KD) | 86.4% | Worse | 关系蒸馏在小 task 上收益有限 |
| 9 | LossD (KD variant) | 86.3% | Worse | 同上 |
| 10 | Exemplar herding | 76.0% | Bad | 旧选择策略严重不足 |
| 11 | Exemplar herdingminmax | 82.8% | Moderate | 最终选择策略，但仍不如好 baseline |

**Phase 0 核心教训：** 单数据集 CIL 已经被"解决"了——好的训练策略（Cosine scheduler）就能到 89.9%，复杂的 KD/fusion/adapter 都是负贡献。**真正的问题在跨域。**

---

### Phase 1: Post-hoc Classifier Exploration (S2CM, Marathon, frozen backbone)

| # | Method | Avg TAg | Δ SHINE | Trento | Houston | MUUFL | Problem Exposed |
|---|--------|---------|---------|--------|---------|-------|-----------------|
| 12 | Frozen NCM | 59.3% | -13.5 | 83.9% | 44.2% | 66.4% | Cross-domain catastrophe: Houston 崩溃 |
| 13 | **SHINE** (domain whitening) | **72.8%** | **baseline** | 87.0% | 68.6% | 70.7% | First-order shift 可修复 +13.5pp |
| 14 | SHINE+PCA64 | 74.8% | +2.0 | 88.3% | 71.1% | 72.4% | PCA 微小提升，不值得复杂度 |
| 15 | SHINE+TIDE (tangent capsule) | 73.0% | +0.2 | 87.2% | 68.8% | 70.9% | 可变形原型几乎无效 |
| 16 | SHINE+BRACE | 71.4% | -1.4 | 83.1% | 67.8% | 69.9% | Branch re-centering 反而伤害 Trento |
| 17 | SHINE+BRIO (adaptive weight) | 71.5% | -1.3 | 83.1% | 67.8% | 70.3% | 自适应分支权重没有帮助 |
| 18 | SHINE+COPE (pairwise expert) | 72.7% | -0.1 | 87.0% | 68.6% | 70.5% | 成对专家无效 |
| 19 | SHINE+LEID (endmember decompose) | 55.3% | **-17.5** | 69.0% | 56.7% | 46.0% | **灾难性**——复杂分解破坏特征 |
| 20 | SHINE+SRC (sparse coding) | 65.8% | -7.0 | 81.4% | 57.8% | 68.0% | 稀疏编码不适合此场景 |
| 21 | SHINE+NMF32 | 64.0% | -8.8 | 61.4% | 66.4% | 62.1% | NMF 严重损害 Trento |

**Phase 1 核心教训：** SHINE 是 post-hoc ceiling。在 frozen backbone 上，**没有任何分类器/后处理能超越简单的 domain whitening + cosine NCM**。10+ 方法全部失败。信息不在分类器层面——是 representation 本身不行。

---

### Phase 2: Feature Transport (SART)

| # | Method | Avg TAg | Trento | Houston | MUUFL | Problem Exposed |
|---|--------|---------|--------|---------|-------|-----------------|
| 22 | SART v2 (有 exemplar replay) | **82.0%** | 83.9% | 87.5% | 73.4% | **首次超过 iCaRL！但用了 replay** |
| 23 | SART+SHINE v2 | **82.6%** | 87.0% | 87.5% | 73.4% | SHINE 修 Trento，SART 修 Houston |
| 24 | SART v4 (exemplar-free) | 77.5% | 83.9% | 78.5% | 72.8% | 去掉 replay 后 Houston 掉 9pp |
| 25 | SART+SHINE v4 | 78.1% | 87.0% | 78.5% | 72.8% | 仍然好于 AnchorLoRA |

**Phase 2 核心教训：** Feature transport 本身是有效的——SART v2 是全项目最好的方法之一 (82.6%)。但去掉 exemplar replay (v4) 后掉了 4.5pp。**Replay 是关键成分，不是 transport。** SART v4 的复杂 transport 机制（70K params）只换来比简单 LoRA (36K params, 75.4%) 多 2.7pp。

---

### Phase 3: PEFT Methods

| # | Method | Avg TAg | Δ SHINE | Params | Problem Exposed |
|---|--------|---------|---------|--------|-----------------|
| 26 | **AnchorLoRA+SHINE** (best) | **75.4%** | +2.6 | 36K | 简单 LoRA 就够了 |
| 27 | AnchorLoRA+SHINE (seed 1) | 75.9% | +4.5 | 36K | Seed 间差异 ~0.5pp |
| 28 | + ortho constraint | 74.4% | +1.6 | 36K | 正交约束反而限制可塑性 |
| 29 | + spectral KD | 75.2% | +2.4 | 36K | KD 几乎无效 (Δ=0.2pp) |
| 30 | Symmetric rank (r4/4) | 71.5% | -1.3 | 24K | **LiDAR 需要 2× rank** |
| 31 | Larger rank (r8/16) | 74.5% | +1.7 | 74K | 2× params 没有收益 |
| 32 | warmup=2 (vs 3) | 74.3% | +1.5 | 43K | Warmup 影响小 |
| 33 | warmup=1, THM | 76.3% | +3.2 | 36K | 更少 warmup 反而好？(fewer tasks) |
| 34 | MTH ordering | 78.7% | +4.4 | 36K | **顺序影响 >3pp** |
| 35 | HMT ordering | 73.6% | +5.8 | 36K | Houston 首先也还行 |
| 36 | DualSpeed+SHINE | 72.3% | -0.5 | 36K | 双分类头过度复杂化 |
| 37 | SpecRoute+SHINE (THM) | 72.1% | -0.7 | 36K | **路由 < 简单累积** |
| 38 | SpecRoute+SHINE (MHT) | 67.7% | — | 36K | MHT 顺序下路由更差 |
| 39 | SDBT+SHINE | 72.8% | ±0 | — | Drift compensation ≈ SHINE |
| 40 | SAFC full+SHINE | 73.8% | +1.0 | — | 3 组件堆叠仅 +1pp |
| 41 | SAFC no-scrr+SHINE | 74.8% | +2.0 | — | 去掉 SCRR 反而更好？EFC alone |

**Phase 3 核心教训：**
1. **简单 > 复杂**：unconstrained LoRA (75.4%) > 所有花式设计 (DualSpeed, SpecRoute, SDBT, SAFC)
2. **KD、正交、路由、drift compensation 都没用**
3. **唯一有用的 insight：LiDAR 需要 2× rank（asymmetric）**
4. **顺序影响巨大（3-5pp）——这可能比方法本身更重要**

---

### Phase 4: Latent Replay & Baselines

| # | Method | Avg TAg | Δ SHINE | Problem Exposed |
|---|--------|---------|---------|-----------------|
| 42 | EWC+SHINE | 65.9% | -6.9 | EWC 在跨域下完全失效 |
| 43 | LwF+SHINE | 68.5% | -4.3 | LwF 也不行 |
| 44 | Analytic+SHINE | 72.7% | -0.1 | Analytic = SHINE (分类器不是瓶颈) |
| 45 | PSCEN+SHINE | 72.7% | -0.1 | 同上 |
| 46 | FOSTER+SHINE | 72.7% | -0.1 | 同上 |
| 47 | DCPRN+SHINE | 74.6% | +1.8 | Dual KD 微小提升 |
| 48 | **iCaRL** (exemplar) | **81.4%** | — | Exemplar 是硬实力 |
| 49 | iCaRL+SHINE | 82.1% | — | SHINE 对 exemplar 方法也有效 |
| 50 | **LUCIR** (exemplar) | **83.9%** | — | 最强 exemplar baseline |
| 51 | LUCIR+SHINE | 83.0% | — | SHINE 微伤 LUCIR |
| 52 | LatentReplay n10+SHINE | 76.4% | +3.6 | 仅 10 maps/class 就有效 |
| 53 | LatentReplay n20+SHINE | 76.9% | +4.1 | 微增 |
| 54 | **LatentReplay n50** | **80.4%** | **+7.6** | **不需要 SHINE！replay 本身解决域 shift** |
| 55 | LatentReplay n50+SHINE | 79.3% | +6.5 | SHINE 反而伤害 replay |
| 56 | LatentReplay n100 | 78.7% | — | 更多 map 反而更差 (过拟合?) |
| 57 | LatentReplay n100+SHINE | 79.2% | — | — |
| 58 | ViT-frozen | 66.6% | — | ViT 不如 S2CM |
| 59 | ViT-anchor (AnchorLoRA) | 46.7% | — | **AnchorLoRA 在 ViT 上完全失败** |

**Phase 4 核心教训：**
1. **Latent Replay n50 (80.4%) 是最好的 exemplar-free 方法**——不需要 SHINE
2. **SHINE 和 replay 冲突**——replay 自己解决了域 shift，SHINE 反而干扰
3. **AnchorLoRA 是 architecture-specific**——在 ViT 上完全失败 (46.7%)
4. **Exemplar methods (82-84%) 仍是 ceiling**——但 gap 缩小到 3-4pp

---

## 3. Patterns: What Works vs What Doesn't

### What Works
| Insight | Evidence | Strength |
|---------|----------|----------|
| **Domain whitening (SHINE)** | +13.5pp on frozen backbone | Strong |
| **Spatial LoRA (spectral frozen)** | +2.6pp over SHINE | Moderate |
| **Asymmetric rank (LiDAR 2×)** | +3.9pp vs symmetric | Strong |
| **Latent replay in pre-LoRA space** | 80.4% (best exemplar-free) | Strong |
| **Feature transport + replay** | 82.6% (best overall) | Strong |

### What Doesn't Work
| Attempted | Result | Why It Failed |
|-----------|--------|---------------|
| 正交 LoRA 约束 | -1.0pp | Rank-4 太小，约束限制可塑性 |
| Spectral KD | -0.2pp | 光谱分支已冻结，KD 没有目标 |
| Dual classification heads | -3.1pp | Over-engineering，简单 NCM 更好 |
| Spectral routing (MoE) | -3.3pp | 路由误差 > 累积误差 |
| Domain-selective LoRA | -4.6pp | 跨域 LoRA 的 diversity 是有益的 |
| Drift compensation (SDBT) | ±0pp | SHINE 已经解决了一阶 drift |
| SAFC (3 组件堆叠) | +1.0pp | 3 个组件加起来只有 1pp，不值得 |
| All 10+ post-hoc classifiers | ≤ SHINE | Frozen backbone = information bottleneck |
| EWC, LwF (classic CL) | < SHINE | 跨域场景下传统 CL 方法全部失败 |
| 复杂特征分解 (LEID, NMF, SRC) | Catastrophic | 破坏了原始特征空间结构 |

---

## 4. Fundamental Problems Exposed

### Problem 1: The 75-80% Wall (Exemplar-Free)
所有 exemplar-free 方法都卡在 75-80% 之间。LoRA、transport、replay 各自贡献 2-8pp，但无法叠加。

**根因假说：** Frozen spectral branch 只有 64 维，跨域时信息丢失不可恢复。空间分支即使有 LoRA 也只能弥补一部分。

### Problem 2: Houston Forgetting (-24.6pp)
Houston 是 "最难的受害者"：从 Task 3 的 92.5% 跌到 Task 8 的 67.9%。
- 这不是方法问题——**所有方法都有严重的 Houston 遗忘**
- 只有 replay (存储旧特征) 能有效缓解

### Problem 3: SART v2 > Everything, But It Uses Replay
SART v2 + SHINE (82.6%) 是全项目最好成绩，超过 iCaRL (82.1%)。但它使用了 exemplar replay。去掉 replay 后 (v4) 降到 78.1%。

**这说明：** 真正有效的不是 transport 机制本身，而是 replay。

### Problem 4: Architecture Specificity
AnchorLoRA 在 S2CM 上 +9.5pp，在 ViT 上 -20pp。方法完全依赖 backbone 的 3-branch 结构。

### Problem 5: Ordering Sensitivity (3-5pp)
MTH (78.7%) vs HMT (73.6%)。**训练顺序的影响 (5pp) 大于大多数方法改进 (2-3pp)**。

---

## 5. Honest Assessment

### What We Have (可以写进论文的)
1. **Marathon CIL Protocol** — 首个跨域多模态 RS CIL benchmark (novelty: high)
2. **SHINE** — 有效的 domain canonicalization baseline (novelty: low, but useful)
3. **Empirical finding: spectral 比 spatial 稳定** — drift analysis 支持 (novelty: moderate)
4. **AnchorLoRA** — asymmetric rank insight (novelty: moderate)
5. **Systematic ablation** — 30+ methods, 极其完整 (novelty: moderate, 但有价值)

### What We Don't Have
1. **没有一个方法能 clean beat exemplar baselines** (LatentReplay 80.4% vs iCaRL 82.1%)
2. **没有真正的 "方法论创新"** — 所有尝试都是已知技术的组合
3. **单 backbone (S2CM) + AnchorLoRA 在 ViT 上失败** — 泛化性不足

### The Hard Truth
| Asset | Status | Paper-Ready? |
|-------|--------|-------------|
| Marathon CIL benchmark | Done | Yes (contribution 1) |
| SHINE baseline | Done | Yes (minor contribution) |
| Drift analysis | Done | Yes (diagnostic tool) |
| AnchorLoRA | Done | Marginal (+2.6pp over SHINE) |
| Latent Replay | Done | Best exemplar-free (80.4%) |
| SART v2 | Done | Best overall (82.6%) but uses replay |
| Beat exemplar methods | **Not achieved** | Gap: 3-4pp |
| Novelty > 6/10 | **Questionable** | 组合创新，非原理创新 |

### Possible Paths Forward
1. **写 benchmark + analysis 论文** (TGRS): Marathon protocol + 30 methods systematic comparison + drift analysis。不 claim 方法突破，claim "first comprehensive cross-domain multimodal CIL study"。
2. **做 LatentReplay + AnchorLoRA 组合**：目前没试过。如果 replay + LoRA 能到 83+%，那就有 clean beat exemplar 的 story。
3. **换方向**：如果这个领域确实没有 meaningful 的 open problem，及时止损。

---

## Appendix: Method-to-File Mapping

| Method | Server Path |
|--------|------------|
| HyperKD baselines | `/root/autodl-tmp/JC/compare/compare_results/` |
| SHINE | `/root/autodl-tmp/results/s2cm/shine_gate_results.json` |
| BRACE/BRIO/COPE | `/root/autodl-tmp/results/s2cm/brace/` |
| TIDE | `/root/autodl-tmp/results/s2cm/tide/` |
| LEID | `/root/autodl-tmp/results/s2cm/leid/` |
| SART v1-v4 | `/root/autodl-tmp/results/s2cm/sart/` |
| AnchorLoRA | `/root/autodl-tmp/results/s2cm/anchor_lora/` |
| DualSpeed | `/root/autodl-tmp/results/s2cm/dualspeed/` |
| SpecRoute | `/root/autodl-tmp/results/s2cm/specroute/` |
| SDBT | `/root/autodl-tmp/results/s2cm/sdbt/` |
| SAFC | `/root/autodl-tmp/results/s2cm/safc/` |
| Latent Replay | `/root/autodl-tmp/results/s2cm/latent_replay/` |
| CIL Baselines | `/root/autodl-tmp/results/s2cm/baselines/` |
| ViT Baselines | `/root/autodl-tmp/results/s2cm/vit_baselines/` |
| Drift Analysis | `/root/autodl-tmp/results/s2cm/drift_analysis/` |
| Single Dataset | `/root/autodl-tmp/results/s2cm/single_dataset/` |
