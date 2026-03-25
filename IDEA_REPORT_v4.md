# Idea Report v4: Closing the Exemplar Gap

**Goal:** Exemplar-free > Exemplar-based (82-83%) on Marathon CIL
**Current:** AnchorLoRA 75.7% | Gap: ~7pp
**Date:** 2026-03-24

---

## 推荐方案：Spectral-Anchored Feature Consolidation (SAFC)

三个组件组合，各解决一个子问题：

### 组件 1: Spectral-Conditioned Residual Replay (SCRR)

**解决**: 旧类空间特征消失 → Houston -24.6pp 遗忘

**机制**:
- 每个旧类存: spectral prototype (64-d) + 空间残差 PCA basis (k×64-d, k=5)
- 训练新 task 时: `synthetic_spatial = g(spectral_proto) + Σ(α_i × basis_i)` 采样合成特征
- 合成特征和真实新数据混合训练 LoRA

**Novelty**: 7/10
- 最近竞争者: FeTrIL (WACV'23) — 但用几何平移，不是跨模态条件生成
- PASS (CVPR'21) — 但用各向同性高斯噪声，不是低秩基底
- 我们: **跨模态条件生成** (spectral → spatial) + **低秩残差基底** = 无先例

**Reviewer 可能攻击**: "这就是 FeTrIL + RS 包装"
**防御**: FeTrIL 用单模态几何平移，我们用跨模态条件生成 + 结构化残差

### 组件 2: Per-Branch Empirical Feature Consolidation (EFC)

**解决**: LoRA 更新破坏旧类特征方向

**机制**:
- 每个 task 后，计算 per-branch Empirical Feature Matrix (EFM)
- EFM 衡量 "哪些特征方向对旧类重要"
- 训练新 LoRA 时: `L_efc = Σ EFM_b ⊙ (f_b^new - f_b^old)²` 只惩罚重要方向

**Novelty**: 6/10
- 最近竞争者: EFC++ (ICLR 2024) — 但单模态，不 per-branch
- GPM (ICLR'21), Adam-NSCL (CVPR'21) — 但用存储的激活值，不是 prototype
- 我们: **per-branch asymmetric drift 正则化** + **spectral anchor 驱动** = 有区分度

**Reviewer 可能攻击**: "这是 EFC++ 改成 per-branch"
**防御**: per-branch 不是 trivial — spectral branch 不需要正则化，只需正则化 spatial

### 组件 3: Analytic Classifier Head

**解决**: 分类器遗忘

**机制**:
- 替换 NCM 为 recursive ridge regression (REAL/DS-AL 风格)
- `W = (X^T X + λI)^{-1} X^T Y` 闭式解
- 数学保证: 新类加入不改变旧类权重

**Novelty**: 3/10 (本身不新，但在我们系统里是新组件)
- 最近竞争者: REAL (2024), DS-AL (AAAI 2024), CrossACL (TGRS 2025)
- 我们已经有 analytic baseline (65.3%)，需要和 SCRR + EFC 结合才有意义

---

## 存储开销

| 组件 | Per-class | 32 classes total |
|------|-----------|-----------------|
| Spectral prototype | 64 floats = 256B | 8KB |
| 残差 PCA basis (k=5) | 5×64 = 1.3KB | 41KB |
| EFM matrix (shared) | 64×64×3 branches = 49KB | 49KB (shared) |
| Analytic W matrix | 192×32 = 24KB | 24KB (shared) |
| **Total** | | **~122KB** |

vs iCaRL: 5 samples/class × 32 classes × ~1KB/sample = **~160KB** (comparable!)

---

## 预期效果分析

| 组件 | 预期提升 | 依据 |
|------|---------|------|
| SCRR (残差回放) | +4-6pp | FeTrIL 在标准 CIL 上 +5-8pp over frozen |
| EFC (drift 正则化) | +1-2pp | EFC++ 在 CIFAR-100 上 +2-3pp over no reg |
| Analytic head | +0-1pp | 已有数据: analytic 65.3% ≈ frozen 59.2% + 6pp (raw) |
| **Combined** | **+5-9pp** | **75.7 + 7 = 82.7%** (target range) |

---

## 实现计划

| 阶段 | 内容 | 代码量 | GPU 时间 |
|------|------|--------|---------|
| 1 | SCRR: 残差存储 + 采样 + 回放训练 | ~200 行 | 2h pilot |
| 2 | EFC: per-branch EFM 计算 + drift loss | ~150 行 | 1h pilot |
| 3 | Analytic head: 替换 NCM | ~50 行 | 30min pilot |
| 4 | 组合 + 调优 | ~100 行 | 4h full |
| **Total** | | **~500 行** | **~8h** |

---

## 论文叙事 (representation drift narrative)

> **Observation (drift analysis):** Spectral features are relatively stable across domains (drift 1×), while spatial features drift significantly (2.3-2.5×).
>
> **Design Principle:** Freeze the stable spectral branch → use it as (1) semantic anchor for feature generation, (2) conditioning signal for normalization, (3) drift measurement baseline.
>
> **Method (SAFC):**
> 1. Spectral-Conditioned Residual Replay: spectral anchors generate synthetic spatial features for old classes
> 2. Per-Branch Feature Consolidation: protect old-class-sensitive spatial directions proportional to measured drift
> 3. Analytic Classifier: mathematically guaranteed zero-forgetting at the classification layer
>
> **Result:** First exemplar-free method to match exemplar-based performance on cross-domain multimodal CIL.

这个叙事从 drift observation → method design → result，逻辑链完整。
