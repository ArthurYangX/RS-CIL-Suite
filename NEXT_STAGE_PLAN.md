# Stage 3: Spectral-Anchored Feature Consolidation (SAFC)

**Goal:** Exemplar-free ≥ 82% Avg TAg (beat iCaRL 82.1%, approach LUCIR 83.0%)
**Current:** AnchorLoRA 75.7% | Gap: ~7pp
**Date:** 2026-03-24

---

## 方法概述

在 AnchorLoRA 基础上增加三个组件：

```
SAFC = AnchorLoRA + SCRR + Per-Branch EFC + Analytic Head
```

### 组件 1: Spectral-Conditioned Residual Replay (SCRR)

**解决**: 旧类空间特征消失 → Houston -24.6pp

**核心**:
- 在**特征空间**生成旧类伪特征（不是像素空间 model inversion）
- 利用冻结 spectral branch 作为跨模态生成条件
- 存储: spectral prototype + 空间残差 PCA basis (per class)
- 生成: `f_spatial_synthetic = W_proj × f_spectral_proto + Σ(α_i × basis_i)`
- α_i ~ N(0, λ_i) 其中 λ_i 是对应特征值

**与 HyperSC (TGRS 2025) 的区别**:
| | HyperSC | SCRR |
|---|---------|------|
| 生成空间 | 输入像素 (model inversion, 迭代优化) | **特征空间** (单次前向, 无迭代) |
| 跨模态 | 无 (单模态 spectral stats) | **有** (spectral → spatial 条件生成) |
| 多样性 | 高斯噪声 (各向同性) | **PCA 基底** (各向异性, 保留真实结构) |
| 动机 | spectral 带级统计一致性 | **drift-aware**: spectral 稳定 → 做生成条件 |

**实现步骤**:
1. 每个 task 结束后:
   - 提取当前 task 所有类的 3-branch 特征
   - 存 spectral prototype (均值)
   - 计算空间特征关于 spectral 投影的残差
   - PCA 取 top-k 残差方向作为 basis
2. 训练新 task 时:
   - 对每个旧类，从 spectral proto + 残差 basis 采样 N 个伪特征
   - 伪特征与真实新数据混合训练 LoRA
   - 加 margin loss 保持旧类边界

### 组件 2: Per-Branch Empirical Feature Consolidation (EFC)

**解决**: LoRA 更新破坏旧类特征方向

**核心**:
- 基于 EFC++ (ICLR 2024) 的 Empirical Feature Matrix
- **关键区别**: per-branch 计算，spectral branch 不需要正则化
- 只在 HSI-spatial 和 LiDAR-spatial 上施加 drift 正则化

**实现**:
```python
# 每个 task 结束后
for branch in [hsi_spa, lid_spa]:
    EFM[branch] = compute_empirical_feature_matrix(features[branch], labels)

# 训练新 task 时
loss_efc = 0
for branch in [hsi_spa, lid_spa]:
    drift = features_new[branch] - features_old[branch]  # on synthetic replay data
    loss_efc += (EFM[branch] * drift**2).sum()
```

### 组件 3: Analytic Classifier Head

**解决**: 分类器遗忘

**核心**:
- 替换 cosine NCM 为 recursive ridge regression
- `W = (X^T X + λI)^{-1} X^T Y` 递增更新
- 数学保证: 新类加入不改变旧类权重
- 已有 baseline 实现 (`baselines_experiment.py --method analytic`)

---

## 实验计划

### Phase 3.1: 单独验证每个组件 (~4h GPU)

| 实验 | 命令思路 | 预期 |
|------|---------|------|
| AnchorLoRA + SCRR only | 加残差回放，不加 EFC | +3-5pp (主要提升) |
| AnchorLoRA + EFC only | 加 drift 正则化，不加 SCRR | +1-2pp |
| AnchorLoRA + Analytic | 换分类器 | +0-1pp |
| **AnchorLoRA + SCRR + EFC + Analytic** | **全组合** | **+5-8pp → 81-83%** |

### Phase 3.2: 调优 SCRR 参数 (~2h GPU)

| 参数 | 候选值 |
|------|--------|
| PCA basis k | 3, 5, 10, 20 |
| 每类采样数 N | 20, 50, 100, 200 |
| Replay weight λ_replay | 0.1, 0.3, 0.5, 1.0 |
| Spectral → spatial 投影 | Linear / MLP(64→64) |

### Phase 3.3: 全面对比 (~3h GPU)

- 5 seeds mean±std
- 6 orderings
- vs 所有 baseline (已有数据)
- vs HyperSC (如果能复现)

---

## 代码实现计划

### 新文件: `code/safc_experiment.py`

```
imports from anchor_lora_experiment (data, features, metrics)

class ResidualMemory:
    """Per-class spectral proto + spatial residual PCA basis."""
    def store(class_id, spectral_feats, spatial_feats)
    def sample(class_id, n_samples) → synthetic_spatial_feats

class EmpiricalFeatureMatrix:
    """Per-branch feature importance matrix."""
    def compute(features, labels)
    def drift_loss(old_features, new_features) → loss

class AnalyticHead:
    """Recursive ridge regression classifier."""
    def update(features, labels)
    def predict(features) → logits

def train_safc_task(model, loader, memory, efm, ...):
    """Train LoRA with SCRR + EFC."""
    # Generate synthetic old-class features
    synthetic = memory.sample_all_old_classes(n=50)
    # Mix with real new data
    # Train with: CE + replay_CE + EFC_drift_loss + margin_loss

def run_safc_marathon(net, device, args):
    """Full marathon with SAFC."""
    # Same structure as anchor_lora marathon
    # + ResidualMemory update after each task
    # + EFM computation after each task
    # + Analytic head incremental update
```

**代码量**: ~500 行新代码（基于现有 anchor_lora 架构）
**GPU 时间**: pilot ~2h, full ~8h

---

## 论文叙事更新

从 "AnchorLoRA: spectral freeze + LoRA"
升级为 "SAFC: drift-aware exemplar-free CIL"

```
§1 Intro:
  Observation: spectral features are stable, spatial features drift
  → Design: freeze spectral as anchor, adapt spatial with LoRA
  → Problem: still forgets because spatial features for old classes vanish
  → Solution: use spectral anchor to GENERATE synthetic spatial features for old classes

§3 Method:
  3.1 Marathon CIL Protocol
  3.2 Drift Analysis (LSGA-ViT + S2CM)
  3.3 S2CM Backbone (spectral-spatial separation)
  3.4 AnchorLoRA (freeze + asymmetric LoRA)
  3.5 SCRR (spectral-conditioned residual replay)  ← NEW
  3.6 Per-Branch Feature Consolidation              ← NEW
  3.7 Analytic Classifier                           ← NEW

§4 Experiments:
  Table 1: SAFC vs 9 baselines + exemplar-based → 目标 ≥ 82%
  Table 2: Component ablation (AnchorLoRA → +SCRR → +EFC → +Analytic)
  Table 3: Cross-backbone (S2CM vs ViT)
  Table 4: Ordering robustness
  Table 5: Houston forgetting analysis (before/after SCRR)
```

---

## 风险评估

| 风险 | 概率 | 缓解 |
|------|------|------|
| SCRR 生成的伪特征太差 | 中 | 增加 basis k, 加 diversity loss |
| EFC 过度约束阻碍新类学习 | 低 | asymmetric: spectral 不加, spatial 轻加 |
| Analytic head 不兼容 LoRA 特征 | 低 | 已有 analytic baseline 65.3%, 证明可行 |
| 组合后不如单独 | 低 | 逐步加组件，有消融数据 |
| 仍然打不过 exemplar-based | 中 | 即使 80%+ 也是 strong result (接近) |

---

## 时间线

| 天 | 内容 |
|----|------|
| Day 1 | 实现 SCRR + pilot 测试 |
| Day 2 | 实现 EFC + Analytic + 组合测试 |
| Day 3 | 调优 + 5 seeds + orderings |
| Day 4 | 论文更新 + re-review |
| **Total** | **4 天** |
