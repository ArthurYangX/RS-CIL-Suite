# 下阶段方案设计：融合 MACL + Prototype Meta-Prompt 的 MOSAIC 升级

## 两篇参考论文的核心思路

### 论文 1: MACL — Modal-Aware Contrastive Learning (Image & Vision Computing, 2025)

**核心创新：把 HSI+LiDAR 的对比学习拆成两个分支**

```
传统做法：HSI+LiDAR → 融合 → 一个统一的对比学习
MACL做法：拆成两个对比学习分支

光谱分支 (Spectral Branch):
  · 以 HSI 的光谱信息为主
  · LiDAR 作为"上下文"辅助
  · 对比学习目标：区分不同材质的光谱签名
  · 关键：LiDAR-grounded spectral — 用 LiDAR 的空间信息引导 HSI 光谱提取

空间分支 (Spatial Branch):
  · 以空间纹理/结构为主
  · HSI 和 LiDAR 都贡献空间信息
  · 对比学习目标：区分不同空间模式
  · 关键：HSI-grounded spatial — 用 HSI 的全局信息引导空间特征
```

**Modal-Aligned Sample Pair Construction**:
- 光谱样本对：同一空间位置的 HSI 光谱向量 + LiDAR 高度值
- 空间样本对：同一区域的 HSI 空间 patch + LiDAR 空间 patch
- 保证两种样本对的数据结构一致

**MAFF (Multimodal Attentional Feature Fusion)**:
- 光谱特征 + 空间特征通过注意力机制融合
- 比简单拼接效果更好

### 论文 2: Prototype-based Meta-Prompt Tuning (IEEE TIP, 2026)

**核心创新：Incremental Prototype Contrastive Loss**

```
问题：增量学习中，旧类原型在嵌入空间中会"漂移"（semantic drift）
      同时，新旧类原型可能"重叠"（prototype overlap）

Incremental Prototype Contrastive Loss 同时解决两个问题：
  1. 减少 semantic drift: 拉近当前特征与旧类原型的距离
  2. 防止 prototype overlap: 推远不同类原型之间的距离

loss = Σ_c [ -log( exp(sim(f, p_c)/τ) / Σ_k exp(sim(f, p_k)/τ) ) ]

其中：
  f = 当前样本特征
  p_c = 该样本所属类的原型
  p_k = 所有已知类的原型（包括旧类存储的原型）
  τ = 温度参数
```

**Meta-Prompt Tuning**:
- 冻结 backbone，只调 prompt tokens
- Prototype 引导 prompt 的更新方向
- Rehearsal-free：不需要存储旧样本

---

## 融合设计：MOSAIC v2

### 核心理念

把 MACL 的 **spectral/spatial 分支** 思想和 Meta-Prompt 的 **Incremental Prototype Contrastive Loss** 融入 MOSAIC 框架。

### 非对称体现在三个层面（不只是超参数）

#### 层面 1: 架构非对称（来自 MACL）

```
不是简单的 HSI/LiDAR 两个分支，而是 Spectral/Spatial 两个分支：

Spectral Branch (稳定):
  · 输入: HSI 光谱向量（LiDAR 引导的空间注意力池化）
  · 编码器: 1D 光谱编码（band 维度处理）
  · 特点: 物理不变性 → 自然稳定 → 适合冻结
  · 增量学习角色: 旧类记忆的锚点

Spatial Branch (可塑):
  · 输入: HSI 空间 patch + LiDAR 空间 patch
  · 编码器: 2D VSSBlock（Mamba 空间扫描）
  · 特点: 空间上下文随新类变化 → 需要适应
  · 增量学习角色: 新类学习的主力
```

**为什么这比"HSI/LiDAR 两分支"更好**：
- HSI 既有光谱信息又有空间信息，强行冻结整个 HSI 分支会限制空间适应
- Spectral/Spatial 分解让我们**只冻结光谱部分，空间部分两个模态都可以适应**

#### 层面 2: 对比学习非对称（来自 MACL + Meta-Prompt）

```
Spectral Contrastive Loss (强约束):
  · 对比对象: 光谱原型
  · 目标: 保持光谱特征的类间区分度
  · 包含旧类原型作为负样本 → 防止 prototype overlap
  · 温度 τ_spec 低（尖锐）→ 强区分

Spatial Contrastive Loss (弱约束):
  · 对比对象: 空间原型
  · 目标: 学习新类的空间模式
  · 旧类原型作为参考但不强制约束
  · 温度 τ_spat 高（柔和）→ 允许调整

Incremental Prototype Contrastive Loss:
  · 当新任务到来时:
    - 拉近新类样本与新类原型
    - 推远新类原型与旧类原型
    - 保持旧类原型位置稳定
  · 这直接解决了我们实验中观察到的"旧类 TAg 崩溃"问题
```

#### 层面 3: 蒸馏非对称（原 MOSAIC）

```
Spectral KD: 强（w_spec ~ 0.7）
  · 教师的光谱特征 → 学生的光谱特征
  · cos_distance, 低温度

Spatial KD: 弱（w_spat ~ 0.3）
  · 教师的空间特征 → 学生的空间特征
  · cos_distance, 高温度

Logit KD: 正常
  · 旧类 logits 蒸馏
```

### 完整方法架构

```
输入: HSI patch (B,C,H,W) + LiDAR patch (B,L,H,W)
        │                        │
  ┌─────┴────────────────────────┴─────┐
  │          LiDAR-Guided Pooling       │
  │   attn = σ(Conv(LiDAR))            │
  │   x_spec = Σ(HSI × attn)  → (B,C) │ ← MACL 的 modal-aligned 思想
  └──────────────┬─────────────────────┘
                 │
  ┌──────────────▼──────────────┐    ┌─────────────────────────────┐
  │    Spectral Branch 🔒冻结    │    │    Spatial Branch 🔓可塑     │
  │                              │    │                             │
  │  1D Conv → GELU → 1D Conv   │    │  HSI: Conv1x1 → 2D空间特征  │
  │  → GAP → Linear             │    │  LiDAR: Conv3x3 → 2D空间特征 │
  │  → spec_feat (B, D_s)       │    │  Cat → VSSBlock×N → GAP     │
  │                              │    │  → spat_feat (B, D_p)       │
  │  Task 0 训练, Task 1+ 冻结   │    │  始终可训练                   │
  └──────────────┬──────────────┘    └──────────────┬──────────────┘
                 │                                   │
  ┌──────────────▼───────────────────────────────────▼──────────────┐
  │                    Fusion + Classification                       │
  │                                                                  │
  │  feat = [spec_feat ; gate ⊙ spat_feat]  → MLP → class logits   │
  │                                                                  │
  │  Incremental Prototype Contrastive Loss:                        │
  │    · 拉近 spec_feat 与旧类光谱原型（保持旧类区分度）                │
  │    · 推远不同类原型之间的距离（防止重叠）                           │
  │    · 空间原型可以适度调整（允许新类适应）                           │
  └─────────────────────────────────────────────────────────────────┘

训练损失:
  L = L_CE                                    # 新类分类
    + λ_kd × L_logit_KD                       # 旧类 logit 蒸馏
    + w_spec × L_spec_KD + w_spat × L_spat_KD # 非对称特征蒸馏
    + λ_ipc × L_incremental_proto_contrastive  # 增量原型对比
```

### 与当前实验的问题对应

| 当前问题 | 原因 | 新方案如何解决 |
|---------|------|-------------|
| 旧类 TAg 崩溃到 0% | 遗忘太严重，没有有效的旧类保持机制 | Incremental Proto Contrastive Loss 直接防止原型漂移和重叠 |
| 可塑参数多→漂移大 | backbone 冻结后仍有大量参数变化 | Spectral branch 完全冻结提供稳定锚，Spatial branch 受控可塑 |
| 非对称只是超参数差异 | w_h/w_d 不够构成创新 | 三层非对称（架构+对比学习+蒸馏），是方法论级别的创新 |
| E-Mamba 太小/太大都不行 | 稳定-可塑平衡点难找 | Spectral/Spatial 分解天然提供了平衡机制 |

### 创新点梳理

1. **Spectral-Spatial 分解的增量学习框架**（来自 MACL 思想，首次用于 CIL）
   - 光谱分支冻结 → 旧类锚
   - 空间分支可塑 → 新类适应

2. **Incremental Prototype Contrastive Loss**（来自 Meta-Prompt 论文，适配到我们的 spectral/spatial 分解）
   - 同时解决 semantic drift 和 prototype overlap
   - 在光谱空间上做强约束，空间空间上做弱约束

3. **三层非对称知识蒸馏**
   - 架构级（冻结/可塑分支）
   - 对比学习级（强/弱温度）
   - 特征蒸馏级（高/低权重）

4. **CIL-E-Mamba backbone**
   - 基于 E-Mamba 的增量学习专用骨干
   - 内置 stable/plastic 分界线

### 目标数字

| 指标 | 当前最好 | 目标 |
|------|---------|------|
| Houston TAg Avg | ~62% (RFCIL) | **85%** |
| Houston Forgetting | ~32% | **<10%** |
| Trento HM-TAg | 87.7% (E-Mamba) | **90%+** |
| MUUFL HM-TAg | 45.7% (HALM) | **80%+** |

### 实现计划

| 组件 | 基于 | 新代码量 |
|------|------|---------|
| Spectral Branch | saga_augment.py (已有) | ~30 行修改 |
| Spatial Branch | CIL-E-Mamba VSSBlock | ~50 行修改 |
| Incremental Proto Contrastive Loss | 新写 | ~60 行 |
| Spectral/Spatial 分解融合 | 新写 | ~40 行 |
| 非对称 KD | mosaic.py (已有) | ~20 行修改 |
| **总新代码** | | **~200 行** |
