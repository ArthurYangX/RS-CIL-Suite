# 文献调研报告：CIL 中的 Backbone 更新策略

**日期：** 2026-03-23
**背景：** 当前 S²CM 项目在 Task 0（仅 2 类）就完全冻结 backbone，后续 30 类只做 inference。这个设定不合理，需要调研更好的 backbone 更新策略。

---

## 一、核心问题

S²CM backbone 有 3 个分支（spectral 64d, HSI-spatial 64d, LiDAR-spatial 64d），在 Trento Task 0 上通过对比学习训练后**完全冻结**。

阶段 1 的实验已经证明：
- 跨域性能崩塌（Houston 从 84% 跌到 44%）
- 10+ 种 post-hoc 方法无法突破 SHINE (z-score whitening) 天花板
- SART (feature-space transport) 有提升 (+5.4pp) 但不够一致（MUUFL 仅 +2.1pp）

**根本原因：** frozen backbone 的表征能力被锁死在极小的训练集上，feature space 的补丁无法弥补 representation 层面的缺失。

---

## 二、方向 1: Parameter-Efficient Fine-Tuning (PEFT) for CIL

在冻结 backbone 的基础上，通过极少量可训练参数实现增量适应。

### 2.1 Prompt Tuning

| 方法 | 会议 | 核心思路 | Exemplar-free |
|------|------|---------|:---:|
| L2P | CVPR 2022 | 冻结 ViT + 可学习 prompt pool，query 机制选择 task 相关 prompt | Yes |
| DualPrompt | ECCV 2022 | G-Prompt (通用共享) + E-Prompt (任务专用) | Yes |
| CODA-Prompt | CVPR 2023 | 注意力加权 prompt 组合，端到端可优化 | Yes |
| S-Prompt | 2022-2024 | 每个 task 独立 prompt，推理时用 K-means 识别 task | Yes |
| HiDe-Prompt | NeurIPS 2023 | 层级分解，不同层分离 task-shared/task-specific 知识 | Yes |
| CPP | 2024 | 动态扩展 prompt pool + 对比学习保证多样性 | Yes |

**局限：** Prompt tuning 主要为 ViT 设计，S²CM 是 Mamba 架构，直接套用不太自然。

### 2.2 LoRA / Adapter

| 方法 | 会议 | 核心思路 | Exemplar-free |
|------|------|---------|:---:|
| **InfLoRA** | **CVPR 2024** | LoRA 在旧 task 特征表示的**零空间**展开，保证无干扰 | **Yes** |
| **O-LoRA** | **ICML 2024** | 正交约束 LoRA，新 task 的 LoRA 正交于旧 task | **Yes** |
| **EASE** | **CVPR 2024** | 每个 task 加一个轻量 adapter，子空间集成推理 | **Yes** |
| LAE | 2024 | 累积 LoRA 模块 + Fisher 加权合并 | Yes |
| MoRAL | 2024 | MoE 架构，每个 expert 是一个 LoRA，router 选择/组合 | Yes |
| ADC | 2024 | 双 adapter：stability adapter (冻结) + plasticity adapter (更新) | Yes |

**关键观察：**
- LoRA/Adapter 是**架构无关的**，可以直接用于 S²CM 的 Mamba 分支
- 正交约束（InfLoRA/O-LoRA）提供理论上的零遗忘保证
- 参数量极小（每个 task 仅增加几千参数），满足 exemplar-free 要求
- 参数量随 task 线性增长，但每个 LoRA 很小

### 2.3 与我们项目的联系

S²CM 的 3 分支结构天然适合 PEFT：
- **光谱分支冻结**（最稳定，Fisher weight 50-80%）→ 作为 anchor
- **空间分支加 LoRA**（漂移最大）→ 每个 task 学习域适应的低秩更新
- 这比 SART 的 feature-space transport 更根本——直接在 representation 层面适应新域

**潜在方案：**
```
光谱分支: 完全冻结 (stable, spectral anchor)
HSI 空间分支: + LoRA_t per task (正交约束)
LiDAR 空间分支: + LoRA_t per task (正交约束)
分类器: branch cosine NCM + SHINE
```

---

## 三、方向 2: Progressive / Selective Freezing

不是一刀切冻结，而是**渐进式**或**选择性**地决定哪些部分冻结、何时冻结。

### 3.1 核心方法

| 方法 | 会议 | 核心思路 | Backbone 更新 |
|------|------|---------|:---:|
| Selective Freezing | ICCVW 2023 | 按层的重要性选择性冻结，平衡稳定-可塑 | 部分层 |
| **PTLF** | **arXiv 2023** | **按层间 task 相关性渐进冻结，浅层先冻（更通用）、深层后冻（更任务特定）** | **渐进减少** |
| **MIST** | **CVPRW 2025** | **稀疏参数更新 (<5%)，互信息引导选择更新哪些参数** | **极少量 (<5%)** |
| FOCIL | arXiv 2024 | 每个 task 随机剪枝出稀疏子网络 → 训练 → 冻结，子网络不重叠 | 不重叠稀疏掩码 |
| CPS | 2022-2023 | 迭代剪枝 + 参数隔离，按重要性分数冻结 | 部分参数 |

**关键发现（MIST, CVPRW 2025）：**
> "Full freezing is suboptimal under distribution shift. Updating even <5% of backbone parameters yields +6-11% gains."

这直接支持了你的直觉——完全冻结不合理。

### 3.2 与我们项目的联系

S²CM 的 3 分支结构可以做**分支级别的选择性冻结**：

| 分支 | 冻结策略 | 依据 |
|------|---------|------|
| 光谱分支 | 始终冻结 | Fisher weight 最高，跨域最稳定 |
| HSI 空间分支 | 渐进冻结 / 稀疏更新 | 中等漂移 |
| LiDAR 空间分支 | 最后冻结 / 更多更新 | 漂移最严重 (≈2× HSI) |

这种**非对称冻结策略**比"全冻结"和"全微调"都更有物理意义：
- 光谱信息（材质签名）跨域不变 → 冻结合理
- 空间信息（纹理/结构）跨域变化大 → 需要适应

**PTLF 的渐进冻结思想**也可以用：随着 task 增加，逐步冻结空间分支的浅层（已学到通用空间模式），只保留深层可塑。

---

## 四、方向 3: Stable-Plastic 双网络架构

明确将网络分为"稳定"和"可塑"两个部分。

### 4.1 核心方法

| 方法 | 会议 | 核心思路 |
|------|------|---------|
| **DualNet** | **NeurIPS 2021** | **慢网络 (SSL, 渐进学习通用表征) + 快网络 (supervised, 任务适应)**，灵感来自互补学习系统 (CLS) 理论 |
| **ANCL** | **CVPR 2023** | **主网络 (stable, 正则化) + 辅助网络 (plastic, 无约束)**，正则化器在两个目标间插值 |
| DLCPA | 2024 | 双学习器：plastic learner 自由学习 + stable learner 累积参数平均 |
| RDBP | arXiv 2025 | ReLUDown + 递减反向传播，浅层梯度渐进衰减（隐式渐进冻结） |
| **DNE** | **CVPR 2023** | **每个 task 加小 expert 网络，通过 dense connection 连接到旧 expert 的中间特征** |

### 4.2 Network Expansion

| 方法 | 会议 | 核心思路 |
|------|------|---------|
| DNE | CVPR 2023 | 旧 expert 冻结 + 新 expert 通过 cross-task attention 密集连接旧 expert |
| EASE | CVPR 2024 | 冻结 backbone + 每 task 加一个 adapter，子空间集成 |
| SEAL | arXiv 2025 | NAS 自动搜索每个 task 该扩展哪些部分、加多少容量 |

### 4.3 与我们项目的联系

**S²CM 的 3 分支结构天然就是一个 stable-plastic 架构：**

```
Stable (光谱分支):          Plastic (空间分支):
- 物理不变性               - 域相关性强
- 跨域最稳定               - 漂移最严重
- Fisher weight 最高        - 需要适应新域
- 作为语义锚点              - 作为学习主力
        ↓                          ↓
  始终冻结，提供             每个 task 可更新，
  旧类记忆的锚               学习新域的空间模式
```

这个 narrative 比 SART 的"feature transport"更有说服力：
- SART 说的是"在 feature space 校正漂移"——本质上是打补丁
- Stable-plastic 说的是"光谱分支天然稳定，空间分支按需适应"——是架构层面的设计

**结合 DualNet 的 CLS 理论框架：**
- 光谱分支 = 新皮层（慢学习，存储结构化知识）
- 空间分支 = 海马体（快学习，适应新经验）
- 这个类比在神经科学上有据可循，审稿人容易接受

---

## 五、方向 4: RS Foundation Model + KD 新进展

### 5.1 遥感基础模型

| 模型 | 会议 | 模态 | HSI 支持 |
|------|------|------|:---:|
| **SpectralGPT** | **TPAMI 2024** | HSI | **Yes (原生)** |
| **DOFA** | **2024** | 多模态 (HSI/SAR/MS) | **Yes** |
| SatMAE | NeurIPS 2022 | 多光谱 | 部分 |
| Scale-MAE | ICCV 2023 | RGB/多光谱 | No |
| SkySense | CVPR 2024 | 光学+SAR+时序 | No |
| RingMo | TGRS 2023 | RGB | No |
| Prithvi | IBM/NASA 2023 | 多光谱 | No |

**关键空白：HSI+LiDAR 联合基础模型不存在。**

### 5.2 Foundation Model + CIL

| 方法 | 会议 | 核心发现 |
|------|------|---------|
| **SimpleCIL** | **2023** | 冻结强预训练 ViT + NCM = 打败大多数复杂 CIL 方法 |
| **RanPAC** | **NeurIPS 2023** | 随机投影 + ridge regression，零遗忘 |
| **ADAM** | **ICLR 2024** | 聚合多个预训练模型，模型多样性 > 抗遗忘技巧 |

**启示：** 如果预训练够强，CIL 问题大幅简化。但 S²CM 只在 2 个类上训练，远不够"强预训练"。

### 5.3 KD 新进展 (2024-2025)

| 方法 | 核心思路 | 与我们的关联 |
|------|---------|-------------|
| DKD (Decoupled KD) | 分解为 target-class KD + non-target-class KD，后者是主要驱动力 | 可改进 SART 的 proto consistency loss |
| Self-Distillation | 模型自己做老师，无需存储旧模型 | 减少存储开销 |
| Cross-Modal KD | 从多模态教师蒸馏到单模态学生 | 可用于光谱→空间的跨模态知识迁移 |
| **KD + PEFT** | **热点：LoRA + KD，冻结模型做 anchor distillation** | **与路线 A 直接结合** |

**关键空白：** RS 多模态 (HSI+LiDAR) 的跨模态 KD + CIL = 零论文。

---

## 六、三个方向如何融合

三个方向不是互斥的，而是可以统一到一个框架中：

```
┌─────────────────────────────────────────────────────────────┐
│              S²CM Backbone (3 branches)                      │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ 光谱分支 🔒   │  │ HSI空间 🔓    │  │ LiDAR空间 🔓  │       │
│  │ (Stable)     │  │ (Plastic)    │  │ (Plastic)    │       │
│  │              │  │ + LoRA_t     │  │ + LoRA_t     │       │
│  │ 完全冻结     │  │ 正交约束      │  │ 正交约束      │       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │
│         │                 │                 │                │
│    方向 3:             方向 1:           方向 1:              │
│    Stable anchor       PEFT (LoRA)      PEFT (LoRA)         │
│                                                              │
│         │                 │                 │                │
│         ▼                 ▼                 ▼                │
│    ┌────────────────────────────────────────────┐            │
│    │     方向 2: Selective Freezing               │            │
│    │  光谱: 始终冻结                               │            │
│    │  空间浅层: 渐进冻结 (PTLF)                    │            │
│    │  空间深层: 保持可塑 (LoRA)                     │            │
│    └────────────────────────────────────────────┘            │
│                          │                                   │
│                          ▼                                   │
│    ┌────────────────────────────────────────────┐            │
│    │     KD: Spectral Anchor Distillation        │            │
│    │  冻结光谱分支 → 约束空间分支 LoRA 更新         │            │
│    │  (你的 KD 经验在这里发挥作用)                  │            │
│    └────────────────────────────────────────────┘            │
│                          │                                   │
│                          ▼                                   │
│              Branch Cosine NCM + SHINE                       │
└─────────────────────────────────────────────────────────────┘
```

### 融合框架的 narrative

> **"光谱-空间非对称持续学习 (Spectral-Spatial Asymmetric Continual Learning)"**
>
> 多模态遥感的光谱信息具有物理不变性（材质签名不因传感器/场景变化），而空间信息高度域相关。因此我们提出非对称持续学习策略：
>
> 1. **Stable spectral anchor** (方向 3): 光谱分支在初始对比学习后完全冻结，作为跨域语义锚
> 2. **Plastic spatial adaptation** (方向 1): 空间分支通过正交 LoRA 逐 task 适应，保证新域学习不干扰旧 task
> 3. **Progressive consolidation** (方向 2): 空间分支的浅层随 task 增加渐进冻结，深层保持可塑
> 4. **Spectral anchor distillation** (KD): 冻结的光谱特征指导空间 LoRA 更新方向，防止空间适应偏离语义

### 与现有工作的差异化

| 现有方法 | 他们做的 | 我们的不同 |
|---------|---------|-----------|
| InfLoRA (CVPR 2024) | ViT 上的正交 LoRA for CIL | 多模态 Mamba + 分支非对称冻结 |
| ANCL (CVPR 2023) | 辅助网络促进可塑性 | 光谱分支是天然的 stable anchor，不是人为设计 |
| PTLF (2023) | 按相关性渐进冻结层 | 按物理意义冻结分支（光谱不变 vs 空间可变） |
| PMPT (TIP 2026) | Prompt tuning for RS CIL | LoRA 更灵活；光谱锚 + 非对称设计是新的 |
| SART (我们之前) | Feature-space transport 补丁 | 从 representation 层面解决，不再打补丁 |

---

## 七、文献空白总结

| 领域交叉 | 现有论文数量 | 机会 |
|---------|:---:|------|
| PEFT + CIL (自然图像) | 大量 (20+) | 饱和 |
| PEFT + CIL (遥感) | 极少 (2-3) | 有空间 |
| **PEFT + CIL (多模态 RS, HSI+LiDAR)** | **0** | **完全空白** |
| Selective freezing + CIL | 少量 (5-8) | 有空间 |
| **分支非对称冻结 + 多模态 CIL** | **0** | **完全空白** |
| Stable-plastic + CIL | 中等 (10+) | 有空间 |
| **物理驱动的 stable-plastic (光谱/空间)** | **0** | **完全空白** |
| RS Foundation Model + CIL | 几乎 0 | 大空白 |
| **Cross-modal KD + PEFT + CIL** | **0** | **完全空白** |

---

## 八、推荐的下一步

1. **精读 3-4 篇核心论文：** InfLoRA (CVPR 2024), ANCL (CVPR 2023), MIST (CVPRW 2025), PTLF (2023)
2. **设计融合方案：** 把 LoRA + 分支选择性冻结 + spectral anchor distillation 统一成一个方法
3. **快速验证：** 在 S²CM 空间分支加最简单的 per-task LoRA，看能否超过 SART v4d
4. **写 story：** 用 CLS 理论 (stable-plastic) + 物理不变性 (spectral anchor) 构建 narrative

---

## 九、参考文献索引

### PEFT for CIL
- [L2P] Wang et al., "Learning to Prompt for Continual Learning," CVPR 2022
- [DualPrompt] Wang et al., "DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning," ECCV 2022
- [CODA-Prompt] Smith et al., "CODA-Prompt: COntinual Decomposed Attention-based Prompting," CVPR 2023
- [HiDe-Prompt] Wang et al., "Hierarchical Decomposition of Prompt-Based Continual Learning," NeurIPS 2023
- [InfLoRA] Liang & Li, "InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning," CVPR 2024
- [O-LoRA] "Orthogonal Low-Rank Adaptation for Continual Learning," ICML 2024
- [EASE] Zhou et al., "Expandable Subspace Ensemble for Pre-Trained Model-Based Class-Incremental Learning," CVPR 2024

### Progressive / Selective Freezing
- [Selective Freezing] Sorrenti et al., "Selective Freezing for Efficient Continual Learning," ICCVW 2023
- [PTLF] Yang et al., "Efficient Self-Supervised Continual Learning with Progressive Task-Correlated Layer Freezing," arXiv 2023
- [MIST] "Beyond Freezing: Sparse Tuning Enhances Plasticity in Continual Learning with Pre-Trained Models," CVPRW 2025
- [FOCIL] Yildirim et al., "Finetune-and-Freeze for Online Class Incremental Learning," arXiv 2024

### Stable-Plastic / Dual Network
- [DualNet] Pham et al., "DualNet: Continual Learning, Fast and Slow," NeurIPS 2021
- [ANCL] Kim et al., "Achieving a Better Stability-Plasticity Trade-off via Auxiliary Networks in Continual Learning," CVPR 2023
- [DLCPA] "Towards Plastic and Stable Exemplar-Free Incremental Learning — A Dual-Learner Framework with Cumulative Parameter Averaging," 2024
- [DNE] Hu et al., "Dense Network Expansion for Class Incremental Learning," CVPR 2023

### RS Foundation Models
- [SpectralGPT] Hong et al., "SpectralGPT: Spectral Remote Sensing Foundation Model," TPAMI 2024
- [DOFA] Xiong et al., "Dynamic One-For-All: Adapting to Multiple RS Modalities," 2024
- [SkySense] Guo et al., "SkySense: Multi-Modal Multi-Temporal RS Foundation Model," CVPR 2024

### Foundation Model + CIL
- [SimpleCIL] Zhou et al., 2023
- [RanPAC] McDonnell et al., "RanPAC: Random Projections and Pre-trained Models for CIL," NeurIPS 2023
- [ADAM] Zhou et al., "Revisiting CIL with Pre-Trained Models: Aggregating Diverse Models," ICLR 2024

### KD for CIL
- [DKD] Zhao et al., "Decoupled Knowledge Distillation," CVPR 2022
- [IJCAI Survey] Zhou et al., "Continual Learning with Pre-Trained Models: A Survey," IJCAI 2024
