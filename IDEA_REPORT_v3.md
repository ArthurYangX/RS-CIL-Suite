# Idea Discovery Report v3: 光谱-空间非对称持续学习

**方向：** 多模态遥感 (HSI+LiDAR) CIL 中融合 PEFT + Selective Freezing + Stable-Plastic 架构
**日期：** 2026-03-23
**Pipeline：** literature-survey → idea-generation (GPT-5.4) → novelty-check (GPT-5.4 × 3)

---

## Executive Summary

在多模态 HSI+LiDAR 类增量学习的 PEFT/选择性冻结/稳定-可塑交叉领域，生成了 9 个 ideas，筛选出 **3 个推荐方向**。经 GPT-5.4 深度 novelty check 确认：三个 ideas 均具有**实质性新颖性**（novelty 7-8/10），核心交叉点无直接先例。

**推荐实施顺序：**
1. **AnchorLoRA** (最简洁，快速验证) → 建立 baseline
2. **DualSpeed-S²CM** (完整论文方法) → 主投稿方案
3. **SpecRoute** (高风险高回报) → 如果前两个成功，扩展为更强贡献

---

## 一、文献全景

### 1.1 确认的空白领域

| 交叉领域 | 现有论文 | 确认方式 |
|---------|:---:|------|
| HSI+LiDAR CIL | **0** | 两轮独立搜索 (arXiv, IEEE, Scholar) |
| PEFT + HSI CIL | **0** | PMPT 最接近但用 prompt 非 LoRA |
| Mamba + LoRA + CIL | **0** | Mamba-CL 和 InfLoRA 各自存在但无交叉 |
| 物理光谱不变性作为 CL anchor | **0** | 所有 "spectral CL" 指矩阵谱非物理光谱 |
| 跨域 marathon CIL | **0** | 评估协议完全新颖 |
| 模态非对称冻结 for CIL | **0** | BranchLoRA 最接近但用于 LLM |

### 1.2 最相关竞争者

| 论文 | 会议 | 关联 | 差异 |
|------|------|------|------|
| **PMPT** | TIP 2026 | 多模态 RS FSCIL + frozen backbone + prompt | 用 prompt 非 LoRA，全冻结非非对称冻结 |
| **InfLoRA** | CVPR 2024 | 正交 LoRA for CIL | 单模态 ViT，无光谱锚 |
| **Mamba-FSCIL** | arXiv 2024 | frozen/dynamic 双分支 Mamba | base/novel 分离非光谱/空间分离 |
| **MSLoRA-CR** | ACM MM 2025 | 模态特定 LoRA for 增量学习 | 所有模态都加 LoRA，无非对称冻结 |
| **BranchLoRA** | ACL 2025 | 非对称 branch tuning-freezing | LLM 指令调优，非 RS |
| **FEICA-CIL** | PR 2025 | 空间-光谱 HSI CIL | 无 PEFT，无多模态 |
| **AMoED** | TCSVT 2026 | RS 增量学习 + domain expert | 无 LoRA，无 HSI+LiDAR |
| **D-MoLE** | ICML 2025 | 多模态 LoRA expert routing | LLM 指令调优，非 RS |

### 1.3 HSI-only CIL 方法（需作为 baseline 对比）

| 论文 | 会议 | 方法 | Backbone 策略 |
|------|------|------|-------------|
| CrossACL | TGRS 2025 | 分析式 feature cross + ridge regression | Frozen |
| FEICA-CIL | PR 2025 | Feature expansion/compression + spectral-spatial aug | Partial FT |
| PSCEN | Expert Syst. 2025 | Prototype similarity enhancement (training-free) | Frozen |
| HSI-CIL | Franklin Inst. 2024 | Analytic linear classifier | Frozen |
| DCPRN | TGRS 2024 | Dual KD + prototype representation | Full FT |

---

## 二、推荐 Ideas 详细说明

---

### 🏆 Idea 1: DualSpeed-S²CM — Stable Spectral, Plastic Spatial Incremental Learning

**Novelty Score: 8/10** (GPT-5.4 verified)

#### Thesis
在多模态 HSI+LiDAR CIL 中，显式利用物理驱动的模态非对称性：光谱分支作为慢学习器 (stable)，空间分支作为快学习器 (plastic)，结合正交 LoRA 和双分类头设计。

#### Method

```
┌─────────────────────────────────────────────────────────┐
│                    S²CM Backbone                         │
│                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ Spectral 🔒  │  │ HSI-Spa 🔓   │  │ LiDAR-Spa 🔓 │     │
│  │ (Slow path) │  │ (Fast path) │  │ (Fast path) │     │
│  │             │  │ + LoRA_t    │  │ + LoRA_t    │     │
│  │ Warm-up     │  │ rank=r_h    │  │ rank=r_l    │     │
│  │ 2-3 tasks   │  │ (smaller)   │  │ (larger)    │     │
│  │ then freeze │  │ Ortho const │  │ Ortho const │     │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │
│         │                │                │              │
│         ▼                ▼                ▼              │
│  ┌─────────────┐  ┌────────────────────────────┐        │
│  │ Stable Head │  │   Plastic Fusion Head      │        │
│  │ (spectral   │  │ (all 3 branches,           │        │
│  │  NCM only)  │  │  cosine NCM)               │        │
│  └──────┬──────┘  └──────────────┬─────────────┘        │
│         │                        │                       │
│         ▼                        ▼                       │
│  ┌──────────────────────────────────────────┐           │
│  │  Calibrated Blending + Spectral→Plastic KD│           │
│  │  (spectral head 约束 plastic head on      │           │
│  │   old classes; plastic head 学习新类)      │           │
│  └──────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────┘
```

**具体步骤：**
1. **Warm-up (Task 0-2):** 光谱分支允许轻微更新（小 lr），空间分支正常训练
2. **Freeze spectral (Task 3+):** 光谱分支完全冻结，维护 per-class 光谱原型
3. **Spatial LoRA:** 每个 task 在 HSI-spatial 和 LiDAR-spatial 的 Mamba block 中插入正交 LoRA
   - LiDAR LoRA rank > HSI LoRA rank（匹配 LiDAR ≈ 2× 漂移）
   - 正交约束（InfLoRA 式零空间展开）保证旧 task 不受干扰
4. **Progressive spatial freezing:** 空间分支浅层随 task 增加渐进冻结（Fisher/drift 稳定后冻结）
5. **Dual head:**
   - Stable head: 仅用光谱特征做 cosine NCM（旧类保持）
   - Plastic head: 用全部 3 分支做 cosine NCM（新类学习）
   - 推理时 calibrated blending
6. **Spectral→Plastic KD:** 冻结光谱分支的 logits/prototype distances 蒸馏到 plastic head，防止旧类漂移

#### Novelty 分析 (GPT-5.4 verified)

**新颖的部分：**
- ✅ 模态定义的 stable/plastic 分离（非 backbone 复制）：无先例
- ✅ 物理光谱不变性驱动冻结策略：无先例
- ✅ 非对称模态冻结速度（spectral→HSI-spa→LiDAR-spa）：无先例
- ✅ Mamba + LoRA + CIL 三者结合：无先例

**已有的部分：**
- ⚠ Fast/slow CL 概念来自 DualNet (NeurIPS 2021)
- ⚠ 正交 LoRA 来自 InfLoRA (CVPR 2024)
- ⚠ Mamba 增量学习来自 Mamba-FSCIL (arXiv 2024)

**3 个最近竞争者及差异：**

| 最近竞争者 | 差异 |
|-----------|------|
| MSLoRA-CR (ACM MM 2025) | 所有模态都加 LoRA，无 stable/plastic 分离，无光谱锚，非 Mamba |
| DualNet (NeurIPS 2021) | Fast/slow 通过 backbone 复制实现，非模态分离；非多模态 RS |
| Mamba-FSCIL (arXiv 2024) | Frozen/dynamic 分离基于 base/novel class，非 spectral/spatial；单模态 |

**审稿人可能的 gotcha 论文：**
- PMPT (TIP 2026): 最近的应用级竞争者
- BranchLoRA (ACL 2025): 非对称 branch 调优概念
- MoDE (NeurIPS 2025): 模态解耦 expert + KD
- FEICA-CIL (PR 2025): 空间-光谱 HSI CIL

#### 风险与缓解

| 风险 | 缓解 |
|------|------|
| 光谱锚 warm-up 不够导致 anchor 太弱 | 用 3 个 task warm-up；对比 1/2/3 task warm-up |
| Dual head calibration 不稳定 | 先跑 single head 版本作为 ablation |
| 正交约束限制后期 task 可塑性 | 监控可用子空间维度；必要时放松约束 |

#### 实施复杂度
- **难度：** Medium
- **新代码量：** ~300 行（LoRA 模块 + 双头 + KD loss + 渐进冻结）
- **GPU 时间估计：** ~2-3h per run (1× RTX 4090)

#### 投稿定位
- **首选：** IEEE TIP / CVPR
- **备选：** IEEE TGRS

---

### 🥈 Idea 2: AnchorLoRA — Spectral-Anchored Orthogonal LoRA for Multimodal CIL

**Novelty Score: 7/10** (GPT-5.4 verified)

#### Thesis
以物理稳定的光谱分支作为冻结锚点，将所有任务特定的可塑性限制在空间分支的正交 LoRA 模块中，LiDAR LoRA rank 高于 HSI（匹配漂移比例）。

#### Method

```
Task 0-2 (Warm-up):
  全部分支训练，光谱分支用小 lr

Task 3+ (Incremental):
  光谱分支: 完全冻结 → spectral prototypes 存储
  HSI-spatial: + LoRA_t (rank=r, 正交约束)
  LiDAR-spatial: + LoRA_t (rank=2r, 正交约束)
  浅层空间: 渐进冻结 (Fisher/drift 稳定后)
  
  Loss = CE_new + λ_kd × L_spectral_KD + λ_proto × L_proto_consistency
  
  Spectral KD: 光谱 head 的 old-class logits → plastic head
  Proto consistency: transported 旧类原型的 pairwise geometry 保持
```

**与 DualSpeed 的区别：**
- AnchorLoRA 是 DualSpeed 的**简化版**：单分类头（融合 head），无 dual-head calibration
- 更容易实现和调试，适合作为第一个实验验证

#### Novelty 分析 (GPT-5.4 verified)

**新颖的部分：**
- ✅ LoRA 仅加在空间分支（非全模态）：无先例
- ✅ 物理驱动的 rank 非对称（LiDAR > HSI）：无先例
- ✅ Mamba + LoRA + CIL：无先例
- ✅ 冻结一个模态分支 + LoRA 另一个模态分支 in CIL：无先例

**已有的部分：**
- ⚠ 正交 LoRA: InfLoRA (CVPR 2024), O-LoRA (ICML 2024)
- ⚠ Rank adaptation: CoDyRA (arXiv 2024)
- ⚠ 多模态 RS FSCIL + PEFT: PMPT (TIP 2026, 用 prompt 非 LoRA)

**3 个最近竞争者及差异：**

| 最近竞争者 | 差异 |
|-----------|------|
| PMPT (TIP 2026) | 用 prompt 非 LoRA；全冻结非非对称冻结；无 rank 非对称 |
| InfLoRA (CVPR 2024) | 单模态 ViT；LoRA 加在所有层；无光谱锚 |
| Mamba-FSCIL (arXiv 2024) | 无 LoRA；frozen/dynamic 是 base/novel 非 spectral/spatial |

**审稿人可能的 gotcha 论文：**
- SD-LoRA (ICLR 2025): 强 LoRA CIL baseline
- CoACT (TMLR 2025): LoRA for foundation model FSCIL
- CoDyRA (arXiv 2024): adaptive rank 削弱 rank 非对称的 novelty
- CPE-CLIP (ICCVW 2023): 多模态 FSCIL + PEFT

#### 实施复杂度
- **难度：** Easy-Medium
- **新代码量：** ~200 行
- **GPU 时间估计：** ~1.5-2h per run

#### 推荐 claim 措辞
> "A physically grounded multimodal CIL framework that uses the stable spectral branch as an anchor and confines plasticity to orthogonal low-rank updates on spatial branches, with modality-asymmetric adaptation strength."

---

### 🥉 Idea 3: SpecRoute — Spectral-Routed Adapter Banks for Task-Free Multimodal CIL

**Novelty Score: 8/10** (GPT-5.4 verified)

#### Thesis
利用光谱不变性驱动 adapter 路由——冻结的光谱分支作为天然的 task/domain 识别器，在推理时无需 task ID 即可选择正确的空间 adapter expert。

#### Method

```
Task 0-2 (Warm-up):
  全部分支训练

Task 3+ (Incremental):
  光谱分支: 完全冻结
  每个 task t:
    HSI-spatial: 新增 LoRA expert E_t^h
    LiDAR-spatial: 新增 LoRA expert E_t^l
    旧 expert: 冻结

  推理时:
    1. 提取光谱特征 z_spec (frozen)
    2. 计算 z_spec 与各 task 光谱 centroid 的相似度
    3. 用相似度作为 softmax 权重，加权组合空间 experts
    4. 或: 用小型 router MLP(z_spec) → expert weights

  训练:
    Loss = CE_new + λ_kd × L_spectral_KD + λ_route × L_route_diversity
```

```
推理流程:
                    ┌──────────────┐
  Input ──────────→ │ Spectral 🔒  │──→ z_spec
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   Router     │
                    │ sim(z_spec,  │
                    │  centroids)  │
                    └──────┬───────┘
                           │ weights: [w1, w2, ..., wT]
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         ┌────────┐  ┌────────┐  ┌────────┐
         │Expert 1│  │Expert 2│  │Expert T│   (spatial LoRA banks)
         └────┬───┘  └────┬───┘  └────┬───┘
              └────────────┼────────────┘
                           ▼
                    Weighted sum → prediction
```

#### Novelty 分析 (GPT-5.4 verified)

**新颖的部分（最强）：**
- ✅ 用物理光谱不变性做 expert 路由：**完全无先例** (strongest novel angle)
- ✅ 一个冻结模态分支路由另一个模态的 adapter：无先例
- ✅ MoE + LoRA + CIL in RS sensor fusion：无先例

**已有的部分：**
- ⚠ Centroid-based task inference: S-Prompts (NeurIPS 2022)
- ⚠ MoE adapter for CL: MoE-Adapters (CVPR 2024)
- ⚠ LoRA expert routing: D-MoLE (ICML 2025)

**3 个最近竞争者及差异：**

| 最近竞争者 | 差异 |
|-----------|------|
| S-Prompts (NeurIPS 2022) | 用通用 frozen feature 做 task 识别，非物理光谱；用 prompt 非 LoRA |
| MoE-Adapters (CVPR 2024) | Distribution-based routing，非 spectral physics；非跨模态路由 |
| D-MoLE (ICML 2025) | Layer-wise instruction routing for LLM，非 spectral-driven spatial routing |

**审稿人可能的 gotcha 论文：**
- AMoED (TCSVT 2026): RS 增量学习 + domain expert（威胁 "first MoE in RS incremental" 的 claim）
- MixtureRS (Remote Sensing 2025): HSI+LiDAR + MoE（非 CIL 但 MoE 概念重叠）
- Towards Cross-domain CIL for RS (TGRS 2024): RS CIL under distribution shift

#### 风险与缓解

| 风险 | 缓解 |
|------|------|
| Task-level spectral centroids 太接近导致路由崩溃 | 用 domain-level 而非 task-level centroids；加 diversity loss |
| Expert 数量线性增长 | 每个 expert 很小 (LoRA ~几K params)；可加 expert merging |
| 比 AnchorLoRA/DualSpeed 复杂很多 | 先验证简单版本（AnchorLoRA）再扩展 |

#### 实施复杂度
- **难度：** Hard
- **新代码量：** ~400 行
- **GPU 时间估计：** ~3-4h per run

#### 投稿定位
- **首选：** CVPR / NeurIPS（如果路由效果好，概念贡献最强）
- **备选：** TIP

---

## 三、被淘汰的 Ideas

| Idea | Score | 淘汰原因 |
|------|:---:|---------|
| Freeze-by-Drift | 512 | 好想法但 drift metric 引入过多超参，且核心 novelty 弱于 top 3 |
| AsymLoRA | 504 | 是 AnchorLoRA 的子集，单独不够构成论文 |
| TrustFuse | 504 | gate 可能退化为 domain ID；与 SpecRoute 竞争但 novelty 更低 |
| ProtoResidual | 448 | 太接近 SART（已有的 feature transport），novelty 不够 |
| OrthoSubspace | 448 | 正交约束过于严格，可能限制后期可塑性 |
| AnchorKD | 441 | 是 AnchorLoRA 的 KD-only 简化版，单独 novelty 不够 |

---

## 四、三个 Ideas 的关系与实施路线

```
AnchorLoRA (简化版, Easy)
    │
    │ 加 dual head + calibration
    ▼
DualSpeed-S²CM (完整版, Medium)      ← 主论文方法
    │
    │ 加 expert bank + spectral routing
    ▼
SpecRoute (扩展版, Hard)              ← 如果效果好，升级贡献
```

**关键洞察：** 三个 ideas 是同一个 narrative 的递进版本——

> 光谱信息具有物理不变性，是多模态遥感持续学习的天然锚点。
> 空间信息域相关性强，需要按需适应。

- AnchorLoRA: 最简实现（光谱冻结 + 空间 LoRA）
- DualSpeed: 完整框架（显式 stable-plastic + dual head + KD）
- SpecRoute: 最强扩展（光谱驱动路由，task-free 推理）

### 实施计划

| 阶段 | 内容 | 预计时间 | 目标 |
|------|------|---------|------|
| **Phase 1** | 实现 AnchorLoRA，在 Marathon CIL 上验证 | 2-3 天 | 确认 LoRA > SART |
| **Phase 2** | 扩展为 DualSpeed-S²CM，加 dual head + KD | 2-3 天 | 确认 dual head 有贡献 |
| **Phase 3** | 实现 SpecRoute routing | 3-4 天 | 确认 spectral routing 有效 |
| **Phase 4** | 消融实验 + multi-seed + ordering rotation | 3-4 天 | 论文所需全部实验 |
| **Phase 5** | 写论文 | 5-7 天 | 投稿 |

### 必须回答的关键实验问题

1. **AnchorLoRA vs SART:** LoRA (representation-level) 是否优于 transport (feature-level)?
2. **Warm-up 长度:** 1/2/3 task warm-up 对光谱锚质量的影响?
3. **LoRA rank 非对称:** r_lidar = 2 × r_hsi 是否最优? sweep rank ratio
4. **Dual head vs single head:** dual head + calibration 是否优于简单 fusion head?
5. **Spectral routing accuracy:** 光谱 centroid 能否准确识别 task/domain?
6. **Progressive freezing:** 哪些空间层应该先冻结? Fisher/drift criterion 有效吗?

---

## 五、投稿策略

### 安全路线 (TIP/TGRS)
- 主方法: DualSpeed-S²CM
- 消融: AnchorLoRA 作为简化版 ablation
- 贡献: (1) Marathon CIL 协议 (2) 物理驱动非对称 stable-plastic 框架 (3) 系统性消融

### 冲刺路线 (CVPR/NeurIPS)
- 主方法: SpecRoute (spectral routing 概念最强)
- DualSpeed 作为不含路由的 ablation
- 需要额外: (1) 更多数据集 (2) 更强 baseline 对比 (3) 路由可视化

### 推荐 claim 措辞
> "We present the first physics-guided stable-plastic framework for multimodal remote sensing class-incremental learning. Leveraging the physical invariance of spectral signatures across acquisition campaigns, we freeze the spectral branch as a semantic anchor and confine task-specific plasticity to orthogonal low-rank adaptations on spatial branches, with modality-asymmetric adaptation strength proportional to observed cross-domain drift."

---

## 六、参考文献

### 核心竞争者
- [InfLoRA] Liang & Li, CVPR 2024 — Interference-Free Low-Rank Adaptation for CIL
- [O-LoRA] ICML 2024 — Orthogonal Low-Rank Adaptation for CL
- [SD-LoRA] ICLR 2025 — Scalable Decoupled LoRA for CIL
- [Mamba-FSCIL] Li et al., arXiv 2024 — Dual Selective SSM for FSCIL
- [Mamba-CL] arXiv 2024 — Learning Mamba as a Continual Learner
- [MSLoRA-CR] Zhang et al., ACM MM 2025 — Modality-Specific LoRA with Contrastive Regularization
- [BranchLoRA] Zhang et al., ACL 2025 — Asymmetric Tuning-Freezing for Multimodal CL
- [ProgLoRA] ACL Findings 2025 — Progressive LoRA for Multimodal Continual Tuning
- [D-MoLE] ICML 2025 — Dynamic Mixture of LoRA Experts for Multimodal CL
- [PMPT] TIP 2026 — Prototype-Based Meta-Prompt Tuning for RS FSCIL
- [FEICA-CIL] Pattern Recognition 2025 — Feature Expansion/Compression for HSI CIL
- [CrossACL] TGRS 2025 — Analytic CL via Feature Cross for HSI
- [AMoED] TCSVT 2026 — RS Incremental Learning with Domain-Specific Experts

### Stable-Plastic / Dual Network
- [DualNet] Pham et al., NeurIPS 2021
- [ANCL] Kim et al., CVPR 2023
- [DLCPA] 2024

### PEFT for CIL
- [EASE] Zhou et al., CVPR 2024
- [CODA-Prompt] Smith et al., CVPR 2023
- [S-Prompts] Wang et al., NeurIPS 2022
- [MoE-Adapters] Yu et al., CVPR 2024
- [CoACT] TMLR 2025 — LoRA for Foundation Model FSCIL
- [CoDyRA] arXiv 2024 — Dynamic Rank-Selective LoRA

### RS Foundation Models
- [SpectralGPT] Hong et al., TPAMI 2024
- [DOFA] Xiong et al., 2024

### HSI CIL
- [HSI-CIL] Franklin Inst. 2024
- [DCPRN] TGRS 2024
- [PSCEN] Expert Syst. 2025
- [CIL-KD] Remote Sensing 2022

### General CIL
- [SimpleCIL] Zhou et al., 2023
- [RanPAC] McDonnell et al., NeurIPS 2023
- [ADAM] Zhou et al., ICLR 2024
