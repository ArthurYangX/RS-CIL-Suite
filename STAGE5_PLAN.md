# Stage 5: 实验回顾与下一步计划

## Context

经过 4 个月、58 个方法变体、两条研究线（MACIL/RFCIL + S²CM Marathon），本文档从过去所有实验中提取**真正有效的组件**，诊断核心问题，并规划下一阶段实验。

---

## 一、两条研究线的核心发现

### 研究线 1: MACIL/RFCIL（approach/ 正式方法）

**最佳结果**: TAw=83.17%, TAg=51.18%, TAg Forgetting=22.01%（phase2, cap_state_d16_x4_s0）

**确定有效的组件**:
| 组件 | 效果 | 机制 |
|------|------|------|
| **Logits KD (LwF-style)** | 必需，移除后全线崩溃 | 保持旧类 logit 分布 |
| **Cross-modal InfoNCE** | 单独最强 addon，+22 TAg | 跨模态对比学习强化表征 |
| **ssm_expand=4** | 最强架构改动，+14~17 TAg | 容量提升比任何 loss 都重要 |
| **advanced_cosine scheduler** | +2 TAg | 更好的学习率调度 |

**确定无效/有害的组件**:
| 组件 | 效果 | 原因 |
|------|------|------|
| RKD / feature KD / MGD（叠加在 logits+InfoNCE 上）| TAg 暴跌至 36.86% | 过度约束，抑制可塑性 |
| normalize_heads | TAg forgetting +16.69pp | 错误的校准代理 |
| hybrid_proto InfoNCE | 不如 batch InfoNCE | 复杂度增加无收益 |
| virtual replay | 只降 forgetting，不提 TAg | forgetting 控制器，非精度提升器 |
| MGD（任何形式）| 只降 forgetting，从不提 TAg | forgetting 和 TAg 是 tradeoff |
| 更大 embed_dim / 更深 backbone / drop_path | 全部有害 | 小数据过拟合 |

**TAw vs TAg 差距的根因（~32pp gap）**:
1. **无跨头校准**: TAg 直接拼接所有 head logits 做 argmax，新 head 产生更大 magnitude logits → 系统性抢夺旧类预测
2. **Checkpoint 选择错位**: 用 valid_loss（KD/contrastive 混合）选 best model，与 TAg 目标不一致
3. **Old/new competition 是真正瓶颈**: 低 forgetting 不保证高 TAg（相关性仅 -0.366）

### 研究线 2: S²CM Marathon（顶层实验脚本）

**最佳 exemplar-free**: Latent Replay n50 = 80.4%
**最佳 overall**: SART v2 + SHINE = 82.6%（但用了 replay）

**确定有效的组件**:
| 组件 | 效果 | 机制 |
|------|------|------|
| **SHINE (domain whitening)** | +13.5pp | z-score 域归一化，修复一阶 shift |
| **Latent Replay (n=50)** | +7.6pp over SHINE，达 80.4% | 存 pre-LoRA feature maps，训练时回放 |
| **Asymmetric LoRA rank (LiDAR 2x)** | +3.9pp vs symmetric | LiDAR 空间漂移 > HSI 空间漂移 |
| **Spatial-only LoRA (spectral frozen)** | +2.6pp over SHINE | 光谱稳定→冻结，空间→适配 |

**确定无效/有害的组件**:
| 组件 | 效果 | 原因 |
|------|------|------|
| Orthogonal LoRA constraint | -1.0pp | rank-4 太小，约束杀可塑性 |
| Spectral KD | -0.2pp | 光谱已冻结，KD 无目标 |
| Dual heads (DualSpeed) | -3.1pp | 双头校准噪声 > 收益 |
| Spectral routing (SpecRoute) | -3.3pp | 路由误差 > 简单累积 |
| Domain-selective LoRA | -4.6pp | 跨域 LoRA diversity 有益 |
| Drift compensation (SDBT) | ±0pp | SHINE 已解决一阶 drift |
| SAFC 3 组件堆叠 | +1.0pp | 3 模块只换 1pp，负 ROI |
| EWC / LwF | -6.9pp / -4.3pp | 传统 CL 在跨域下完全失败 |
| 所有 10+ post-hoc classifiers | ≤ SHINE | Frozen backbone = information bottleneck |
| LEID / NMF / SRC | -8~17pp | 破坏特征空间结构 |

---

## 二、跨研究线的 5 条通用规律

### Rule 1: 简单 > 复杂
- unconstrained LoRA > orthogonal LoRA, routing, dual heads, drift compensation
- CE + logits KD + InfoNCE > 任何更重的 loss stack
- 每增加一个组件，都需要独立证明 >1pp 增益才值得

### Rule 2: 容量/调度 > Loss 设计
- ssm_expand=4 (+14 TAg) > 任何 loss 改进
- advanced_cosine (+2 TAg) > 任何 KD 变体
- 好的训练策略比花式方法更重要（Phase 0: cosine scheduler 89.9% vs 复杂方法 78-86%）

### Rule 3: Replay 是唯一能打破 75-80% wall 的策略
- Latent Replay n50: 80.4%（不需要 SHINE）
- SART v2 + replay: 82.6%（超过 iCaRL）
- 去掉 replay 后 SART v4: 78.1%（-4.5pp）
- **有效的不是 transport 机制本身，是 replay**

### Rule 4: 训练顺序 > 大多数方法改进
- MTH (78.7%) vs HMT (73.6%)，差 5.1pp
- 这比除 SHINE 和 replay 外的所有方法改进都大

### Rule 5: TAw/TAg gap 的核心是校准问题
- MACIL: TAw=83.17%, TAg=51.18% → ~32pp gap
- S²CM Marathon: 不存在此问题（用 cosine NCM 而非 multi-head）
- **NCM/prototype 分类器天然避免 old/new head competition**

---

## 三、核心问题诊断与已知解法

### Problem 1: Houston Forgetting (-24.6pp)

| 尝试过的解法 | 效果 | 结论 |
|-------------|------|------|
| Orthogonal LoRA | 无效 | rank 太小 |
| Spectral KD | 无效 | 冻结分支无信号 |
| SDBT drift compensation | 无效 | SHINE 已处理一阶 |
| SAFC (EFC + SCRR) | +1pp | 不值得复杂度 |
| Domain-selective LoRA | 有害 (-4.6pp) | 反而限制跨域共享 |
| MGD (MACIL) | 降 forgetting 但降 TAg | Tradeoff，非解决方案 |
| Virtual replay (MACIL) | 降 forgetting 不提 TAg | 同上 |
| **Latent Replay** | **有效 (+7.6pp)** | **唯一真正有效的策略** |
| **SART v2 (exemplar replay)** | **有效 (+9.8pp)** | **但需要存原始特征** |

### Problem 2: TAw vs TAg 不一致（MACIL 线特有）

| 尝试过的解法 | 效果 | 结论 |
|-------------|------|------|
| normalize_heads | catastrophic | 错误的代理 |
| hybrid_proto | 不如 baseline | 复杂无收益 |
| A_visibility proxy | -3.82 TAg | 系统性有害 |
| **cosine NCM (S²CM 线)** | **无此问题** | **避免 multi-head 就解决了** |

### Problem 3: 75-80% Exemplar-Free Wall

| 方法 | Avg TAg | 差距 to iCaRL (82.1%) |
|------|---------|----------------------|
| SHINE only | 72.8% | -9.3pp |
| AnchorLoRA+SHINE | 75.7% | -6.4pp |
| SART v4 (EF) | 78.1% | -4.0pp |
| Latent Replay n50 | 80.4% | -1.7pp |
| **AnchorLoRA + Latent Replay (未测试)** | **~82%?** | **可能 clean beat** |

---

## 四、下一步：AnchorLoRA + Latent Replay 组合实验

### 目标
验证 AnchorLoRA（空间适配 +2.6pp）和 Latent Replay（旧类记忆 +7.6pp）是否可叠加。
- 预期: 82%+ → clean beat iCaRL (82.1%)
- 如果成功，后续可以对 latent 做压缩（PCA/量化/distill）降低存储

### 实现方案
在 `latent_replay_experiment.py` 基础上改（它已经有 LatentStore + AnchorLoRA forward_from_maps 支持）：
- 确认 LoRA 训练阶段同时使用 latent replay loss
- replay 的 feature maps 通过当前 LoRA stack 前向，产生当前空间的旧类特征
- 与新类真实特征混合做 CE + prototype consistency

### 实验矩阵（~2h GPU）

| Run | Method | 预期 |
|-----|--------|------|
| 1 | AnchorLoRA only (baseline) | ~75.7% |
| 2 | Latent Replay n=50 only (baseline) | ~80.4% |
| 3 | **AnchorLoRA + Latent Replay n=50** | **82%+ ?** |
| 4 | AnchorLoRA + Latent Replay n=20 | 看存储/精度 tradeoff |

全部用 THM 顺序, seed=0, warmup=3

### 关键代码修改
- `latent_replay_experiment.py`: 确认 LoRA 训练 loop 中 replay loss 正确接入
- 确保 `forward_from_maps()` 通过当前所有 LoRA（不只是最新的）
- 评估时同时报告 with/without SHINE

### 后续压缩路线（如果组合有效）
1. **PCA 压缩**: 对 pre-LoRA maps 做 Global PCA，n=50 → 可能 <10MB
2. **量化**: fp32 → fp16/int8
3. **Prototype distill**: 用存储的 maps 蒸馏出每类 Gaussian statistics（均值+协方差），变成纯 statistics memory（~32KB）
4. **Herding 选择**: 减少 n（n=50 → n=10~20）看精度下降多少

---

## 五、论文定位（待实验结果确认）

### 如果 AnchorLoRA + Latent Replay ≥ 82%

**Title**: "A Systematic Study of Cross-Domain Multimodal Class-Incremental Learning: What Works, What Doesn't, and Why Replay Matters"

**Contributions**:
1. 首个跨域 HSI+LiDAR CIL benchmark (Marathon protocol + 30+ methods)
2. 系统性发现：简单 > 复杂，容量 > Loss，replay 是唯一破墙策略
3. AnchorLoRA + Latent Replay 组合超越 exemplar-based methods
4. Latent replay 压缩方案（后续工作）

**Target**: TGRS（benchmark+analysis 为主）或 TIP（如果方法新意足够）

### 如果 AnchorLoRA + Latent Replay < 82%

**Title**: "Marathon CIL: A Cross-Domain Multimodal Benchmark Reveals the Limits of Exemplar-Free Continual Learning"

**Contributions**:
1. Benchmark + 30+ methods systematic comparison
2. 负面结果的价值：证明跨域 CIL 的难度
3. "Simple beats complex" 的实证结论
4. 为后续研究提供 baseline 和 protocol

**Target**: TGRS
