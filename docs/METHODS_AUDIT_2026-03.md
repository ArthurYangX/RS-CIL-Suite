# JC 方法实现梳理与实验审计（2026-03）

## 1. 阅读说明

本文只覆盖当前主代码树 `/root/autodl-tmp/jc`，不纳入早期分支 `hyperkd/`。目标不是罗列“理论上存在过什么想法”，而是梳理仓库里**确实有实现**的方法，并给出它们在当前仓库中的证据强度、实验效果和主要问题。

本文把条目分成三类：

- `正式主线方法`：`train_hypermamba.py` 可直接接入的 `approach/*.py`。
- `实验方法/专项试验`：顶层 `*_experiment.py`、`*_pilot.py`、`*_eval.py` 中实现的方法族，通常带自己的训练/评测流程。
- `历史/过渡变体`：保留在仓库中、但已经明显被后续版本替代或只用于局部验证的实现。

证据等级：

- `A`：有系统分析报告支撑，主要来自 `outputs/plan_2026-03-13/phase2-5/*_analysis_report.md`、`ARCH_SUMMARY_2026-03-13.md`、`docs/RFCIL_METHOD.md`。
- `B`：有结果日志或可追溯数值，但缺少系统报告，主要来自 `results/*.log`。
- `C`：只有脚本内注释、打印口径、对比流程，没有可直接引用的稳定结果。
- `D`：能确认代码实现，但没有找到可靠实验记录。

状态定义：

- `主线候选`：当前代码中最值得继续迭代的方向。
- `可参考`：有方法价值，但还不是当前最强主线。
- `已弱化`：结论不稳定、效果一般或已被别的方法覆盖。
- `历史残留`：主要用于追溯或对照，不建议继续作为主线。

## 2. 总览

### 2.1 正式主线方法总表

| 方法 | 代码位置 | 主要模型/入口 | 证据 | 当前判断 |
| --- | --- | --- | --- | --- |
| `our` | `approach/our.py` | 旧式 KD + exemplars | D | 历史残留 |
| `jcmethod` | `approach/jcmethod.py` | logits KD + 可选 feature KD / NCM / prototype IPCL | B | 可参考 |
| `macil` | `approach/macil.py` | MACILNet / `train_hypermamba.py` 默认主线之一 | A | 主线候选 |
| `linkedmem` | `approach/linkedmem.py` | HALM-CIL linked memory | D | 已弱化 |
| `mosaic` | `approach/mosaic.py` | asymmetric KD + proto reconstruction | D | 已弱化 |
| `agora` | `approach/agora.py` | spectral anchor + virtual replay + proto recon | B | 可参考 |
| `rfcil` | `approach/rfcil.py` | RFCILNet rehearsal-free 主线 | A/B | 主线候选 |
| `rfcil_matryoshka` | `approach/rfcil_matryoshka.py` | RFCIL + auxiliary core head | D | 已弱化 |
| `rfcil_postmat` | `approach/rfcil_postmat.py` | RFCIL + post-fusion matryoshka | D | 已弱化 |
| `rfcil_backup_pre_mosaic` | `approach/rfcil_backup_pre_mosaic.py` | RFCIL 早期备份 | D | 历史残留 |
| `s2cm` | `approach/s2cm.py` | S²CMNet + MASGC classifier | C | 可参考 |

### 2.2 实验方法/专项试验总表

| 方法族 | 代码位置 | 核心方向 | 证据 | 当前判断 |
| --- | --- | --- | --- | --- |
| `AnchorLoRA` | `anchor_lora_experiment.py` | 冻结谱支 + 任务 LoRA + SHINE/NCM | C | 可参考 |
| `DualSpeed-S²CM` | `dualspeed_experiment.py` | 双头稳定/可塑分离 + progressive freezing | C | 可参考 |
| `SpecRoute` | `specroute_experiment.py` | 基于谱特征的 LoRA expert routing | C | 可参考 |
| `SDBT` | `drift_compensate_experiment.py` | spectral drift basis transport | C | 可参考 |
| `Latent Replay` | `latent_replay_experiment.py` | frozen pre-LoRA latent replay | D | 已弱化 |
| `SAFC` | `safc_experiment.py` | residual replay + per-branch consolidation | D | 已弱化 |
| `TIDE` | `tide_experiment.py` | SHINE + tangent-deformable capsule prototypes | C | 可参考 |
| `BRACE` | `brace_experiment.py` | SHINE + BRIO + COPE | C | 可参考 |
| `SART v2` | `sart_experiment.py` | spectral-anchored residual transport | C | 已弱化 |
| `SART v4b` | `sart_v4_experiment.py` | exemplar-free SART 改进版 | C | 可参考 |
| `LEID` | `leid_experiment.py` | learned decomposition + Gaussian replay | C | 可参考 |
| `LEID-Analytical` | `leid_analytical.py` | training-free decomposition baselines | C | 可参考 |
| `Baseline Comparison` | `baseline_comparison.py` | Frozen-NCM / SHINE / iCaRL / Replay-Herding | C | 评测脚本 |
| `Baselines Experiment` | `baselines_experiment.py` | EWC / LwF 等 marathon baseline | C | 评测脚本 |
| `ViT Baselines` | `vit_baselines_experiment.py` | ViT track baselines + L2P | C | 评测脚本 |
| `SHINE Gate` | `shine_gate.py` | post-hoc domain whitening | C | 评测/后处理 |
| `Bilateral KD Pilot` | `bilateral_kd_pilot.py` | `M_0 + M_prev` feature KD | C | 小范围 pilot |
| `Single-dataset CIL` | `single_dataset_cil.py` | 单数据集 Baseline / SHINE / SART | C | 协议脚本 |
| `Marathon CIL` | `marathon_cil.py` | 9-task marathon baseline protocol | C | 协议脚本 |
| `Posthoc Eval` | `posthoc_eval.py` | S²CM/MASGC 后验参数扫描 | C | 后处理工具 |

### 2.3 不计入“方法”的文件

以下文件更像工具、数据检查、对比脚本或运维脚本，不单列为方法：

- `check_keys.py`
- `check_keys_more.py`
- `check_muufl_keys.py`
- `detect_outliers.py`
- `drift_analysis.py`
- `fewshot_test.py`
- `get_dataset_counts.py`
- `grid_search.py`
- `compare_mmcs.py`
- `sync_wandb.py`
- `test_HSILiDAR.py`
- `tide_gamma_sweep.py`
- `train_life_long.py`

## 3. 正式主线方法

### 3.1 `our`

- 类别：正式实现中的旧方法
- 代码位置：`approach/our.py`
- 核心机制：
  - 老模型蒸馏到当前模型。
  - 支持 exemplar memory。
  - 训练时使用当前 head CE + 旧头 KD。
- 实验效果：
  - 没有找到当前仓库里针对 `our` 的系统报告或专门日志。
  - 仅能确认它是一条早期 KD + exemplar 风格实现。
- 主要问题：
  - 训练循环仍按 `for images, targets in loader` 写法实现，而当前主数据流是 `(x_hsi, x_lidar, y)` 三元组，接口明显落后于当前多模态主线。
  - 没有现成多模态辅助手段，也没有新的文档或测试覆盖。
  - 更像从旧版单输入增量学习模板遗留而来。
- 状态判定：`历史残留`

### 3.2 `jcmethod`

- 类别：正式主线方法
- 代码位置：`approach/jcmethod.py`
- 核心机制：
  - 以 logits KD 为主干。
  - 可选 `feature KD`、`RKD`、`NCM classifier`。
  - 可选 `sample_prototype_ipcl` 和 `sample_sample_proto_ipcl`，并复用 RFCIL 中的原型对比实现。
- 实验效果：
  - 有单元测试覆盖 NCM 和两类 IPCL 注入逻辑：`tests/test_jcmethod_ncm_ipcl.py`。
  - 仓库中存在 `compare_mmcs.py` 等针对 `jcmethod` 的比较脚本，但没有像 MACIL 那样的 phase2-4 系统报告。
  - 因此只能判断其为“持续可运行、可扩展的中间主线”，没有足够证据证明它是当前最优。
- 主要问题：
  - 结果证据主要停留在日志和周边脚本层，缺少统一分析报告。
  - 方法本体较依赖旧式 KD 框架，后续新机制很多已经迁入 RFCIL/MACIL。
  - 在仓库结构上已经成为“被后继方法吸收部分思想”的基线，而不是当前实验主轴。
- 状态判定：`可参考`

### 3.3 `macil`

- 类别：正式主线方法
- 代码位置：`approach/macil.py`
- 核心机制：
  - 当前可运行 MACIL 主线。
  - `CE_new`。
  - 跨模态监督对比 `InfoNCE`。
  - `CMA-MGD` 生成式重建蒸馏。
  - `feature KD`、`RKD`。
  - `prototype virtual replay`。
  - 支持单头、router、fusion ablation、pretrain protocol。
- 实验效果：
  - 这是当前仓库中实验记录最完整的主线之一。
  - `ARCH_SUMMARY_2026-03-13.md` 给出的较优单次结果：
    - `TAw Avg = 80.19%`
    - `TAg Avg = 48.64%`
    - `TAw Forget Avg = 16.91%`
    - `TAg Forget Avg = 6.70%`
  - `phase2_analysis_report.md` 的核心结论：
    - 当前最强稳定配方仍是 `CE + logits KD + InfoNCE`。
    - phase2 最优 seed-0 点是容量调整，而不是更重的蒸馏栈：
      - `cap_state_d16_x4_s0 = 83.17 / 51.18 / 12.00 / 22.01`
    - 纯 `logits + InfoNCE` 网格里较优点：
      - `kd_logits_infonce_a0.1_g1.0_T2.0_s0 = 83.25 / 50.03 / 13.84 / 26.05`
    - `RKD / feature KD / MGD` 多数情况下不能继续提升 `TAg`，甚至会抑制可塑性。
  - `phase3_analysis_report.md`：
    - `expand=4` 依旧是强正向因素。
    - `advanced_cosine` 比普通 `cosine` 更利于 `TAg`。
    - 修复后的 `hybrid_proto` 可以训练，但仍不如最佳 baseline。
    - `MGD` 更像“降遗忘正则”，不是“提精度主力”。
  - `phase4_analysis_report.md`：
    - phase4 没有产生新主线，控制线仍然最好：
      - `p4_base_main_advcos_s0 = 81.26 / 48.44 / 38.31`
    - `normalize_heads` 不是好的校准方案。
    - 当前瓶颈不在轻量 prototype visibility proxy，也不在轻量 head normalization。
- 主要问题：
  - `valid_loss` 仍被用于 best checkpoint 选择，但 `ARCH_SUMMARY` 已明确指出这与最终 `TAg` 目标不对齐。
  - `TAg` 校准仍偏弱，后期任务 old/new competition 明显。
  - 重蒸馏栈不稳定，`best_stack` 复现实验不鲁棒：
    - `rep_best_stack: TAw=81.74 ± 1.34, TAg=32.27 ± 4.89`
  - 很多增强项只是在降低遗忘和提升精度之间做交换，没有真正推高主线前沿。
- 状态判定：`主线候选`

### 3.4 `linkedmem`

- 类别：正式主线方法
- 代码位置：`approach/linkedmem.py`
- 核心机制：
  - HALM-CIL：存储 HSI / LiDAR / fused prototype。
  - 通过 ridge mapping 做 `HSI -> LiDAR` linked memory。
  - 估计模态漂移并做 analytic prototype classification。
  - 可选 spectral anchor 拼到 HSI 特征前缀。
- 实验效果：
  - 没有发现专门日志或 phase 报告。
  - 代码本身较完整，说明其不是半成品，但缺少可靠对外实验结论。
- 主要问题：
  - 没有标准化结果输出或复现实验记录。
  - 方法很依赖 backbone 能返回统一 aux 结构，兼容层较多，维护成本高。
  - 已被 RFCIL / MACIL 主线遮蔽，没有后续实验推进证据。
- 状态判定：`已弱化`

### 3.5 `mosaic`

- 类别：正式主线方法
- 代码位置：`approach/mosaic.py`
- 核心机制：
  - asymmetric KD：按 online drift 估计给 HSI / LiDAR 不同 KD 权重。
  - logits KD + feature KD。
  - per-class prototype reconstruction。
  - 可对 backbone 做冻结策略切换。
- 实验效果：
  - 没有单独的系统报告。
  - 但 RFCIL 中已经内嵌 `mosaic_asymmetric_kd`、`mosaic_lidar_recon_discount` 等开关，说明 MOSAIC 的有效想法已被吸收入 RFCIL 主线做可选部件，而不是独立主线继续推进。
- 主要问题：
  - 独立版本缺少完整结果闭环。
  - 思想已局部并入 RFCIL，使得单独维护其价值下降。
  - 没有专门测试覆盖它的独立训练逻辑。
- 状态判定：`已弱化`

### 3.6 `agora`

- 类别：正式主线方法
- 代码位置：`approach/agora.py`
- 核心机制：
  - 光谱支作为稳定 anchor。
  - 虚拟回放 + 原型重构。
  - `IPCL` 改到 classifier space。
  - proto recon 后做 weight alignment。
  - 支持无 gate、无 anchor、full-freeze 等 ablation。
- 实验效果：
  - 有较多日志，但没有 phase2-4 那样的统一系统报告，证据等级以 `B` 为主。
  - 从 `results/agora_*.log` 看：
    - `agora_ff_full_s0` 最终平衡较好，`HM-TAg=47.56%`，`Avg-Forg(TAg)=23.60%`
    - `agora_best_s17.log` 和 `agora_best_s42.log` 的最终 `HM-TAg` 分别约为 `44.32%` 和 `43.55%`
    - `agora_v6_*` 一系列配置普遍出现明显的任务无关遗忘，`Avg-Forg(TAg)` 常在 `30%` 到 `60%+`
    - 某些配置在最后一个任务上 `TAg` 很高，但前面任务竞争严重，说明全局校准不稳定
  - 结论上看，AGORA 有一定可行性，但没有形成比 MACIL phase2 最优结果更强、更稳定的证据链。
- 主要问题：
  - seed 波动与配置敏感性较强。
  - 旧任务与新任务头之间的校准问题依然显著。
  - 日志里很多实验看起来像“大量手调版本”，但缺少像 phase2-4 那样的结构化对照。
  - 部分数据集日志呈现训练集拟合很高、验证或跨任务 `TAg` 很低的现象。
- 状态判定：`可参考`

### 3.7 `rfcil`

- 类别：正式主线方法
- 代码位置：`approach/rfcil.py`
- 相关网络：`networks/rfcil_net.py`
- 核心机制：
  - LiDAR-guided shared/private 表征。
  - shared-only class-prototype InfoNCE。
  - separation loss。
  - old-head logits KD。
  - shared feature KD。
  - optional gate KD。
  - rehearsal-free vector-only prototypes。
  - 支持单头增长、core/post-core 变体、mosaic-style asymmetric shared KD、shortcut ablation。
- 实验效果：
  - `docs/RFCIL_METHOD.md` 给出了方法到代码的映射和最小 ablation 命令，是 RFCIL 的主要规范文档。
  - `results/rfcil_*.log` 表明至少做过多轮实跑：
    - `rfcil_emamba_s0.log` 中期汇总可见 `avg_TAw=64.0%, avg_TAg=42.9%`，基础表现尚可。
    - `rfcil_own_s0.log` 末段约为 `avg_TAw=72.5%, avg_TAg=39.8%`
    - `rfcil_own_v2_s0.log` 末段约为 `avg_TAw=73.0%, avg_TAg=40.3%`
  - 当前 RFCIL 没有像 MACIL phase2-4 那样的大规模分析报告，但方法文档、日志和测试都比较完整。
  - 测试覆盖是当前仓库里最强的一组：
    - `test_rfcil_cross_kd.py`
    - `test_rfcil_fused_kd.py`
    - `test_rfcil_infonce_unittest.py`
    - `test_rfcil_matryoshka.py`
    - `test_rfcil_ncm_classifier.py`
    - `test_rfcil_proto_virtual.py`
    - `test_rfcil_shortcut.py`
    - `test_rfcil_weight_align.py`
- 主要问题：
  - 从已存日志看，`TAg` 仍明显落后于 MACIL phase2 最优点。
  - 虽然测试充分，但“大规模稳定实验结论”不足，更多还是可审计的工程化实现。
  - 变体和开关很多，复杂度高，存在过度配置化风险。
- 状态判定：`主线候选`

### 3.8 `rfcil_matryoshka`

- 类别：正式主线变体
- 代码位置：`approach/rfcil_matryoshka.py`
- 核心机制：
  - 在 RFCIL 上增加低维 `z_core` 辅助分支和辅助分类头。
  - 主推理仍走 full head，core head 主要承担压缩和稳定性约束。
- 实验效果：
  - 有单元测试，能确认 core head 添加、虚拟回放范围等逻辑。
  - 但没有看到成体系的实验报告或日志归档。
- 主要问题：
  - 需要 `--single_head` 且要求 `--rfcil_core_dim > 0`，使用条件更苛刻。
  - 没有证据证明它明显优于 base RFCIL。
  - 代码复杂度增加明显。
- 状态判定：`已弱化`

### 3.9 `rfcil_postmat`

- 类别：正式主线变体
- 代码位置：`approach/rfcil_postmat.py`
- 核心机制：
  - 在 post-fusion 后做两尺度 Matryoshka-lite。
  - `z_post_full` 用于主推理，`z_post_core` 用于辅助压缩/蒸馏。
- 实验效果：
  - 有完整代码实现。
  - 没有找到专门结果报告。
- 主要问题：
  - 需要 `--single_head`、`--rfcil_post_matyoshka` 和正的 `--rfcil_post_core_dim`。
  - 与 `rfcil_matryoshka` 一样，复杂度上升快，但缺少实证收益。
- 状态判定：`已弱化`

### 3.10 `rfcil_backup_pre_mosaic`

- 类别：历史/过渡变体
- 代码位置：`approach/rfcil_backup_pre_mosaic.py`
- 核心机制：
  - 这是 RFCIL 在吸收 MOSAIC 相关改造前的整份备份。
  - 主结构和当前 RFCIL 高度相似，但缺少后续合并点。
- 实验效果：
  - 未发现专门结果。
- 主要问题：
  - 明显属于历史快照，不应继续并行维护。
  - 保留价值主要在于追溯改动。
- 状态判定：`历史残留`

### 3.11 `s2cm`

- 类别：正式主线方法
- 代码位置：`approach/s2cm.py`
- 相关网络：`networks/s2cm_net.py`
- 核心机制：
  - Task 0 用 `CE + spatial contrastive + spectral SupCon`。
  - 增量阶段冻结 backbone。
  - 存储 per-class Gaussian capsules。
  - 推理时使用 `MASGC` 或 `cosine_ncm`。
- 实验效果：
  - `marathon_cil.py`、`single_dataset_cil.py`、`posthoc_eval.py`、多种顶层实验都在复用 S²CM/S2CMNet，因此它是整个顶层实验生态的“底座”。
  - 但作为独立 `approach/s2cm.py` 的系统结果没有被统一汇总到 phase2-4 报告。
  - 顶层 `tide_experiment.py` 中将 SHINE 标记为“validated, +24% Houston”，说明 S²CM 轨道上至少产生过较强的 domain normalization 观察，但这更属于 SHINE/TIDE 系列，而非 `approach/s2cm.py` 单独贡献。
- 主要问题：
  - 作为主线 `approach` 的实验闭环不完整。
  - 很多更激进的新想法没有继续在纯 S²CM 分支上推进，而是在实验脚本层各自分叉。
- 状态判定：`可参考`

## 4. 实验方法与专项试验

### 4.1 AnchorLoRA

- 代码位置：`anchor_lora_experiment.py`
- 核心机制：
  - 冻结谱支作为 stable anchor。
  - 空间分支挂 task-wise LoRA。
  - HSI/LiDAR 采用不对称 rank。
  - 使用 SHINE + cosine NCM 做预测。
- 实验效果：
  - 脚本提供 marathon 模式与对比打印，包含 `AnchorLoRA+SHINE vs SHINE` 的差值输出。
  - 但仓库内没有现成归档结果或系统分析。
- 主要问题：
  - 代码体量很大，和 SART/SpecRoute/DualSpeed 共享大量复制逻辑。
  - 结果管理没有并入主仓库统一实验表。
- 状态判定：`可参考`

### 4.2 DualSpeed-S²CM

- 代码位置：`dualspeed_experiment.py`
- 核心机制：
  - 稳定谱头 + 可塑融合头双头推理。
  - progressive spatial freezing。
  - spectral-to-plastic KD。
- 实验效果：
  - 只有脚本级实现，没有统一结论。
- 主要问题：
  - 与 AnchorLoRA 高度耦合。
  - 没有证据表明双头校准已解决 old/new competition。
- 状态判定：`可参考`

### 4.3 SpecRoute

- 代码位置：`specroute_experiment.py`
- 核心机制：
  - 用谱特征做 soft routing。
  - 为每个任务维护 LoRA experts。
  - 目标是 task-free inference。
- 实验效果：
  - 只有代码和命令入口，未见可靠结果归档。
- 主要问题：
  - router 机制增加系统复杂度，但当前仓库中没有证据证明它相对简单主线更有效。
- 状态判定：`可参考`

### 4.4 SDBT

- 代码位置：`drift_compensate_experiment.py`
- 核心机制：
  - 观测 LoRA 更新引入的空间漂移。
  - 用 frozen spectral feature 预测漂移系数并在推理期补偿。
- 实验效果：
  - 代码里描述清楚，属于明确的 zero-storage drift compensation 想法。
  - 没有系统结果记录。
- 主要问题：
  - 训练/推理链路额外维护 drift basis，复杂度不低。
  - 未见与现有主线的公平对比结果。
- 状态判定：`可参考`

### 4.5 Latent Replay

- 代码位置：`latent_replay_experiment.py`
- 核心机制：
  - 在 pre-LoRA 冻结空间做 latent replay，避免 raw exemplar。
- 实验效果：
  - 只确认有实现，缺少归档结果。
- 主要问题：
  - 后续仓库重心已转向 prototype / vector-only replay 和 RFCIL/MACIL 系列。
- 状态判定：`已弱化`

### 4.6 SAFC

- 代码位置：`safc_experiment.py`
- 核心机制：
  - spectral-anchored residual replay。
  - per-branch empirical feature consolidation。
- 实验效果：
  - 没有可引用实验记录。
- 主要问题：
  - 被后续更多结构化方案覆盖，没有形成持续推进的实验线。
- 状态判定：`已弱化`

### 4.7 TIDE

- 代码位置：`tide_experiment.py`
- 核心机制：
  - SHINE domain canonicalization。
  - tangent-deformable capsule prototypes。
- 实验效果：
  - 脚本注释直接写明 `SHINE — Domain Canonicalization (validated, +24% Houston)`。
  - 这是仓库里少数对 SHINE 效果有明确正向描述的地方，但仍然属于脚本级证据，不是统一报告。
- 主要问题：
  - “+24% Houston” 没有在统一分析文档里复核。
  - TIDE 本体是否稳定优于更简单的 SHINE baseline，没有当前仓库内统一结论。
- 状态判定：`可参考`

### 4.8 BRACE

- 代码位置：`brace_experiment.py`
- 核心机制：
  - SHINE 域白化。
  - BRIO query-adaptive branch weighting。
  - COPE ambiguity pairwise experts。
- 实验效果：
  - 脚本自带 baseline / SHINE / BRACE 多模式比较。
  - 说明做过细粒度消融，但未见结构化结果归档。
- 主要问题：
  - 仍停留在脚本级复现实验，缺少主线化。
  - 复杂模块较多，但没有统一对照证明它优于简单方案。
- 状态判定：`可参考`

### 4.9 SART v2

- 代码位置：`sart_experiment.py`
- 核心机制：
  - spectral-anchored residual transport。
  - 对跨域任务训练 transport module。
  - 与 SHINE 组合使用。
- 实验效果：
  - 脚本会打印 `SART vs SHINE` 的最终差值。
  - 代码中明确写了如果提升不强就打印 `Weak, needs strong ablation`。
  - 这说明作者自己也把它视为“需要更强消融才能站住”的方案。
- 主要问题：
  - 训练依赖 seen-so-far cross-domain 数据，后续被 exemplar-free 版本重新修正。
  - 自身注释已经承认证据不够强。
- 状态判定：`已弱化`

### 4.10 SART v4b

- 代码位置：`sart_v4_experiment.py`
- 核心机制：
  - 去掉累计真实特征回放，回到 exemplar-free 口径。
  - transport 参数跨任务持续学习。
  - 只保留 prototype + domain stats + transport weights。
- 实验效果：
  - 相比 v2，设计上更干净，也更贴合 exemplar-free 叙事。
  - 但脚本同样保留了 `Weak, needs strong ablation` 的结论口径，说明作者对收益并不满意。
- 主要问题：
  - 虽然口径更合理，但仍缺统一报告。
  - 运输模块的收益似乎没有稳定超过 SHINE baseline。
- 状态判定：`可参考`

### 4.11 LEID

- 代码位置：`leid_experiment.py`
- 核心机制：
  - 学习式分解 encoder/decoder。
  - 在 SHINE 空间上对当前任务和高斯伪回放联合训练。
  - 可做 simplex/no-simplex/no-interaction 等 ablation。
- 实验效果：
  - 从脚本结构看，做了 marathon、few-shot、ablation 全套流程。
  - 但没有在仓库里找到统一结果表。
- 主要问题：
  - 额外模块较重。
  - 伪特征回放与现有主线之间的公平性和稳定性没有统一证据。
- 状态判定：`可参考`

### 4.12 LEID-Analytical

- 代码位置：`leid_analytical.py`
- 核心机制：
  - 不训练新模块，直接对 SHINE 特征做 PCA / WPCA / NMF / SRC 等解析式分解。
- 实验效果：
  - 更像对 LEID 思路的“training-free sanity check”。
- 主要问题：
  - 不是独立主线，只是辅助判断 learned decomposition 是否必要。
- 状态判定：`可参考`

### 4.13 Baseline / 协议 / 后处理类方法

#### `baseline_comparison.py`

- 内容：Frozen-NCM、SHINE、iCaRL、Replay-Herding 的横向比较。
- 价值：给早期多模态 marathon 设置一个基线面。
- 问题：主要是对比脚本，不是新的方法实现。

#### `baselines_experiment.py`

- 内容：EWC、LwF 等传统 CIL baseline 的 marathon 版本。
- 价值：补传统 CL 对照。
- 问题：结果没有并入主线分析报告。

#### `vit_baselines_experiment.py`

- 内容：ViT 轨 baseline、简化版 L2P。
- 价值：提供不同 backbone 范式的参考。
- 问题：和当前 MACIL/RFCIL 主线连接不强。

#### `shine_gate.py`

- 内容：post-hoc per-domain whitening。
- 价值：是对“校准/域归一化是否足够”这个问题的直接后处理试验。
- 问题：后处理能说明瓶颈在哪，但不是完整训练方法。

#### `bilateral_kd_pilot.py`

- 内容：`M_0 + M_prev` 双教师 feature KD pilot。
- 价值：验证更强 feature anchoring 是否值得继续做。
- 问题：定位就是 pilot，不应被当成稳定主线。

#### `single_dataset_cil.py`

- 内容：单数据集协议下跑 Baseline / SHINE / SART / SART+SHINE。
- 价值：提供单数据集对照环境。
- 问题：更像 protocol script。

#### `marathon_cil.py`

- 内容：Trento -> Houston -> MUUFL 9-task 协议脚本。
- 价值：给 S²CM 系列实验提供 marathon 载体。
- 问题：不是新方法本身。

#### `posthoc_eval.py`

- 内容：冻结特征上的后验参数扫值，主要面向 S²CM/MASGC。
- 价值：帮助理解后处理空间。
- 问题：属于分析工具，不是方法实现。

## 5. 历史变体与方法谱系

### 5.1 主线演化脉络

按仓库现状，方法演化大致可以读成四条线：

1. `our -> jcmethod -> MACIL / RFCIL`
   - 从旧式 KD + exemplar 逐渐演化到多模态 KD、原型对比、rehearsal-free 向量记忆。

2. `S²CM -> SHINE/TIDE/BRACE/SART/AnchorLoRA/...`
   - `S2CMNet` 成为大量顶层实验的底座。
   - 这条线的重点不在 `approach/`，而在顶层专项脚本。

3. `MOSAIC -> RFCIL 中的 mosaic 开关`
   - MOSAIC 的不对称 KD 思想没有消失，而是被吸收到 RFCIL 里做可选模块。

4. `RFCIL -> Matryoshka / PostMat`
   - 当前 RFCIL 在工程上继续扩展，但扩展分支的实验收益尚未被证实。

### 5.2 当前最值得继续维护的线

- `MACIL`：证据最完整，phase2-4 已经跑出比较清晰的正反结论。
- `RFCIL`：代码成熟、文档和测试最好，适合继续做工程化主线。
- `S²CM-based experiment family`：不一定是正式主线，但它支撑了大量概念验证。

### 5.3 明显不宜继续作为独立主线的线

- `our`
- `rfcil_backup_pre_mosaic`
- `SART v2`
- 若没有新证据，`rfcil_matryoshka` / `rfcil_postmat` 也不应优先于 base RFCIL

## 6. 实验结论总表

### 6.1 当前证据最强的结论

1. `MACIL` 线上，`CE + logits KD + InfoNCE` 仍是最稳定的核心配方。
2. 更重的蒸馏栈不是自动更好，`RKD + feature KD + MGD` 往往会牺牲 `TAg`。
3. 容量改动比额外 loss stack 更有价值，尤其是 `expand=4`。
4. `advanced_cosine` 比普通 `cosine` 更值得作为后续默认 scheduler 候选。
5. phase3、phase4 都没有超过 phase2 最佳容量线：
   - `cap_state_d16_x4_s0 = 83.17 / 51.18 / 12.00 / 22.01`
6. `RFCIL` 工程质量高、测试充分，但目前能看到的成绩仍偏中等，没有压过 MACIL 的 phase2 前沿。

### 6.2 已被否定或暂时不值得继续深挖的方向

1. `normalize_heads` 不是好的 task-agnostic calibration 方案。
2. 当前 `hybrid_proto` visibility proxy 没有打赢 baseline。
3. `MGD + hybrid_proto` 组合没有显示出加和收益。
4. `SART` 系列在脚本自身判断里都还属于“提升不强，需要更强消融”的状态。

### 6.3 缺证据但仍有研究价值的方向

1. `linkedmem`
2. `MOSAIC` 独立线
3. `AnchorLoRA / DualSpeed / SpecRoute / SDBT`
4. `LEID`

这些方向共同的问题不是“看起来一定不行”，而是“仓库里没有足够规范的实验归档支持继续相信它们”。

## 7. 共性问题与技术债

### 7.1 结果管理不统一

- `MACIL` / `RFCIL` 系列已有较规范的 phase 报告、日志和测试。
- 顶层大量实验脚本仍靠各自打印、局部 JSON 或最终控制台摘要保存结果。
- 直接后果是很多方法“确实实现了”，但几个月后已经很难判断它们究竟赢没赢。

### 7.2 方法成熟度差异大

- `approach/` 不等于都处于同一稳定层级。
- `our.py` 这种旧接口还在仓库里，会让人误以为它仍是当前有效主线。
- 一些历史变体只是为了备份或迁移，应该在文档和命名上更明确。

### 7.3 任务无关精度 (`TAg`) 依然是主瓶颈

从 `ARCH_SUMMARY_2026-03-13.md` 和 phase2-4 报告看，共性问题集中在：

- old/new class competition
- head calibration
- 过重约束抑制 plasticity
- `valid_loss` 与最终 `TAg` 目标不一致

这不是单个方法的问题，而是当前主线方法族共享的问题。

### 7.4 重复代码较多

顶层实验脚本普遍重复：

- `PaddedDataset`
- `extract_branch_features`
- `SHINE` 相关逻辑
- marathon / single-dataset 协议构建

这会提高验证成本，也会让实验间难以做到真正公平比较。

### 7.5 测试覆盖极不均衡

- RFCIL 家族测试最完整。
- JCMethod 有少量针对性测试。
- MACIL 有基础测试。
- 大部分顶层实验脚本几乎没有自动化测试，只能靠人工运行。

## 8. 建议优先级

### 8.1 如果目标是“继续推主线结果”

优先级建议：

1. 继续沿 `MACIL` 主线做 `TAg` 校准、checkpoint 选择和轻量机制升级。
2. 把 `RFCIL` 作为第二主线，补大规模对照和统一结果表。
3. 暂停没有统一证据的重型分支，避免再次把精力分散到十几个脚本。

### 8.2 如果目标是“清理仓库结构”

优先级建议：

1. 明确把 `our.py`、`rfcil_backup_pre_mosaic.py` 标成 legacy。
2. 给顶层实验脚本加统一结果落盘模板。
3. 把重复的 `PaddedDataset / SHINE / feature extraction` 抽到共享模块。

### 8.3 如果目标是“补论文/汇报材料”

最值得引用的证据源是：

1. `ARCH_SUMMARY_2026-03-13.md`
2. `outputs/plan_2026-03-13/phase2/phase2_analysis_report.md`
3. `outputs/plan_2026-03-13/phase3/phase3_analysis_report.md`
4. `outputs/plan_2026-03-13/phase4/phase4_analysis_report.md`
5. `docs/RFCIL_METHOD.md`

## 9. 证据来源索引

核心文档：

- `README.md`
- `ARCH_SUMMARY_2026-03-13.md`
- `docs/RFCIL_METHOD.md`

主线分析：

- `outputs/plan_2026-03-13/kd_analysis_report.md`
- `outputs/plan_2026-03-13/phase2/phase2_analysis_report.md`
- `outputs/plan_2026-03-13/phase3/phase3_analysis_report.md`
- `outputs/plan_2026-03-13/phase4/phase4_analysis_report.md`
- `outputs/plan_2026-03-13/phase5/phase5_design.md`

结果日志：

- `results/agora_*.log`
- `results/ablation_*.log`
- `results/rfcil_*.log`
- `results/*Houston*`
- `results/*MUUFL*`
- `results/*Trento*`

测试：

- `tests/test_jcmethod_ncm_ipcl.py`
- `tests/test_macil_*.py`
- `tests/test_rfcil_*.py`
- `tests/test_phase5_*.py`

