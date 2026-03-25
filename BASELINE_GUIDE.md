# Baseline Experiment Guide

**文件**: `code/baselines_experiment.py`
**Backbone**: S2CM (Spectral-Spatial Contrastive Mamba, embed_dim=64)
**协议**: Marathon CIL — 3 域 × 9 tasks × 32 classes (Trento→Houston→MUUFL)

---

## 方法总览

| # | Method | 类型 | Backbone 状态 | 需要 Exemplar? | 参考文献 |
|---|--------|------|--------------|---------------|---------|
| 1 | `frozen` | Lower bound | 冻结 | 否 | — |
| 2 | `ewc` | 正则化 | Fine-tune | 否 | Kirkpatrick et al., PNAS 2017 |
| 3 | `lwf` | 知识蒸馏 | Fine-tune | 否 | Li & Hoiem, TPAMI 2017 |
| 4 | `analytic` | 解析解 | 冻结 | 否 | CrossACL-style, TGRS 2025 |
| 5 | `pscen` | 原型校准 | 冻结 | 否 | PSCEN-style, Expert Syst. 2025 |
| 6 | `foster` | 特征适配 | 冻结+Adapter | 否 | FOSTER-style, ECCV 2022 |
| 7 | `dcprn` | 双蒸馏+伪回放 | Fine-tune | 否(伪特征) | DCPRN-style, TGRS 2024 |
| 8 | `icarl` | 样本回放+蒸馏 | Fine-tune | **是** | Rebuffi et al., CVPR 2017 |
| 9 | `lucir` | 余弦头+遗忘约束 | Fine-tune | **是** | Hou et al., CVPR 2019 |

---

## 运行命令

### 通用格式

```bash
eval "$(/root/miniconda3/bin/conda shell.bash hook)" && conda activate jc && cd /root/autodl-tmp/jc

PYTHONUNBUFFERED=1 python baselines_experiment.py \
    --method <METHOD> \
    --mode marathon \
    --batch_size 128 \
    --seed 0 \
    [方法特定参数]
```

### 逐方法命令

**Frozen (下界，无需训练)**
```bash
python baselines_experiment.py --method frozen --mode marathon --seed 0
```

**EWC**
```bash
python baselines_experiment.py --method ewc --mode marathon --seed 0 \
    --lambda_ewc 1000 --train_epochs 30 --train_lr 1e-3
```

**LwF**
```bash
python baselines_experiment.py --method lwf --mode marathon --seed 0 \
    --lambda_kd 1.0 --train_epochs 30 --train_lr 1e-3
```

**Analytic (CrossACL-style, 无需训练)**
```bash
python baselines_experiment.py --method analytic --mode marathon --seed 0 \
    --ridge_lambda 1.0
```

**PSCEN (训练无关，仅原型校准)**
```bash
python baselines_experiment.py --method pscen --mode marathon --seed 0
```

**FOSTER**
```bash
python baselines_experiment.py --method foster --mode marathon --seed 0 \
    --train_epochs 30 --train_lr 1e-3
```

**DCPRN**
```bash
python baselines_experiment.py --method dcprn --mode marathon --seed 0 \
    --train_epochs 30 --train_lr 1e-3
```

**iCaRL (需要 exemplar)**
```bash
python baselines_experiment.py --method icarl --mode marathon --seed 0 \
    --n_exemplars 20 --train_epochs 30 --train_lr 1e-3
```

**LUCIR (需要 exemplar)**
```bash
python baselines_experiment.py --method lucir --mode marathon --seed 0 \
    --n_exemplars 20 --train_epochs 30 --train_lr 1e-3
```

---

## 批量执行脚本

### 全部跑完 (~3-4 小时，分 3 批)

```bash
#!/bin/bash
eval "$(/root/miniconda3/bin/conda shell.bash hook)" && conda activate jc
cd /root/autodl-tmp/jc
mkdir -p /root/autodl-tmp/results/s2cm/baselines

# Batch 1: Exemplar-free, 无训练/轻量训练 (3 并发, ~20 min)
for m in frozen analytic pscen; do
  screen -dmS bl_$m bash -lc "eval \"\$(/root/miniconda3/bin/conda shell.bash hook)\" && conda activate jc && cd /root/autodl-tmp/jc && PYTHONUNBUFFERED=1 python baselines_experiment.py --method $m --mode marathon --batch_size 128 --seed 0 2>&1 | tee /root/autodl-tmp/results/s2cm/baselines/${m}.log"
done
echo "Batch 1 launched: frozen, analytic, pscen"

# 等 Batch 1 完成后...

# Batch 2: Exemplar-free, fine-tune (3 并发, ~40 min)
for m in ewc lwf foster; do
  screen -dmS bl_$m bash -lc "eval \"\$(/root/miniconda3/bin/conda shell.bash hook)\" && conda activate jc && cd /root/autodl-tmp/jc && PYTHONUNBUFFERED=1 python baselines_experiment.py --method $m --mode marathon --batch_size 128 --seed 0 2>&1 | tee /root/autodl-tmp/results/s2cm/baselines/${m}.log"
done
echo "Batch 2 launched: ewc, lwf, foster"

# 等 Batch 2 完成后...

# Batch 3: Exemplar-based + DCPRN (3 并发, ~40 min)
for m in dcprn icarl lucir; do
  screen -dmS bl_$m bash -lc "eval \"\$(/root/miniconda3/bin/conda shell.bash hook)\" && conda activate jc && cd /root/autodl-tmp/jc && PYTHONUNBUFFERED=1 python baselines_experiment.py --method $m --mode marathon --batch_size 128 --seed 0 --n_exemplars 20 2>&1 | tee /root/autodl-tmp/results/s2cm/baselines/${m}.log"
done
echo "Batch 3 launched: dcprn, icarl, lucir"
```

### 快速验证 (MVP 模式，4 tasks)

```bash
python baselines_experiment.py --method ewc --mode mvp --seed 0
```

---

## 超参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch_size` | 128 | 训练和评估的 batch size |
| `--train_epochs` | 30 | 每个 task 的训练 epoch 数 |
| `--train_lr` | 1e-3 | AdamW 学习率 |
| `--lambda_ewc` | 1000 | EWC Fisher 正则化强度 |
| `--lambda_kd` | 1.0 | LwF/iCaRL/DCPRN 蒸馏损失权重 |
| `--n_exemplars` | 20 | iCaRL/LUCIR 每类存储样本数 |
| `--ridge_lambda` | 1.0 | Analytic ridge regression 正则化 |

---

## 各方法核心原理

### EWC (Elastic Weight Consolidation)
```
训练后: 计算 Fisher 信息矩阵 F (对角近似)
训练时: Loss = CE_new + λ × Σ F_i × (θ_i - θ*_i)²
```
Fisher 矩阵标记重要参数，防止大幅改变。

### LwF (Learning without Forgetting)
```
训练前: 保存旧模型快照
训练时: Loss = CE_new + λ × cosine_distillation(new_features, old_features)
```
用旧模型的特征方向作为软标签蒸馏。

### Analytic (CrossACL-style)
```
每个 task: 提取冻结特征 → 累积 X^T X 和 X^T Y
预测: W = (X^T X + λI)^{-1} × X^T Y (闭式解)
```
无梯度训练，纯数学求解。

### PSCEN (Prototype Similarity Calibration)
```
无训练，仅调整推理:
score = NCM_similarity - α × prototype_inter_similarity
```
惩罚与其他原型太相似的类（减少混淆）。

### FOSTER (Feature Boosting + Compression)
```
每个 task: 训练小型 adapter 做特征增强
adapter: features → features + MLP(features)
压缩: KD 约束防止 adapter 偏离太远
```
Backbone 冻结，只训练轻量 adapter。

### DCPRN (Dual KD + Prototype Replay)
```
训练时:
  1. CE on 真实数据 + 伪特征 (从原型 + 噪声生成)
  2. Feature-level KD: MSE(new_feat, old_feat)
  3. 原型回放: 旧类伪特征参与训练
```
无需存储真实样本，用原型统计量生成伪特征。

### iCaRL
```
每个 task:
  1. Fine-tune backbone: CE + KD
  2. Herding: 选择最接近类均值的 20 个样本存储
推理: NCM on stored exemplars
```
需要存储真实样本。

### LUCIR
```
每个 task:
  1. Cosine normalized classifier
  2. Less-forget: 保持旧特征方向 (cosine distillation)
  3. Inter-class margin: 拉大新旧类的特征距离
```
需要存储真实样本。

---

## 输出格式

每个实验输出 JSON 到 `/root/autodl-tmp/results/s2cm/baselines/<method>_seed<N>.json`:

```json
{
  "config": { ... },
  "results": {
    "<method>": [
      {"task": 0, "dataset": "Trento", "n_seen": 2, "avg_tag": 0.998, "per_ds": {"Trento": 0.998}},
      ...
    ],
    "<method>+SHINE": [ ... ]
  }
}
```

每个方法同时报告 raw 和 +SHINE 两个版本的结果。

---

## 预期结果表格

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
| **AnchorLoRA+SHINE** | **EF** | **~92%** | **~70%** | **~75%** | **~75.7%** |

EF=Exemplar-Free, PR=Pseudo-Replay, EB=Exemplar-Based

---

## 检查实验进度

```bash
# 查看所有 screen
screen -ls

# 查看某个实验日志
tail -20 /root/autodl-tmp/results/s2cm/baselines/ewc.log

# 查看是否完成
grep "^FINAL:" /root/autodl-tmp/results/s2cm/baselines/*.log

# GPU 使用
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
```
