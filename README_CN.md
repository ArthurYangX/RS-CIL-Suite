<p align="center">
  <h1 align="center">RS-CIL-Suite</h1>
  <p align="center">
    面向遥感高光谱图像的标准化类增量学习基准平台
  </p>
</p>

<p align="center">
  <a href="./README.md">English</a> &bull;
  <a href="#快速开始">快速开始</a> &bull;
  <a href="#数据集">数据集</a> &bull;
  <a href="#方法">方法</a> &bull;
  <a href="#骨干网络">骨干网络</a> &bull;
  <a href="#可视化">可视化</a> &bull;
  <a href="#添加新方法">扩展</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/方法-17-blue" alt="Methods">
  <img src="https://img.shields.io/badge/数据集-10-green" alt="Datasets">
  <img src="https://img.shields.io/badge/协议-15%2B-orange" alt="Protocols">
  <img src="https://img.shields.io/badge/骨干网络-5-purple" alt="Backbones">
  <img src="https://img.shields.io/badge/测试-66%20通过-brightgreen" alt="Tests">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python">
</p>

---

## 核心特性

- **10 个公开 HSI 数据集** — 自动下载与预处理（PCA、归一化、patch 提取）
- **17 种类增量学习方法** — 覆盖正则化、回放、蒸馏、解析、梯度投影五大类
- **5 种骨干网络** — 从轻量 CNN（0.1M）到 ViT-Small（10.7M）
- **15 个内置评估协议**（场景内 + 跨场景）+ 自定义 YAML 协议
- **8 种样本选择策略**（herding、k-center、entropy、k-means 等）
- **任务感知评估** — 保存精确空间坐标，支持分类图可视化
- **论文级可视化** — 任务精度矩阵、反馈曲线、HyperKD 风格分类图
- **YAML 配置系统** — 支持 CLI 覆盖、wandb 集成、自描述 checkpoint
- **66 个单元测试** + GitHub Actions CI

---

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 下载数据集
python benchmark/download.py --dataset IndianPines --root ~/datasets/rs_cil

# 运行实验
python benchmark/run.py \
    --protocol A_IndianPines \
    --method icarl \
    --data_root ~/datasets/rs_cil \
    --seed 0 \
    --output results/icarl_A_IP.json \
    --plot

# 对比方法
python benchmark/compare.py results/ --latex
```

---

## 数据集

全部 10 个数据集通过 `download.py` 自动下载，无需注册。

| 数据集 | 模态 | 类别数 | HSI 尺寸 | 辅助数据尺寸 |
|--------|------|:------:|----------|-------------|
| [Trento](https://github.com/tyust-dayu/Trento) | HSI + LiDAR | 6 | 166 x 600 x 63 | 166 x 600 x 1 |
| [Houston 2013](https://github.com/songyz2019/rs-fusion-datasets-dist) | HSI + LiDAR | 15 | 349 x 1905 x 144 | 349 x 1905 x 1 |
| [MUUFL](https://github.com/GatorSense/MUUFLGulfport) | HSI + LiDAR | 11 | 325 x 220 x 64 | 325 x 220 x 2 |
| [Augsburg](https://github.com/songyz2019/rs-fusion-datasets-dist) | HSI + SAR | 8 | 332 x 485 x 180 | 332 x 485 x 4 |
| [Houston 2018](https://github.com/songyz2019/rs-fusion-datasets-dist) | HSI + LiDAR | 20 | 1202 x 4768 x 50 | 1202 x 4768 x 1 |
| [Indian Pines](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) | HSI | 16 | 145 x 145 x 200 | -- |
| [Pavia University](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) | HSI | 9 | 610 x 340 x 103 | -- |
| [Salinas](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) | HSI | 16 | 512 x 217 x 204 | -- |
| [Berlin](https://github.com/songyz2019/rs-fusion-datasets-dist) | HSI + SAR | 8 | 476 x 1723 x 244 | 476 x 1723 x 4 |
| [WHU-Hi-LongKou](https://huggingface.co/datasets/danaroth/whu_hi) | 无人机 HSI | 9 | 550 x 400 x 270 | -- |

预处理流程（仅首次运行，结果自动缓存）：
`原始 .mat` &rarr; PCA（默认 36 波段，仅在训练像素上拟合）&rarr; 归一化（训练像素统计量）&rarr; 镜像填充 &rarr; patch 提取（默认 7x7）&rarr; `.npz` 缓存

PCA 和归一化仅基于训练像素拟合，避免测试分布泄露。Patch 尺寸和 PCA 维度可通过 `--patch_size` 和 `--pca_components` 自定义。

---

## 方法

| 方法 | 类别 | 需要样本库 | 参考文献 |
|------|------|:----------:|----------|
| `joint` | 上界 | -- | 联合训练（所有数据同时可见） |
| `finetune` | 下界 | -- | 顺序微调（无持续学习机制） |
| `ncm` | 原型 | -- | 最近类均值 |
| `ewc` | 正则化 | -- | EWC (Kirkpatrick et al., PNAS 2017) |
| `si` | 正则化 | -- | SI (Zenke et al., ICML 2017) |
| `lwf` | 知识蒸馏 | -- | LwF (Li & Hoiem, ECCV 2016) |
| `gpm` | 梯度投影 | -- | GPM (Saha et al., NeurIPS 2021) |
| `acil` | 解析学习 | -- | ACIL (Zhuang et al., NeurIPS 2022) |
| `er` | 回放 | Yes | Experience Replay（基础回放基线） |
| `er_ace` | 回放 + 非对称 CE | Yes | ER-ACE (Caccia et al., ICLR 2022) |
| `icarl` | 回放 + 蒸馏 | Yes | iCaRL (Rebuffi et al., CVPR 2017) |
| `lucir` | 回放 + 余弦分类器 | Yes | LUCIR (Hou et al., CVPR 2019) |
| `bic` | 回放 + 偏差校正 | Yes | BiC (Wu et al., CVPR 2019) |
| `wa` | 回放 + 权重对齐 | Yes | WA (Zhao et al., CVPR 2020) |
| `podnet` | 回放 + 池化蒸馏 | Yes | PODNet (Douillard et al., ECCV 2020) |
| `der` | 回放 + logit KD | Yes | DER++ (Buzzega et al., NeurIPS 2020) |
| `gdumb` | 回放（重训练） | Yes | GDumb (Prabhu et al., ECCV 2020) |

---

## 骨干网络

所有方法共享可插拔的骨干网络，通过配置切换：

```bash
python benchmark/run.py --method icarl --protocol A_IndianPines \
    --data_root ~/data --opts model.backbone=resnet18_hsi
```

| 骨干网络 | 参数量 | 类型 | 说明 |
|---------|-------:|------|------|
| `simple_encoder` | 0.1M | CNN | 两层卷积 + BN + GELU（默认） |
| `vit_tiny_hsi` | 1.8M | Transformer | ViT-Tiny（embed=192, depth=4, heads=3） |
| `resnet18_hsi` | 11.3M | CNN | 适配 7x7 HSI patch 的 ResNet-18 |
| `vit_small_hsi` | 10.7M | Transformer | ViT-Small（embed=384, depth=6, heads=6） |
| `resnet34_hsi` | 21.4M | CNN | ResNet-34 变体 |

通过 `@register_backbone("name")` 装饰器在 `benchmark/models/` 中添加自定义骨干网络。

---

## 评估协议

### 协议 A — 场景内类增量

同一数据集内的类别按任务递增地到达。

| 协议 | 数据集 | 任务数 | 每任务类别数 |
|------|--------|:-----:|:----------:|
| `A_Trento` | Trento | 3 | 2 |
| `A_Houston2013` | Houston 2013 | 5 | 3 |
| `A_MUUFL` | MUUFL | 4 | 3 |
| `A_Augsburg` | Augsburg | 3 | 3/3/2 |
| `A_Houston2018` | Houston 2018 | 5 | 4 |
| `A_IndianPines` | Indian Pines | 4 | 4 |
| `A_PaviaU` | Pavia University | 3 | 3 |
| `A_Salinas` | Salinas | 4 | 4 |
| `A_Berlin` | Berlin | 3 | 3 |
| `A_WHUHiLongKou` | WHU-Hi-LongKou | 3 | 3 |

### 协议 B — 跨场景类增量

不同数据集依次到达，同时引入新类别和域偏移。

| 协议 | 数据集序列 | 任务数 |
|------|-----------|:-----:|
| `B1` | Trento &rarr; Houston 2013 &rarr; MUUFL | 9 |
| `B2` | Trento &rarr; Houston 2013 &rarr; MUUFL &rarr; Augsburg | 12 |
| `B3` | Indian Pines &rarr; Pavia U &rarr; Salinas | 11 |
| `B4` | 全部 5 个 HSI+LiDAR 数据集 | 17 |
| `B5` | Indian Pines &rarr; Pavia U &rarr; Salinas &rarr; Berlin &rarr; WHU-Hi | 17 |

### 自定义协议（YAML）

无需修改代码，通过 YAML 文件定义任务流：

```yaml
# configs/protocols/my_protocol.yaml
name: MyExperiment
type: cross_scene          # 或 "within_scene"
dataset_order: [Trento, Houston2013, MUUFL]
class_splits:
  Trento: [3, 3]
  Houston2013: [5, 5, 5]
  MUUFL: [6, 5]
train_ratio: 0.15          # 覆盖默认 10% 训练比例
shuffle_classes: true       # 随机化类别顺序
class_order_seed: 42
```

```bash
python benchmark/run.py --protocol configs/protocols/my_protocol.yaml \
    --method icarl --data_root ~/data
```

---

## 可视化

使用 `--plot` 自动生成论文级图表：

### 任务精度矩阵

行 = 被评估的任务，列 = 学习完第 *t* 个任务后。清晰展示遗忘（数值从左到右递减）和可塑性（对角线值）。

<p align="center">
  <img src="assets/icarl_A_IndianPines_seed0_task_matrix.png" width="45%" alt="iCaRL 任务矩阵">
  &nbsp;&nbsp;
  <img src="assets/ewc_A_IndianPines_seed0_task_matrix.png" width="45%" alt="EWC 任务矩阵">
</p>
<p align="center">
  <em>左：iCaRL（回放）保留了部分旧任务精度。右：EWC（正则化）出现灾难性遗忘——仅对角线存活。</em>
</p>

### 任务反馈曲线

三条线分解 OA 在任务序列中的变化：所有已见任务均值、旧任务均值（遗忘指标）、当前任务精度（可塑性指标）。

<p align="center">
  <img src="assets/icarl_A_Trento_seed0_task_feedback_curve.png" width="55%" alt="iCaRL 反馈曲线">
</p>
<p align="center">
  <em>iCaRL 在 A_Trento 上的表现：当前任务 OA（橙色）与旧任务均值（绿色）之间的差距揭示了稳定性-可塑性权衡。</em>
</p>

### 分类图

Ground Truth + 每个任务后的预测图，使用精确像素坐标。支持多方法并排对比。

<p align="center">
  <img src="assets/icarl_A_IndianPines_maps_per_task.png" width="95%" alt="iCaRL IndianPines 分类图">
</p>
<p align="center">
  <em>iCaRL 在 A_IndianPines 上的表现（16 类，4 个任务）：分类图从 95.1% 退化到 61.4%，误分类区域随任务增加明显增大。</em>
</p>

<p align="center">
  <img src="assets/icarl_A_Trento_maps_per_task.png" width="95%" alt="iCaRL Trento 分类图">
</p>
<p align="center">
  <em>iCaRL 在 A_Trento 上的表现（6 类，3 个任务）：OA 从 100% 降至 89%。</em>
</p>

```bash
# 生成全部图表
python benchmark/run.py --protocol A_IndianPines --method icarl \
    --data_root ~/data --output results/icarl.json --plot --plot_maps

# 从已保存结果批量生成
python -c "from benchmark.eval.plots import plot_suite; plot_suite('results/')"
```

---

## 实验管理

### Weights & Biases

[Weights & Biases](https://wandb.ai) 是一个免费的实验追踪平台。设置步骤：

```bash
# 1. 安装
pip install wandb

# 2. 在 https://wandb.ai/site 注册免费账号并获取 API key

# 3. 登录（仅需一次）
wandb login

# 4. 启用追踪运行实验
python benchmark/run.py --protocol A_IndianPines --method icarl \
    --data_root ~/data --wandb --wandb_project rs-cil-suite
```

所有指标（每个任务的 OA、AA、Kappa、BWT）会自动记录到你的 wandb 仪表盘。

### 多种子运行

```bash
python benchmark/run.py --protocol A_IndianPines --method icarl \
    --seeds 0,1,2 --data_root ~/data --output results/icarl_A_IP.json
```

### 结果对比表格

```bash
python benchmark/compare.py results/ --latex > table.tex
python benchmark/compare.py results/ --markdown
```

---

## 配置系统

所有超参数通过 YAML 配置管理，支持 CLI 覆盖：

```bash
# 覆盖任意参数
python benchmark/run.py --method icarl --protocol B1 --data_root ~/data \
    --opts training.lr=0.0005 method.memory_size=5000 model.backbone=resnet18_hsi

# 使用自定义配置文件
python benchmark/run.py --method icarl --config my_config.yaml --protocol B1
```

配置优先级：`defaults.yaml` &larr; `{method}.yaml` &larr; `--config` &larr; `--opts`

---

## 添加新方法

1. 复制模板：

```bash
cp benchmark/methods/_template.py benchmark/methods/my_method.py
```

2. 实现 `train_task()` 和 `predict()`：

```python
@register_method("my_method")
class MyMethod(CILMethod):
    def train_task(self, task, train_loader):
        ...
    def predict(self, loader):
        ...
```

3. 直接运行 — 自动发现，无需手动注册：

```bash
python benchmark/run.py --method my_method --protocol B1 --data_root ~/data
```

详见 `methods/_template.py`，包含完整可运行示例（样本库、知识蒸馏辅助、中间特征提取）。

---

## 样本选择策略

回放类方法共享可插拔的 `ExemplarMemory`，提供 8 种选择策略：

| 策略 | 无需模型 | 说明 |
|------|:--------:|------|
| `herding` | 否 | 迭代最近均值选择（iCaRL 默认） |
| `closest` | 否 | 非迭代最近均值（更快的 herding 替代） |
| `k_center` | 否 | 贪心核心集（最大化最小距离覆盖） |
| `entropy` | 否 | 选择特征空间中不确定性最高的样本（距类均值最远） |
| `kmeans` | 否 | K-Means++ 聚类，每个簇中心选一个样本 |
| `random` | 是 | 均匀随机采样 |
| `reservoir` | 是 | 类别平衡的蓄水池采样（在线） |
| `ring` | 是 | 每类 FIFO 环形缓冲 |

通过 CLI 切换：`--opts method.exemplar_strategy=entropy`

---

## Checkpoint 与推理

```bash
# 每个任务后保存 checkpoint
python benchmark/run.py --protocol B1 --method icarl \
    --data_root ~/data --save_checkpoints --checkpoint_dir ckpts/

# 从 checkpoint 独立推理
python benchmark/infer.py \
    --checkpoint ckpts/icarl_B1_seed0/task_8.pt \
    --protocol B1 --method icarl --data_root ~/data --save_maps
```

---

## 项目结构

```
benchmark/
├── configs/              # YAML 配置（默认 + 17 个方法配置 + 自定义协议）
├── models/               # 骨干网络注册表（SimpleEncoder、ResNet、ViT）
├── datasets/             # 10 个数据集加载器 + 预处理流水线
├── methods/              # 17 种 CIL 方法 + 基类 + 模板
├── protocols/            # 15 个内置协议 + YAML 加载器
├── eval/                 # 评估指标（OA/AA/Kappa/BWT/Plasticity）+ 14 个绘图函数 + 调色板
├── utils/                # ExemplarMemory、优化器、调度器
├── config.py             # YAML 配置加载器（支持 CLI 合并）
├── download.py           # 数据集自动下载器
├── run.py                # 实验运行主程序
├── infer.py              # 独立推理脚本
└── compare.py            # 结果聚合 + LaTeX/Markdown 表格
tests/                    # 64 个单元测试
.github/workflows/ci.yml # CI（Python 3.10/3.11/3.12）
```

---

## 引用

```bibtex
@misc{yang2026rscilsuite,
    author       = {Yang, Junchao},
    title        = {RS-CIL-Suite: A Standardized Class-Incremental Learning Suite for Remote Sensing Hyperspectral Imagery},
    year         = {2026},
    organization = {GitHub},
    url          = {https://github.com/ArthurYangX/RS-CIL-Suite},
}
```

## 致谢

感谢所有数据集提供者公开其数据。本项目参考了 [PyCIL](https://github.com/G-U-N/PyCIL)、[FACIL](https://github.com/mmasana/FACIL) 及 Zhou et al. (TPAMI 2024) 的类增量学习综述中的设计思路。
