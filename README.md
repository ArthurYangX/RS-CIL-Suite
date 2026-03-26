# RS-CIL Benchmark

A standardized Class-Incremental Learning benchmark for Remote Sensing hyperspectral imagery, supporting 10 public datasets, 15 CIL methods, and fully customizable evaluation protocols.

---

## Quick Start

```bash
# 1. Download datasets
python benchmark/download.py --dataset all --root ~/datasets/rs_cil

# 2. Preprocess (.mat → .npz cache, one-time)
python benchmark/download.py --dataset all --root ~/datasets/rs_cil --preprocess

# 3. Run a method
python benchmark/run.py --protocol A_IndianPines --method icarl \
    --data_root ~/datasets/rs_cil --seed 0

# 4. Compare results
python benchmark/compare.py "results/*.json" --latex
```

---

## Datasets

All 10 datasets download automatically via `download.py` (no registration required).

| Dataset | Modality | Classes | HSI Size | Aux Size | Source |
|---------|----------|---------|----------|----------|--------|
| [Trento](https://github.com/tyust-dayu/Trento) | HSI + LiDAR | 6 | 166×600×63 | 166×600×1 | [tyust-dayu/Trento](https://github.com/tyust-dayu/Trento) |
| [Houston2013](https://github.com/songyz2019/rs-fusion-datasets-dist) | HSI + LiDAR | 15 | 349×1905×144 | 349×1905×1 | [rs-fusion-datasets-dist](https://github.com/songyz2019/rs-fusion-datasets-dist) |
| [MUUFL](https://github.com/GatorSense/MUUFLGulfport) | HSI + LiDAR | 11 | 325×220×64 | 325×220×2 | [GatorSense/MUUFLGulfport](https://github.com/GatorSense/MUUFLGulfport) |
| [Augsburg](https://github.com/songyz2019/rs-fusion-datasets-dist) | HSI + SAR | 8 | 332×485×180 | 332×485×4 | [rs-fusion-datasets-dist](https://github.com/songyz2019/rs-fusion-datasets-dist) |
| [Houston2018](https://github.com/songyz2019/rs-fusion-datasets-dist) | HSI + LiDAR | 20 | 1202×4768×50 | 1202×4768×1 | [rs-fusion-datasets-dist](https://github.com/songyz2019/rs-fusion-datasets-dist) |
| [IndianPines](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) | HSI | 16 | 145×145×200 | — | [EHU/GIC](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) |
| [PaviaU](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) | HSI | 9 | 610×340×103 | — | [EHU/GIC](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) |
| [Salinas](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) | HSI | 16 | 512×217×204 | — | [EHU/GIC](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) |
| [Berlin](https://github.com/songyz2019/rs-fusion-datasets-dist) | HSI + SAR | 8 | 476×1723×244 | 476×1723×4 | [rs-fusion-datasets-dist](https://github.com/songyz2019/rs-fusion-datasets-dist) |
| [WHU-Hi-LongKou](https://huggingface.co/datasets/danaroth/whu_hi) | UAV HSI | 9 | 550×400×270 | — | [danaroth/whu_hi](https://huggingface.co/datasets/danaroth/whu_hi) |

> WHU-Hi-LongKou uses ENVI `.bsq` format. Requires: `pip install spectral`

We thank all dataset providers for making their data publicly available.

---

## Methods

| Method | Type | Exemplar | Reference |
|--------|------|----------|-----------|
| `joint` | Upper bound | — | Joint training on all data |
| `finetune` | Lower bound | — | Sequential fine-tuning (no CL) |
| `ncm` | Prototype | — | Nearest Class Mean |
| `ewc` | Regularization | — | EWC (Kirkpatrick et al., PNAS 2017) |
| `si` | Regularization | — | SI (Zenke et al., ICML 2017) |
| `lwf` | Distillation | — | LwF (Li & Hoiem, ECCV 2016) |
| `gpm` | Gradient | — | GPM (Saha et al., NeurIPS 2021) |
| `acil` | Analytic | — | ACIL (Zhuang et al., NeurIPS 2022) |
| `icarl` | Replay | ✓ | iCaRL (Rebuffi et al., CVPR 2017) |
| `lucir` | Replay + Cosine | ✓ | LUCIR (Hou et al., CVPR 2019) |
| `podnet` | Replay + Distill | ✓ | PODNet (Douillard et al., ECCV 2020) |
| `der` | Replay + KD | ✓ | DER++ (Buzzega et al., NeurIPS 2020) |
| `gdumb` | Replay (greedy) | ✓ | GDumb (Prabhu et al., ECCV 2020) |
| `bic` | Replay + Bias Correction | ✓ | BiC (Wu et al., CVPR 2019) |
| `wa` | Replay + Weight Align | ✓ | WA (Zhao et al., CVPR 2020) |

---

## Protocols

### Protocol A — Within-scene CIL
Classes from a single dataset split incrementally.

| Protocol | Dataset | Tasks | Classes per task |
|----------|---------|-------|-----------------|
| `A_IndianPines` | IndianPines | 4 | 4 |
| `A_PaviaU` | PaviaU | 3 | 3 |
| `A_Salinas` | Salinas | 4 | 4 |
| `A_Trento` | Trento | 3 | 2 |
| `A_Houston2013` | Houston2013 | 5 | 3 |
| `A_MUUFL` | MUUFL | 4 | 3 |
| `A_Augsburg` | Augsburg | 3 | 3/3/2 |
| `A_Houston2018` | Houston2018 | 5 | 4 |
| `A_Berlin` | Berlin | 3 | 3 |
| `A_WHUHiLongKou` | WHU-Hi-LongKou | 3 | 3 |

### Protocol B — Cross-scene CIL
Entire datasets arrive sequentially (domain shift across tasks).

| Protocol | Datasets | Tasks |
|----------|---------|-------|
| `B1` | Trento → Houston2013 → MUUFL | 3 |
| `B2` | Trento → Houston2013 → MUUFL → Augsburg | 4 |
| `B3` | IndianPines → PaviaU → Salinas | 3 |
| `B4` | All 5 HSI+LiDAR datasets | 5 |
| `B5` | IndianPines → PaviaU → Salinas → Berlin → WHU-Hi-LongKou | 5 |

### Custom Protocols (YAML)

Beyond the built-in protocols, you can define your own task flow via a YAML file — **no Python code changes needed**.

```yaml
# configs/protocols/my_protocol.yaml
name: MyExperiment
type: cross_scene                  # or "within_scene"

dataset_order:
  - Trento
  - Houston2013
  - MUUFL

class_splits:
  Trento: [3, 3]                   # 2 tasks, 3 classes each
  Houston2013: [5, 5, 5]           # 3 tasks, 5 classes each
  MUUFL: [6, 5]                    # 2 tasks, 6+5 classes

# Optional
train_ratio: 0.15                  # override default 10% train split
shuffle_classes: true              # randomize class order within each dataset
class_order_seed: 42               # seed for reproducible shuffling
```

Run it directly:

```bash
# By file path
python benchmark/run.py --protocol configs/protocols/my_protocol.yaml \
    --method icarl --data_root ~/datasets/rs_cil

# Or place in configs/protocols/ and use the name
python benchmark/run.py --protocol my_protocol --method icarl \
    --data_root ~/datasets/rs_cil
```

**Within-scene example** (single dataset, fine-grained task splits):

```yaml
name: IndianPines_8Tasks
type: within_scene
dataset_order: [IndianPines]
class_splits:
  IndianPines: [2, 2, 2, 2, 2, 2, 2, 2]   # 8 tasks, 2 classes each
```

**Key parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `dataset_order` | List of dataset names in the order they appear | (required) |
| `class_splits` | Dict mapping dataset → list of classes per task | (required) |
| `type` | `"cross_scene"` or `"within_scene"` | `cross_scene` |
| `train_ratio` | Fraction of labelled pixels for training | `0.1` |
| `shuffle_classes` | Randomize class order within each dataset | `false` |
| `class_order_seed` | Seed for reproducible shuffling | `42` |

See `configs/protocols/` for more examples.

---

## Metrics

| Metric | Description |
|--------|-------------|
| **OA** | Overall Accuracy — weighted by class frequency |
| **AA** | Average Accuracy — mean per-class accuracy |
| **Kappa** | Cohen's Kappa coefficient |
| **BWT** | Backward Transfer (forgetting) — negative = forgetting |
| **FWT** | Forward Transfer (plasticity) — how well model learns new tasks |

Results are reported as final-task averages across all seen classes.

---

## File Structure

```
benchmark/
├── configs/
│   ├── defaults.yaml       # Shared hyperparameters
│   ├── {method}.yaml       # Per-method configs (15 files)
│   └── protocols/          # Custom YAML protocols
├── models/
│   ├── __init__.py         # Backbone registry + build_backbone()
│   └── simple_encoder.py   # SimpleEncoder (with intermediate feature hooks)
├── datasets/
│   ├── base.py             # RSDataset + PatchDataset + gt_map property
│   ├── hsi_lidar.py        # Trento, Houston2013, MUUFL, Augsburg, Houston2018
│   ├── hsi_only.py         # IndianPines, PaviaU, Salinas, Berlin, WHU-Hi-LongKou
│   ├── preprocess.py       # PCA, normalize, mirror pad, patch extraction
│   └── registry.py         # Dataset name → class mapping
├── methods/
│   ├── base.py             # CILMethod base + @register_method + checkpoint + KD helpers
│   ├── _template.py        # Template for adding new methods
│   ├── wa.py               # WA (Weight Aligning)
│   └── {13 more methods}   # acil, bic, der, ewc, finetune, gdumb, gpm, icarl, ...
├── protocols/
│   └── cil.py              # 15 built-in protocols + YAML protocol loader
├── eval/
│   ├── metrics.py          # OA, AA, Kappa, BWT, FWT
│   ├── plots.py            # 11 plot functions (curves, maps, confusion, radar, ...)
│   └── colors.py           # Color palettes for all 10 datasets
├── utils/
│   ├── training.py         # build_optimizer, build_scheduler, remap_labels
│   └── exemplars.py        # Shared ExemplarMemory (8 selection strategies)
├── config.py               # YAML config loader (defaults ← method ← CLI)
├── download.py             # Dataset downloader
├── run.py                  # Experiment runner (config, checkpoints, wandb)
├── infer.py                # Standalone inference from checkpoint
└── compare.py              # Results aggregation + LaTeX + Markdown leaderboard
```

---

## Preprocessing Pipeline

Each dataset is preprocessed once and cached as `.npz`:

```
raw .mat files
    → PCA (36 components, whitened)
    → per-band min-max normalization
    → mirror padding (patch//2 pixels)
    → 7×7 patch extraction
    → .cache/benchmark_<hash>.npz
```

Subsequent runs load directly from cache (skip PCA).

---

## Running on Server

```bash
# 1. Download locally (server may have no internet)
python benchmark/download.py --dataset all --preprocess

# 2. Copy to server
scp -r ~/datasets/rs_cil/ your-server:/path/to/datasets/

# 3. Run on server
ssh your-server
python benchmark/run.py --protocol B1 --method icarl \
    --data_root /path/to/datasets --seed 0
```

---

## Weights & Biases Integration

Track experiments with [wandb](https://wandb.ai):

```bash
# Enable wandb logging
python benchmark/run.py --protocol A_IndianPines --method icarl \
    --data_root ~/datasets/rs_cil --wandb --wandb_project rs-cil-benchmark

# Custom project name
python benchmark/run.py --protocol B1 --method der \
    --data_root ~/datasets/rs_cil --wandb --wandb_project my-project
```

Logged metrics per task:
- OA, AA, Kappa
- Per-dataset accuracy
- BWT, FWT (final)
- Task-feedback matrix values
- Confusion matrices (as wandb tables)

> Requires `pip install wandb` and `wandb login`.

---

## Multi-seed + Comparison

```bash
# Run with 3 seeds
python benchmark/run.py --protocol A_IndianPines --method icarl --seeds 0,1,2 \
    --data_root ~/datasets/rs_cil --output results/icarl_A_IP.json

# Generate comparison table
python benchmark/compare.py results/ --group-by method
python benchmark/compare.py results/ --latex > table.tex
python benchmark/compare.py results/ --markdown          # GitHub leaderboard
python benchmark/compare.py results/ --leaderboard       # save LEADERBOARD.md
```

---

## YAML Config System

All hyperparameters are externalized into YAML configs. You can override any parameter from the CLI:

```bash
# Use default config for iCaRL
python benchmark/run.py --method icarl --protocol B1 --data_root ~/data

# Override learning rate and memory size
python benchmark/run.py --method icarl --protocol B1 --data_root ~/data \
    --opts training.lr=0.0005 method.memory_size=5000

# Use a fully custom config file
python benchmark/run.py --method icarl --protocol B1 --data_root ~/data \
    --config my_config.yaml
```

Config resolution order: `defaults.yaml` ← `{method}.yaml` ← `--config` ← `--opts`

---

## Checkpoints & Inference

Save model checkpoints after each task, then run standalone inference:

```bash
# Train with checkpoints
python benchmark/run.py --protocol B1 --method icarl \
    --data_root ~/data --save_checkpoints --checkpoint_dir ckpts/

# Inference from checkpoint
python benchmark/infer.py \
    --checkpoint ckpts/icarl_B1_seed0/task_8.pt \
    --protocol B1 --method icarl --data_root ~/data

# Generate classification maps
python benchmark/infer.py \
    --checkpoint ckpts/icarl_B1_seed0/task_8.pt \
    --protocol B1 --method icarl --data_root ~/data \
    --save_maps --output_dir maps/
```

---

## Adding a New Method

Create **one file** — the benchmark auto-discovers it:

```bash
cp benchmark/methods/_template.py benchmark/methods/my_method.py
```

Edit `my_method.py`:

```python
from .base import CILMethod, register_method
from benchmark.models import SimpleEncoder
from benchmark.protocols.cil import Task

@register_method("my_method")              # <-- auto-registered
class MyMethod(CILMethod):
    name = "MyMethod"

    def __init__(self, hsi_channels, lidar_channels, num_classes, device,
                 d=128, epochs=50, lr=1e-3, my_param=1.0, **kwargs):
        backbone = SimpleEncoder(hsi_channels, lidar_channels, d)
        super().__init__(backbone, device, num_classes)
        self.epochs = epochs
        self.lr = lr
        self.my_param = my_param
        self.head = torch.nn.Linear(d, num_classes).to(device)

    def train_task(self, task, train_loader):
        # Your training logic here
        ...

    def predict(self, loader):
        # Your inference logic here
        ...
```

Optionally add a YAML config:

```yaml
# configs/my_method.yaml
method:
  my_param: 1.0
```

Run it:

```bash
python benchmark/run.py --method my_method --protocol B1 --data_root ~/data
```

**KD method helpers** available in the base class:

```python
# Snapshot old model for knowledge distillation
self._snapshot_old_model()       # → self._old_model (frozen copy)

# Intermediate features from backbone
feat, intermediates = self.model(xh, xl, return_features=True)
# intermediates = {"conv1": ..., "conv2": ..., "pooled": ..., "pre_norm": ...}

# Shared exemplar memory
from benchmark.utils.exemplars import ExemplarMemory
self.memory = ExemplarMemory(budget=2000, strategy="herding")
self.memory.update(self.model, hsi, lidar, labels, device)
replay_loader = self.memory.get_loader(batch_size=64)

# Epoch-level logging (auto-routed to wandb when --wandb is set)
self._log({"train/loss": loss, "epoch": ep, "task_id": task.task_id})
```

See `methods/_template.py` for a complete runnable example.

---

## Exemplar Selection Strategies

Replay-based methods (iCaRL, LUCIR, PODNet, WA, DER++, BiC, GDumb) use exemplar memory to store a subset of old-task data. The benchmark provides **8 selection strategies** via the shared `ExemplarMemory` module — switch with one line in YAML or CLI:

```bash
# CLI override
python benchmark/run.py --method wa --protocol B1 --data_root ~/data \
    --opts method.exemplar_strategy=entropy

# Or in YAML config
# configs/wa.yaml
# method:
#   exemplar_strategy: kmeans
```

| Strategy | Needs Model | Speed | Description |
|----------|:-----------:|:-----:|-------------|
| `herding` | Yes | Medium | Iterative closest-to-mean in feature space (iCaRL default) |
| `closest` | Yes | Fast | Non-iterative closest-to-mean (faster herding alternative) |
| `random` | No | Fast | Uniform random sampling |
| `k_center` | Yes | Medium | Greedy coreset — maximise min-distance coverage |
| `entropy` | Yes | Fast | Select highest-entropy (most uncertain) samples |
| `kmeans` | Yes | Medium | K-Means++ clustering — one sample per centroid |
| `reservoir` | No | Fast | Class-balanced reservoir sampling (online) |
| `ring` | No | Fast | FIFO ring buffer per class |

**Usage in your own method:**

```python
from benchmark.utils.exemplars import ExemplarMemory, list_strategies

# See all available strategies
print(list_strategies())

# Create memory with any strategy
memory = ExemplarMemory(budget=2000, strategy="herding")

# Update after each task (selects exemplars for new classes, trims old ones)
memory.update(self.model, hsi, lidar, labels, device,
              new_class_ids=task.global_class_ids)

# Get replay data
replay_loader = memory.get_loader(batch_size=64)
hsi, lidar, labels = memory.get_data()  # raw tensors

# Checkpoint support
state = memory.state_dict()    # save
memory.load_state_dict(state)  # restore
```

Also update the file structure entry:

```
└── exemplars.py        # Shared ExemplarMemory (8 strategies)
```

---

## Visualization

Generate paper-quality figures from results:

```bash
# After training with --save_checkpoints
python benchmark/run.py --protocol B1 --method icarl \
    --data_root ~/data --save_checkpoints --plot --plot_maps

# Or from the plot suite
python -c "from benchmark.eval.plots import plot_suite; plot_suite('results/')"
```

Available plots: task progression curves, method comparison bars, methods overlay, forgetting heatmap, classification maps, confusion matrices, per-class accuracy bars, radar charts, comparison tables.
