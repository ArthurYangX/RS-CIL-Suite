# RS-CIL Benchmark

A standardized Class-Incremental Learning benchmark for Remote Sensing hyperspectral imagery, supporting 10 public datasets, 14 CIL methods, and 15 evaluation protocols.

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
python benchmark/compare.py results/ --latex
```

---

## Datasets

All 10 datasets download automatically via `download.py` (no registration required).

| Dataset | Modality | Classes | Size | Source |
|---------|----------|---------|------|--------|
| Trento | HSI + LiDAR | 6 | 166×600 | GitHub (tyust-dayu) |
| Houston2013 | HSI + LiDAR | 15 | 349×1905 | rs-fusion-datasets-dist |
| MUUFL | HSI + LiDAR | 11 | 325×220 | GatorSense GitHub |
| Augsburg | HSI + SAR | 8 | 332×485 | rs-fusion-datasets-dist |
| Houston2018 | HSI + LiDAR | 20 | 1202×4768 | rs-fusion-datasets-dist |
| IndianPines | HSI | 16 | 145×145 | EHU/GIC |
| PaviaU | HSI | 9 | 610×340 | EHU/GIC |
| Salinas | HSI | 16 | 512×217 | EHU/GIC |
| Berlin | HSI + SAR | 8 | 476×1723 | rs-fusion-datasets-dist |
| WHU-Hi-LongKou | UAV HSI | 9 | 550×400 | HuggingFace (danaroth/whu_hi) |

> WHU-Hi-LongKou uses ENVI `.bsq` format. Requires: `pip install spectral`

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
| `A_Augsburg` | Augsburg | 4 | 2 |
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
├── datasets/
│   ├── base.py          # RSDataset + PatchDataset (lazy load + .npz cache)
│   ├── hsi_lidar.py     # Trento, Houston2013, MUUFL, Augsburg, Houston2018
│   ├── hsi_only.py      # IndianPines, PaviaU, Salinas, Berlin, WHU-Hi-LongKou
│   ├── preprocess.py    # PCA, normalize, mirror pad, patch extraction
│   └── registry.py      # Dataset name → class mapping
├── methods/
│   ├── base.py          # CILMethod abstract base
│   ├── acil.py          # Analytic CIL
│   ├── der.py           # DER++
│   ├── ewc.py           # EWC
│   ├── finetune.py      # Sequential fine-tuning
│   ├── gdumb.py         # GDumb
│   ├── gpm.py           # Gradient Projection Memory
│   ├── icarl.py         # iCaRL
│   ├── joint.py         # Joint training
│   ├── lucir.py         # LUCIR
│   ├── lwf.py           # LwF
│   ├── ncm.py           # NCM
│   ├── podnet.py        # PODNet
│   └── si.py            # Synaptic Intelligence
├── protocols/
│   └── cil.py           # 15 protocols (A × 10 datasets + B1–B5)
├── eval/
│   └── metrics.py       # OA, AA, Kappa, BWT, FWT
├── download.py          # Dataset downloader (--dataset all --preprocess)
├── run.py               # Experiment runner
└── compare.py           # Results aggregation + LaTeX table
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
# 1. Download locally (server has no internet)
python benchmark/download.py --dataset all --preprocess

# 2. Copy to server
scp -r ~/datasets/rs_cil/ gpu-server:/root/autodl-tmp/datasets/

# 3. Run on server
ssh gpu-server
eval "$(/root/miniconda3/bin/conda shell.bash hook)" && conda activate jc
python benchmark/run.py --protocol B1 --method icarl \
    --data_root /root/autodl-tmp/datasets --seed 0
```

---

## Multi-seed + Comparison

```bash
# Run with 3 seeds
python benchmark/run.py --protocol A_IndianPines --method icarl --seeds 0,1,2 \
    --data_root ~/datasets/rs_cil --out_dir results/

# Generate comparison table
python benchmark/compare.py results/ --group-by method
python benchmark/compare.py results/ --latex > table.tex
```
