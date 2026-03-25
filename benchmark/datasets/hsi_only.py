"""HSI-only dataset implementations (LiDAR channel zeroed out).

These single-modality datasets serve as:
  - Single-modal CIL baselines
  - Within-scene Protocol A validation
  - Comparison with HSI+LiDAR methods

Supported:
  - IndianPines   (16 classes, AVIRIS, 145×145)
  - PaviaU        (9 classes,  ROSIS,  610×340)
  - Salinas       (16 classes, AVIRIS, 512×217)
  - Berlin        (8 classes,  EnMAP HSI + Sentinel-1 SAR, 1723×476)
  - WHU-Hi-LongKou (9 classes, UAV HSI, 550×400)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.io import loadmat

from .base import RSDataset, DatasetInfo
from .preprocess import preprocess_hsi_lidar, linear_index_to_label_maps


def _zero_lidar(hsi: np.ndarray) -> np.ndarray:
    """Return a single-channel zero array matching HSI spatial dims."""
    return np.zeros((hsi.shape[0], hsi.shape[1], 1), dtype=np.float32)


# ══════════════════════════════════════════════════════════════════
# Indian Pines  (AVIRIS, 1992)
# Download: http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes
# GitHub mirror: https://github.com/danfenghong/IEEE_TGRS_MDL-RS
# ══════════════════════════════════════════════════════════════════

class IndianPines(RSDataset):
    INFO = DatasetInfo(
        name="IndianPines",
        modalities=["hsi"],
        num_classes=16,
        class_names=[
            "Alfalfa", "Corn-notill", "Corn-mintill", "Corn",
            "Grass-pasture", "Grass-trees", "Grass-pasture-mowed",
            "Hay-windrowed", "Oats", "Soybean-notill", "Soybean-mintill",
            "Soybean-clean", "Wheat", "Woods",
            "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers",
        ],
        location="Northwestern Indiana, USA",
        sensor="AVIRIS (220 bands, 0.4–2.5 µm)",
        resolution_m=20.0,
        hsi_bands=200,   # after removing water absorption bands
        lidar_channels=0,
    )

    def _preprocess(self):
        # Standard filenames used across the community
        hsi = loadmat(self.root / "Indian_pines_corrected.mat")["indian_pines_corrected"].astype(np.float32)
        gt  = loadmat(self.root / "Indian_pines_gt.mat")["indian_pines_gt"].astype(np.int32)

        # Standard fixed split: 10% train, 90% test (per class)
        tr, te = _stratified_split(gt, self.INFO.num_classes, train_ratio=0.1)
        lidar = _zero_lidar(hsi)
        return preprocess_hsi_lidar(hsi, lidar, tr, te,
                                    self.INFO.num_classes,
                                    self.patch_size, self.pca_components)


# ══════════════════════════════════════════════════════════════════
# Pavia University  (ROSIS, 2001)
# ══════════════════════════════════════════════════════════════════

class PaviaU(RSDataset):
    INFO = DatasetInfo(
        name="PaviaU",
        modalities=["hsi"],
        num_classes=9,
        class_names=[
            "Asphalt", "Meadows", "Gravel", "Trees",
            "Painted metal sheets", "Bare soil", "Bitumen",
            "Self-blocking bricks", "Shadows",
        ],
        location="Pavia, Italy",
        sensor="ROSIS (103 bands, 0.43–0.86 µm)",
        resolution_m=1.3,
        hsi_bands=103,
        lidar_channels=0,
    )

    def _preprocess(self):
        hsi = loadmat(self.root / "PaviaU.mat")["paviaU"].astype(np.float32)
        gt  = loadmat(self.root / "PaviaU_gt.mat")["paviaU_gt"].astype(np.int32)
        tr, te = _stratified_split(gt, self.INFO.num_classes, train_ratio=0.1)
        lidar = _zero_lidar(hsi)
        return preprocess_hsi_lidar(hsi, lidar, tr, te,
                                    self.INFO.num_classes,
                                    self.patch_size, self.pca_components)


# ══════════════════════════════════════════════════════════════════
# Salinas Valley  (AVIRIS, 1998)
# ══════════════════════════════════════════════════════════════════

class Salinas(RSDataset):
    INFO = DatasetInfo(
        name="Salinas",
        modalities=["hsi"],
        num_classes=16,
        class_names=[
            "Brocoli_green_weeds_1", "Brocoli_green_weeds_2",
            "Fallow", "Fallow_rough_plow", "Fallow_smooth",
            "Stubble", "Celery", "Grapes_untrained",
            "Soil_vinyard_develop", "Corn_senesced_green_weeds",
            "Lettuce_romaine_4wk", "Lettuce_romaine_5wk",
            "Lettuce_romaine_6wk", "Lettuce_romaine_7wk",
            "Vinyard_untrained", "Vinyard_vertical_trellis",
        ],
        location="Salinas Valley, California, USA",
        sensor="AVIRIS (224 bands, 0.4–2.5 µm)",
        resolution_m=3.7,
        hsi_bands=204,   # after water absorption removal
        lidar_channels=0,
    )

    def _preprocess(self):
        hsi = loadmat(self.root / "Salinas_corrected.mat")["salinas_corrected"].astype(np.float32)
        gt  = loadmat(self.root / "Salinas_gt.mat")["salinas_gt"].astype(np.int32)
        tr, te = _stratified_split(gt, self.INFO.num_classes, train_ratio=0.1)
        lidar = _zero_lidar(hsi)
        return preprocess_hsi_lidar(hsi, lidar, tr, te,
                                    self.INFO.num_classes,
                                    self.patch_size, self.pca_components)


# ══════════════════════════════════════════════════════════════════
# Berlin  (EnMAP HSI + Sentinel-1 SAR, DFC2018)
# https://github.com/danfenghong/ISPRS_S2FL
# ══════════════════════════════════════════════════════════════════

class Berlin(RSDataset):
    INFO = DatasetInfo(
        name="Berlin",
        modalities=["hsi", "sar"],
        num_classes=8,
        class_names=[
            "Forest", "Residential area", "Industrial area",
            "Low plants", "Soil", "Allotment", "Commercial area", "Water",
        ],
        location="Berlin, Germany",
        sensor="EnMAP (244 bands HSI) + Sentinel-1 (4 bands SAR)",
        resolution_m=30.0,
        hsi_bands=244,
        lidar_channels=4,   # SAR treated as "LiDAR" channel slot
    )

    def _preprocess(self):
        # rs-fusion-datasets-dist (OUC) format:
        # berlin_hsi.mat   key: berlin_hsi
        # berlin_sar.mat   key: berlin_sar
        # berlin_gt.mat    key: berlin_gt   (H×W label map, 1-indexed)
        # berlin_index.mat keys: berlin_train, berlin_test (linear pixel indices, 1-indexed)
        hsi = loadmat(self.root / "berlin_hsi.mat")["berlin_hsi"].astype(np.float32)
        sar = loadmat(self.root / "berlin_sar.mat")["berlin_sar"].astype(np.float32)
        gt  = loadmat(self.root / "berlin_gt.mat")["berlin_gt"].astype(np.int32)
        idx = loadmat(self.root / "berlin_index.mat")
        tr_linear = idx["berlin_train"].ravel().astype(np.int64) - 1
        te_linear = idx["berlin_test"].ravel().astype(np.int64) - 1
        tr, te = linear_index_to_label_maps(gt, tr_linear, te_linear)
        return preprocess_hsi_lidar(hsi, sar, tr, te,
                                    self.INFO.num_classes,
                                    self.patch_size, self.pca_components)


# ══════════════════════════════════════════════════════════════════
# WHU-Hi-LongKou  (UAV Hyperspectral, Wuhan University)
# https://github.com/nobodyczcz/WHU-Hi-Dataset
# HuggingFace: https://huggingface.co/datasets/WangHongbo/WHU-Hi
# ══════════════════════════════════════════════════════════════════

class WHUHiLongKou(RSDataset):
    INFO = DatasetInfo(
        name="WHU-Hi-LongKou",
        modalities=["hsi"],
        num_classes=9,
        class_names=[
            "Corn", "Cotton", "Sesame", "Broad-leaf soybean",
            "Narrow-leaf soybean", "Rice", "Water", "Roads and houses",
            "Mixed weed",
        ],
        location="Longkou, Hubei, China",
        sensor="Headwall Nano-Hyperspec (270 bands, 0.4–1.0 µm, UAV)",
        resolution_m=0.463,
        hsi_bands=270,
        lidar_channels=0,
    )

    def _preprocess(self):
        # Try .mat first (some sources provide this), fall back to ENVI .bsq
        mat_path = self.root / "WHU_Hi_LongKou.mat"
        if mat_path.exists():
            hsi = loadmat(mat_path)["WHU_Hi_LongKou"].astype(np.float32)
            gt  = loadmat(self.root / "WHU_Hi_LongKou_gt.mat")["WHU_Hi_LongKou_gt"].astype(np.int32)
        else:
            # ENVI .bsq format (HuggingFace danaroth/whu_hi mirror)
            try:
                import spectral
            except ImportError:
                raise ImportError(
                    "WHU-Hi-LongKou uses ENVI .bsq format. "
                    "Install with: pip install spectral"
                )
            hdr = str(self.root / "WHU-Hi-LongKou.hdr")
            hsi = np.array(spectral.open_image(hdr).load(), dtype=np.float32)
            gt_hdr = str(self.root / "WHU-Hi-LongKou_gt.hdr")
            gt = np.array(spectral.open_image(gt_hdr).load()[:, :, 0], dtype=np.int32)
        tr, te = _stratified_split(gt, self.INFO.num_classes, train_ratio=0.1)
        lidar = _zero_lidar(hsi)
        return preprocess_hsi_lidar(hsi, lidar, tr, te,
                                    self.INFO.num_classes,
                                    self.patch_size, self.pca_components)


# ══════════════════════════════════════════════════════════════════
# Shared utility: stratified train/test split from ground-truth map
# ══════════════════════════════════════════════════════════════════

def _stratified_split(gt: np.ndarray, num_classes: int,
                      train_ratio: float = 0.1,
                      min_train: int = 5,
                      seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Split a ground-truth label map into TR and TE maps.

    Args:
        gt:          (H, W) int map; values 1..num_classes (0 = background)
        num_classes: number of classes
        train_ratio: fraction of each class to use for training
        min_train:   minimum training samples per class
        seed:        random seed for reproducibility

    Returns:
        tr_map, te_map: same shape as gt, values 1..num_classes or 0
    """
    rng = np.random.default_rng(seed)
    tr = np.zeros_like(gt)
    te = np.zeros_like(gt)
    for c in range(1, num_classes + 1):
        idx = np.argwhere(gt == c)
        if len(idx) == 0:
            continue
        n_train = max(min_train, int(len(idx) * train_ratio))
        n_train = min(n_train, len(idx))
        perm = rng.permutation(len(idx))
        for i in perm[:n_train]:
            tr[idx[i, 0], idx[i, 1]] = c
        for i in perm[n_train:]:
            te[idx[i, 0], idx[i, 1]] = c
    return tr, te
