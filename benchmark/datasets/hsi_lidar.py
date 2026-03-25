"""Concrete HSI+LiDAR dataset implementations.

Currently supported:
  - Trento       (6 classes)
  - Houston2013  (15 classes)
  - MUUFL        (11 classes)
  - Houston2018  (20 classes)  ← stub, loader TBD
  - Augsburg     (7 classes)   ← stub, loader TBD
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.io import loadmat

from .base import RSDataset, DatasetInfo
from .preprocess import preprocess_hsi_lidar, index_to_label_maps


# ══════════════════════════════════════════════════════════════════
# Trento
# ══════════════════════════════════════════════════════════════════

class Trento(RSDataset):
    INFO = DatasetInfo(
        name="Trento",
        modalities=["hsi", "lidar"],
        num_classes=6,
        class_names=["Apple trees", "Buildings", "Ground", "Woods",
                     "Vineyard", "Roads"],
        location="Trento, Italy",
        sensor="AISA Eagle (HSI) + Optech ALTM 3100C (LiDAR)",
        resolution_m=1.0,
        hsi_bands=63,
        lidar_channels=1,
    )

    def _preprocess(self):
        hsi   = loadmat(self.root / "HSI.mat")["HSI"].astype(np.float32)
        lidar = loadmat(self.root / "LiDAR.mat")["LiDAR"].astype(np.float32)
        tr    = loadmat(self.root / "TRLabel.mat")["TRLabel"].astype(np.int32)
        te    = loadmat(self.root / "TSLabel.mat")["TSLabel"].astype(np.int32)
        return preprocess_hsi_lidar(hsi, lidar, tr, te,
                                    self.INFO.num_classes,
                                    self.patch_size, self.pca_components)


# ══════════════════════════════════════════════════════════════════
# Houston 2013  (GRSS DFC 2013)
# ══════════════════════════════════════════════════════════════════

class Houston2013(RSDataset):
    INFO = DatasetInfo(
        name="Houston2013",
        modalities=["hsi", "lidar"],
        num_classes=15,
        class_names=[
            "Healthy grass", "Stressed grass", "Synthetic grass",
            "Trees", "Soil", "Water", "Residential", "Commercial",
            "Road", "Highway", "Railway", "Parking lot 1",
            "Parking lot 2", "Tennis court", "Running track",
        ],
        location="Houston, TX, USA",
        sensor="ITRES-CASI 1500 (HSI) + Optech Gemini (LiDAR)",
        resolution_m=2.5,
        hsi_bands=144,
        lidar_channels=1,
    )

    def _preprocess(self):
        hsi   = loadmat(self.root / "HSI.mat")["HSI"].astype(np.float32)
        lidar = loadmat(self.root / "LiDAR.mat")["LiDAR"].astype(np.float32)
        tr    = loadmat(self.root / "TRLabel.mat")["TRLabel"].astype(np.int32)
        te    = loadmat(self.root / "TSLabel.mat")["TSLabel"].astype(np.int32)
        return preprocess_hsi_lidar(hsi, lidar, tr, te,
                                    self.INFO.num_classes,
                                    self.patch_size, self.pca_components)


# ══════════════════════════════════════════════════════════════════
# MUUFL Gulfport
# ══════════════════════════════════════════════════════════════════

class MUUFL(RSDataset):
    INFO = DatasetInfo(
        name="MUUFL",
        modalities=["hsi", "lidar"],
        num_classes=11,
        class_names=[
            "Trees", "Mostly grass", "Mixed ground surface", "Dirt and sand",
            "Road", "Water", "Buildings shadow", "Buildings",
            "Sidewalk", "Yellow curb", "Cloth panels",
        ],
        location="Long Beach, Mississippi, USA",
        sensor="ITRES CASI-1500 (HSI) + Optech Gemini (LiDAR)",
        resolution_m=1.0,
        hsi_bands=64,
        lidar_channels=2,
    )

    def _preprocess(self):
        hsi   = loadmat(self.root / "HSI.mat")["HSI"].astype(np.float32)
        lidar = loadmat(self.root / "LiDAR.mat")["LiDAR"].astype(np.float32)
        tr    = loadmat(self.root / "muufl_tr.mat")["training_map"].astype(np.int32)
        te    = loadmat(self.root / "muufl_ts.mat")["testing_map"].astype(np.int32)
        return preprocess_hsi_lidar(hsi, lidar, tr, te,
                                    self.INFO.num_classes,
                                    self.patch_size, self.pca_components)


# ══════════════════════════════════════════════════════════════════
# Augsburg  (DLR HySU + TanDEM-X)
# ══════════════════════════════════════════════════════════════════

class Augsburg(RSDataset):
    INFO = DatasetInfo(
        name="Augsburg",
        modalities=["hsi", "sar"],
        num_classes=8,
        class_names=[
            "Forest", "Residential area", "Industrial area",
            "Low plants", "Soil", "Allotment", "Commercial area", "Water",
        ],
        location="Augsburg, Germany",
        sensor="HySpex (HSI, 180 bands) + Sentinel-1 SAR + DSM",
        resolution_m=30.0,
        hsi_bands=180,
        lidar_channels=4,   # SAR+DSM combined channels
    )

    def _preprocess(self):
        # rs-fusion-datasets-dist format: augsburg_hsi.mat, augsburg_sar.mat,
        # augsburg_gt.mat (label per pixel), augsburg_index.mat (tr/te split coords)
        data = loadmat(self.root / "augsburg_index.mat")
        hsi  = loadmat(self.root / "augsburg_hsi.mat")["augsburg_hsi"].astype(np.float32)
        sar  = loadmat(self.root / "augsburg_sar.mat")["augsburg_sar"].astype(np.float32)
        gt_raw = loadmat(self.root / "augsburg_gt.mat")
        gt_key = [k for k in gt_raw if not k.startswith("_")][0]
        gt   = gt_raw[gt_key].astype(np.int32)    # (H, W) full label map
        # Index mat has coordinate arrays for tr/te splits
        idx_key = [k for k in data if not k.startswith("_")][0]
        idx_data = data[idx_key]
        # Expected: struct with tr_idx, te_idx fields; fall back to label map split
        try:
            tr_idx = idx_data["tr"][0, 0].ravel().astype(np.int32) - 1  # MATLAB 1-indexed
            te_idx = idx_data["te"][0, 0].ravel().astype(np.int32) - 1
            labeled = np.argwhere(gt > 0)
            gt_vals = gt[labeled[:, 0], labeled[:, 1]]
            tr, te = index_to_label_maps(gt_vals, tr_idx, te_idx, labeled,
                                         hsi.shape[0], hsi.shape[1])
        except Exception:
            # Fallback: stratified split
            from .hsi_only import _stratified_split
            tr, te = _stratified_split(gt, self.INFO.num_classes, train_ratio=0.1)
        return preprocess_hsi_lidar(hsi, sar, tr, te,
                                    self.INFO.num_classes,
                                    self.patch_size, self.pca_components)


# ══════════════════════════════════════════════════════════════════
# Houston 2018  (GRSS DFC 2018)
# ══════════════════════════════════════════════════════════════════

class Houston2018(RSDataset):
    INFO = DatasetInfo(
        name="Houston2018",
        modalities=["hsi", "lidar"],
        num_classes=20,
        class_names=[
            "Healthy grass", "Stressed grass", "Artificial turf",
            "Evergreen trees", "Deciduous trees", "Bare earth",
            "Water", "Residential buildings", "Non-residential buildings",
            "Roads", "Sidewalks", "Crosswalks", "Major thoroughfares",
            "Highways", "Railways", "Paved parking lots",
            "Unpaved parking lots", "Cars", "Trains", "Stadium seats",
        ],
        location="Houston, TX, USA",
        sensor="ITRES CASI-1500 (HSI) + Optech Titan MW (LiDAR)",
        resolution_m=1.0,
        hsi_bands=50,
        lidar_channels=1,
    )

    def _preprocess(self):
        # rs-fusion-datasets-dist format: houston_hsi.mat, houston_lidar.mat,
        # houston_gt.mat (H×W label map), houston_index.mat (tr/te coordinate split)
        hsi   = loadmat(self.root / "houston_hsi.mat")["houston_hsi"].astype(np.float32)
        lidar = loadmat(self.root / "houston_lidar.mat")["houston_lidar"].astype(np.float32)
        gt    = loadmat(self.root / "houston_gt.mat")
        gt_key = [k for k in gt if not k.startswith("_")][0]
        gt    = gt[gt_key].astype(np.int32)
        idx   = loadmat(self.root / "houston_index.mat")
        idx_key = [k for k in idx if not k.startswith("_")][0]
        idx_data = idx[idx_key]
        try:
            tr_idx = idx_data["tr"][0, 0].ravel().astype(np.int32) - 1
            te_idx = idx_data["te"][0, 0].ravel().astype(np.int32) - 1
            labeled = np.argwhere(gt > 0)
            gt_vals = gt[labeled[:, 0], labeled[:, 1]]
            tr, te = index_to_label_maps(gt_vals, tr_idx, te_idx, labeled,
                                         hsi.shape[0], hsi.shape[1])
        except Exception:
            from .hsi_only import _stratified_split
            tr, te = _stratified_split(gt, self.INFO.num_classes, train_ratio=0.1)
        return preprocess_hsi_lidar(hsi, lidar, tr, te,
                                    self.INFO.num_classes,
                                    self.patch_size, self.pca_components)
