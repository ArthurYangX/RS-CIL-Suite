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
from .preprocess import preprocess_hsi_lidar, linear_index_to_label_maps


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
        # tyust-dayu/Trento format: Italy_hsi.mat, Italy_lidar.mat, allgrd.mat
        hsi   = loadmat(self.root / "Italy_hsi.mat")["data"].astype(np.float32)
        lidar = loadmat(self.root / "Italy_lidar.mat")["data"].astype(np.float32)
        gt    = loadmat(self.root / "allgrd.mat")["mask_test"].astype(np.int32)
        # allgrd.mat contains the full (train+test) labelled map; use fixed split
        # provided by the community: split ~10% for training
        from .hsi_only import _stratified_split
        tr, te = _stratified_split(gt, self.INFO.num_classes, train_ratio=0.1)
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
        # GatorSense format: one nested struct file
        # muufl_gulfport_campus_1_hsi_220_label.mat, key='hsi'
        # hsi.Data (325×220×64), hsi.Lidar[0].z, hsi.sceneLabels.labels
        mat = loadmat(self.root / "muufl_gulfport_campus_1_hsi_220_label.mat",
                      squeeze_me=True, struct_as_record=False)
        s = mat["hsi"]
        hsi_cube = np.array(s.Data, dtype=np.float32)         # (325, 220, 64)
        lidar_ch = np.array(s.Lidar[0].z, dtype=np.float32)   # (325, 220) first return
        if lidar_ch.ndim == 2:
            lidar_ch = lidar_ch[:, :, np.newaxis]
        gt = np.array(s.sceneLabels.labels, dtype=np.int32)    # (325, 220), 1-indexed
        from .hsi_only import _stratified_split
        tr, te = _stratified_split(gt, self.INFO.num_classes, train_ratio=0.1)
        return preprocess_hsi_lidar(hsi_cube, lidar_ch, tr, te,
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
        # rs-fusion-datasets-dist (OUC) format:
        # augsburg_hsi.mat  key: augsburg_hsi
        # augsburg_sar.mat  key: augsburg_sar
        # augsburg_gt.mat   key: augsburg_gt  (H×W label map, 1-indexed)
        # augsburg_index.mat keys: augsburg_train, augsburg_test (linear pixel indices, 1-indexed)
        hsi = loadmat(self.root / "augsburg_hsi.mat")["augsburg_hsi"].astype(np.float32)
        sar = loadmat(self.root / "augsburg_sar.mat")["augsburg_sar"].astype(np.float32)
        gt  = loadmat(self.root / "augsburg_gt.mat")["augsburg_gt"].astype(np.int32)
        idx = loadmat(self.root / "augsburg_index.mat")
        tr_linear = idx["augsburg_train"].ravel().astype(np.int64) - 1  # MATLAB→0-indexed
        te_linear = idx["augsburg_test"].ravel().astype(np.int64) - 1
        tr, te = linear_index_to_label_maps(gt, tr_linear, te_linear)
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
        # rs-fusion-datasets-dist (OUC) format:
        # houston_hsi.mat   key: houston_hsi
        # houston_lidar.mat key: houston_lidar
        # houston_gt.mat    key: houston_gt  (H×W label map, 1-indexed)
        # houston_index.mat keys: houston_train, houston_test (linear pixel indices, 1-indexed)
        hsi   = loadmat(self.root / "houston_hsi.mat")["houston_hsi"].astype(np.float32)
        lidar = loadmat(self.root / "houston_lidar.mat")["houston_lidar"].astype(np.float32)
        gt    = loadmat(self.root / "houston_gt.mat")["houston_gt"].astype(np.int32)
        idx   = loadmat(self.root / "houston_index.mat")
        tr_linear = idx["houston_train"].ravel().astype(np.int64) - 1
        te_linear = idx["houston_test"].ravel().astype(np.int64) - 1
        tr, te = linear_index_to_label_maps(gt, tr_linear, te_linear)
        return preprocess_hsi_lidar(hsi, lidar, tr, te,
                                    self.INFO.num_classes,
                                    self.patch_size, self.pca_components)
