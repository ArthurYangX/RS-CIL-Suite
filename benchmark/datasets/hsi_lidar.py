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
from .preprocess import preprocess_hsi_lidar


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
        modalities=["hsi", "lidar"],
        num_classes=7,
        class_names=[
            "Forest", "Residential area", "Industrial area",
            "Low plants", "Allotment", "Commercial area", "Water",
        ],
        location="Augsburg, Germany",
        sensor="DLR HySU (HSI, 180 bands) + TanDEM-X (DSM)",
        resolution_m=30.0,
        hsi_bands=180,
        lidar_channels=1,
    )

    def _preprocess(self):
        # Augsburg uses the same file layout as Trento/Houston
        hsi   = loadmat(self.root / "HSI.mat")["HSI"].astype(np.float32)
        lidar = loadmat(self.root / "LiDAR.mat")["LiDAR"].astype(np.float32)
        tr    = loadmat(self.root / "TRLabel.mat")["TRLabel"].astype(np.int32)
        te    = loadmat(self.root / "TSLabel.mat")["TSLabel"].astype(np.int32)
        return preprocess_hsi_lidar(hsi, lidar, tr, te,
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
        sensor="ITRES CASI-1500 (HSI) + Optech Titan MW (multi-spectral LiDAR)",
        resolution_m=1.0,
        hsi_bands=48,
        lidar_channels=3,   # Optech Titan = 3 wavelength channels
    )

    def _preprocess(self):
        # Houston 2018 raw data has a different layout (GeoTIFF or .mat depending on source)
        # This loader assumes data has been converted to the standard HSI.mat / LiDAR.mat layout
        hsi   = loadmat(self.root / "HSI.mat")["HSI"].astype(np.float32)
        lidar = loadmat(self.root / "LiDAR.mat")["LiDAR"].astype(np.float32)
        tr    = loadmat(self.root / "TRLabel.mat")["TRLabel"].astype(np.int32)
        te    = loadmat(self.root / "TSLabel.mat")["TSLabel"].astype(np.int32)
        return preprocess_hsi_lidar(hsi, lidar, tr, te,
                                    self.INFO.num_classes,
                                    self.patch_size, self.pca_components)
