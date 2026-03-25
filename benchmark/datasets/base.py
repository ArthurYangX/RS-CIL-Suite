"""Base class for all RS-CIL benchmark datasets."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class DatasetInfo:
    name: str
    modalities: List[str]          # e.g. ["hsi", "lidar"]
    num_classes: int
    class_names: List[str]
    location: str
    sensor: str
    resolution_m: float            # GSD in metres
    hsi_bands: Optional[int] = None
    lidar_channels: Optional[int] = None


class PatchDataset(Dataset):
    """Generic patch dataset returned by RSDataset loaders.

    Each sample: (hsi_patch, lidar_patch, label)
      hsi_patch   : (C_hsi, H, W) float32  — may be zeros if no HSI
      lidar_patch : (C_lid, H, W) float32  — may be zeros if no LiDAR
      label       : int64  (0-indexed)
    """

    def __init__(
        self,
        hsi: np.ndarray,       # (N, C, H, W)
        lidar: np.ndarray,     # (N, C, H, W)
        labels: np.ndarray,    # (N,)
    ):
        self.hsi    = torch.from_numpy(hsi).float()
        self.lidar  = torch.from_numpy(lidar).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.hsi[idx], self.lidar[idx], self.labels[idx]

    def subset(self, class_ids: List[int]) -> "PatchDataset":
        """Return a new PatchDataset containing only the given classes."""
        mask = torch.zeros(len(self.labels), dtype=torch.bool)
        for c in class_ids:
            mask |= self.labels == c
        idx = mask.nonzero(as_tuple=True)[0]
        new = PatchDataset.__new__(PatchDataset)
        new.hsi    = self.hsi[idx]
        new.lidar  = self.lidar[idx]
        new.labels = self.labels[idx]
        return new


class RSDataset(ABC):
    """Abstract base class for all RS-CIL datasets.

    Subclasses must implement `load()` which returns
    (train_dataset, test_dataset, info).
    """

    INFO: DatasetInfo  # class-level metadata

    def __init__(self, root: str | Path, patch_size: int = 7, pca_components: int = 36):
        self.root = Path(root)
        self.patch_size = patch_size
        self.pca_components = pca_components
        self._train: Optional[PatchDataset] = None
        self._test:  Optional[PatchDataset] = None

    # ── Public API ────────────────────────────────────────────────

    @property
    def train(self) -> PatchDataset:
        if self._train is None:
            self._train, self._test = self._load_and_cache()
        return self._train

    @property
    def test(self) -> PatchDataset:
        if self._test is None:
            self._train, self._test = self._load_and_cache()
        return self._test

    @property
    def info(self) -> DatasetInfo:
        return self.INFO

    @property
    def num_classes(self) -> int:
        return self.INFO.num_classes

    @property
    def class_names(self) -> List[str]:
        return self.INFO.class_names

    # ── Internal ──────────────────────────────────────────────────

    def _load_and_cache(self) -> Tuple[PatchDataset, PatchDataset]:
        import hashlib, os
        cache_dir = self.root / ".cache"
        cache_dir.mkdir(exist_ok=True)
        key = hashlib.md5(
            f"pca{self.pca_components}_patch{self.patch_size}".encode()
        ).hexdigest()[:12]
        cache_file = cache_dir / f"benchmark_{key}.npz"

        if cache_file.exists():
            print(f"[CACHE] {self.INFO.name}: loading from {cache_file}")
            d = np.load(cache_file, allow_pickle=False)
            train_ds = PatchDataset(d["x_train_hsi"], d["x_train_lidar"], d["y_train"])
            test_ds  = PatchDataset(d["x_test_hsi"],  d["x_test_lidar"],  d["y_test"])
            return train_ds, test_ds

        print(f"[LOAD] {self.INFO.name}: preprocessing from {self.root}")
        x_tr_hsi, x_tr_lid, y_tr, x_te_hsi, x_te_lid, y_te = self._preprocess()

        np.savez_compressed(
            cache_file,
            x_train_hsi=x_tr_hsi, x_train_lidar=x_tr_lid, y_train=y_tr,
            x_test_hsi=x_te_hsi,  x_test_lidar=x_te_lid,  y_test=y_te,
        )
        print(f"[CACHE] {self.INFO.name}: saved to {cache_file}")

        train_ds = PatchDataset(x_tr_hsi, x_tr_lid, y_tr)
        test_ds  = PatchDataset(x_te_hsi, x_te_lid, y_te)
        return train_ds, test_ds

    @abstractmethod
    def _preprocess(self) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray,  # train hsi, lidar, labels
        np.ndarray, np.ndarray, np.ndarray,  # test  hsi, lidar, labels
    ]:
        """Load raw data, apply PCA + normalisation + patch extraction."""
        ...
