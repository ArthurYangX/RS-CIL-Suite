"""Shared fixtures for RS-CIL-Bench tests."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

# Ensure benchmark package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def dummy_data():
    """Synthetic 2-task dataset: 6 classes, 50 samples each."""
    np.random.seed(0)
    torch.manual_seed(0)
    n_per_class = 50
    n_classes = 6
    hsi_ch, lid_ch, patch = 36, 1, 7
    N = n_per_class * n_classes

    hsi = torch.randn(N, hsi_ch, patch, patch)
    lidar = torch.randn(N, lid_ch, patch, patch)
    labels = torch.arange(n_classes).repeat_interleave(n_per_class)

    return {
        "hsi": hsi, "lidar": lidar, "labels": labels,
        "hsi_ch": hsi_ch, "lid_ch": lid_ch,
        "n_classes": n_classes, "n_per_class": n_per_class,
    }


@pytest.fixture
def make_loader(dummy_data):
    """Create a DataLoader for a subset of classes."""
    def _make(class_ids: list[int], batch_size: int = 32):
        d = dummy_data
        mask = torch.zeros(len(d["labels"]), dtype=torch.bool)
        for c in class_ids:
            mask |= d["labels"] == c
        ds = TensorDataset(d["hsi"][mask], d["lidar"][mask], d["labels"][mask])
        return DataLoader(ds, batch_size=batch_size, shuffle=False)
    return _make
