"""Smoke tests for all 15 CIL methods: 2-task train + predict cycle."""
import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from benchmark.methods import get_method_registry
from benchmark.protocols.cil import Task


def _make_task(task_id, class_ids, global_class_ids, ds_name="Synth"):
    return Task(task_id=task_id, dataset_name=ds_name,
                class_ids=class_ids, global_class_ids=global_class_ids)


def _make_loader(hsi, lidar, labels, class_ids, batch_size=32):
    mask = torch.zeros(len(labels), dtype=torch.bool)
    for c in class_ids:
        mask |= labels == c
    ds = TensorDataset(hsi[mask], lidar[mask], labels[mask])
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


# Methods that are expected to work with this synthetic setup
# (all 15, but with minimal epochs for speed)
_METHODS = list(get_method_registry().keys())


@pytest.fixture(scope="module")
def synth():
    """Module-scoped synthetic data for speed."""
    torch.manual_seed(42)
    np.random.seed(42)
    n_per_class, n_classes = 30, 6
    hsi_ch, lid_ch, patch = 36, 1, 7
    N = n_per_class * n_classes
    hsi = torch.randn(N, hsi_ch, patch, patch)
    lidar = torch.randn(N, lid_ch, patch, patch)
    labels = torch.arange(n_classes).repeat_interleave(n_per_class)
    return hsi, lidar, labels, hsi_ch, lid_ch, n_classes


@pytest.mark.parametrize("method_name", _METHODS)
def test_method_2task_smoke(method_name, synth):
    """Each method must survive: before_task → train_task → after_task → predict."""
    hsi, lidar, labels, hsi_ch, lid_ch, n_classes = synth
    device = torch.device("cpu")

    registry = get_method_registry()
    method = registry[method_name](
        hsi_channels=hsi_ch, lidar_channels=lid_ch,
        num_classes=n_classes, device=device,
        epochs=2, lr=1e-3,  # minimal epochs for speed
    )

    task0 = _make_task(0, [0, 1, 2], [0, 1, 2])
    task1 = _make_task(1, [3, 4, 5], [3, 4, 5])

    loader0 = _make_loader(hsi, lidar, labels, [0, 1, 2])
    loader1 = _make_loader(hsi, lidar, labels, [3, 4, 5])
    test_loader = _make_loader(hsi, lidar, labels, list(range(n_classes)))

    # Task 0
    method.before_task(task0)
    method.train_task(task0, loader0)
    method.after_task(task0, loader0)
    preds0, targets0 = method.predict(test_loader)
    assert preds0.shape == targets0.shape
    assert len(preds0) > 0

    # Task 1
    method.before_task(task1)
    method.train_task(task1, loader1)
    method.after_task(task1, loader1)
    preds1, targets1 = method.predict(test_loader)
    assert preds1.shape == targets1.shape
    assert len(preds1) > 0
