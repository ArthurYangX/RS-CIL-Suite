"""Base class for all CIL methods in the RS-CIL benchmark."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from benchmark.datasets.base import PatchDataset
from benchmark.protocols.cil import Task


class CILMethod(ABC):
    """Abstract base for CIL methods.

    Lifecycle per task:
        method.before_task(task)
        method.train_task(task, train_loader)
        method.after_task(task, train_loader)
        preds = method.predict(test_loader)
    """

    name: str = "BaseMethod"

    def __init__(self, model: nn.Module, device: torch.device,
                 num_classes_total: int):
        self.model = model.to(device)
        self.device = device
        self.num_classes_total = num_classes_total
        self.seen_classes: List[int] = []

    # ── Lifecycle hooks ───────────────────────────────────────────

    def before_task(self, task: Task):
        """Called before training on a new task (freeze layers, allocate heads, etc.)."""
        self.seen_classes = self.seen_classes + task.global_class_ids

    @abstractmethod
    def train_task(self, task: Task, train_loader: DataLoader):
        """Train on one task."""
        ...

    def after_task(self, task: Task, train_loader: DataLoader):
        """Called after training (update memory, compute Fisher, etc.)."""
        pass

    # ── Inference ─────────────────────────────────────────────────

    @abstractmethod
    @torch.no_grad()
    def predict(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        """Return (predictions, targets) arrays over the full loader."""
        ...

    # ── Helpers ───────────────────────────────────────────────────

    def _extract_features(self, loader: DataLoader
                          ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract (hsi, lidar, labels) tensors from a loader (no grad)."""
        hsi_list, lid_list, lbl_list = [], [], []
        self.model.eval()
        with torch.no_grad():
            for xh, xl, y in loader:
                hsi_list.append(xh)
                lid_list.append(xl)
                lbl_list.append(y)
        return (torch.cat(hsi_list), torch.cat(lid_list), torch.cat(lbl_list))
