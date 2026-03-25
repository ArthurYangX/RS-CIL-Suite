"""Base class for all CIL methods in the RS-CIL benchmark.

Adding a new method
-------------------
1. Create ``benchmark/methods/your_method.py``.
2. Subclass :class:`CILMethod` and decorate with ``@register_method("your_method")``.
3. Implement :meth:`train_task` and :meth:`predict` (required).
4. Optionally override :meth:`before_task`, :meth:`after_task`,
   :meth:`_method_state`, :meth:`_load_method_state`.
5. (Optional) Add ``benchmark/configs/your_method.yaml`` for hyperparams.
6. Run: ``python benchmark/run.py --method your_method --protocol B1``
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from benchmark.datasets.base import PatchDataset
from benchmark.protocols.cil import Task


# ── Method registry ───────────────────────────────────────────────

_METHOD_REGISTRY: dict[str, type] = {}


def register_method(name: str):
    """Class decorator to register a CIL method.

    Example::

        @register_method("ewc")
        class EWC(CILMethod):
            ...
    """
    def wrapper(cls):
        if name in _METHOD_REGISTRY:
            raise ValueError(f"Method '{name}' already registered "
                             f"({_METHOD_REGISTRY[name].__name__})")
        _METHOD_REGISTRY[name] = cls
        return cls
    return wrapper


def get_method_registry() -> dict[str, type]:
    """Return a copy of the method registry."""
    return dict(_METHOD_REGISTRY)


# ── Base class ────────────────────────────────────────────────────

class CILMethod(ABC):
    """Abstract base for CIL methods.

    Lifecycle per task::

        method.before_task(task)          # prepare for new task
        method.train_task(task, loader)   # train on task data
        method.after_task(task, loader)   # post-training (exemplars, Fisher, …)
        preds, targets = method.predict(test_loader)  # inference

    Constructor args passed by the benchmark runner:
        - ``hsi_channels``, ``lidar_channels``, ``num_classes``, ``device``
          are always provided.
        - Additional kwargs come from the YAML config system (training.*
          and method.* sections).  Accept ``**kwargs`` in your ``__init__``
          to silently ignore unknown config keys.
    """

    name: str = "BaseMethod"

    def __init__(self, model: nn.Module, device: torch.device,
                 num_classes_total: int):
        self.model = model.to(device)
        self.device = device
        self.num_classes_total = num_classes_total
        self.seen_classes: List[int] = []
        self.log_fn: Optional[Callable[[dict, Optional[int]], None]] = None

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

    # ── Knowledge Distillation helpers ─────────────────────────

    def _snapshot_old_model(self):
        """Create a frozen copy of the current model for knowledge distillation.

        After calling this, ``self._old_model`` holds a detached, eval-mode
        copy of ``self.model``.  All KD methods (LwF, iCaRL, PODNet, BiC, …)
        need this — using this shared helper eliminates boilerplate.
        """
        from copy import deepcopy
        self._old_model = deepcopy(self.model)
        self._old_model.eval()
        for p in self._old_model.parameters():
            p.requires_grad_(False)

    # ── Logging ─────────────────────────────────────────────────

    def _log(self, metrics: dict, step: int | None = None):
        """Log metrics via callback (set by runner when --wandb is active)."""
        if self.log_fn is not None:
            self.log_fn(metrics, step)

    # ── Checkpointing ──────────────────────────────────────────

    def save_checkpoint(self, path: str | Path, task_id: int):
        """Save model + method state to a .pt file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "model_state_dict": self.model.state_dict(),
            "seen_classes": self.seen_classes,
            "task_id": task_id,
            "method_name": self.name,
        }
        ckpt.update(self._method_state())
        torch.save(ckpt, path)

    def load_checkpoint(self, path: str | Path) -> dict:
        """Restore model + method state from a .pt file."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.seen_classes = ckpt["seen_classes"]
        self._load_method_state(ckpt)
        return ckpt

    def _method_state(self) -> dict:
        """Override to save method-specific state (exemplars, Fisher, …)."""
        return {}

    def _load_method_state(self, ckpt: dict):
        """Override to restore method-specific state."""
        pass

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
