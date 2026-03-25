"""Shared training utilities for CIL methods.

These are convenience helpers that methods can optionally adopt.
Existing methods continue to work without changes.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def build_optimizer(
    params,
    name: str = "adamw",
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    **kwargs,
) -> torch.optim.Optimizer:
    """Create an optimizer by name.

    Args:
        params:       Model parameters (iterable or param groups).
        name:         One of "adamw", "adam", "sgd".
        lr:           Learning rate.
        weight_decay: Weight decay (L2 penalty).
    """
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        momentum = kwargs.get("momentum", 0.9)
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay,
                               momentum=momentum)
    else:
        raise ValueError(f"Unknown optimizer '{name}'. Available: adamw, adam, sgd")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    name: str = "cosine",
    T_max: int = 50,
    **kwargs,
):
    """Create a learning rate scheduler by name.

    Args:
        optimizer: The optimizer to schedule.
        name:      One of "cosine", "step", "none".
        T_max:     Period for cosine annealing.
    """
    name = name.lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    elif name == "step":
        step_size = kwargs.get("step_size", 30)
        gamma = kwargs.get("gamma", 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size,
                                                gamma=gamma)
    elif name == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler '{name}'. Available: cosine, step, none")


def remap_labels(y: torch.Tensor, seen_classes: list[int]) -> torch.Tensor:
    """Map global class IDs to contiguous 0..N-1 indices.

    Args:
        y:            Labels tensor with global class IDs.
        seen_classes: Sorted list of all seen global class IDs.

    Returns:
        Tensor with remapped labels (0-indexed).
    """
    g2l = {g: i for i, g in enumerate(sorted(seen_classes))}
    return torch.tensor([g2l[yi.item()] for yi in y], device=y.device)
