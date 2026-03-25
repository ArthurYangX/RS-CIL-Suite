"""Shared exemplar memory for replay-based CIL methods.

Supports multiple selection strategies: herding, random, k-center.
Used by iCaRL, LUCIR, PODNet, BiC, DER++, GDumb, and future methods.

Usage:
    from benchmark.utils.exemplars import ExemplarMemory

    memory = ExemplarMemory(budget=2000, strategy="herding")
    memory.update(model, hsi, lidar, labels, device)
    replay_loader = memory.get_loader(batch_size=64)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class ExemplarMemory:
    """Centralized exemplar buffer with configurable selection strategies.

    Args:
        budget:    Total number of exemplars to store.
        strategy:  Selection strategy: "herding" | "random" | "k_center".
        per_class: If True, budget is split equally across all seen classes.
                   If False, budget is a hard total limit with balanced allocation.
    """

    def __init__(self, budget: int = 2000, strategy: str = "herding",
                 per_class: bool = True):
        self.budget = budget
        self.strategy = strategy
        self.per_class = per_class

        # Storage: {class_id: {"hsi": Tensor, "lidar": Tensor}}
        self._store: dict[int, dict[str, torch.Tensor]] = {}

    @property
    def n_classes(self) -> int:
        return len(self._store)

    @property
    def n_exemplars(self) -> int:
        return sum(d["hsi"].shape[0] for d in self._store.values())

    @property
    def class_ids(self) -> list[int]:
        return sorted(self._store.keys())

    def k_per_class(self, n_classes: int | None = None) -> int:
        """Number of exemplars per class given current/total class count."""
        nc = n_classes or max(self.n_classes, 1)
        return self.budget // nc

    def update(
        self,
        model: nn.Module,
        hsi: torch.Tensor,
        lidar: torch.Tensor,
        labels: torch.Tensor,
        device: torch.device,
        new_class_ids: list[int] | None = None,
    ):
        """Update the memory with new data.

        Selects exemplars for new classes and optionally reduces existing
        classes to maintain the budget.

        Args:
            model:         Feature extractor (used for herding).
            hsi:           (N, C, H, W) HSI patches.
            lidar:         (N, C, H, W) LiDAR patches.
            labels:        (N,) class labels.
            device:        Torch device.
            new_class_ids: Classes to add. If None, inferred from labels.
        """
        if new_class_ids is None:
            new_class_ids = sorted(set(labels.tolist()))

        all_classes = sorted(set(self.class_ids) | set(new_class_ids))
        k = self.k_per_class(len(all_classes))

        # Select exemplars for new classes
        for c in new_class_ids:
            mask = labels == c
            c_hsi = hsi[mask]
            c_lid = lidar[mask]

            if self.strategy == "herding":
                indices = self._herding_select(model, c_hsi, c_lid, k, device)
            elif self.strategy == "random":
                indices = self._random_select(c_hsi.shape[0], k)
            elif self.strategy == "k_center":
                indices = self._k_center_select(model, c_hsi, c_lid, k, device)
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")

            self._store[c] = {
                "hsi": c_hsi[indices].cpu(),
                "lidar": c_lid[indices].cpu(),
            }

        # Reduce old classes to maintain budget
        for c in self.class_ids:
            if self._store[c]["hsi"].shape[0] > k:
                self._store[c]["hsi"] = self._store[c]["hsi"][:k]
                self._store[c]["lidar"] = self._store[c]["lidar"][:k]

    def get_data(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return all stored exemplars as (hsi, lidar, labels) tensors."""
        if not self._store:
            raise ValueError("ExemplarMemory is empty")

        hsi_list, lid_list, lbl_list = [], [], []
        for c in sorted(self._store):
            n = self._store[c]["hsi"].shape[0]
            hsi_list.append(self._store[c]["hsi"])
            lid_list.append(self._store[c]["lidar"])
            lbl_list.append(torch.full((n,), c, dtype=torch.long))

        return torch.cat(hsi_list), torch.cat(lid_list), torch.cat(lbl_list)

    def get_loader(self, batch_size: int = 64, shuffle: bool = True) -> DataLoader:
        """Return a DataLoader over all stored exemplars."""
        hsi, lidar, labels = self.get_data()
        ds = TensorDataset(hsi, lidar, labels)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    def state_dict(self) -> dict:
        """Serialize for checkpointing."""
        return {
            "budget": self.budget,
            "strategy": self.strategy,
            "store": {c: {k: v.cpu() for k, v in d.items()}
                      for c, d in self._store.items()},
        }

    def load_state_dict(self, state: dict):
        """Restore from checkpoint."""
        self.budget = state.get("budget", self.budget)
        self.strategy = state.get("strategy", self.strategy)
        self._store = {c: {k: v for k, v in d.items()}
                       for c, d in state.get("store", {}).items()}

    # ── Selection strategies ──────────────────────────────────────

    @staticmethod
    @torch.no_grad()
    def _herding_select(
        model: nn.Module,
        hsi: torch.Tensor,
        lidar: torch.Tensor,
        k: int,
        device: torch.device,
    ) -> list[int]:
        """Herding: select exemplars closest to class mean in feature space."""
        model.eval()
        # Extract features in batches to avoid OOM
        feats = []
        bs = 256
        for i in range(0, hsi.shape[0], bs):
            h = hsi[i:i+bs].to(device)
            l = lidar[i:i+bs].to(device)
            f = model(h, l)
            feats.append(f.cpu())
        feats = torch.cat(feats)  # (N, d)

        mean = feats.mean(0)
        selected = []
        selected_sum = torch.zeros_like(mean)
        available = set(range(feats.shape[0]))

        for _ in range(min(k, feats.shape[0])):
            # Pick sample that moves running mean closest to true mean
            best_idx, best_dist = -1, float("inf")
            for idx in available:
                new_sum = selected_sum + feats[idx]
                new_mean = new_sum / (len(selected) + 1)
                dist = (new_mean - mean).pow(2).sum().item()
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            selected.append(best_idx)
            selected_sum += feats[best_idx]
            available.discard(best_idx)

        return selected

    @staticmethod
    def _random_select(n: int, k: int) -> list[int]:
        """Random: uniformly sample k indices from n."""
        k = min(k, n)
        return np.random.choice(n, k, replace=False).tolist()

    @staticmethod
    @torch.no_grad()
    def _k_center_select(
        model: nn.Module,
        hsi: torch.Tensor,
        lidar: torch.Tensor,
        k: int,
        device: torch.device,
    ) -> list[int]:
        """K-center greedy: maximise minimum distance between selected exemplars."""
        model.eval()
        feats = []
        bs = 256
        for i in range(0, hsi.shape[0], bs):
            h = hsi[i:i+bs].to(device)
            l = lidar[i:i+bs].to(device)
            f = model(h, l)
            feats.append(f.cpu())
        feats = torch.cat(feats)
        n = feats.shape[0]
        k = min(k, n)

        # Start with sample closest to mean
        mean = feats.mean(0)
        dists = (feats - mean).pow(2).sum(1)
        first = dists.argmin().item()

        selected = [first]
        min_dists = (feats - feats[first]).pow(2).sum(1)

        for _ in range(k - 1):
            # Pick farthest point from any selected
            idx = min_dists.argmax().item()
            selected.append(idx)
            new_dists = (feats - feats[idx]).pow(2).sum(1)
            min_dists = torch.minimum(min_dists, new_dists)

        return selected
