"""Shared exemplar memory for replay-based CIL methods.

Supports 8 selection strategies:

  Feature-free (no model needed):
    - random:     Uniform random sampling
    - reservoir:  Class-balanced reservoir sampling (online)
    - ring:       FIFO ring buffer per class

  Feature-dependent (one forward pass):
    - herding:    iCaRL-style iterative closest-to-mean (default)
    - closest:    Non-iterative closest-to-mean (faster herding)
    - k_center:   Greedy coreset — maximise min-distance coverage
    - entropy:    Select highest-entropy (most uncertain) samples
    - kmeans:     K-Means clustering — one sample per centroid

Usage:
    from benchmark.utils.exemplars import ExemplarMemory, list_strategies

    print(list_strategies())       # show all available strategies
    memory = ExemplarMemory(budget=2000, strategy="herding")
    memory.update(model, hsi, lidar, labels, device)
    replay_loader = memory.get_loader(batch_size=64)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ── Strategy registry ─────────────────────────────────────────────

_STRATEGIES: dict[str, dict] = {
    "herding":   {"needs_model": True,  "cost": "medium",
                  "desc": "Iterative closest-to-mean in feature space (iCaRL)"},
    "random":    {"needs_model": False, "cost": "fast",
                  "desc": "Uniform random sampling"},
    "k_center":  {"needs_model": True,  "cost": "medium",
                  "desc": "Greedy coreset — maximise min-distance coverage"},
    "closest":   {"needs_model": True,  "cost": "fast",
                  "desc": "Non-iterative closest-to-mean (faster herding)"},
    "entropy":   {"needs_model": True,  "cost": "fast",
                  "desc": "Select highest-entropy (most uncertain) samples"},
    "kmeans":    {"needs_model": True,  "cost": "medium",
                  "desc": "K-Means clustering — one sample per centroid"},
    "reservoir": {"needs_model": False, "cost": "fast",
                  "desc": "Class-balanced reservoir sampling (online)"},
    "ring":      {"needs_model": False, "cost": "fast",
                  "desc": "FIFO ring buffer per class"},
}


def list_strategies() -> dict[str, str]:
    """Return {name: description} for all available strategies."""
    return {k: v["desc"] for k, v in _STRATEGIES.items()}


# ── Feature extraction helper ─────────────────────────────────────

@torch.no_grad()
def _extract_feats(model: nn.Module, hsi: torch.Tensor,
                   lidar: torch.Tensor, device: torch.device,
                   batch_size: int = 256) -> torch.Tensor:
    """Extract features in batches. Returns (N, d) CPU tensor."""
    model.eval()
    feats = []
    for i in range(0, hsi.shape[0], batch_size):
        h = hsi[i:i+batch_size].to(device)
        l = lidar[i:i+batch_size].to(device)
        f = model(h, l)
        feats.append(f.cpu())
    return torch.cat(feats)


# ── ExemplarMemory ────────────────────────────────────────────────

class ExemplarMemory:
    """Centralized exemplar buffer with configurable selection strategies.

    Args:
        budget:    Total number of exemplars to store.
        strategy:  Selection strategy (see ``list_strategies()``).
        per_class: If True, budget is split equally across all seen classes.
    """

    def __init__(self, budget: int = 2000, strategy: str = "herding",
                 per_class: bool = True):
        if strategy not in _STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Available: {sorted(_STRATEGIES)}"
            )
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
        head: nn.Module | None = None,
    ):
        """Update the memory with new data.

        Args:
            model:         Feature extractor (needed for feature-dependent strategies).
            hsi:           (N, C, H, W) HSI patches.
            lidar:         (N, C, H, W) LiDAR patches.
            labels:        (N,) class labels.
            device:        Torch device.
            new_class_ids: Classes to add. If None, inferred from labels.
            head:          Classifier head (needed for entropy strategy).
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
            indices = self._select(model, c_hsi, c_lid, k, device, head)
            self._store[c] = {
                "hsi": c_hsi[indices].cpu(),
                "lidar": c_lid[indices].cpu(),
            }

        # Reduce old classes to maintain budget
        for c in self.class_ids:
            if self._store[c]["hsi"].shape[0] > k:
                self._store[c]["hsi"] = self._store[c]["hsi"][:k]
                self._store[c]["lidar"] = self._store[c]["lidar"][:k]

    def _select(self, model, hsi, lidar, k, device, head=None) -> list[int]:
        """Dispatch to the configured strategy."""
        n = hsi.shape[0]
        k = min(k, n)
        if k <= 0:
            return []

        s = self.strategy
        if s == "herding":
            return _herding(model, hsi, lidar, k, device)
        elif s == "random":
            return _random(n, k)
        elif s == "k_center":
            return _k_center(model, hsi, lidar, k, device)
        elif s == "closest":
            return _closest_to_mean(model, hsi, lidar, k, device)
        elif s == "entropy":
            return _entropy(model, hsi, lidar, k, device, head)
        elif s == "kmeans":
            return _kmeans(model, hsi, lidar, k, device)
        elif s == "reservoir":
            return _random(n, k)  # offline fallback = random
        elif s == "ring":
            return list(range(max(0, n - k), n))  # last k samples (FIFO)
        else:
            raise ValueError(f"Unknown strategy: {s}")

    # ── Data access ───────────────────────────────────────────────

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

    # ── Serialization ─────────────────────────────────────────────

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


# ══════════════════════════════════════════════════════════════════
# Selection strategy implementations
# ══════════════════════════════════════════════════════════════════

def _random(n: int, k: int) -> list[int]:
    """Uniform random sampling."""
    return np.random.choice(n, min(k, n), replace=False).tolist()


@torch.no_grad()
def _herding(model, hsi, lidar, k, device) -> list[int]:
    """iCaRL herding: iteratively pick sample that moves running mean
    closest to the true class mean in feature space.

    Reference: Rebuffi et al., "iCaRL", CVPR 2017.
    """
    feats = _extract_feats(model, hsi, lidar, device)
    mean = feats.mean(0)
    selected = []
    selected_sum = torch.zeros_like(mean)
    available = set(range(feats.shape[0]))

    for _ in range(min(k, feats.shape[0])):
        best_idx, best_dist = -1, float("inf")
        for idx in available:
            new_mean = (selected_sum + feats[idx]) / (len(selected) + 1)
            dist = (new_mean - mean).pow(2).sum().item()
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        selected.append(best_idx)
        selected_sum += feats[best_idx]
        available.discard(best_idx)

    return selected


@torch.no_grad()
def _closest_to_mean(model, hsi, lidar, k, device) -> list[int]:
    """Non-iterative: pick the k samples whose features are individually
    closest to the class mean. Faster than herding, often similar quality.
    """
    feats = _extract_feats(model, hsi, lidar, device)
    mean = feats.mean(0)
    dists = (feats - mean).pow(2).sum(1)
    _, indices = dists.topk(min(k, feats.shape[0]), largest=False)
    return indices.tolist()


@torch.no_grad()
def _k_center(model, hsi, lidar, k, device) -> list[int]:
    """Greedy k-center coreset: maximise minimum distance between
    selected exemplars for maximum coverage of feature space.

    Reference: Nguyen et al., "Variational CL", ICLR 2018.
    """
    feats = _extract_feats(model, hsi, lidar, device)
    n = feats.shape[0]
    k = min(k, n)

    # Start with sample closest to mean
    mean = feats.mean(0)
    dists_to_mean = (feats - mean).pow(2).sum(1)
    first = dists_to_mean.argmin().item()

    selected = [first]
    min_dists = (feats - feats[first]).pow(2).sum(1)

    for _ in range(k - 1):
        idx = min_dists.argmax().item()
        selected.append(idx)
        new_dists = (feats - feats[idx]).pow(2).sum(1)
        min_dists = torch.minimum(min_dists, new_dists)

    return selected


@torch.no_grad()
def _entropy(model, hsi, lidar, k, device, head=None) -> list[int]:
    """Select the k samples with highest prediction entropy (most uncertain).
    If no head is provided, uses feature-norm as a proxy for uncertainty.

    Reference: Entropy-based ER (Eusipco 2020).
    """
    feats = _extract_feats(model, hsi, lidar, device)
    n = feats.shape[0]

    if head is not None:
        # Compute logits → softmax → entropy
        head.eval()
        logits_list = []
        bs = 256
        for i in range(0, n, bs):
            logits_list.append(head(feats[i:i+bs].to(device)).cpu())
        logits = torch.cat(logits_list)
        probs = F.softmax(logits, dim=1)
        entropy = -(probs * (probs + 1e-8).log()).sum(1)
    else:
        # Proxy: samples with features far from mean are more "uncertain"
        mean = feats.mean(0)
        entropy = (feats - mean).pow(2).sum(1)

    _, indices = entropy.topk(min(k, n), largest=True)
    return indices.tolist()


@torch.no_grad()
def _kmeans(model, hsi, lidar, k, device, max_iter: int = 20) -> list[int]:
    """K-Means clustering: cluster features into k groups, select the sample
    closest to each centroid. Ensures coverage of intra-class modes.

    Reference: Multi-criteria selection (Pattern Recognition, 2022).
    """
    feats = _extract_feats(model, hsi, lidar, device)
    n = feats.shape[0]
    k = min(k, n)

    # K-Means++ initialization
    centroids = [feats[np.random.randint(n)]]
    for _ in range(k - 1):
        dists = torch.stack([(feats - c).pow(2).sum(1) for c in centroids]).min(0).values
        probs = dists / (dists.sum() + 1e-8)
        idx = np.random.choice(n, p=probs.numpy())
        centroids.append(feats[idx])
    centroids = torch.stack(centroids)  # (k, d)

    # Iterate
    for _ in range(max_iter):
        # Assign
        dists = torch.cdist(feats, centroids)  # (n, k)
        assignments = dists.argmin(1)
        # Update centroids
        new_centroids = torch.zeros_like(centroids)
        for j in range(k):
            mask = assignments == j
            if mask.any():
                new_centroids[j] = feats[mask].mean(0)
            else:
                new_centroids[j] = centroids[j]
        if (new_centroids - centroids).abs().max() < 1e-6:
            break
        centroids = new_centroids

    # Select closest sample to each centroid
    dists = torch.cdist(feats, centroids)  # (n, k)
    selected = []
    used = set()
    for j in range(k):
        sorted_idx = dists[:, j].argsort()
        for idx in sorted_idx.tolist():
            if idx not in used:
                selected.append(idx)
                used.add(idx)
                break

    return selected
