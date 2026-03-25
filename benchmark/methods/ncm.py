"""Nearest Class Mean (NCM) — exemplar-free upper-bound baseline.

Trains a simple CNN feature extractor per task (fine-tune style),
then classifies via cosine distance to stored class prototypes.
This is the 'Frozen+NCM' style baseline — no forgetting by design
(frozen feature extractor, prototype stored for all seen classes).
"""
from __future__ import annotations
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base import CILMethod, register_method
from benchmark.protocols.cil import Task


# SimpleEncoder moved to benchmark.models — re-exported here for backward compat
from benchmark.models import SimpleEncoder  # noqa: F401


@register_method("ncm")
class NCMMethod(CILMethod):
    """Frozen backbone NCM.

    Strategy:
      - Task 0: train encoder from scratch on first task classes.
      - Tasks 1+: encoder is frozen; only update prototypes.
      - Prediction: cosine NCM over all stored prototypes.
    """

    name = "NCM"

    def __init__(self, hsi_channels: int, lidar_channels: int,
                 num_classes: int, device: torch.device, d: int = 128,
                 epochs: int = 50, lr: float = 1e-3, **kwargs):
        model = SimpleEncoder(hsi_channels, lidar_channels, d)
        super().__init__(model, device, num_classes)
        self.d = d
        self.epochs = epochs
        self.lr = lr
        self.prototypes: dict[int, torch.Tensor] = {}   # gid → (d,)
        self._frozen = False

    def before_task(self, task: Task):
        super().before_task(task)

    def train_task(self, task: Task, train_loader: DataLoader):
        if self._frozen:
            # Just update prototypes, no training
            self._update_prototypes(train_loader)
            return

        # Train encoder on first task
        self.model.train()
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        gids = sorted(task.global_class_ids)
        gid_to_idx = {g: i for i, g in enumerate(gids)}

        for ep in range(self.epochs):
            total, n = 0.0, 0
            for xh, xl, y in train_loader:
                xh, xl, y = xh.to(self.device), xl.to(self.device), y.to(self.device)
                feat = self.model(xh, xl)
                # Build prototype-based logits (online, per batch)
                proto_mat = torch.stack([
                    feat[y == g].mean(0) if (y == g).any()
                    else torch.zeros(self.d, device=self.device)
                    for g in gids
                ])  # (K, d)
                logits = feat @ proto_mat.t() * 10.0
                mapped = torch.tensor([gid_to_idx[yi.item()] for yi in y],
                                      device=self.device)
                loss = F.cross_entropy(logits, mapped)
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item(); n += 1
            sched.step()
            if (ep + 1) % 10 == 0:
                print(f"    [NCM] Epoch {ep+1}/{self.epochs}  loss={total/n:.4f}")

    def after_task(self, task: Task, train_loader: DataLoader):
        self._frozen = True
        self._update_prototypes(train_loader)

    @torch.no_grad()
    def _update_prototypes(self, loader: DataLoader):
        self.model.eval()
        feats_by_class: dict[int, list] = {}
        for xh, xl, y in loader:
            xh, xl = xh.to(self.device), xl.to(self.device)
            f = self.model(xh, xl)
            for feat, label in zip(f, y):
                c = label.item()
                feats_by_class.setdefault(c, []).append(feat.cpu())
        for c, fs in feats_by_class.items():
            self.prototypes[c] = torch.stack(fs).mean(0)

    @torch.no_grad()
    def predict(self, loader: DataLoader):
        if not self.prototypes:
            all_t = [y for _, _, y in loader]
            t = torch.cat(all_t).numpy()
            return np.zeros_like(t), t

        self.model.eval()
        cids = sorted(self.prototypes.keys())
        proto_mat = torch.stack([self.prototypes[c] for c in cids]).to(self.device)

        all_preds, all_targets = [], []
        for xh, xl, y in loader:
            xh, xl = xh.to(self.device), xl.to(self.device)
            f = self.model(xh, xl)
            sims = f @ proto_mat.t()
            pred_idx = sims.argmax(1)
            preds = torch.tensor([cids[i] for i in pred_idx.cpu().tolist()])
            all_preds.append(preds)
            all_targets.append(y)

        return torch.cat(all_preds).numpy(), torch.cat(all_targets).numpy()

    # ── checkpoint ──────────────────────────────────────────────
    def _method_state(self) -> dict:
        return {
            "prototypes": {k: v.cpu() for k, v in self.prototypes.items()},
            "_frozen": self._frozen,
        }

    def _load_method_state(self, ckpt: dict):
        self.prototypes = ckpt["prototypes"]
        self._frozen = ckpt["_frozen"]
