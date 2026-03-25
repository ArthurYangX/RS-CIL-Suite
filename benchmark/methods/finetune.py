"""Fine-tune baseline — catastrophic forgetting lower bound.

Trains a linear head on top of the same SimpleEncoder.
Each new task overwrites the old head (no replay, no regularisation).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base import CILMethod, register_method
from .ncm import SimpleEncoder
from benchmark.protocols.cil import Task


@register_method("finetune")
class FineTune(CILMethod):

    name = "FineTune"

    def __init__(self, hsi_channels: int, lidar_channels: int,
                 num_classes: int, device: torch.device, d: int = 128,
                 epochs: int = 50, lr: float = 1e-3, **kwargs):
        encoder = SimpleEncoder(hsi_channels, lidar_channels, d)
        super().__init__(encoder, device, num_classes)
        self.d = d
        self.epochs = epochs
        self.lr = lr
        self.head = nn.Linear(d, num_classes).to(device)

    def train_task(self, task: Task, train_loader: DataLoader):
        self.model.train()
        self.head.train()
        params = list(self.model.parameters()) + list(self.head.parameters())
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)

        for ep in range(self.epochs):
            total, n = 0.0, 0
            for xh, xl, y in train_loader:
                xh, xl, y = xh.to(self.device), xl.to(self.device), y.to(self.device)
                logits = self.head(self.model(xh, xl))
                loss = F.cross_entropy(logits[:, self.seen_classes],
                                       self._remap(y, self.seen_classes))
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item(); n += 1
            sched.step()
            if (ep + 1) % 10 == 0:
                print(f"    [FT] Epoch {ep+1}/{self.epochs}  loss={total/n:.4f}")

    @torch.no_grad()
    def predict(self, loader: DataLoader):
        self.model.eval(); self.head.eval()
        all_preds, all_targets = [], []
        cids = sorted(self.seen_classes)
        for xh, xl, y in loader:
            xh, xl = xh.to(self.device), xl.to(self.device)
            logits = self.head(self.model(xh, xl))[:, cids]
            pred_idx = logits.argmax(1)
            preds = torch.tensor([cids[i] for i in pred_idx.cpu().tolist()])
            all_preds.append(preds); all_targets.append(y)
        return torch.cat(all_preds).numpy(), torch.cat(all_targets).numpy()

    # ── checkpoint ──────────────────────────────────────────────
    def _method_state(self) -> dict:
        return {"head": self.head.state_dict()}

    def _load_method_state(self, ckpt: dict):
        self.head.load_state_dict(ckpt["head"])

    @staticmethod
    def _remap(y: torch.Tensor, seen: list[int]) -> torch.Tensor:
        g2l = {g: i for i, g in enumerate(sorted(seen))}
        return torch.tensor([g2l[yi.item()] for yi in y], device=y.device)
