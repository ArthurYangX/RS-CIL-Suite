"""Joint Training — upper bound baseline.

Stores ALL training data seen so far and retrains from scratch each task.
Not a practical CIL method, but gives the oracle upper bound.
"""
from __future__ import annotations
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

from .base import CILMethod, register_method
from .ncm import SimpleEncoder
from benchmark.protocols.cil import Task


@register_method("joint")
class JointTraining(CILMethod):
    name = "Joint"

    def __init__(self, hsi_channels, lidar_channels, num_classes, device,
                 d=128, epochs=50, lr=1e-3, **kwargs):
        encoder = SimpleEncoder(hsi_channels, lidar_channels, d)
        super().__init__(encoder, device, num_classes)
        self.d = d
        self.epochs = epochs
        self.lr = lr
        self.head = nn.Linear(d, num_classes).to(device)
        self._all_datasets: list = []   # accumulate all task datasets

    def train_task(self, task: Task, train_loader: DataLoader):
        # Store this task's dataset
        self._all_datasets.append(train_loader.dataset)
        joint_ds = ConcatDataset(self._all_datasets)
        joint_loader = DataLoader(joint_ds, batch_size=256, shuffle=True, num_workers=0)

        self.model.train(); self.head.train()
        params = list(self.model.parameters()) + list(self.head.parameters())
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        cids = sorted(self.seen_classes)
        g2l = {g: i for i, g in enumerate(cids)}

        for ep in range(self.epochs):
            total, n = 0.0, 0
            for xh, xl, y in joint_loader:
                xh, xl, y = xh.to(self.device), xl.to(self.device), y.to(self.device)
                logits = self.head(self.model(xh, xl))[:, cids]
                mapped = torch.tensor([g2l[yi.item()] for yi in y], device=self.device)
                loss = F.cross_entropy(logits, mapped)
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item(); n += 1
            sched.step()
            if (ep + 1) % 10 == 0:
                print(f"    [Joint] Epoch {ep+1}/{self.epochs}  loss={total/n:.4f}")

    @torch.no_grad()
    def predict(self, loader):
        self.model.eval(); self.head.eval()
        cids = sorted(self.seen_classes)
        all_p, all_t = [], []
        for xh, xl, y in loader:
            xh, xl = xh.to(self.device), xl.to(self.device)
            logits = self.head(self.model(xh, xl))[:, cids]
            idx = logits.argmax(1)
            all_p.append(torch.tensor([cids[i] for i in idx.cpu().tolist()]))
            all_t.append(y)
        return torch.cat(all_p).numpy(), torch.cat(all_t).numpy()

    # ── checkpoint ──────────────────────────────────────────────
    def _method_state(self) -> dict:
        return {"head": self.head.state_dict()}

    def _load_method_state(self, ckpt: dict):
        self.head.load_state_dict(ckpt["head"])
