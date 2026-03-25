"""LwF — Learning without Forgetting (Li & Hoiem, ECCV 2016 / TPAMI 2018).

Uses knowledge distillation from a frozen snapshot of the old model
to preserve old-task predictions while learning new classes.
"""
from __future__ import annotations
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base import CILMethod
from .ncm import SimpleEncoder
from benchmark.protocols.cil import Task


class LwF(CILMethod):
    name = "LwF"

    def __init__(self, hsi_channels, lidar_channels, num_classes, device,
                 d=128, epochs=50, lr=1e-3, T=2.0, lwf_lambda=1.0):
        encoder = SimpleEncoder(hsi_channels, lidar_channels, d)
        super().__init__(encoder, device, num_classes)
        self.d = d
        self.epochs = epochs
        self.lr = lr
        self.T = T
        self.lwf_lambda = lwf_lambda
        self.head = nn.Linear(d, num_classes).to(device)
        self._old_model: nn.Module | None = None
        self._old_classes: list[int] = []

    def before_task(self, task: Task):
        super().before_task(task)
        # Snapshot old model before training new task
        if self._old_classes:
            self._old_model = deepcopy(self.model).eval()
            for p in self._old_model.parameters():
                p.requires_grad_(False)

    def train_task(self, task: Task, train_loader: DataLoader):
        self.model.train(); self.head.train()
        params = list(self.model.parameters()) + list(self.head.parameters())
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=1e-4)
        cids = sorted(self.seen_classes)
        g2l = {g: i for i, g in enumerate(cids)}

        for ep in range(self.epochs):
            total, n = 0.0, 0
            for xh, xl, y in train_loader:
                xh, xl, y = xh.to(self.device), xl.to(self.device), y.to(self.device)
                feat = self.model(xh, xl)
                logits = self.head(feat)[:, cids]
                mapped = torch.tensor([g2l[yi.item()] for yi in y], device=self.device)
                loss_ce = F.cross_entropy(logits, mapped)

                # KD on old classes
                loss_kd = torch.tensor(0.0, device=self.device)
                if self._old_model is not None and self._old_classes:
                    with torch.no_grad():
                        old_feat = self._old_model(xh, xl)
                    old_logits = self.head(old_feat)[:, self._old_classes]
                    new_logits = self.head(feat)[:, self._old_classes]
                    loss_kd = F.kl_div(
                        F.log_softmax(new_logits / self.T, dim=1),
                        F.softmax(old_logits.detach() / self.T, dim=1),
                        reduction='batchmean'
                    ) * (self.T ** 2)

                loss = loss_ce + self.lwf_lambda * loss_kd
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item(); n += 1

            if (ep + 1) % 10 == 0:
                print(f"    [LwF] Epoch {ep+1}/{self.epochs}  loss={total/n:.4f}")

    def after_task(self, task: Task, train_loader: DataLoader):
        self._old_classes = list(self.seen_classes)

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
