"""Template for adding a new CIL method to RS-CIL-Suite.

How to add your method:
  1. Copy this file → benchmark/methods/your_method.py
  2. Rename the class and update @register_method("your_method")
  3. Implement train_task() and predict()
  4. (Optional) Create benchmark/configs/your_method.yaml for hyperparams
  5. Run: python benchmark/run.py --method your_method --protocol B1

The benchmark runner will:
  - Auto-discover your method via the @register_method decorator
  - Pass constructor kwargs from YAML config (training.* and method.*)
  - Call the lifecycle hooks in order for each task
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base import CILMethod, register_method
from benchmark.models import build_backbone
from benchmark.protocols.cil import Task


# Uncomment the decorator to register (commented out so the template
# doesn't pollute the registry).
# @register_method("template")
class TemplateMethod(CILMethod):
    """Minimal CIL method example — sequential fine-tuning with a linear head."""

    name = "Template"

    def __init__(self, hsi_channels, lidar_channels, num_classes, device,
                 backbone="simple_encoder", d=128, epochs=50, lr=1e-3, **kwargs):
        backbone = build_backbone(backbone, hsi_ch=hsi_channels, lidar_ch=lidar_channels, d=d)
        super().__init__(backbone, device, num_classes)
        self.d = d
        self.epochs = epochs
        self.lr = lr
        self.head = nn.Linear(d, num_classes).to(device)

    # ── Required: train on one task ─────────────────────────────

    def train_task(self, task: Task, train_loader: DataLoader):
        self.model.train()
        self.head.train()
        params = list(self.model.parameters()) + list(self.head.parameters())
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)

        gids = sorted(self.seen_classes)
        g2l = {g: i for i, g in enumerate(gids)}

        for ep in range(self.epochs):
            total, n = 0.0, 0
            for xh, xl, y in train_loader:
                xh, xl, y = xh.to(self.device), xl.to(self.device), y.to(self.device)
                feat = self.model(xh, xl)
                logits = self.head(feat)[:, :len(gids)]
                mapped = torch.tensor([g2l[yi.item()] for yi in y],
                                      device=self.device)
                loss = F.cross_entropy(logits, mapped)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item()
                n += 1
            sched.step()
            # Log to wandb if callback is set
            self._log({"train/loss": total / n, "epoch": ep,
                        "task_id": task.task_id})

    # ── Required: predict on test set ───────────────────────────

    @torch.no_grad()
    def predict(self, loader: DataLoader):
        self.model.eval()
        self.head.eval()
        gids = sorted(self.seen_classes)
        all_preds, all_targets = [], []
        for xh, xl, y in loader:
            xh, xl = xh.to(self.device), xl.to(self.device)
            logits = self.head(self.model(xh, xl))[:, :len(gids)]
            pred_idx = logits.argmax(1).cpu()
            preds = torch.tensor([gids[i] for i in pred_idx])
            all_preds.append(preds)
            all_targets.append(y)
        return torch.cat(all_preds).numpy(), torch.cat(all_targets).numpy()

    # ── Optional: checkpoint hooks ──────────────────────────────

    def _method_state(self) -> dict:
        return {"head_state_dict": self.head.state_dict()}

    def _load_method_state(self, ckpt: dict):
        if "head_state_dict" in ckpt:
            self.head.load_state_dict(ckpt["head_state_dict"])
