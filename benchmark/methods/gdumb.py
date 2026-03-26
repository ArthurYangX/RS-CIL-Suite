"""GDumb — Greedy Sampler and Dumb Learner (Prabhu et al., ECCV 2020).

Maintains a balanced memory budget across all seen classes.
At test time, trains a fresh model from scratch only on the stored exemplars.
No forgetting by construction — but bounded by memory quality.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .base import CILMethod, register_method
from .ncm import SimpleEncoder
from benchmark.protocols.cil import Task


@register_method("gdumb")
class GDumb(CILMethod):
    name = "GDumb"

    def __init__(self, hsi_channels, lidar_channels, num_classes, device,
                 d=128, epochs_final=100, lr=1e-3, memory_size=2000, **kwargs):
        encoder = SimpleEncoder(hsi_channels, lidar_channels, d)
        super().__init__(encoder, device, num_classes)
        self.hsi_ch = hsi_channels
        self.lidar_ch = lidar_channels
        self.d = d
        self.epochs_final = epochs_final
        self.lr = lr
        self.memory_size = memory_size
        self.head = nn.Linear(d, num_classes).to(device)

        self._buf_hsi:    list[torch.Tensor] = []
        self._buf_lidar:  list[torch.Tensor] = []
        self._buf_labels: list[torch.Tensor] = []

    def train_task(self, task: Task, train_loader: DataLoader):
        """Greedily add new class samples to a balanced memory buffer."""
        k_per_class = max(1, self.memory_size // len(self.seen_classes))

        # Trim existing classes to new budget
        counts: dict[int, int] = {}
        new_h, new_l, new_y = [], [], []
        for h, l, y in zip(self._buf_hsi, self._buf_lidar, self._buf_labels):
            c = y.item()
            if counts.get(c, 0) < k_per_class:
                new_h.append(h); new_l.append(l); new_y.append(y)
                counts[c] = counts.get(c, 0) + 1
        self._buf_hsi, self._buf_lidar, self._buf_labels = new_h, new_l, new_y

        # Add new class samples
        added: dict[int, int] = {c: 0 for c in task.global_class_ids}
        for xh, xl, y in train_loader:
            for i in range(len(y)):
                c = y[i].item()
                if c in added and added[c] < k_per_class:
                    self._buf_hsi.append(xh[i])
                    self._buf_lidar.append(xl[i])
                    self._buf_labels.append(y[i])
                    added[c] += 1

        print(f"    [GDumb] Buffer size: {len(self._buf_hsi)}")

    def after_task(self, task: Task, train_loader: DataLoader):
        """Train fresh model on entire buffer (called after each task)."""
        if not self._buf_hsi:
            return

        # Re-init model for fresh training
        from .ncm import SimpleEncoder
        self.model = SimpleEncoder(self.hsi_ch, self.lidar_ch, self.d).to(self.device)
        self.head = nn.Linear(self.d, self.num_classes_total).to(self.device)

        ds = TensorDataset(
            torch.stack(self._buf_hsi),
            torch.stack(self._buf_lidar),
            torch.stack(self._buf_labels),
        )
        loader = DataLoader(ds, batch_size=256, shuffle=True, num_workers=0)
        cids = sorted(self.seen_classes)
        g2l = {g: i for i, g in enumerate(cids)}

        self.model.train(); self.head.train()
        opt = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.head.parameters()),
            lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs_final)

        for ep in range(self.epochs_final):
            total, n = 0.0, 0
            for xh, xl, y in loader:
                xh, xl, y = xh.to(self.device), xl.to(self.device), y.to(self.device)
                logits = self.head(self.model(xh, xl))[:, cids]
                mapped = torch.tensor([g2l[yi.item()] for yi in y], device=self.device)
                loss = F.cross_entropy(logits, mapped)
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item(); n += 1
            sched.step()
            if (ep + 1) % 20 == 0:
                print(f"    [GDumb] Retrain epoch {ep+1}/{self.epochs_final}  loss={total/n:.4f}")

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
        return {
            "head": self.head.state_dict(),
            "_buf_hsi": [t.cpu() for t in self._buf_hsi],
            "_buf_lidar": [t.cpu() for t in self._buf_lidar],
            "_buf_labels": [t.cpu() for t in self._buf_labels],
        }

    def _load_method_state(self, ckpt: dict):
        if "head" in ckpt:
            self.head.load_state_dict(ckpt["head"])
        self._buf_hsi = ckpt["_buf_hsi"]
        self._buf_lidar = ckpt["_buf_lidar"]
        self._buf_labels = ckpt["_buf_labels"]
