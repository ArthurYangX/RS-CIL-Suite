"""WA — Weight Aligning (Zhao et al., CVPR 2020).

Key idea: After standard fine-tuning with exemplar replay, the classifier
develops a bias towards new classes because they have more training data.
WA corrects this by normalizing the weight norms of the classifier's
old-class and new-class columns to be equal.

This is a simple but strong baseline that demonstrates:
  1. Old-model snapshot (via base class helper)
  2. Exemplar memory (via shared ExemplarMemory)
  3. Standard KD loss + post-hoc weight alignment
"""
from __future__ import annotations
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

from .base import CILMethod, register_method
from .ncm import SimpleEncoder
from benchmark.protocols.cil import Task
from benchmark.utils.exemplars import ExemplarMemory


@register_method("wa")
class WA(CILMethod):
    """Weight Aligning with exemplar replay + KD.

    Training:
      - CE loss on new + exemplar data
      - KD loss from frozen old model on old-class logits
      - After training: align weight norms between old and new class columns

    Inference:
      - Standard softmax classifier over all seen classes
    """

    name = "WA"

    def __init__(self, hsi_channels, lidar_channels, num_classes, device,
                 d=128, epochs=50, lr=1e-3,
                 memory_size=2000, T=2.0, exemplar_strategy="herding",
                 **kwargs):
        encoder = SimpleEncoder(hsi_channels, lidar_channels, d)
        super().__init__(encoder, device, num_classes)
        self.d = d
        self.epochs = epochs
        self.lr = lr
        self.T = T
        self.head = nn.Linear(d, num_classes).to(device)
        self.memory = ExemplarMemory(budget=memory_size, strategy=exemplar_strategy)
        self._old_model = None
        self._old_head = None
        self._old_n_classes = 0

    def before_task(self, task: Task):
        super().before_task(task)
        if task.task_id > 0:
            # Snapshot old model for KD
            self._snapshot_old_model()
            self._old_head = deepcopy(self.head)
            self._old_head.eval()
            for p in self._old_head.parameters():
                p.requires_grad_(False)
            self._old_n_classes = len(
                [c for c in self.seen_classes if c not in task.global_class_ids]
            )

    def train_task(self, task: Task, train_loader: DataLoader):
        self.model.train()
        self.head.train()

        # Merge with exemplars if available
        if self.memory.n_classes > 0:
            replay_loader = self.memory.get_loader(batch_size=64)
            loaders = [train_loader, replay_loader]
        else:
            loaders = [train_loader]

        params = list(self.model.parameters()) + list(self.head.parameters())
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)

        gids = sorted(self.seen_classes)
        g2l = {g: i for i, g in enumerate(gids)}
        n_seen = len(gids)

        for ep in range(self.epochs):
            total, n = 0.0, 0
            for loader in loaders:
                for xh, xl, y in loader:
                    xh = xh.to(self.device)
                    xl = xl.to(self.device)
                    y = y.to(self.device)

                    feat = self.model(xh, xl)
                    logits = self.head(feat)[:, :n_seen]
                    mapped = torch.tensor([g2l[yi.item()] for yi in y],
                                          device=self.device)
                    loss_ce = F.cross_entropy(logits, mapped)

                    # KD loss on old-class logits
                    loss_kd = torch.tensor(0.0, device=self.device)
                    if self._old_model is not None and self._old_n_classes > 0:
                        with torch.no_grad():
                            old_feat = self._old_model(xh, xl)
                            old_logits = self._old_head(old_feat)[:, :self._old_n_classes]
                        new_logits_old = logits[:, :self._old_n_classes]
                        loss_kd = F.kl_div(
                            F.log_softmax(new_logits_old / self.T, dim=1),
                            F.softmax(old_logits / self.T, dim=1),
                            reduction="batchmean",
                        ) * (self.T ** 2)

                    loss = loss_ce + loss_kd
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    total += loss.item()
                    n += 1

            sched.step()
            self._log({"train/loss": total / max(n, 1), "epoch": ep,
                        "task_id": task.task_id})
            if (ep + 1) % 10 == 0:
                print(f"    [WA] Epoch {ep+1}/{self.epochs}  "
                      f"loss={total/max(n,1):.4f}")

        # Weight Aligning: normalize old vs new class weight norms
        self._weight_align(task)

    def _weight_align(self, task: Task):
        """Post-hoc weight alignment between old and new class columns."""
        if self._old_n_classes == 0:
            return

        gids = sorted(self.seen_classes)
        with torch.no_grad():
            w = self.head.weight[:len(gids)]
            # Norms of old-class and new-class weight vectors
            old_norms = w[:self._old_n_classes].norm(dim=1)
            new_norms = w[self._old_n_classes:].norm(dim=1)

            if new_norms.numel() > 0 and old_norms.numel() > 0:
                gamma = old_norms.mean() / (new_norms.mean() + 1e-8)
                w[self._old_n_classes:] *= gamma
                print(f"    [WA] Weight aligned: gamma={gamma:.4f}")

    def after_task(self, task: Task, train_loader: DataLoader):
        # Update exemplar memory
        hsi_all, lid_all, lbl_all = self._extract_features(train_loader)
        self.memory.update(
            self.model, hsi_all, lid_all, lbl_all, self.device,
            new_class_ids=task.global_class_ids,
        )

    @torch.no_grad()
    def predict(self, loader: DataLoader):
        self.model.eval()
        self.head.eval()
        gids = sorted(self.seen_classes)
        g2l = {g: i for i, g in enumerate(gids)}
        n_seen = len(gids)

        all_preds, all_targets = [], []
        for xh, xl, y in loader:
            xh, xl = xh.to(self.device), xl.to(self.device)
            feat = self.model(xh, xl)
            logits = self.head(feat)[:, :n_seen]
            pred_idx = logits.argmax(1).cpu()
            preds = torch.tensor([gids[i] for i in pred_idx.tolist()])
            all_preds.append(preds)
            all_targets.append(y)
        return torch.cat(all_preds).numpy(), torch.cat(all_targets).numpy()

    # ── Checkpoint ────────────────────────────────────────────────

    def _method_state(self) -> dict:
        state = {
            "head_state_dict": self.head.state_dict(),
            "memory": self.memory.state_dict(),
            "_old_n_classes": self._old_n_classes,
        }
        return state

    def _load_method_state(self, ckpt: dict):
        if "head_state_dict" in ckpt:
            self.head.load_state_dict(ckpt["head_state_dict"])
        if "memory" in ckpt:
            self.memory.load_state_dict(ckpt["memory"])
        self._old_n_classes = ckpt.get("_old_n_classes", 0)
