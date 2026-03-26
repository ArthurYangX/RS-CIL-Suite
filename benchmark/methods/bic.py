"""BiC — Bias Correction for Class-Incremental Learning.
(Wu et al., CVPR 2019)

Key idea: after each incremental task, the linear classifier develops a recency
bias — new-class logits are systematically larger because more of the training
data (exemplars + new data) belongs to new classes. BiC adds a per-task bias
correction layer (just two scalars: alpha, beta per task group) trained on a
small held-out validation split from old-class exemplars.

Correction: logit_corrected = alpha * logit + beta  (applied to new-class cols)

Everything else (feature extractor, head training, herding) mirrors iCaRL.
"""
from __future__ import annotations
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .base import CILMethod, register_method
from .ncm import SimpleEncoder
from benchmark.protocols.cil import Task


class _BiasLayer(nn.Module):
    """Learnable alpha/beta correction for a contiguous range of logit columns."""
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta  = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * x + self.beta


@register_method("bic")
class BiC(CILMethod):
    name = "BiC"

    def __init__(self, hsi_channels, lidar_channels, num_classes, device,
                 d=128, epochs=50, lr=1e-3, memory_size=2000, T=2.0,
                 bias_epochs=200, bias_lr=1e-3, val_ratio=0.1, **kwargs):
        encoder = SimpleEncoder(hsi_channels, lidar_channels, d)
        super().__init__(encoder, device, num_classes)
        self.d = d
        self.epochs = epochs
        self.lr = lr
        self.memory_size = memory_size
        self.T = T
        self.bias_epochs = bias_epochs
        self.bias_lr = bias_lr
        self.val_ratio = val_ratio

        self.head = nn.Linear(d, num_classes, bias=True).to(device)
        self._old_model: nn.Module | None = None
        self._old_head: nn.Linear | None = None

        # Exemplar store: {class_id: (hsi, lidar, labels)}
        self._exemplars: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        # Bias layer: one per task group (task_idx → BiasLayer)
        self._bias_layers: list[_BiasLayer] = []
        # Task class ranges: list of sorted class_id lists, one per task
        self._task_class_ranges: list[list[int]] = []

    # ── helpers ───────────────────────────────────────────────────

    def _exemplar_loader(self, exclude_val: bool = False) -> DataLoader | None:
        if not self._exemplars:
            return None
        hsi_list, lid_list, lbl_list = [], [], []
        for c, (h, l, y) in self._exemplars.items():
            if exclude_val:
                # hold out val_ratio for bias correction training
                n = max(1, int(len(h) * (1 - self.val_ratio)))
                h, l, y = h[:n], l[:n], y[:n]
            hsi_list.append(h); lid_list.append(l); lbl_list.append(y)
        ds = TensorDataset(torch.cat(hsi_list), torch.cat(lid_list), torch.cat(lbl_list))
        return DataLoader(ds, batch_size=256, shuffle=True, num_workers=0)

    def _val_loader(self) -> DataLoader | None:
        """Small validation set from old exemplars (for bias correction)."""
        if not self._exemplars:
            return None
        hsi_list, lid_list, lbl_list = [], [], []
        for c, (h, l, y) in self._exemplars.items():
            n_val = max(1, int(len(h) * self.val_ratio))
            hsi_list.append(h[-n_val:]); lid_list.append(l[-n_val:]); lbl_list.append(y[-n_val:])
        ds = TensorDataset(torch.cat(hsi_list), torch.cat(lid_list), torch.cat(lbl_list))
        return DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)

    @torch.no_grad()
    def _extract_all(self, loader: DataLoader):
        self.model.eval()
        feats, hsi_all, lid_all, lbl_all = [], [], [], []
        for xh, xl, y in loader:
            xh_d, xl_d = xh.to(self.device), xl.to(self.device)
            feats.append(self.model(xh_d, xl_d).cpu())
            hsi_all.append(xh); lid_all.append(xl); lbl_all.append(y)
        return (torch.cat(feats), torch.cat(hsi_all),
                torch.cat(lid_all), torch.cat(lbl_all))

    def _herding_select(self, feats: torch.Tensor, k: int) -> torch.Tensor:
        mean = feats.mean(0)
        selected, current_sum, remaining = [], torch.zeros_like(mean), list(range(len(feats)))
        for _ in range(min(k, len(remaining))):
            best_i, best_score = -1, float('inf')
            for idx in remaining:
                score = ((mean - (current_sum + feats[idx]) / (len(selected) + 1)) ** 2).sum().item()
                if score < best_score:
                    best_score, best_i = score, idx
            selected.append(best_i); current_sum += feats[best_i]; remaining.remove(best_i)
        return torch.tensor(selected)

    def _corrected_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply per-task bias correction layers to full logit vector."""
        if not self._bias_layers:
            return logits
        out = logits.clone()
        for i, (layer, cids) in enumerate(zip(self._bias_layers, self._task_class_ranges)):
            # skip the last (current) task — no bias layer yet during training
            out[:, cids] = layer(logits[:, cids])
        return out

    # ── lifecycle ────────────────────────────────────────────────

    def before_task(self, task: Task):
        super().before_task(task)
        if len(self.seen_classes) > len(task.global_class_ids):
            self._old_model = deepcopy(self.model).eval()
            self._old_head = deepcopy(self.head).eval()
            for p in self._old_model.parameters():
                p.requires_grad_(False)
            for p in self._old_head.parameters():
                p.requires_grad_(False)

    def train_task(self, task: Task, train_loader: DataLoader):
        # ── Stage 1: train feature extractor + head (same as iCaRL) ──
        ex_loader = self._exemplar_loader(exclude_val=True)
        all_loaders = [train_loader] + ([ex_loader] if ex_loader else [])

        self.model.train(); self.head.train()
        params = list(self.model.parameters()) + list(self.head.parameters())
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=1e-4)
        cids = sorted(self.seen_classes)

        for ep in range(self.epochs):
            total, n = 0.0, 0
            for ldr in all_loaders:
                for xh, xl, y in ldr:
                    xh, xl, y = xh.to(self.device), xl.to(self.device), y.to(self.device)
                    feat = self.model(xh, xl)
                    logits = self.head(feat)[:, cids]
                    one_hot = torch.zeros(len(y), len(cids), device=self.device)
                    for bi, yi in enumerate(y):
                        if yi.item() in cids:
                            one_hot[bi, cids.index(yi.item())] = 1.0

                    if self._old_model is not None and self._old_head is not None:
                        with torch.no_grad():
                            old_feat = self._old_model(xh, xl)
                            old_logits = self._old_head(old_feat)
                        old_classes_idx = [cids.index(c) for c in cids
                                           if c not in task.global_class_ids
                                           and c < old_logits.shape[1]]
                        if old_classes_idx:
                            old_cids = [c for c in cids
                                        if c not in task.global_class_ids
                                        and c < old_logits.shape[1]]
                            one_hot[:, old_classes_idx] = torch.sigmoid(
                                old_logits[:, old_cids]).detach()

                    loss = F.binary_cross_entropy_with_logits(logits, one_hot)
                    opt.zero_grad(); loss.backward(); opt.step()
                    total += loss.item(); n += 1

            if (ep + 1) % 10 == 0:
                print(f"    [BiC] Epoch {ep+1}/{self.epochs}  loss={total/n:.4f}")

        # ── Stage 2: train bias correction on held-out val ────────
        # (only meaningful from task 2 onward, when we have old exemplars)
        val_loader = self._val_loader()
        if val_loader is not None:
            # Add a bias layer for the current task's class range
            new_bias = _BiasLayer().to(self.device)
            self._bias_layers.append(new_bias)
            self._task_class_ranges.append(sorted(task.global_class_ids))

            # Freeze everything except bias layers
            self.model.eval(); self.head.eval()
            bias_params = [p for bl in self._bias_layers for p in bl.parameters()]
            bias_opt = torch.optim.Adam(bias_params, lr=self.bias_lr)

            for ep in range(self.bias_epochs):
                total, n = 0.0, 0
                for xh, xl, y in val_loader:
                    xh, xl, y = xh.to(self.device), xl.to(self.device), y.to(self.device)
                    with torch.no_grad():
                        feat = self.model(xh, xl)
                        raw_logits = self.head(feat)
                    corrected = self._corrected_logits(raw_logits)
                    loss = F.cross_entropy(corrected[:, cids],
                                           torch.tensor([cids.index(yi.item()) for yi in y],
                                                        device=self.device))
                    bias_opt.zero_grad(); loss.backward(); bias_opt.step()
                    total += loss.item(); n += 1

    def after_task(self, task: Task, train_loader: DataLoader):
        k_per_class = max(1, self.memory_size // len(self.seen_classes))
        for c in list(self._exemplars):
            h, l, y = self._exemplars[c]
            self._exemplars[c] = (h[:k_per_class], l[:k_per_class], y[:k_per_class])

        feats, hsi, lidar, labels = self._extract_all(train_loader)
        for c in task.global_class_ids:
            mask = labels == c
            if mask.sum() == 0:
                continue
            c_feats = feats[mask]
            idx = self._herding_select(c_feats, k_per_class)
            self._exemplars[c] = (hsi[mask][idx], lidar[mask][idx], labels[mask][idx])

    @torch.no_grad()
    def predict(self, loader):
        self.model.eval(); self.head.eval()
        cids = sorted(self.seen_classes)
        all_p, all_t = [], []
        for xh, xl, y in loader:
            xh, xl = xh.to(self.device), xl.to(self.device)
            feat = self.model(xh, xl)
            logits = self.head(feat)
            logits = self._corrected_logits(logits)
            idx = logits[:, cids].argmax(1)
            all_p.append(torch.tensor([cids[i] for i in idx.cpu().tolist()]))
            all_t.append(y)
        return torch.cat(all_p).numpy(), torch.cat(all_t).numpy()

    # ── checkpoint ──────────────────────────────────────────────
    def _method_state(self) -> dict:
        return {
            "head": self.head.state_dict(),
            "_exemplars": {c: (h.cpu(), l.cpu(), y.cpu())
                           for c, (h, l, y) in self._exemplars.items()},
            "_bias_layers": [bl.state_dict() for bl in self._bias_layers],
            "_task_class_ranges": self._task_class_ranges,
        }

    def _load_method_state(self, ckpt: dict):
        self.head.load_state_dict(ckpt["head"])
        self._exemplars = ckpt["_exemplars"]
        self._bias_layers = []
        for sd in ckpt["_bias_layers"]:
            bl = _BiasLayer().to(self.device)
            bl.load_state_dict(sd)
            self._bias_layers.append(bl)
        self._task_class_ranges = ckpt["_task_class_ranges"]
