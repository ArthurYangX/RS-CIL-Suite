"""iCaRL — Class-Incremental Learning via Classification, Exemplar and Distillation.
(Rebuffi et al., CVPR 2017)

Key components:
  1. Feature extractor trained with binary cross-entropy + KD distillation
  2. Herding-based exemplar selection (K exemplars per class, K=budget//n_classes)
  3. Nearest-mean-of-exemplars (NME) classifier at test time
"""
from __future__ import annotations
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .base import CILMethod
from .ncm import SimpleEncoder
from benchmark.protocols.cil import Task


class iCaRL(CILMethod):
    name = "iCaRL"

    def __init__(self, hsi_channels, lidar_channels, num_classes, device,
                 d=128, epochs=50, lr=1e-3, memory_size=2000, T=2.0):
        encoder = SimpleEncoder(hsi_channels, lidar_channels, d)
        super().__init__(encoder, device, num_classes)
        self.d = d
        self.epochs = epochs
        self.lr = lr
        self.memory_size = memory_size
        self.T = T

        self.head = nn.Linear(d, num_classes).to(device)
        self._old_model: nn.Module | None = None

        # Exemplar store: {class_id: (hsi, lidar, labels)} tensors on CPU
        self._exemplars: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        # Class means in feature space: {class_id: (d,) tensor}
        self._class_means: dict[int, torch.Tensor] = {}

    # ── helpers ───────────────────────────────────────────────────

    def _exemplar_loader(self) -> DataLoader | None:
        if not self._exemplars:
            return None
        hsi_list, lid_list, lbl_list = [], [], []
        for h, l, y in self._exemplars.values():
            hsi_list.append(h); lid_list.append(l); lbl_list.append(y)
        ds = TensorDataset(torch.cat(hsi_list), torch.cat(lid_list), torch.cat(lbl_list))
        return DataLoader(ds, batch_size=256, shuffle=True, num_workers=0)

    @torch.no_grad()
    def _extract_all(self, loader: DataLoader):
        """Return (feats, hsi, lidar, labels) for an entire loader."""
        self.model.eval()
        feats, hsi_all, lid_all, lbl_all = [], [], [], []
        for xh, xl, y in loader:
            xh_d, xl_d = xh.to(self.device), xl.to(self.device)
            feats.append(self.model(xh_d, xl_d).cpu())
            hsi_all.append(xh); lid_all.append(xl); lbl_all.append(y)
        return (torch.cat(feats), torch.cat(hsi_all),
                torch.cat(lid_all), torch.cat(lbl_all))

    def _herding_select(self, feats: torch.Tensor, k: int) -> torch.Tensor:
        """Select k exemplars via herding (greedy class-mean matching)."""
        mean = feats.mean(0)
        selected = []
        current_sum = torch.zeros_like(mean)
        remaining = list(range(len(feats)))
        for _ in range(min(k, len(remaining))):
            best_i, best_score = -1, float('inf')
            for idx in remaining:
                new_sum = current_sum + feats[idx]
                score = ((mean - new_sum / (len(selected) + 1)) ** 2).sum().item()
                if score < best_score:
                    best_score, best_i = score, idx
            selected.append(best_i)
            current_sum += feats[best_i]
            remaining.remove(best_i)
        return torch.tensor(selected)

    # ── lifecycle ────────────────────────────────────────────────

    def before_task(self, task: Task):
        super().before_task(task)
        if len(self.seen_classes) > len(task.global_class_ids):
            self._old_model = deepcopy(self.model).eval()
            for p in self._old_model.parameters():
                p.requires_grad_(False)

    def train_task(self, task: Task, train_loader: DataLoader):
        # Merge current task data with exemplars
        ex_loader = self._exemplar_loader()
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

                    # Binary cross entropy over all seen classes (iCaRL style)
                    logits = self.head(feat)[:, cids]
                    one_hot = torch.zeros(len(y), len(cids), device=self.device)
                    for bi, yi in enumerate(y):
                        if yi.item() in cids:
                            one_hot[bi, cids.index(yi.item())] = 1.0

                    # KD targets from old model for old classes
                    if self._old_model is not None:
                        with torch.no_grad():
                            old_logits = self.head(self._old_model(xh, xl))
                        old_classes_idx = [cids.index(c) for c in cids
                                           if c not in task.global_class_ids
                                           and c < old_logits.shape[1]]
                        if old_classes_idx:
                            old_soft = torch.sigmoid(old_logits[:, old_classes_idx])
                            one_hot[:, old_classes_idx] = old_soft.detach()

                    loss = F.binary_cross_entropy_with_logits(logits, one_hot)
                    opt.zero_grad(); loss.backward(); opt.step()
                    total += loss.item(); n += 1

            if (ep + 1) % 10 == 0:
                print(f"    [iCaRL] Epoch {ep+1}/{self.epochs}  loss={total/n:.4f}")

    def after_task(self, task: Task, train_loader: DataLoader):
        """Update exemplar set and class means using herding."""
        k_per_class = max(1, self.memory_size // len(self.seen_classes))

        # Re-balance existing exemplars
        for c in list(self._exemplars):
            h, l, y = self._exemplars[c]
            self._exemplars[c] = (h[:k_per_class], l[:k_per_class], y[:k_per_class])

        # Add new classes from current task
        feats, hsi, lidar, labels = self._extract_all(train_loader)
        for c in task.global_class_ids:
            mask = labels == c
            if mask.sum() == 0:
                continue
            c_feats = feats[mask]
            idx = self._herding_select(c_feats, k_per_class)
            self._exemplars[c] = (hsi[mask][idx], lidar[mask][idx], labels[mask][idx])

        # Update class means for NME classifier
        self._update_class_means()

    @torch.no_grad()
    def _update_class_means(self):
        self.model.eval()
        self._class_means = {}
        for c, (h, l, y) in self._exemplars.items():
            feats = []
            bs = 256
            for i in range(0, len(h), bs):
                xh = h[i:i+bs].to(self.device)
                xl = l[i:i+bs].to(self.device)
                feats.append(self.model(xh, xl).cpu())
            self._class_means[c] = torch.cat(feats).mean(0)

    @torch.no_grad()
    def predict(self, loader):
        self.model.eval()
        if not self._class_means:
            all_t = [y for _, _, y in loader]
            t = torch.cat(all_t).numpy()
            return np.zeros_like(t), t

        cids = sorted(self._class_means.keys())
        mean_mat = torch.stack([self._class_means[c] for c in cids]).to(self.device)
        mean_mat = F.normalize(mean_mat, dim=1)

        all_p, all_t = [], []
        for xh, xl, y in loader:
            xh, xl = xh.to(self.device), xl.to(self.device)
            f = F.normalize(self.model(xh, xl), dim=1)
            sims = f @ mean_mat.t()
            idx = sims.argmax(1)
            all_p.append(torch.tensor([cids[i] for i in idx.cpu().tolist()]))
            all_t.append(y)
        return torch.cat(all_p).numpy(), torch.cat(all_t).numpy()
