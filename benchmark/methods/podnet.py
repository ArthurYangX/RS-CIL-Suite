"""PODNet — Pooled Outputs Distillation Network (Douillard et al., ECCV 2020).

Key ideas:
  1. Pooled intermediate feature distillation (spatial + channel pooling at every layer)
  2. Large-scale class cosine classifier (NCA loss)
  3. Balanced exemplar replay
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


@register_method("podnet")
class PODNet(CILMethod):
    name = "PODNet"

    def __init__(self, hsi_channels, lidar_channels, num_classes, device,
                 d=128, epochs=50, lr=1e-3,
                 memory_size=2000, pod_lambda=1.0, T=2.0, **kwargs):
        encoder = SimpleEncoder(hsi_channels, lidar_channels, d)
        super().__init__(encoder, device, num_classes)
        self.d = d
        self.epochs = epochs
        self.lr = lr
        self.memory_size = memory_size
        self.pod_lambda = pod_lambda
        self.T = T

        # Cosine classifier (weights = unit vectors per class)
        self.cls_weights = nn.Parameter(
            torch.randn(num_classes, d, device=device))
        nn.init.xavier_uniform_(self.cls_weights.unsqueeze(0))

        self._old_model: nn.Module | None = None
        self._exemplars: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def _cosine_logits(self, feat: torch.Tensor, class_ids: list[int],
                       scale: float = 10.0) -> torch.Tensor:
        w = F.normalize(self.cls_weights[class_ids], dim=1)
        f = F.normalize(feat, dim=1)
        return scale * (f @ w.t())

    def _exemplar_loader(self):
        if not self._exemplars:
            return None
        h, l, y = zip(*[(v[0], v[1], v[2]) for v in self._exemplars.values()])
        ds = TensorDataset(torch.cat(h), torch.cat(l), torch.cat(y))
        return DataLoader(ds, batch_size=256, shuffle=True, num_workers=0)

    def before_task(self, task: Task):
        super().before_task(task)
        if len(self.seen_classes) > len(task.global_class_ids):
            self._old_model = deepcopy(self.model).eval()
            for p in self._old_model.parameters():
                p.requires_grad_(False)

    def train_task(self, task: Task, train_loader: DataLoader):
        ex_loader = self._exemplar_loader()
        all_loaders = [train_loader] + ([ex_loader] if ex_loader else [])

        self.model.train()
        params = list(self.model.parameters()) + [self.cls_weights]
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=1e-4)
        cids = sorted(self.seen_classes)
        g2l = {g: i for i, g in enumerate(cids)}

        for ep in range(self.epochs):
            total, n = 0.0, 0
            for ldr in all_loaders:
                for xh, xl, y in ldr:
                    xh, xl, y = xh.to(self.device), xl.to(self.device), y.to(self.device)
                    feat = self.model(xh, xl)
                    logits = self._cosine_logits(feat, cids)
                    mapped = torch.tensor([g2l[yi.item()] for yi in y], device=self.device)
                    loss_ce = F.cross_entropy(logits, mapped)

                    # POD distillation on frozen old model features
                    loss_pod = torch.tensor(0.0, device=self.device)
                    if self._old_model is not None:
                        with torch.no_grad():
                            f_old = self._old_model(xh, xl)
                        # Spatial pooling distillation (simplified: L2 on normalised features)
                        loss_pod = F.mse_loss(F.normalize(feat, dim=1),
                                              F.normalize(f_old.detach(), dim=1))

                    loss = loss_ce + self.pod_lambda * loss_pod
                    opt.zero_grad(); loss.backward(); opt.step()
                    total += loss.item(); n += 1

            if (ep + 1) % 10 == 0:
                print(f"    [PODNet] Epoch {ep+1}/{self.epochs}  loss={total/n:.4f}")

    def _herding_select(self, feats: torch.Tensor, k: int) -> torch.Tensor:
        """Select k exemplars via herding (greedy class-mean approximation)."""
        mean = feats.mean(0)
        selected, current_sum = [], torch.zeros_like(mean)
        remaining = list(range(len(feats)))
        for _ in range(min(k, len(remaining))):
            scores = [
                ((mean - (current_sum + feats[i]) / (len(selected) + 1)) ** 2).sum().item()
                for i in remaining
            ]
            best = remaining[int(np.argmin(scores))]
            selected.append(best)
            current_sum += feats[best]
            remaining.remove(best)
        return torch.tensor(selected)

    def after_task(self, task: Task, train_loader: DataLoader):
        k_per_class = max(1, self.memory_size // len(self.seen_classes))
        # Trim existing exemplars to new budget
        for c in list(self._exemplars):
            h, l, y = self._exemplars[c]
            self._exemplars[c] = (h[:k_per_class], l[:k_per_class], y[:k_per_class])
        # Collect all patches for new task classes, then apply herding
        self.model.eval()
        class_patches: dict[int, tuple] = {}
        with torch.no_grad():
            for xh, xl, y in train_loader:
                for c in task.global_class_ids:
                    mask = y == c
                    if mask.sum() == 0 or c in self._exemplars:
                        continue
                    xh_c, xl_c, y_c = xh[mask], xl[mask], y[mask]
                    if c not in class_patches:
                        class_patches[c] = ([], [], [])
                    class_patches[c][0].append(xh_c)
                    class_patches[c][1].append(xl_c)
                    class_patches[c][2].append(y_c)
            for c, (hs, ls, ys) in class_patches.items():
                xh_c = torch.cat(hs); xl_c = torch.cat(ls); y_c = torch.cat(ys)
                feats = self.model(xh_c.to(self.device), xl_c.to(self.device)).cpu()
                idx = self._herding_select(feats, k_per_class)
                self._exemplars[c] = (xh_c[idx], xl_c[idx], y_c[idx])

    @torch.no_grad()
    def predict(self, loader):
        self.model.eval()
        cids = sorted(self.seen_classes)
        all_p, all_t = [], []
        for xh, xl, y in loader:
            xh, xl = xh.to(self.device), xl.to(self.device)
            feat = self.model(xh, xl)
            logits = self._cosine_logits(feat, cids)
            idx = logits.argmax(1)
            all_p.append(torch.tensor([cids[i] for i in idx.cpu().tolist()]))
            all_t.append(y)
        return torch.cat(all_p).numpy(), torch.cat(all_t).numpy()

    # ── checkpoint ──────────────────────────────────────────────
    def _method_state(self) -> dict:
        return {
            "cls_weights": self.cls_weights.data.cpu(),
            "_exemplars": {c: (h.cpu(), l.cpu(), y.cpu())
                           for c, (h, l, y) in self._exemplars.items()},
        }

    def _load_method_state(self, ckpt: dict):
        self.cls_weights.data.copy_(ckpt["cls_weights"])
        self._exemplars = ckpt["_exemplars"]
