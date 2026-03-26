"""ACIL — Analytic Class-Incremental Learning (Zhuang et al., NeurIPS 2022).

Key idea: replace gradient-based training of the linear head with a closed-form
recursive least-squares update. No exemplars, no forgetting, mathematically equivalent
to joint training on all seen data.

The feature extractor is frozen (pre-trained or trained on task 0 only).
Only the linear readout is updated analytically.
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


@register_method("acil")
class ACIL(CILMethod):
    name = "ACIL"

    def __init__(self, hsi_channels, lidar_channels, num_classes, device,
                 backbone="simple_encoder", d=128, epochs_base=50, lr=1e-3, ridge=1.0, **kwargs):
        encoder = build_backbone(backbone, hsi_ch=hsi_channels, lidar_ch=lidar_channels, d=d)
        super().__init__(encoder, device, num_classes)
        self.d = d
        self.epochs_base = epochs_base
        self.lr = lr
        self.ridge = ridge

        # Analytic state: R = (F^T F + λI)^{-1}, accumulated feature-label products
        self._R: torch.Tensor | None = None          # (d, d)
        self._FtY: torch.Tensor | None = None        # (d, num_classes)
        self._W: torch.Tensor | None = None          # (d, num_classes) — analytic weights
        self._frozen = False

    def train_task(self, task: Task, train_loader: DataLoader):
        if not self._frozen:
            # Task 0: train feature extractor with supervised loss
            self.model.train()
            head = nn.Linear(self.d, len(task.global_class_ids)).to(self.device)
            params = list(self.model.parameters()) + list(head.parameters())
            opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=1e-4)
            g2l = {g: i for i, g in enumerate(sorted(task.global_class_ids))}
            for ep in range(self.epochs_base):
                total, n = 0.0, 0
                for xh, xl, y in train_loader:
                    xh, xl, y = xh.to(self.device), xl.to(self.device), y.to(self.device)
                    logits = head(self.model(xh, xl))
                    mapped = torch.tensor([g2l[yi.item()] for yi in y], device=self.device)
                    loss = F.cross_entropy(logits, mapped)
                    opt.zero_grad(); loss.backward(); opt.step()
                    total += loss.item(); n += 1
                if (ep + 1) % 10 == 0:
                    print(f"    [ACIL] Base epoch {ep+1}/{self.epochs_base}  loss={total/n:.4f}")
            self._frozen = True

        # Analytic update: accumulate F^T F and F^T Y
        self._accumulate(train_loader, task)

    def _accumulate(self, loader: DataLoader, task: Task):
        """Accumulate sufficient statistics for recursive ridge regression."""
        self.model.eval()
        feats, labels = [], []
        with torch.no_grad():
            for xh, xl, y in loader:
                xh, xl = xh.to(self.device), xl.to(self.device)
                feats.append(self.model(xh, xl).cpu())
                labels.append(y)
        F_mat = torch.cat(feats).float()    # (N, d)
        y_all = torch.cat(labels).long()    # (N,)

        # One-hot encode with global class IDs
        Y = torch.zeros(len(y_all), self.num_classes_total)
        for i, yi in enumerate(y_all):
            Y[i, yi.item()] = 1.0

        # Recursive update of (F^T F) and (F^T Y)
        FtF = F_mat.t() @ F_mat             # (d, d)
        FtY = F_mat.t() @ Y                 # (d, num_classes)

        if self._R is None:
            # First task: initialise with ridge regularisation
            A = FtF + self.ridge * torch.eye(self.d)
            self._R   = torch.linalg.inv(A)
            self._FtY = FtY
        else:
            # Recursive update: R_new = (A_old_inv + FtF_new)^{-1}
            # Ridge was already added in the first task's A; do NOT
            # re-add it here (that would double/triple the regularisation)
            A_new = torch.linalg.inv(
                torch.linalg.inv(self._R) + FtF
            )
            self._R   = A_new
            self._FtY = self._FtY + FtY

        # Compute analytic weights
        self._W = self._R @ self._FtY       # (d, num_classes)

    @torch.no_grad()
    def predict(self, loader):
        if self._W is None:
            all_t = [y for _, _, y in loader]
            t = torch.cat(all_t).numpy()
            return np.zeros_like(t), t

        self.model.eval()
        W = self._W.to(self.device)         # (d, num_classes)
        cids = sorted(self.seen_classes)

        all_p, all_t = [], []
        for xh, xl, y in loader:
            xh, xl = xh.to(self.device), xl.to(self.device)
            f = self.model(xh, xl)           # (B, d)
            logits = f @ W                   # (B, num_classes)
            idx = logits[:, cids].argmax(1)
            all_p.append(torch.tensor([cids[i] for i in idx.cpu().tolist()]))
            all_t.append(y)
        return torch.cat(all_p).numpy(), torch.cat(all_t).numpy()

    # ── checkpoint ──────────────────────────────────────────────
    def _method_state(self) -> dict:
        state = {}
        if self._R is not None:
            state["_R"] = self._R.cpu()
        if self._FtY is not None:
            state["_FtY"] = self._FtY.cpu()
        if self._W is not None:
            state["_W"] = self._W.cpu()
        state["_frozen"] = self._frozen
        return state

    def _load_method_state(self, ckpt: dict):
        self._R = ckpt.get("_R")
        self._FtY = ckpt.get("_FtY")
        self._W = ckpt.get("_W")
        self._frozen = ckpt.get("_frozen", False)
