"""DER++ — Dark Experience Replay (Buzzega et al., NeurIPS 2020).

Stores (input, logits) pairs as exemplars.
Loss = CE(new) + α·MSE(stored_logits, current_logits) + β·CE(stored_labels, current_logits)
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


@register_method("der")
class DERpp(CILMethod):
    name = "DER++"

    def __init__(self, hsi_channels, lidar_channels, num_classes, device,
                 backbone="simple_encoder", d=128, epochs=50, lr=1e-3,
                 memory_size=2000, alpha=0.1, beta=0.5, **kwargs):
        encoder = build_backbone(backbone, hsi_ch=hsi_channels, lidar_ch=lidar_channels, d=d)
        super().__init__(encoder, device, num_classes)
        self.d = d
        self.epochs = epochs
        self.lr = lr
        self.memory_size = memory_size
        self.alpha = alpha
        self.beta = beta
        self.head = nn.Linear(d, num_classes).to(device)

        # Reservoir: (hsi, lidar, label, logits_snapshot)
        self._buf_hsi:    list[torch.Tensor] = []
        self._buf_lidar:  list[torch.Tensor] = []
        self._buf_labels: list[torch.Tensor] = []
        self._buf_logits: list[torch.Tensor] = []
        self._n_seen = 0

    def _reservoir_add(self, xh, xl, y, logits):
        """Reservoir sampling: each new sample has memory_size/n_seen chance to enter."""
        for i in range(len(y)):
            self._n_seen += 1
            if len(self._buf_hsi) < self.memory_size:
                self._buf_hsi.append(xh[i].cpu())
                self._buf_lidar.append(xl[i].cpu())
                self._buf_labels.append(y[i].cpu())
                self._buf_logits.append(logits[i].cpu().detach())
            else:
                idx = torch.randint(0, self._n_seen, (1,)).item()
                if idx < self.memory_size:
                    self._buf_hsi[idx]    = xh[i].cpu()
                    self._buf_lidar[idx]  = xl[i].cpu()
                    self._buf_labels[idx] = y[i].cpu()
                    self._buf_logits[idx] = logits[i].cpu().detach()

    def _sample_buffer(self, k: int):
        if not self._buf_hsi:
            return None
        n = len(self._buf_hsi)
        idx = torch.randperm(n)[:k]
        return (torch.stack([self._buf_hsi[i] for i in idx]),
                torch.stack([self._buf_lidar[i] for i in idx]),
                torch.stack([self._buf_labels[i] for i in idx]),
                torch.stack([self._buf_logits[i] for i in idx]))

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
                logits = self.head(feat)
                mapped = torch.tensor([g2l[yi.item()] for yi in y], device=self.device)
                loss = F.cross_entropy(logits[:, cids], mapped)

                buf = self._sample_buffer(min(len(y), 64))
                if buf is not None:
                    bh, bl, by, blog = [t.to(self.device) for t in buf]
                    b_feat = self.model(bh, bl)
                    b_logits = self.head(b_feat)
                    # α·MSE(old logits, current logits for old samples)
                    loss += self.alpha * F.mse_loss(b_logits, blog.to(self.device))
                    # β·CE(old labels, current logits)
                    bmapped = torch.tensor([g2l.get(yi.item(), 0) for yi in by],
                                           device=self.device)
                    valid = torch.tensor([yi.item() in g2l for yi in by], device=self.device)
                    if valid.any():
                        loss += self.beta * F.cross_entropy(
                            b_logits[valid][:, cids], bmapped[valid])

                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item(); n += 1

            if (ep + 1) % 10 == 0:
                print(f"    [DER++] Epoch {ep+1}/{self.epochs}  loss={total/n:.4f}")

    def after_task(self, task: Task, train_loader: DataLoader):
        # Update reservoir once per task with final model's logits
        self.model.eval(); self.head.eval()
        with torch.no_grad():
            for xh, xl, y in train_loader:
                xh, xl, y = xh.to(self.device), xl.to(self.device), y.to(self.device)
                logits = self.head(self.model(xh, xl))
                self._reservoir_add(xh, xl, y, logits)

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
            "_buf_logits": [t.cpu() for t in self._buf_logits],
            "_n_seen": self._n_seen,
        }

    def _load_method_state(self, ckpt: dict):
        self.head.load_state_dict(ckpt["head"])
        self._buf_hsi = ckpt["_buf_hsi"]
        self._buf_lidar = ckpt["_buf_lidar"]
        self._buf_labels = ckpt["_buf_labels"]
        self._buf_logits = ckpt["_buf_logits"]
        self._n_seen = ckpt["_n_seen"]
