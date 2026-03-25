"""EWC — Elastic Weight Consolidation (Kirkpatrick et al., PNAS 2017).

After each task, estimate the Fisher information matrix diagonal
and add a quadratic penalty on weight changes away from the old optimum.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base import CILMethod, register_method
from .ncm import SimpleEncoder
from benchmark.protocols.cil import Task


@register_method("ewc")
class EWC(CILMethod):
    name = "EWC"

    def __init__(self, hsi_channels, lidar_channels, num_classes, device,
                 d=128, epochs=50, lr=1e-3, ewc_lambda=5000.0, **kwargs):
        encoder = SimpleEncoder(hsi_channels, lidar_channels, d)
        super().__init__(encoder, device, num_classes)
        self.d = d
        self.epochs = epochs
        self.lr = lr
        self.ewc_lambda = ewc_lambda
        self.head = nn.Linear(d, num_classes).to(device)

        # Accumulated Fisher diagonals and optimal params per task
        self._fisher:  list[dict[str, torch.Tensor]] = []
        self._opt_params: list[dict[str, torch.Tensor]] = []

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
                logits = self.head(self.model(xh, xl))[:, cids]
                mapped = torch.tensor([g2l[yi.item()] for yi in y], device=self.device)
                loss_ce = F.cross_entropy(logits, mapped)

                # EWC penalty
                loss_ewc = torch.tensor(0.0, device=self.device)
                for fisher, opt_p in zip(self._fisher, self._opt_params):
                    for name, p in list(self.model.named_parameters()) + \
                                   list(self.head.named_parameters()):
                        if name in fisher:
                            loss_ewc += (fisher[name] *
                                         (p - opt_p[name].to(self.device)).pow(2)).sum()

                loss = loss_ce + self.ewc_lambda * loss_ewc
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item(); n += 1

            if (ep + 1) % 10 == 0:
                print(f"    [EWC] Epoch {ep+1}/{self.epochs}  loss={total/n:.4f}")

    def after_task(self, task: Task, train_loader: DataLoader):
        """Compute Fisher information diagonal on the current task data."""
        self.model.eval(); self.head.eval()
        cids = sorted(self.seen_classes)
        g2l = {g: i for i, g in enumerate(cids)}

        fisher = {n: torch.zeros_like(p)
                  for n, p in list(self.model.named_parameters()) +
                               list(self.head.named_parameters())}

        for xh, xl, y in train_loader:
            xh, xl, y = xh.to(self.device), xl.to(self.device), y.to(self.device)
            self.model.zero_grad(); self.head.zero_grad()
            logits = self.head(self.model(xh, xl))[:, cids]
            mapped = torch.tensor([g2l[yi.item()] for yi in y], device=self.device)
            loss = F.cross_entropy(logits, mapped)
            loss.backward()
            for n, p in list(self.model.named_parameters()) + \
                        list(self.head.named_parameters()):
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2) / len(train_loader)

        self._fisher.append({n: f.detach() for n, f in fisher.items()})
        self._opt_params.append({
            n: p.detach().clone().cpu()
            for n, p in list(self.model.named_parameters()) +
                        list(self.head.named_parameters())
        })

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
            "_fisher": [{n: f.cpu() for n, f in fd.items()} for fd in self._fisher],
            "_opt_params": [{n: p.cpu() for n, p in od.items()} for od in self._opt_params],
        }

    def _load_method_state(self, ckpt: dict):
        self.head.load_state_dict(ckpt["head"])
        self._fisher = ckpt["_fisher"]
        self._opt_params = ckpt["_opt_params"]
